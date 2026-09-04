import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

def dice_focal_loss(
        pred_mask: torch.Tensor, 
        true_mask: torch.Tensor, 
        alpha: float =0.25, 
        gamma: float =2.0, 
        smooth: float =1e-6
    ) -> torch.Tensor:
    """
    L_dice_focal = L_focal - L_dice
    
    - Focal Loss: Adds a focusing knowb (gamma) that excludes noisy background
    - Dice Loss: calculates the overlap percentage between the true mask and
        predicted mask directly
    """
    pred_flat = pred_mask.view(-1)
    target_flat = true_mask.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice_score = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    dice_loss = 1.0 - dice_score

    bce = F.binary_cross_entropy(pred_flat, target_flat, reduction="none")
    p_t = target_flat * pred_flat + (1 - target_flat) * (1 - pred_flat)
    alpha_factor = target_flat * alpha + (1 - target_flat) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma

    focal_loss = (alpha_factor * modulating_factor * bce).mean()
    total_loss = focal_loss + dice_loss

    return total_loss


def extract_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Pure PyTorch GPU-accelerated morphological boundary extraction.
    Uses Max Pooling to perform morphological dilation and erosion on device.
    """
    orig_shape = mask.shape
    if mask.ndim == 2:
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask_4d = mask.unsqueeze(1)
    else:
        mask_4d = mask

    pad = kernel_size // 2
    # Morphological Dilation on GPU
    dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=pad)
    # Morphological Erosion on GPU
    eroded = 1.0 - F.max_pool2d(1.0 - mask_4d, kernel_size=kernel_size, stride=1, padding=pad)

    # Boundary is the morphological gradient (Dilation - Erosion)
    boundary = dilated - eroded
    return boundary.view(orig_shape)


def boundary_loss(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Pure PyTorch GPU Boundary Loss:
    Measures boundary alignment between predicted and ground truth masks in microseconds.
    """
    pred_boundary = extract_boundary(pred_mask)
    true_boundary = extract_boundary(true_mask)

    intersection = (pred_boundary * true_boundary).sum()
    union = pred_boundary.sum() + true_boundary.sum()

    boundary_dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - boundary_dice


class CytogeneticsLoss(torch.nn.Module):
    """
    Combined Cytogenetics Instance Segmentation Loss:
        L_total = L_Dice_Focal + (boundary_weight * L_boundary)
    Fully accelerated on GPU/MPS with zero CPU transfers.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, boundary_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight

    def forward(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        # 1. Compute Dice-Focal loss
        df_loss = dice_focal_loss(pred_mask=pred_mask, true_mask=true_mask, alpha=self.alpha, gamma=self.gamma)

        # 2. Compute Pure GPU Boundary loss
        b_loss = boundary_loss(pred_mask=pred_mask, true_mask=true_mask)

        return df_loss + self.boundary_weight * b_loss