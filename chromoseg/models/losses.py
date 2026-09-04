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


def compute_sdf(true_mask: np.ndarray) -> np.ndarray:
    """
    Computes Signed Distance Function (SDF) for a binary mask
        - Negative: inside the chromosome
        - Positive: outside the chromosome
    """
    postmask = true_mask.astype(bool)
    negmask = ~postmask

    # Distance to the boundary from iside and outside
    out_dist = distance_transform_edt(negmask)
    in_dist = distance_transform_edt(postmask)

    # Combine into Signed Distance Map
    sdf = out_dist - in_dist

    return sdf

def boundary_loss(pred_mask: torch.Tensor, sdf_map: torch.Tensor):
    """
    Integral of predicted probabilities multiplied by distance map
    """

    return (pred_mask * sdf_map).mean()

class CytogeneticsLoss(torch.nn.Module):
    """
    Combined Cytogenetics Instance Segmentation Loss:
        L_total = L_Dice_Focal + (boundary_weight * L_boundary)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, boundary_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight

    def forward(self, pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
        # Compute Dice-Focla loss
        df_loss = dice_focal_loss(pred_mask=pred_mask, true_mask=true_mask, alpha=self.alpha, gamma=self.gamma)

        # Compute Signed Distance Function for the ground truth
        true_mask_np = true_mask.detach().cpu().numpy()
        sdf_np = compute_sdf(true_mask_np)
        sdf_tensor = torch.from_numpy(sdf_np).float().to(pred_mask.device)

        # Compute boundary loss
        b_loss = boundary_loss(pred_mask, sdf_tensor)

        return df_loss + self.boundary_weight * b_loss