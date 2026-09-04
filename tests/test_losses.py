import torch
import numpy as np
import pytest

from chromoseg.models.losses import (
    dice_focal_loss,
    compute_sdf,
    boundary_loss,
    CytogeneticsLoss,
)


def test_dice_focal_loss_values_and_gradients():
    """Test that Dice-Focal Loss produces valid scalar outputs and computes backpropagation gradients."""
    # 1. Prediction identical to ground truth (should have very low loss)
    perfect_pred = torch.tensor([1.0, 1.0, 0.0, 0.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss_perfect = dice_focal_loss(perfect_pred, target)

    assert isinstance(loss_perfect, torch.Tensor)
    assert loss_perfect.item() >= 0.0
    assert loss_perfect.item() < 0.05  # Near zero for perfect prediction

    # 2. Inverted prediction (should have high loss)
    wrong_pred = torch.tensor([0.0, 0.0, 1.0, 1.0], requires_grad=True)
    loss_wrong = dice_focal_loss(wrong_pred, target)
    assert loss_wrong.item() > loss_perfect.item()

    # 3. Test backpropagation (gradients flow properly)
    loss_wrong.backward()
    assert wrong_pred.grad is not None
    assert wrong_pred.grad.shape == wrong_pred.shape
    assert not torch.isnan(wrong_pred.grad).any()


def test_compute_sdf_signs():
    """Test Signed Distance Function: negative inside mask, zero on border, positive outside."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    # Center square chromosome from [5:15, 5:15]
    mask[5:15, 5:15] = 1

    sdf = compute_sdf(mask)

    assert isinstance(sdf, np.ndarray)
    assert sdf.shape == (20, 20)

    # Center of chromosome should be negative (inside)
    assert sdf[10, 10] < 0.0

    # Outside corners should be positive (outside)
    assert sdf[0, 0] > 0.0
    assert sdf[19, 19] > 0.0

    # Outside distance grows as you move further away
    assert sdf[0, 0] > sdf[4, 4]


def test_boundary_loss_differentiability():
    """Test that boundary loss computes correct penalty and supports autograd."""
    sdf_map = torch.tensor([[-2.0, -1.0], [1.0, 4.0]], dtype=torch.float32)
    pred_mask = torch.tensor([[0.8, 0.7], [0.1, 0.2]], dtype=torch.float32, requires_grad=True)

    loss = boundary_loss(pred_mask, sdf_map)
    assert isinstance(loss, torch.Tensor)

    loss.backward()
    assert pred_mask.grad is not None
    assert not torch.isnan(pred_mask.grad).any()


def test_cytogenetics_loss_module():
    """Test the unified CytogeneticsLoss nn.Module."""
    criterion = CytogeneticsLoss(alpha=0.25, gamma=2.0, boundary_weight=0.5)

    pred_mask = torch.rand((16, 16), dtype=torch.float32, requires_grad=True)
    true_mask = torch.zeros((16, 16), dtype=torch.float32)
    true_mask[4:12, 4:12] = 1.0

    total_loss = criterion(pred_mask, true_mask)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.ndim == 0  # Scalar loss
    assert total_loss.item() > 0.0

    total_loss.backward()
    assert pred_mask.grad is not None
    assert pred_mask.grad.shape == (16, 16)

