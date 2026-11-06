import pytest
import torch

from src.losses import BCEDiceLoss, DiceLoss
from src.model import UNet, count_parameters


@pytest.mark.parametrize("size", [64, 384, 512])
def test_output_shape_matches_input(size):
    model = UNet(base_channels=4).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, size, size))
    assert out.shape == (2, 1, size, size)


def test_default_parameter_count_is_within_the_target_range():
    assert 2_000_000 <= count_parameters(UNet()) <= 8_000_000


def test_input_not_divisible_by_16_gives_a_clear_error():
    with pytest.raises(ValueError, match="divisible by 16"):
        UNet(base_channels=4)(torch.zeros(1, 1, 100, 100))


def test_every_parameter_receives_a_gradient():
    model = UNet(base_channels=4)
    loss = BCEDiceLoss()(model(torch.randn(2, 1, 32, 32)), torch.ones(2, 1, 32, 32))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_dice_loss_is_near_zero_for_a_perfect_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1
    logits = (target * 2 - 1) * 20
    assert DiceLoss()(logits, target).item() == pytest.approx(0.0, abs=1e-3)


def test_dice_loss_is_near_one_for_a_completely_wrong_prediction():
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:24] = 1
    logits = (1 - target * 2) * 20
    assert DiceLoss()(logits, target).item() > 0.95


def test_dice_loss_on_an_empty_page_rewards_an_empty_prediction():
    target = torch.zeros(1, 1, 16, 16)
    empty = DiceLoss()(torch.full_like(target, -20.0), target).item()
    full = DiceLoss()(torch.full_like(target, 20.0), target).item()
    assert empty == pytest.approx(0.0, abs=1e-3)
    assert full > 0.9


def test_dice_is_not_fooled_by_predicting_all_background_on_imbalanced_data():
    target = torch.zeros(1, 1, 64, 64)
    target[..., :4, :4] = 1  # Foreground is about 0.4% of pixels.
    logits = torch.full_like(target, -6.0)
    assert torch.nn.BCEWithLogitsLoss()(logits, target).item() < 0.05
    assert DiceLoss()(logits, target).item() > 0.9
