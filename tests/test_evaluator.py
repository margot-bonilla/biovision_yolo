from pathlib import Path
import cv2
import numpy as np
import pytest

from chromoseg.engine.evaluator import compute_count_metrics, compute_overlap_metrics


def test_evaluator_metrics_structure(tmp_path: Path):
    """Test that evaluator output dictionary contains all required clinical metrics."""
    # Create mock validation structure
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # 1. Create a dummy image and label
    mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
    img_file = images_dir / "spread_0001.png"
    cv2.imwrite(str(img_file), mock_img)

    label_file = labels_dir / "spread_0001.txt"
    # 2 chromosome polygons in label
    label_file.write_text(
        "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"
        "0 0.6 0.6 0.8 0.6 0.8 0.8 0.6 0.8\n"
    )

    # Use pretrained weights to test inference pipeline
    weights_path = "weights/yolo11n-seg.pt"
    if not Path(weights_path).exists():
        pytest.skip("Pretrained weights not found at weights/yolo11n-seg.pt")

    # Run count metrics
    count_res = compute_count_metrics(
        weights=weights_path,
        val=str(images_dir),
        img_size=64,
    )

    assert "mae_count" in count_res
    assert "exact_count_acc" in count_res
    assert "within_1_count_acc" in count_res
    assert isinstance(count_res["mae_count"], (int, float))

    # Run overlap metrics
    overlap_res = compute_overlap_metrics(
        weights=weights_path,
        val=str(images_dir),
        labels_dir=str(labels_dir),
        img_size=64,
    )

    assert "isolated_mean_iou" in overlap_res
    assert "overlapping_mean_iou" in overlap_res
    assert "overlap_degradation_gap" in overlap_res
    assert "n_isolated" in overlap_res
    assert "n_overlapping" in overlap_res

