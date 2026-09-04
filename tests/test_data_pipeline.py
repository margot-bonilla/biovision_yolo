from pathlib import Path
import numpy as np
import pytest

from chromoseg.data.extract_npz import extract_dataset
from chromoseg.data.parser import parse_to_yolo
from chromoseg.data.split import create_split


def test_extract_npz(tmp_path: Path):
    """Test extracting images and masks from a mock NPZ file."""
    npz_file = tmp_path / "mock_chromosomes.npz"
    mock_images = np.random.randint(0, 255, size=(2, 64, 64), dtype=np.uint8)
    mock_masks = np.zeros((2, 64, 64), dtype=np.uint8)
    mock_masks[0, 10:25, 10:25] = 1
    mock_masks[1, 20:35, 20:35] = 1

    np.savez_compressed(npz_file, data=mock_images, mask=mock_masks)

    output_dir = tmp_path / "data"
    extract_dataset(npz_path=str(npz_file), output_dir=str(output_dir))

    images_dir = output_dir / "raw" / "images"
    masks_dir = output_dir / "raw" / "masks"

    assert images_dir.exists()
    assert masks_dir.exists()
    assert len(list(images_dir.glob("*.png"))) == 2
    assert len(list(masks_dir.glob("*.png"))) == 2


def test_parser_area_filter_and_normalization():
    """Test that parser extracts valid chromosomes and normalizes coordinates to [0, 1]."""
    # Background is white (255)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    image = np.full((100, 100), 255, dtype=np.uint8)

    # Add dark chromosome (area = 20x20 = 400 pixels >= 100 threshold)
    mask[20:40, 20:40] = 0
    image[20:40, 20:40] = 100

    polygons = parse_to_yolo(image=image, mask=mask)

    assert len(polygons) >= 1

    parts = polygons[0].split()
    assert parts[0] == "0"  # Class ID is 0 for chromosome

    coords = [float(x) for x in parts[1:]]
    assert len(coords) >= 6  # At least 3 (x,y) points for a valid polygon

    # Check that all coordinates are normalized within [0, 1]
    for c in coords:
        assert 0.0 <= c <= 1.0


def test_parser_2class_chromosomes_and_overlap():
    """Test extracting 2 distinct chromosomes (Class 0) and their overlap junction (Class 1)."""
    # 3-channel mask, 100x100
    mask = np.full((100, 100, 3), 255, dtype=np.uint8)
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    # Chromosome 1 in Channel 0: [20:60, 20:40] (area = 800 px)
    mask[20:60, 20:40, 0] = 50
    # Chromosome 2 in Channel 1: [30:50, 20:60] (area = 800 px)
    mask[30:50, 20:60, 1] = 50
    # Overlap intersection: [30:50, 20:40] (area = 400 px)

    polygons = parse_to_yolo(image=image, mask=mask)

    # Should contain 2 chromosomes (Class 0) and 1 overlap (Class 1)
    class_0 = [p for p in polygons if p.startswith("0 ")]
    class_1 = [p for p in polygons if p.startswith("1 ")]

    assert len(class_0) == 2
    assert len(class_1) == 1


def test_split_dataset_no_leakage(tmp_path: Path):
    """Test deterministic split with zero data leakage and 1:1 image-label matching."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "processed"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create 10 mock image-label pairs
    for i in range(10):
        stem = f"spread_{i:04d}"
        (images_dir / f"{stem}.png").write_bytes(b"dummy image")
        (labels_dir / f"{stem}.txt").write_text("0 0.1 0.1 0.2 0.2 0.1 0.2\n")

    create_split(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        output_dir=str(output_dir),
        train_ratio=0.8,
    )

    train_imgs = sorted([p.stem for p in (output_dir / "images" / "train").glob("*.png")])
    val_imgs = sorted([p.stem for p in (output_dir / "images" / "val").glob("*.png")])

    train_lbls = sorted([p.stem for p in (output_dir / "labels" / "train").glob("*.txt")])
    val_lbls = sorted([p.stem for p in (output_dir / "labels" / "val").glob("*.txt")])

    # Check 80/20 split
    assert len(train_imgs) == 8
    assert len(val_imgs) == 2

    # Check 1:1 image to label alignment
    assert train_imgs == train_lbls
    assert val_imgs == val_lbls

    # Verify zero data leakage (disjoint sets)
    assert set(train_imgs).isdisjoint(set(val_imgs))

