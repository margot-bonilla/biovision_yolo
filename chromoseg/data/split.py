import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def create_split(
    images_dir: str, labels_dir: str, output_dir: str, train_ratio: float = 0.8
):
    """
    Create a train/validation split of the dataset.

    Args:
        images_dir (str): Path to the directory containing raw images.
        labels_dir (str): Path to the directory containing raw labels.
        output_dir (str): Path to the output directory where the split dataset will be saved.
        train_ratio (float): Ratio of training data. The rest will be used for validation.
    """

    label_paths = list(Path(labels_dir).glob("*.txt"))
    if not label_paths:
        print(
            f"No label files found in {labels_dir}. "
            f"Please ensure that the directory contains .txt files."
        )
        return

    print(
        f"Found {len(label_paths)} label files in {labels_dir}. Proceeding with dataset split."
    )

    # Create output directories for train and validation sets
    train_images_dir = Path(output_dir) / "images" / "train"
    val_images_dir = Path(output_dir) / "images" / "val"
    train_labels_dir = Path(output_dir) / "labels" / "train"
    val_labels_dir = Path(output_dir) / "labels" / "val"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for img_path in sorted(Path(images_dir).glob("*.png")):
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            # Create empty label file for negative/background samples
            label_path.touch()
            pairs.append((img_path, label_path))

    if not pairs:
        print(
            f"No valid image/label pairs found between {images_dir} and {labels_dir}."
        )
        return

    # Shuffle the dataset
    random.seed(42)
    random.shuffle(pairs)

    # Split into training and validation sets
    split_index = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    # Copy files to respective directories
    for img_file, label_file in tqdm(train_pairs, desc="Copying training files"):
        shutil.copy(img_file, train_images_dir / img_file.name)
        shutil.copy(label_file, train_labels_dir / label_file.name)

    for img_file, label_file in tqdm(val_pairs, desc="Copying validation files"):
        shutil.copy(img_file, val_images_dir / img_file.name)
        shutil.copy(label_file, val_labels_dir / label_file.name)

    print(
        f"Dataset split completed. Training set: {len(train_pairs)} images, "
        f"Validation set: {len(val_pairs)} images."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a train/validation split of the dataset."
    )
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Path to raw images directory."
    )
    parser.add_argument(
        "--labels_dir", type=str, required=True, help="Path to raw labels directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for the split dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8).",
    )

    args = parser.parse_args()
    create_split(args.images_dir, args.labels_dir, args.output_dir, args.train_ratio)
