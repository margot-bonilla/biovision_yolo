import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_to_yolo(mask: np.ndarray, class_id: int = 0) -> list:
    """
    Convert a binary mask to YOLO polygon format.

    Args:
        mask (np.ndarray): Binary mask image.
        class_id (int): Class ID for the object in the mask.

    Returns:
        list: List of polygons in YOLO format.
    """

    yolo_lines = []
    height, width = mask.shape

    # 1. Find all unique pixel intensities in the image
    # (excluding background, assumed to be 0)
    unique_values = np.unique(mask)

    for value in unique_values:
        if value == 0 or value >= 255:
            continue

        # 2. Isolate the current chromosome
        instance_mask = np.uint8(mask == value) * 255

        # 3. Find contours of the isolated chromosome
        contours, _ = cv2.findContours(
            instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 4. For each contour, convert it to YOLO polygon format
        for contour in contours:
            # Filter out dust and artifacts
            if cv2.contourArea(contour) < 50:
                continue

            # Normalize the contour points to [0, 1] range
            normalized_contour = contour.reshape(-1, 2) / np.array([width, height])

            # Flatten the normalized contour points and convert to string
            polygon_str = " ".join(
                [f"{point:.6f}" for point in normalized_contour.flatten()]
            )

            # Create YOLO line with class_id and polygon points
            yolo_line = f"{class_id} {polygon_str}"
            yolo_lines.append(yolo_line)

    # Placeholder for conversion logic
    return yolo_lines


def process_dataset(images_dir: str, masks_dir: str, output_dir: str):
    """
    Process the dataset by converting binary masks to YOLO polygon format.

    Args:
        images_dir (str): Path to the directory containing raw images.
        masks_dir (str): Path to the directory containing binary masks.
        output_dir (str): Path to the output directory for YOLO polygon annotations.
    """
    # Placeholder for processing logic
    print(
        f"Processing dataset from {images_dir}, masks from {masks_dir}, "
        f"saving output to {output_dir}."
    )
    # Here you would implement the logic to convert binary masks to YOLO polygon format.

    os.makedirs(output_dir, exist_ok=True)
    mask_paths = list(Path(masks_dir).glob("*.png"))

    print(f"Found {len(mask_paths)} masks to process.")

    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}. Skipping.")
                continue
            yolo_lines = parse_to_yolo(mask)
            if yolo_lines:
                output_mask_path = Path(output_dir) / mask_path.with_suffix(".txt").name
                with open(output_mask_path, "w") as f:
                    f.write("\n".join(yolo_lines))
            else:
                print(
                    f"Warning: No valid contours found in mask {mask_path}. Skipping."
                )
        except Exception as e:
            print(f"Error processing mask {mask_path}: {e}")

    print(f"Processing completed. YOLO polygon annotations saved to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert binary masks to YOLO polygon format."
    )
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Path to raw images directory."
    )
    parser.add_argument(
        "--masks_dir", type=str, required=True, help="Path to binary masks directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for YOLO polygon annotations.",
    )

    args = parser.parse_args()
    process_dataset(args.images_dir, args.masks_dir, args.output_dir)
