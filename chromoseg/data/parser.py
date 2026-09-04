import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_to_yolo(image: np.ndarray, mask: np.ndarray) -> list:
    """
    Extracts 2 distinct classes into YOLO normalized polygon format:
      - Class 0: 'chromosome' (Individual chromosome contours from Channel 0 & Channel 1)
      - Class 1: 'overlap' (Dense crossover / overlap junction from Channel 0 & Channel 1 intersection)
    """
    yolo_lines = []
    height, width = mask.shape[:2]

    # Handle multi-channel vs single-channel masks
    if mask.ndim == 3 and mask.shape[2] >= 2:
        # Channel 0: Chromosome 1, Channel 1: Chromosome 2
        c1_binary = (mask[:, :, 0] < 200).astype(np.uint8) * 255
        c2_binary = (mask[:, :, 1] < 200).astype(np.uint8) * 255
        overlap_binary = ((c1_binary > 0) & (c2_binary > 0)).astype(np.uint8) * 255

        # Class 0: Chromosome 1 contours
        cnts1, _ = cv2.findContours(c1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts1:
            if cv2.contourArea(contour) < 50:
                continue
            norm_pts = contour.reshape(-1, 2) / np.array([width, height])
            poly_str = " ".join([f"{p:.6f}" for p in norm_pts.flatten()])
            yolo_lines.append(f"0 {poly_str}")

        # Class 0: Chromosome 2 contours
        cnts2, _ = cv2.findContours(c2_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts2:
            if cv2.contourArea(contour) < 50:
                continue
            norm_pts = contour.reshape(-1, 2) / np.array([width, height])
            poly_str = " ".join([f"{p:.6f}" for p in norm_pts.flatten()])
            yolo_lines.append(f"0 {poly_str}")

        # Class 1: Overlap Junction contours
        cnts_ov, _ = cv2.findContours(overlap_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts_ov:
            if cv2.contourArea(contour) < 20:
                continue
            norm_pts = contour.reshape(-1, 2) / np.array([width, height])
            poly_str = " ".join([f"{p:.6f}" for p in norm_pts.flatten()])
            yolo_lines.append(f"1 {poly_str}")

    else:
        # Fallback for single-channel masks
        gray_mask = mask[:, :, 0] if mask.ndim == 3 else mask
        chrom_binary = (gray_mask < 200).astype(np.uint8) * 255
        chrom_contours, _ = cv2.findContours(
            chrom_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in chrom_contours:
            if cv2.contourArea(contour) < 50:
                continue
            norm_pts = contour.reshape(-1, 2) / np.array([width, height])
            poly_str = " ".join([f"{p:.6f}" for p in norm_pts.flatten()])
            yolo_lines.append(f"0 {poly_str}")

    return yolo_lines


def process_dataset(images_dir: str, masks_dir: str, output_dir: str):
    """
    Process the dataset by converting masks to 2-class YOLO polygon format.

    Args:
        images_dir (str): Path to the directory containing raw images.
        masks_dir (str): Path to the directory containing masks.
        output_dir (str): Path to the output directory for YOLO polygon annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_paths = list(Path(masks_dir).glob("*.png"))

    print(f"Processing {len(mask_paths)} masks to 2-class YOLO format (0: chromosome, 1: overlap)...")

    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        try:
            # Read multi-channel mask preserving all channels
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            img_path = Path(images_dir) / mask_path.name
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

            if mask is None or image is None:
                continue

            yolo_lines = parse_to_yolo(image=image, mask=mask)
            if yolo_lines:
                output_mask_path = Path(output_dir) / mask_path.with_suffix(".txt").name
                with open(output_mask_path, "w") as f:
                    f.write("\n".join(yolo_lines))
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
