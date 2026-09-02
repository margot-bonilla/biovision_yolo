import numpy as np
import cv2
import os
import argparse

from pathlib import Path

def extract_dataset(npz_path: str, output_dir: str):
    """
    Extracts images and masks from a .npz file and saves them to the specified output directory.

    Args:
        npz_path (str): Path to the .npz file.
        output_dir (str): Directory where the extracted images and masks will be saved.
    """
    # Load the .npz file
    data = np.load(npz_path)

    # Create output directories for images and masks
    images_dir = Path(output_dir) / "raw" / "images"
    masks_dir = Path(output_dir) / "raw" / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = data['images'] if "images" in data.files else data[data.files[0]]
        masks = data['masks'] if "masks" in data.files else data[data.files[1]]

        print(f'Found {len(images)} images and {len(masks)} masks in the .npz file.')

        # Extract and save images and masks
        for i in range(len(images)):
            image = images[i]
            mask = masks[i]

            # Save image
            image_path = os.path.join(images_dir, f"spread_{i:04d}.png")
            cv2.imwrite(str(image_path), image)

            # Save mask
            mask_path = os.path.join(masks_dir, f"spread_{i:04d}.png")
            cv2.imwrite(str(mask_path), mask)

        print(f"Extraction completed. Images saved to {images_dir}, masks saved to {masks_dir}.")

    except Exception as e:
        print(f"An error occurred while extracting the dataset: {e}")
    finally:
        data.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract images and masks from a .npz file.")
    parser.add_argument("--npz_path", type=str, help="Path to the .npz file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the extracted images and masks.")
    args = parser.parse_args()

    npz_file_path = args.npz_path
    output_directory = args.output_dir
    extract_dataset(npz_file_path, output_directory)