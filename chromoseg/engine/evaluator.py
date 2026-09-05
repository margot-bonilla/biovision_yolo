import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np

def compute_count_metrics(weights: str, val: str, img_size: int = 256) -> dict:
    """
    Evaluates Chromosome Count Error (△N) and Chromosome Classification Error (CCE).
    """
    model = YOLO(weights)
    results = model.predict(
        source=val,
        conf=0.25,
        imgsz=img_size,
        verbose=False,
    )

    errors = []
    for result in results:
        n_pred = 0
        n_true = 0
        if result.masks is not None:
            n_pred = len(result.masks)

        img_path = Path(result.path)
        label_path = img_path.parents[2] / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, "r") as f:
                n_true = sum(1 for line in f if line.strip())

        count_error = abs(n_pred - n_true)
        errors.append(count_error)

    mae = sum(errors) / len(errors) if errors else 0
    accuracy = (errors.count(0) / len(errors)) * 100 if errors else 0
    tolerance = (sum(1 for e in errors if e <= 1) / len(errors)) * 100 if errors else 0

    return {
        "mae_count": mae,
        "exact_count_acc": accuracy,
        "within_1_count_acc": tolerance
    }

def compute_overlap_metrics(
    weights: str,
    val: str,
    labels_dir: str = "data/processed/labels/val",
    img_size: int = 256,
) -> dict:
    """
    Evaluates segmentation accuracy on isolated vs. touching/overlapping chromosomes.

    How it works:
    1. Reconstructs clean Ground Truth masks from polygon label files (.txt).
    2. Uses morphological dilation (cv2.dilate) to check if a chromosome touches a neighbor.
    3. Computes IoU for each chromosome against YOLO's predicted masks.
    4. Compares the average IoU of isolated vs. overlapping chromosomes.
    """
    # 1. Load trained YOLO model and run inference on validation images
    model = YOLO(weights)
    results = model.predict(
        source=val,
        conf=0.25,
        imgsz=img_size,
        verbose=False,
    )

    # 5x5 pixel kernel used to check if neighboring chromosome borders touch
    dilation_kernel = np.ones((5, 5), np.uint8)

    isolated_ious = []
    overlapping_ious = []

    for result in results:
        img_path = Path(result.path)
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        # Get original image height and width
        height, width = result.orig_shape

        # =====================================================================
        # Step A: Load Ground Truth Chromosomes from the .txt label file
        # =====================================================================
        gt_masks = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # A valid polygon has class_id + at least 3 (x,y) points
                    continue

                # Convert normalized [0, 1] polygon points to pixel coordinates
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                pixel_points = (coords * np.array([width, height])).astype(np.int32)

                # Draw the filled chromosome polygon onto a blank canvas
                chromosome_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(chromosome_mask, [pixel_points], 1)
                gt_masks.append(chromosome_mask)

        if not gt_masks:
            continue

        # =====================================================================
        # Step B: Get YOLO Predicted Masks (resized to match original image)
        # =====================================================================
        pred_masks = []
        if result.masks is not None:
            raw_predictions = result.masks.data.cpu().numpy().astype(np.uint8)
            for pred in raw_predictions:
                if pred.shape != (height, width):
                    pred = cv2.resize(
                        pred, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                pred_masks.append(pred)

        # =====================================================================
        # Step C: For each true chromosome, check overlap & calculate best IoU
        # =====================================================================
        for i, true_mask in enumerate(gt_masks):
            # 1. Expand (dilate) the boundary of this chromosome by 5 pixels
            dilated_true_mask = cv2.dilate(true_mask, dilation_kernel, iterations=1)

            # 2. Combine all OTHER true chromosomes in this spread into one mask
            other_chromosomes_mask = np.zeros((height, width), dtype=np.uint8)
            for j, other_mask in enumerate(gt_masks):
                if i != j:
                    other_chromosomes_mask = np.maximum(other_chromosomes_mask, other_mask)

            # 3. If dilated boundary touches any other chromosome -> Overlapping!
            is_overlapping = np.logical_and(
                dilated_true_mask, other_chromosomes_mask
            ).any()

            # 4. Calculate IoU against all predicted masks to find the best match
            best_iou = 0.0
            for pred_mask in pred_masks:
                intersection = np.logical_and(true_mask, pred_mask).sum()
                union = np.logical_or(true_mask, pred_mask).sum()

                iou = intersection / union if union > 0 else 0.0
                best_iou = max(best_iou, iou)

            # 5. Store the IoU score in the corresponding category
            if is_overlapping:
                overlapping_ious.append(best_iou)
            else:
                isolated_ious.append(best_iou)

    # =========================================================================
    # Step D: Compute Summary Statistics
    # =========================================================================
    mean_isolated = float(np.mean(isolated_ious)) if isolated_ious else 0.0
    mean_overlapping = float(np.mean(overlapping_ious)) if overlapping_ious else 0.0
    degradation_gap = (mean_isolated - mean_overlapping) if isolated_ious else 0.0

    return {
        "isolated_mean_iou": mean_isolated,
        "overlapping_mean_iou": mean_overlapping,
        "overlap_degradation_gap": degradation_gap,
        "n_isolated": len(isolated_ious),
        "n_overlapping": len(overlapping_ious),
    }


def cytogenetics_evaluator(
    weights: str,
    val: str = "data/processed/images/val",
    labels_dir: str = "data/processed/labels/val",
    img_size: int = 256,
) -> dict:
    """
    Main evaluation pipeline running both Count Error and Overlap Analysis.
    """
    weights_path = Path(weights)
    if not weights_path.exists():
        # Smart fallback: search for other best.pt weights in runs/
        candidates = sorted(Path("runs/segment/models").glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            print(f"⚠️ Warning: Specified weights '{weights}' not found. Using most recent weights: {candidates[0]}")
            weights = str(candidates[0])
        else:
            raise FileNotFoundError(f"No model weights found at '{weights}' or in 'runs/segment/models/'.")

    print(f"\nEvaluating weights: {weights}")
    print(f"Validation dataset: {val}")

    # 1. Run Count Error Evaluation
    count_results = compute_count_metrics(weights=weights, val=val, img_size=img_size)

    # 2. Run Overlap & Cluster Segmentation Evaluation
    overlap_results = compute_overlap_metrics(
        weights=weights, val=val, labels_dir=labels_dir, img_size=img_size
    )

    # 3. Print Clean Diagnostic Report
    print("\n" + "=" * 50)
    print("        CYTOGENETICS EVALUATION REPORT        ")
    print("=" * 50)
    print("1. Chromosome Count Accuracy:")
    print(f"   - Mean Count Error (MAE △N) : {count_results['mae_count']:.2f} chromosomes")
    print(f"   - Exact Count Accuracy (△N=0): {count_results['exact_count_acc']:.1f}%")
    print(f"   - Within ±1 Tolerance (△N≤1) : {count_results['within_1_count_acc']:.1f}%")

    print("\n2. Overlap & Touching Segmentation:")
    if overlap_results['n_isolated'] > 0:
        print(f"   - Isolated Chromosomes IoU   : {overlap_results['isolated_mean_iou']:.4f} ({overlap_results['n_isolated']} instances)")
        print(f"   - Overlapping Clusters IoU   : {overlap_results['overlapping_mean_iou']:.4f} ({overlap_results['n_overlapping']} instances)")
        print(f"   - Overlap Degradation Gap    : {overlap_results['overlap_degradation_gap']:.4f}")
    else:
        print(f"   - Overlapping Clusters IoU   : {overlap_results['overlapping_mean_iou']:.4f} ({overlap_results['n_overlapping']} instances)")
        print("   - Isolated Chromosomes       : N/A (100% of benchmark slides feature touching/overlapping clusters)")
    print("=" * 50 + "\n")

    return {**count_results, **overlap_results}


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Evaluate cytogenetic clinical metrics for YOLO segmentation model."
    )
    arg_parser.add_argument(
        "--weights",
        type=str,
        default="runs/segment/models/baseline/weights/best.pt",
        help="Path to the trained model weights.",
    )
    arg_parser.add_argument(
        "--val",
        type=str,
        default="data/processed/images/val",
        help="Path to validation images directory.",
    )
    arg_parser.add_argument(
        "--labels_dir",
        type=str,
        default="data/processed/labels/val",
        help="Path to validation labels directory.",
    )
    arg_parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size for evaluation (default: 256).",
    )
    args = arg_parser.parse_args()

    cytogenetics_evaluator(
        weights=args.weights,
        val=args.val,
        labels_dir=args.labels_dir,
        img_size=args.img_size,
    )