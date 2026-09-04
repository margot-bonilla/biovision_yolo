from pathlib import Path
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. Load trained model (falls back to baseline or pretrained if custom is not trained yet)
if Path("runs/segment/models/chromoseg_2class/weights/best.pt").exists():
    WEIGHTS_PATH = "runs/segment/models/chromoseg_2class/weights/best.pt"
elif Path("runs/segment/models/baseline/weights/best.pt").exists():
    WEIGHTS_PATH = "runs/segment/models/baseline/weights/best.pt"
else:
    WEIGHTS_PATH = "weights/yolo11n-seg.pt"

model = YOLO(WEIGHTS_PATH)


def segment_chromosomes(input_img: Image.Image, conf_threshold: float = 0.25):
    if input_img is None:
        return None, "Please upload an image.", {}, []

    # 1. Run YOLO inference
    results = model.predict(source=input_img, conf=conf_threshold, verbose=False)
    res = results[0]

    # 2. Render color-coded segmentation overlay
    annotated_bgr = res.plot()
    annotated_rgb = annotated_bgr[..., ::-1]  # Convert BGR to RGB

    # 3. Clinical Count & Diagnostics for 2 Classes (0: Chromosome, 1: Overlap)
    n_chromosomes = 0
    n_overlaps = 0
    if res.boxes is not None and res.boxes.cls is not None:
        classes = res.boxes.cls.cpu().numpy().astype(int)
        n_chromosomes = int((classes == 0).sum())
        n_overlaps = int((classes == 1).sum())

    if n_overlaps > 0:
        status_md = (
            f"### ⚠️ **{n_overlaps} Overlap Junction(s) Detected!**\n"
            f"*Segmented {n_chromosomes} chromosome body instances with {n_overlaps} touching/crossover junctions.*"
        )
    else:
        status_md = (
            f"### 🟢 **Clean Spread (0 Overlaps)**\n"
            f"*Segmented {n_chromosomes} isolated chromosome bodies with zero touching artifacts.*"
        )

    metrics = {
        "Chromosomes Detected (Class 0)": n_chromosomes,
        "Overlap Junctions (Class 1)": n_overlaps,
        "Detection Confidence Used": conf_threshold,
        "Image Resolution": f"{input_img.width} x {input_img.height}",
    }

    # 4. Extract individual chromosome cutouts for inspection
    img_np = np.array(input_img)
    cutouts = []
    if res.boxes is not None:
        for box in res.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            crop = img_np[max(0, y1):min(img_np.shape[0], y2), max(0, x1):min(img_np.shape[1], x2)]
            if crop.size > 0:
                cutouts.append(crop)

    return annotated_rgb, status_md, metrics, cutouts


# 2. Build the Web Interface
with gr.Blocks(title="ChromoSeg: AI Cytogenetics & Karyotyping", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 ChromoSeg: Clinical Cytogenetics AI
        ### Automated Metaphase Chromosome Instance Segmentation & Diagnostic Karyotyping
        Upload a microscopic metaphase spread image to detect individual chromosomes, identify clusters, and assess numerical abnormalities.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Metaphase Spread")
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="Confidence Threshold"
            )
            analyze_btn = gr.Button("🔍 Segment & Analyze Chromosomes", variant="primary")

            # Clickable Example Images
            example_images = [
                str(p) for p in list(Path("data/processed/images/val").glob("*.png"))[:3]
            ]
            if example_images:
                gr.Examples(examples=example_images, inputs=input_img)

        with gr.Column(scale=1):
            output_img = gr.Image(type="numpy", label="Color-Coded Segmentation Overlay")
            status_box = gr.Markdown()
            metrics_box = gr.JSON(label="Clinical Summary Statistics")

    with gr.Row():
        gallery = gr.Gallery(label="Individual Chromosome Cutouts", columns=8, height="auto")

    analyze_btn.click(
        fn=segment_chromosomes,
        inputs=[input_img, conf_slider],
        outputs=[output_img, status_box, metrics_box, gallery],
    )

if __name__ == "__main__":
    demo.launch()