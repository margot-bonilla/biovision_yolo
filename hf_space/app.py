import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. Load trained YOLO segmentation model
model = YOLO("best.pt")


def _predict(input_img: Image.Image, conf_threshold: float = 0.25):
    """
    Runs YOLO inference on the uploaded spread and computes clinical karyotype stats.
    """
    if input_img is None:
        return None, "Please upload an image.", {}, []

    # A. Run YOLO prediction
    results = model.predict(source=input_img, conf=conf_threshold, verbose=False)
    res = results[0]

    # B. Render color-coded segmentation overlay
    annotated_bgr = res.plot()
    annotated_rgb = annotated_bgr[..., ::-1]  # Convert BGR (OpenCV) to RGB (PIL/Gradio)

    # C. Clinical Diagnostic Count & Interpretation
    n_count = len(res.masks) if res.masks is not None else 0

    if n_count == 46:
        status_md = (
            f"### 🟢 **Karyotype Status: Normal Diploid (2n = 46)**\n"
            f"*Standard human chromosome complement detected.*"
        )
    elif n_count == 47:
        status_md = (
            f"### 🔴 **Karyotype Status: Hyperdiploid / Trisomy Risk (2n = 47)**\n"
            f"*Extra chromosome detected. Detailed band analysis recommended.*"
        )
    elif n_count == 45:
        status_md = (
            f"### 🔴 **Karyotype Status: Hypodiploid / Monosomy Risk (2n = 45)**\n"
            f"*Missing chromosome detected. Potential numerical abnormality.*"
        )
    else:
        status_md = (
            f"### ⚪ **Karyotype Status: Partial / Clustered Spread (N = {n_count})**\n"
            f"*Metaphase spread detected with {n_count} segmented chromosome instances.*"
        )

    metrics = {
        "Total Chromosomes Detected": n_count,
        "Detection Confidence Used": conf_threshold,
        "Image Dimensions": f"{input_img.width} x {input_img.height}",
    }

    # D. Extract individual chromosome cutouts for cytogenetic inspection
    img_np = np.array(input_img)
    cutouts = []
    if res.boxes is not None:
        for box in res.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            crop = img_np[max(0, y1) : min(img_np.shape[0], y2), max(0, x1) : min(img_np.shape[1], x2)]
            if crop.size > 0:
                cutouts.append(crop)

    return annotated_rgb, status_md, metrics, cutouts


# Decorate with spaces.GPU if running on Hugging Face ZeroGPU
if has_spaces:
    segment_chromosomes = spaces.GPU(_predict)
else:
    segment_chromosomes = _predict


# 2. Build the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🔬 ChromoSeg: Clinical Cytogenetics AI
        ### Automated Metaphase Chromosome Instance Segmentation & Diagnostic Karyotyping
        Upload a microscopic metaphase spread image to detect individual chromosomes, separate touching clusters, and assess numerical abnormalities.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Metaphase Spread")
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.25, step=0.05, label="Detection Confidence"
            )
            analyze_btn = gr.Button("🔍 Segment & Analyze Chromosomes", variant="primary")

            # Clickable Example Slides (flat root filenames)
            gr.Examples(
                examples=[
                    ["spread_0135.png", 0.25],
                    ["spread_0309.png", 0.25],
                    ["spread_0464.png", 0.25],
                ],
                inputs=[input_img, conf_slider],
                label="Click an Example Slide to Test:",
            )

        with gr.Column(scale=1):
            output_img = gr.Image(type="numpy", label="Color-Coded Segmentation Overlay")
            status_box = gr.Markdown()
            metrics_box = gr.JSON(label="Clinical Summary Statistics")

    with gr.Row():
        gallery = gr.Gallery(label="Individual Segmented Chromosome Cutouts", columns=8, height="auto")

    analyze_btn.click(
        fn=segment_chromosomes,
        inputs=[input_img, conf_slider],
        outputs=[output_img, status_box, metrics_box, gallery],
    )

if __name__ == "__main__":
    demo.launch()
