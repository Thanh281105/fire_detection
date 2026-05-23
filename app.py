#!/usr/bin/env python3
"""Streamlit inference UI for the final fire/smoke YOLO11 detector."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path(
    "weights/fire-detection-yolo11-stage12/"
    "runs/final/yolo11x_detect_fire_smoke_l4_finetune/weights/best.pt"
)
SAMPLE_DIR = Path("reports/sliced_eval/stage2_test/samples")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISPLAY_NAMES = {0: "fire_smoke"}


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def list_sample_images() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(path for path in SAMPLE_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def read_uploaded_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    return image.convert("RGB")


def read_sample_image(sample_path: Path) -> Image.Image:
    with Image.open(sample_path) as image:
        return image.convert("RGB")


def extract_detections(result) -> list[dict[str, object]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections: list[dict[str, object]] = []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for idx, (box, conf, cls_id) in enumerate(zip(xyxy, confs, classes), start=1):
        class_name = DISPLAY_NAMES.get(int(cls_id), result.names.get(int(cls_id), str(cls_id)))
        x1, y1, x2, y2 = [float(value) for value in box]
        detections.append(
            {
                "#": idx,
                "class": class_name,
                "confidence": round(float(conf), 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )
    return detections


def draw_detections(image: Image.Image, detections: list[dict[str, object]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    line_width = max(2, min(image.size) // 240)

    for detection in detections:
        x1 = float(detection["x1"])
        y1 = float(detection["y1"])
        x2 = float(detection["x2"])
        y2 = float(detection["y2"])
        label = f'{detection["class"]} {detection["confidence"]:.2f}'

        draw.rectangle((x1, y1, x2, y2), outline=(230, 45, 45), width=line_width)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle((x1, label_y, x1 + text_w + 8, label_y + text_h + 6), fill=(230, 45, 45))
        draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)

    return annotated


def main() -> None:
    st.set_page_config(page_title="Fire/Smoke YOLO11 Inference", layout="wide")
    st.title("Fire/Smoke YOLO11 Inference")

    env_model_path = os.getenv("MODEL_PATH")
    model_path = Path(env_model_path) if env_model_path else DEFAULT_MODEL_PATH

    with st.sidebar:
        st.header("Model")
        model_path_text = st.text_input("Checkpoint", value=str(model_path))
        model_path = Path(model_path_text)

        st.header("Inference")
        conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
        iou = st.slider("NMS IoU", 0.0, 1.0, 0.70, 0.01)
        imgsz = st.select_slider("Image size", options=[320, 480, 640, 768, 1024, 1280], value=1024)
        max_det = st.number_input("Max detections", min_value=1, max_value=1000, value=300, step=10)
        device = st.text_input("Device", value="cpu")

    if not model_path.exists():
        st.error(f"Checkpoint not found: {model_path}")
        st.stop()

    with st.spinner("Loading model..."):
        model = load_model(str(model_path))

    samples = list_sample_images()
    source = st.radio("Image source", ["Upload", "Sample"], horizontal=True)

    image: Image.Image | None = None
    caption = ""

    if source == "Upload":
        uploaded_file = st.file_uploader("Upload image", type=sorted(ext.lstrip(".") for ext in IMAGE_EXTS))
        if uploaded_file is not None:
            image = read_uploaded_image(uploaded_file)
            caption = uploaded_file.name
    else:
        if not samples:
            st.warning(f"No sample images found in {SAMPLE_DIR}")
            st.stop()
        sample_path = st.selectbox("Sample image", samples, format_func=lambda path: path.name)
        image = read_sample_image(sample_path)
        caption = sample_path.name

    if image is None:
        st.stop()

    with st.spinner("Running inference..."):
        result = model.predict(
            np.array(image),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=int(max_det),
            device=device.strip() or "cpu",
            verbose=False,
        )[0]

    detections = extract_detections(result)
    annotated = draw_detections(image, detections)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Detections", len(detections))
    metric_cols[1].metric("Image", f"{image.width} x {image.height}")
    metric_cols[2].metric("Checkpoint", model_path.name)

    image_cols = st.columns(2)
    image_cols[0].image(image, caption=f"Input: {caption}", width="stretch")
    image_cols[1].image(annotated, caption="Prediction", width="stretch")

    if detections:
        st.dataframe(detections, width="stretch", hide_index=True)
    else:
        st.info("No detections at the current threshold.")


if __name__ == "__main__":
    main()
