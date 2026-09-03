"""Streamlit demo: upload an MRI slice, run every trained model, compare detections."""
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Brain Tumor YOLO Comparison", layout="wide")
st.title("Brain Tumor Detection: YOLO Model Comparison")
st.caption("Not a diagnostic tool. Research comparison only.")


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


model_paths = sorted(MODELS_DIR.glob("*.pt"))
if not model_paths:
    st.error(f"No .pt weights found in {MODELS_DIR}")
    st.stop()

conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
uploaded = st.file_uploader("Upload an MRI slice", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    cols = st.columns(len(model_paths))

    for col, weights_path in zip(cols, model_paths):
        model_name = weights_path.stem.removesuffix("_best")
        with col:
            st.subheader(model_name)
            model = load_model(str(weights_path))
            result = model.predict(image, conf=conf, verbose=False)[0]
            st.image(result.plot()[:, :, ::-1], use_container_width=True)
            if result.boxes:
                for box in result.boxes:
                    label = result.names[int(box.cls)]
                    st.text(f"{label}: {float(box.conf):.2f}")
            else:
                st.text("No detections")
else:
    st.info("Upload an image to run all trained models.")
