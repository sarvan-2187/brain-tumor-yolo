"""Streamlit demo: upload an MRI slice, run every trained model, compare detections."""
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from fpdf import FPDF
from PIL import Image, ImageOps
from ultralytics import YOLO    

MODELS_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Brain Tumor YOLO Comparison", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap');
    *:not([data-testid*="Icon"]):not([data-testid*="icon"]) { font-family: 'Geist', sans-serif !important; }
    [data-testid*="Icon"], [data-testid*="icon"] { font-family: 'Material Symbols Rounded' !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
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
    grid_cols = 3
    summary_rows = []
    annotated_images = {}

    for row_start in range(0, len(model_paths), grid_cols):
        row_paths = model_paths[row_start : row_start + grid_cols]
        cols = st.columns(grid_cols)
        for col, weights_path in zip(cols, row_paths):
            model_name = weights_path.stem.removesuffix("_best")
            with col:
                st.subheader(model_name)
                with st.spinner(f"Running {model_name}..."):
                    model = load_model(str(weights_path))
                    result = model.predict(image, conf=conf, verbose=False)[0]
                    annotated = Image.fromarray(result.plot()[:, :, ::-1])
                st.image(annotated, use_container_width=True)
                square = ImageOps.pad(annotated, (max(annotated.size),) * 2, color=(0, 0, 0))
                png_buf = io.BytesIO()
                square.save(png_buf, format="PNG")
                png_buf.seek(0)
                annotated_images[model_name] = png_buf
                if result.boxes:
                    for box in result.boxes:
                        label = result.names[int(box.cls)]
                        confidence = float(box.conf)
                        st.text(f"{label}: {confidence:.2f}")
                        summary_rows.append({"model": model_name, "class": label, "confidence": confidence})
                else:
                    st.text("No detections")
                    summary_rows.append({"model": model_name, "class": "No detections", "confidence": None})

    st.subheader("Summary")
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    def build_pdf_report() -> bytes:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Brain Tumor Detection: YOLO Model Comparison", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, "Not a diagnostic tool. Research comparison only.", ln=True)
        pdf.cell(0, 8, f"Confidence threshold: {conf:.2f}", ln=True)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for row in summary_rows:
            conf_str = f"{row['confidence']:.2f}" if row["confidence"] is not None else "-"
            pdf.cell(0, 6, f"{row['model']}: {row['class']} ({conf_str})", ln=True)

        pdf.add_page()
        cell_w = pdf.epw / grid_cols
        img_w = cell_w - 6
        for i, (model_name, annotated) in enumerate(annotated_images.items()):
            col_i = i % grid_cols
            if col_i == 0 and i > 0:
                pdf.ln(img_w + 8)
            if pdf.get_y() + img_w + 8 > pdf.page_break_trigger and col_i == 0:
                pdf.add_page()
            x = pdf.l_margin + col_i * cell_w
            y = pdf.get_y()
            pdf.set_xy(x, y)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(img_w, 6, model_name, ln=0)
            pdf.set_xy(x, y + 6)
            pdf.image(annotated, x=x, y=y + 6, w=img_w)

        return bytes(pdf.output())

    with st.spinner("Building PDF report..."):
        pdf_bytes = build_pdf_report()

    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name="brain_tumor_yolo_comparison.pdf",
        mime="application/pdf",
    )
else:
    st.info("Upload an image to run all trained models.")
