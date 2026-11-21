import os
import re
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- App configuration ----------------
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 Skin Lesion Classifier")
st.write(
    "Upload a skin-lesion image for an educational Benign/Malignant model prediction."
)

# Your shared Google Drive file ID.
# You can override it in Streamlit Cloud > App settings > Secrets:
# MODEL_FILE_ID = "your_drive_file_id"
DEFAULT_MODEL_FILE_ID = "1QhyLc0K6_H8FoS0VvEPkxHT6DcurkWeA"
MODEL_FILE_ID = st.secrets.get("MODEL_FILE_ID", DEFAULT_MODEL_FILE_ID)

# The downloaded file is stored in the app's local writable directory.
MODEL_PATH = Path("best_skin_cancer_model.keras")


def download_from_google_drive(file_id: str, destination: Path) -> None:
    """Download a publicly shared Google Drive file without the gdown package."""
    url = "https://drive.google.com/uc"
    session = requests.Session()
    response = session.get(
        url,
        params={"export": "download", "id": file_id},
        stream=True,
        timeout=(30, 180),
    )
    response.raise_for_status()

    # Large Drive files may show a virus-scan confirmation page.
    confirm_token = None
    for cookie_name, cookie_value in session.cookies.items():
        if cookie_name.startswith("download_warning"):
            confirm_token = cookie_value
            break

    if confirm_token:
        response.close()
        response = session.get(
            url,
            params={
                "export": "download",
                "id": file_id,
                "confirm": confirm_token,
            },
            stream=True,
            timeout=(30, 180),
        )
        response.raise_for_status()

    # Google sometimes returns an HTML permission/error page instead of the file.
    content_type = response.headers.get("Content-Type", "").lower()
    first_chunk = next(response.iter_content(chunk_size=1024), b"")
    if "text/html" in content_type or b"<html" in first_chunk.lower():
        response.close()
        raise RuntimeError(
            "Google Drive returned a web page instead of the model file. "
            "Set the Drive file access to 'Anyone with the link – Viewer' "
            "and verify the file ID."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with temp_path.open("wb") as file_out:
            if first_chunk:
                file_out.write(first_chunk)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_out.write(chunk)
        if not temp_path.exists() or temp_path.stat().st_size < 1024:
            raise RuntimeError("Downloaded file is empty or too small to be a Keras model.")
        temp_path.replace(destination)
    finally:
        response.close()
        if temp_path.exists():
            temp_path.unlink()


@st.cache_resource(show_spinner=False)
def load_model_cached(file_id: str):
    """Download once per app process and cache the loaded Keras model."""
    if not MODEL_PATH.exists():
        with st.spinner("Downloading model from Google Drive (about 58 MB)..."):
            download_from_google_drive(file_id, MODEL_PATH)

    with st.spinner("Loading Keras model..."):
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


# ---------------- Model loading ----------------
model = None
try:
    model = load_model_cached(MODEL_FILE_ID)
    st.success("Model loaded successfully.")
except Exception as error:
    st.error("The model could not be downloaded or loaded.")
    st.exception(error)
    st.info(
        "Check that the Google Drive file is shared as "
        "'Anyone with the link – Viewer'. Also confirm the Drive file ID "
        "points to your .keras model."
    )

# ---------------- Image upload and prediction ----------------
uploaded_file = st.file_uploader(
    "Choose a skin image...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if model is None:
            st.warning("Prediction is unavailable until the model loads successfully.")
        else:
            # Keep the original app's 224x224 RGB and [0, 1] preprocessing.
            resized = image.resize((224, 224))
            image_array = np.asarray(resized, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0) / 255.0

            with st.spinner("Analyzing image..."):
                prediction = np.asarray(model.predict(image_array, verbose=0)).reshape(-1)

            if prediction.size == 0:
                raise ValueError("The model returned an empty prediction.")

            probability = float(prediction[0])
            if not np.isfinite(probability):
                raise ValueError("The model returned an invalid prediction value.")

            # This preserves the original app's sigmoid-output assumption.
            # >= 0.5 = Malignant; < 0.5 = Benign.
            probability = min(max(probability, 0.0), 1.0)
            if probability >= 0.5:
                label = "Malignant"
                confidence = probability * 100
                color = "red"
            else:
                label = "Benign"
                confidence = (1.0 - probability) * 100
                color = "green"

            st.markdown(f"### Prediction: :{color}[{label}]")
            st.markdown(f"### Confidence: **{confidence:.2f}%**")
            st.warning(
                "Educational use only. This result is not a medical diagnosis "
                "and must not replace assessment by a qualified clinician."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
