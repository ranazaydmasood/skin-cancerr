import os
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -------------------- App configuration --------------------
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 Skin Lesion Classifier")
st.write(
    "Upload a clinical image of a skin lesion to classify it as "
    "**Benign** or **Malignant**."
)

# Google Drive file ID from your shared link.
# You can override it by setting the MODEL_FILE_ID environment variable.
MODEL_FILE_ID = os.getenv(
    "MODEL_FILE_ID",
    "1QhyLc0K6_H8FoS0VvEPkxHT6DcurkWeA",
)
MODEL_PATH = Path("best_skin_cancer_model.keras")


@st.cache_resource
def load_keras_model():
    """Download the model from Google Drive (if needed), then load it."""
    if not MODEL_PATH.exists():
        with st.spinner("Downloading the 58 MB model from Google Drive..."):
            downloaded_path = gdown.download(
                id=MODEL_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False,
                fuzzy=True,
            )

        if not downloaded_path or not MODEL_PATH.exists():
            raise RuntimeError(
                "Google Drive model download failed. Check that the file is "
                "'Anyone with the link' accessible and that the file ID is correct."
            )

    return tf.keras.models.load_model(str(MODEL_PATH))


# -------------------- Load model --------------------
model = None
try:
    model = load_keras_model()
    st.success("Model downloaded/loaded successfully!")
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info(
        "Make sure the Google Drive file is shared with "
        "'Anyone with the link – Viewer' access."
    )

# -------------------- Image upload and prediction --------------------
uploaded_file = st.file_uploader(
    "Choose a skin image...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        # Convert to RGB so the model always receives 3 color channels.
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if model is None:
            st.warning("Prediction is unavailable because the model did not load.")
        else:
            # Keep the preprocessing used in your original app.
            img_resized = img.resize((224, 224))
            img_array = np.asarray(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            with st.spinner("Analyzing image..."):
                prediction = model.predict(img_array, verbose=0)
                prediction_prob = float(np.asarray(prediction).reshape(-1)[0])

            # Existing app assumes a sigmoid output:
            # >= 0.5 = Malignant, < 0.5 = Benign.
            if prediction_prob >= 0.5:
                label = "Malignant"
                confidence = prediction_prob * 100
                color = "red"
            else:
                label = "Benign"
                confidence = (1 - prediction_prob) * 100
                color = "green"

            st.markdown(f"### Prediction: :{color}[{label}]")
            st.markdown(f"### Confidence: **{confidence:.2f}%**")

            st.warning(
                "Disclaimer: This model is for educational purposes only "
                "and does not replace diagnosis by a qualified medical professional."
            )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
