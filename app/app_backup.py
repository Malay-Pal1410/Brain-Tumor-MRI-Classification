from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_best.keras"


# ============================================================
# CLASS NAMES
# IMPORTANT: Same order used during training
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = image_array.astype(np.float32)

    # IMPORTANT:
    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image, model):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, predictions


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🧠 Brain Tumor MRI Classification System")

st.write(
    "Upload a brain MRI image to classify it using "
    "our trained MobileNetV2 deep learning model."
)

st.info(
    "Supported classes: Glioma, Meningioma, "
    "No Tumor, and Pituitary Tumor"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    model = load_model()

    st.success("✅ MobileNetV2 model loaded successfully!")

except Exception as e:

    st.error("❌ Could not load the trained model.")

    st.error(str(e))

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.subheader("Uploaded MRI")

    st.image(
        image,
        caption="Uploaded MRI Image",
        width=400
    )

    st.success("MRI image uploaded successfully!")

    st.divider()


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze MRI",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing MRI using MobileNetV2..."
        ):

            predicted_class, confidence, probabilities = (
                predict_image(
                    image,
                    model
                )
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.header("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Predicted Class",
                value=predicted_class
            )

        with col2:

            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader("Class Probabilities")

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            percentage = float(probability) * 100

            st.write(
                f"**{class_name}: {percentage:.2f}%**"
            )

            st.progress(
                min(
                    max(float(probability), 0.0),
                    1.0
                )
            )


        # ====================================================
        # FINAL MESSAGE
        # ====================================================

        st.divider()

        if predicted_class == "No Tumor":

            st.success(
                "Model prediction: No Tumor"
            )

        else:

            st.warning(
                f"Model prediction: {predicted_class}"
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ Research/Educational Use Only — "
    "This AI system is not a substitute for professional "
    "medical diagnosis."
)