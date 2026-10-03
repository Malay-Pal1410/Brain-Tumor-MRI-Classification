import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Brain Tumor MRI Classification System")

st.write(
    "Upload a brain MRI image to classify it using the "
    "trained MobileNetV2 deep learning model."
)

st.info(
    "Supported classes: Glioma, Meningioma, No Tumor, "
    "and Pituitary Tumor"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/mobilenet_best.keras")

IMAGE_SIZE = (224, 224)

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

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


try:

    model = load_model()

    st.success(
        "✅ MobileNetV2 model loaded successfully!"
    )

except Exception as e:

    st.error(
        "❌ Unable to load MobileNetV2 model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(IMAGE_SIZE)

    # IMPORTANT:
    # Keep pixels in 0-255 range.
    #
    # MobileNetV2 preprocess_input() is already inside
    # the trained model.

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# FIND MOBILENETV2 FEATURE EXTRACTOR
# ============================================================

def find_feature_extractor(model):

    """
    Find the nested MobileNetV2 model inside the
    complete classification model.
    """

    for layer in model.layers:

        # First try identifying by model type/name
        if isinstance(layer, tf.keras.Model):

            layer_name = layer.name.lower()

            if "mobilenet" in layer_name:

                return layer

    return None


# ============================================================
# GRAD-CAM HEATMAP
# ============================================================

def generate_gradcam(
    model,
    image_array,
    class_index
):

    """
    Generate Grad-CAM using the nested MobileNetV2
    feature extractor.

    This approach avoids constructing a Functional model
    from model.inputs -> nested_model.output, which can
    produce a disconnected-graph error after loading a
    saved Keras model.
    """

    # --------------------------------------------------------
    # FIND MOBILENETV2 FEATURE EXTRACTOR
    # --------------------------------------------------------

    feature_extractor = find_feature_extractor(
        model
    )

    if feature_extractor is None:

        raise ValueError(
            "Could not locate MobileNetV2 "
            "feature extractor."
        )

    # --------------------------------------------------------
    # FIND POSITION OF FEATURE EXTRACTOR
    # --------------------------------------------------------

    feature_index = None

    for index, layer in enumerate(model.layers):

        if layer.name == feature_extractor.name:

            feature_index = index

            break

    if feature_index is None:

        raise ValueError(
            "MobileNetV2 layer position "
            "could not be determined."
        )

    # --------------------------------------------------------
    # INPUT TENSOR
    # --------------------------------------------------------

    x = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # GRADIENT CALCULATION
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # Run all layers BEFORE MobileNetV2.
        #
        # This includes:
        # data augmentation
        # MobileNetV2 preprocess_input
        # and any other preprocessing layers.

        for layer in model.layers[
            1:feature_index
        ]:

            x = layer(
                x,
                training=False
            )

        # --------------------------------------------
        # MOBILENETV2 FEATURE MAPS
        # --------------------------------------------

        feature_maps = feature_extractor(
            x,
            training=False
        )

        # Explicitly watch feature maps
        tape.watch(
            feature_maps
        )

        # --------------------------------------------
        # RUN CLASSIFICATION HEAD
        # --------------------------------------------

        x = feature_maps

        for layer in model.layers[
            feature_index + 1:
        ]:

            x = layer(
                x,
                training=False
            )

        predictions = x

        # Selected predicted class score
        class_score = predictions[
            :,
            class_index
        ]

    # --------------------------------------------------------
    # GRADIENTS
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:

        raise ValueError(
            "Grad-CAM gradients are None. "
            "Gradient path could not be computed."
        )

    # --------------------------------------------------------
    # GLOBAL AVERAGE POOL GRADIENTS
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    feature_maps = feature_maps[0]

    # --------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        feature_maps
        * pooled_gradients,
        axis=-1
    )

    # Keep only positive contributions
    heatmap = tf.nn.relu(
        heatmap
    )

    # --------------------------------------------------------
    # NORMALIZE 0 -> 1
    # --------------------------------------------------------

    maximum = tf.reduce_max(
        heatmap
    )

    if float(maximum) > 0:

        heatmap = (
            heatmap / maximum
        )

    return heatmap.numpy()

    # --------------------------------------------------------
    # MODEL FROM INPUT -> FEATURE MAPS
    # --------------------------------------------------------

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=feature_extractor.output
    )

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    feature_input = tf.keras.Input(
        shape=feature_extractor.output.shape[1:]
    )

    x = feature_input

    feature_layer_found = False

    for layer in model.layers:

        if layer.name == feature_extractor.name:

            feature_layer_found = True
            continue

        if feature_layer_found:

            x = layer(x)

    classifier_model = tf.keras.Model(
        feature_input,
        x
    )

    # --------------------------------------------------------
    # GRADIENT CALCULATION
    # --------------------------------------------------------

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        feature_maps = feature_model(
            image_tensor,
            training=False
        )

        tape.watch(feature_maps)

        predictions = classifier_model(
            feature_maps,
            training=False
        )

        class_score = predictions[
            :,
            class_index
        ]

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:

        raise ValueError(
            "Unable to calculate Grad-CAM gradients."
        )

    # --------------------------------------------------------
    # GLOBAL AVERAGE POOL GRADIENTS
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    feature_maps = feature_maps[0]

    # Weight feature maps by importance
    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1
    )

    # Keep positive influence only
    heatmap = tf.nn.relu(
        heatmap
    )

    # Normalize
    maximum = tf.reduce_max(
        heatmap
    )

    if float(maximum) > 0:

        heatmap = heatmap / maximum

    heatmap = heatmap.numpy()

    return heatmap


# ============================================================
# RESIZE GRAD-CAM HEATMAP
# ============================================================

def resize_heatmap(heatmap):

    """
    Resize the small Grad-CAM feature map to the
    original model input resolution.

    Bilinear interpolation makes the 7x7 Grad-CAM
    visualization smoother.
    """

    heatmap_tensor = tf.convert_to_tensor(
        heatmap[..., np.newaxis],
        dtype=tf.float32
    )

    heatmap_tensor = tf.image.resize(
        heatmap_tensor,
        IMAGE_SIZE,
        method="bilinear"
    )

    resized = heatmap_tensor.numpy()

    resized = np.squeeze(
        resized
    )

    resized = np.clip(
        resized,
        0.0,
        1.0
    )

    return resized


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap,
    alpha=0.40
):

    """
    Combine the original MRI and Grad-CAM heatmap.
    """

    original_image = image.convert(
        "RGB"
    ).resize(
        IMAGE_SIZE
    )

    original_array = np.array(
        original_image,
        dtype=np.float32
    )

    # Get jet colormap
    colormap = plt.get_cmap(
        "jet"
    )

    colored_heatmap = colormap(
        heatmap
    )

    # Remove alpha channel
    colored_heatmap = colored_heatmap[
        ...,
        :3
    ]

    # Convert 0-1 -> 0-255
    colored_heatmap = (
        colored_heatmap * 255
    ).astype(
        np.float32
    )

    # Blend images
    overlay = (
        (1 - alpha) * original_array
        + alpha * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    return overlay


# ============================================================
# FILE UPLOADER
# ============================================================

st.divider()

st.subheader(
    "Upload Brain MRI"
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        )

        st.subheader(
            "Uploaded MRI"
        )

        st.image(
            image,
            caption="Uploaded MRI Image",
            width=400
        )

        if st.button(
            "🔍 Classify MRI",
            type="primary"
        ):

            with st.spinner(
                "Analyzing MRI using MobileNetV2..."
            ):

                # ============================================
                # PREPROCESS
                # ============================================

                processed_image = preprocess_image(
                    image
                )

                # ============================================
                # PREDICTION
                # ============================================

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )

                probabilities = predictions[0]

                predicted_index = int(
                    np.argmax(
                        probabilities
                    )
                )

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )


            # =================================================
            # RESULTS
            # =================================================

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            result_col1, result_col2 = st.columns(
                2
            )

            with result_col1:

                st.success(
                    f"Predicted Class: "
                    f"{predicted_class}"
                )

            with result_col2:

                st.metric(
                    label="Confidence",
                    value=(
                        f"{confidence * 100:.2f}%"
                    )
                )


            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            st.subheader(
                "Class Probabilities"
            )

            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities
            ):

                probability = float(
                    probability
                )

                st.write(
                    f"**{class_name}: "
                    f"{probability * 100:.2f}%**"
                )

                st.progress(
                    max(
                        0.0,
                        min(
                            1.0,
                            probability
                        )
                    )
                )


            # =================================================
            # GRAD-CAM
            # =================================================

            st.divider()

            st.subheader(
                "🔥 Explainable AI — Grad-CAM"
            )

            st.write(
                "Grad-CAM highlights image regions that "
                "contributed strongly to the model's "
                "classification."
            )

            try:

                with st.spinner(
                    "Generating Grad-CAM explanation..."
                ):

                    # Generate original Grad-CAM
                    heatmap = generate_gradcam(
                        model,
                        processed_image,
                        predicted_index
                    )

                    # Resize smoothly
                    resized_heatmap = resize_heatmap(
                        heatmap
                    )

                    # Generate overlay
                    overlay = create_gradcam_overlay(
                        image,
                        resized_heatmap,
                        alpha=0.40
                    )


                # =============================================
                # DISPLAY GRAD-CAM
                # =============================================

                col1, col2, col3 = st.columns(
                    3
                )

                with col1:

                    st.markdown(
                        "#### Original MRI"
                    )

                    st.image(
                        image,
                        caption="Original MRI",
                        use_container_width=True
                    )


                with col2:

                    st.markdown(
                        "#### Grad-CAM Heatmap"
                    )

                    fig, ax = plt.subplots()

                    ax.imshow(
                        resized_heatmap,
                        cmap="jet",
                        vmin=0,
                        vmax=1
                    )

                    ax.axis(
                        "off"
                    )

                    fig.tight_layout(
                        pad=0
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    plt.close(
                        fig
                    )


                with col3:

                    st.markdown(
                        "#### Grad-CAM Overlay"
                    )

                    st.image(
                        overlay,
                        caption=(
                            f"{predicted_class} "
                            f"({confidence * 100:.2f}%)"
                        ),
                        use_container_width=True
                    )


                st.info(
                    "🔥 Red/yellow regions indicate areas "
                    "that contributed more strongly to the "
                    "model's prediction. Grad-CAM provides "
                    "classification explainability and should "
                    "not be interpreted as tumor segmentation."
                )


            except Exception as gradcam_error:

                st.warning(
                    "Prediction succeeded, but Grad-CAM "
                    "could not be generated."
                )

                with st.expander(
                    "Grad-CAM Error Details"
                ):

                    st.exception(
                        gradcam_error
                    )


            # =================================================
            # TECHNICAL DETAILS
            # =================================================

            with st.expander(
                "Technical Details"
            ):

                st.write(
                    "Model: MobileNetV2"
                )

                st.write(
                    "Input size: 224 × 224"
                )

                st.write(
                    "Model input shape:",
                    model.input_shape
                )

                st.write(
                    "Model output shape:",
                    model.output_shape
                )

                feature_extractor = (
                    find_feature_extractor(
                        model
                    )
                )

                if feature_extractor is not None:

                    st.write(
                        "Grad-CAM feature extractor:",
                        feature_extractor.name
                    )

                    st.write(
                        "Feature map shape:",
                        feature_extractor.output.shape
                    )

                st.write(
                    "Predicted class index:",
                    predicted_index
                )

                st.write(
                    "Raw prediction:",
                    predictions
                )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.warning(
                "⚠️ This system is an academic "
                "machine-learning project and is not "
                "intended to provide a medical diagnosis."
            )


    except Exception as e:

        st.error(
            "❌ An error occurred while processing "
            "the image."
        )

        st.exception(
            e
        )


else:

    st.info(
        "👆 Upload an MRI image to begin classification."
    )