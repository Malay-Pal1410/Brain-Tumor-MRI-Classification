# ============================================================
# GRAD-CAM VISUALIZATION
# Brain Tumor MRI Classification
# MobileNetV2 Explainable AI
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mobilenet_best.keras"
TEST_DIR = BASE_DIR / "dataset" / "test"

OUTPUT_DIR = BASE_DIR / "results" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def load_model():

    print("=" * 70)
    print("GRAD-CAM - BRAIN TUMOR MRI EXPLAINABILITY")
    print("=" * 70)

    print("\nLoading MobileNetV2 model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = keras.models.load_model(MODEL_PATH)

    print("✓ Model loaded successfully")

    return model


# ============================================================
# LOAD AND PREPARE IMAGE
# ============================================================

def load_image(image_path):

    image = keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = keras.utils.img_to_array(image)

    # Keep original image for visualization
    original_image = image_array.copy()

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return original_image, image_array


# ============================================================
# FIND MOBILENETV2 FEATURE LAYER
# ============================================================

def find_feature_model(model):
    """
    Find the MobileNetV2 backbone inside the complete model.
    """

    print("\nSearching for MobileNetV2 feature extractor...")

    # First try to locate MobileNetV2 by name
    for layer in model.layers:

        if "mobilenet" in layer.name.lower():

            print(
                f"✓ Feature extractor found: {layer.name}"
            )

            print(
                f"  Output shape: {layer.output.shape}"
            )

            return layer

    # Fallback: find a nested model with convolutional output
    for layer in model.layers:

        if isinstance(layer, keras.Model):

            try:
                output_shape = layer.output.shape

                if (
                    len(output_shape) == 4
                    and output_shape[1] < 224
                    and output_shape[2] < 224
                ):

                    print(
                        f"✓ Feature extractor found: {layer.name}"
                    )

                    print(
                        f"  Output shape: {output_shape}"
                    )

                    return layer

            except Exception:
                continue

    raise ValueError(
        "Could not find MobileNetV2 feature extractor."
    )


# ============================================================
# CREATE GRAD-CAM HEATMAP
# ============================================================

def make_gradcam_heatmap(
    image_array,
    model,
    feature_model,
    predicted_class
):
    """
    Generate Grad-CAM heatmap for a nested MobileNetV2 model.

    This version is designed for the project structure:
    Input
      -> data augmentation
      -> MobileNetV2
      -> GlobalAveragePooling2D
      -> Dropout
      -> Dense
      -> BatchNormalization
      -> Dropout
      -> Predictions
    """

    # ---------------------------------------------------------
    # 1. Find where MobileNetV2 sits in the outer model
    # ---------------------------------------------------------

    feature_index = None

    for i, layer in enumerate(model.layers):

        if layer.name == feature_model.name:
            feature_index = i
            break

    if feature_index is None:
        raise ValueError(
            "MobileNetV2 feature model was not found "
            "inside the main model."
        )

    # ---------------------------------------------------------
    # 2. Create the classifier head
    #
    # MobileNetV2 output will be passed through all layers
    # that come AFTER MobileNetV2.
    # ---------------------------------------------------------

    feature_input = keras.Input(
        shape=feature_model.output_shape[1:]
    )

    x = feature_input

    for layer in model.layers[feature_index + 1:]:
        x = layer(x, training=False)

    classifier_model = keras.Model(
        inputs=feature_input,
        outputs=x,
        name="gradcam_classifier"
    )

    # ---------------------------------------------------------
    # 3. Prepare the image for MobileNetV2
    #
    # Run any layers BEFORE MobileNetV2 first.
    # This includes data augmentation if present.
    # ---------------------------------------------------------

    x = image_array

    for layer in model.layers[1:feature_index]:
        x = layer(x, training=False)

    # ---------------------------------------------------------
    # 4. Calculate feature maps + gradients
    # ---------------------------------------------------------

    with tf.GradientTape() as tape:

        feature_maps = feature_model(
            x,
            training=False
        )

        tape.watch(feature_maps)

        predictions = classifier_model(
            feature_maps,
            training=False
        )

        class_score = predictions[
            :,
            predicted_class
        ]

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients are None. "
            "Could not connect feature maps "
            "to the classifier output."
        )

    # ---------------------------------------------------------
    # 5. Global-average-pool the gradients
    # ---------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    feature_maps = feature_maps[0]

    # ---------------------------------------------------------
    # 6. Weight feature channels
    # ---------------------------------------------------------

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1
    )

    # ---------------------------------------------------------
    # 7. ReLU
    # ---------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # ---------------------------------------------------------
    # 8. Normalize between 0 and 1
    # ---------------------------------------------------------

    maximum = tf.reduce_max(heatmap)

    if maximum > 0:
        heatmap = heatmap / maximum

    return heatmap.numpy()


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap
):

    # Resize heatmap to MRI image size
    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_image = keras.utils.array_to_img(
        np.expand_dims(
            heatmap,
            axis=-1
        )
    )

    heatmap_image = heatmap_image.resize(
        IMAGE_SIZE
    )

    heatmap = keras.utils.img_to_array(
        heatmap_image
    )

    heatmap = heatmap.squeeze()

    # Use matplotlib colormap
    colormap = plt.get_cmap("jet")

    colored_heatmap = colormap(
        heatmap / 255.0
    )[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    )

    # Blend original MRI and heatmap
    overlay = (
        0.6 * original_image
        +
        0.4 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype("uint8")

    return overlay


# ============================================================
# PREDICT + GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path,
    model,
    feature_model
):

    print("\n" + "=" * 70)
    print("ANALYZING MRI IMAGE")
    print("=" * 70)

    print(f"\nImage: {image_path}")

    original_image, image_array = load_image(
        image_path
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_class]
    )

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    print(
        f"\nPrediction : {predicted_name}"
    )

    print(
        f"Confidence : {confidence * 100:.2f}%"
    )

    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(
        image_array,
        model,
        feature_model,
        predicted_class
    )

    overlay = create_overlay(
        original_image,
        heatmap
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    fig = plt.figure(
        figsize=(15, 5)
    )

    # Original image
    ax1 = fig.add_subplot(1, 3, 1)

    ax1.imshow(
        original_image.astype("uint8")
    )

    ax1.set_title(
        "Original MRI"
    )

    ax1.axis("off")

    # Heatmap
    ax2 = fig.add_subplot(1, 3, 2)

    ax2.imshow(
        heatmap,
        cmap="jet"
    )

    ax2.set_title(
        "Grad-CAM Heatmap"
    )

    ax2.axis("off")

    # Overlay
    ax3 = fig.add_subplot(1, 3, 3)

    ax3.imshow(
        overlay
    )

    ax3.set_title(
        f"Prediction: {predicted_name}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    ax3.axis("off")

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR /
        f"gradcam_{Path(image_path).stem}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n✓ Grad-CAM visualization saved:"
    )

    print(output_file)

    return predicted_name, confidence


# ============================================================
# FIND SAMPLE TEST IMAGE
# ============================================================

def find_sample_image():

    supported_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    # Try each class
    for class_folder in TEST_DIR.iterdir():

        if not class_folder.is_dir():
            continue

        for extension in supported_extensions:

            images = list(
                class_folder.glob(extension)
            )

            if images:

                return images[0]

    raise FileNotFoundError(
        f"No MRI images found inside:\n{TEST_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model = load_model()

    feature_model = find_feature_model(
        model
    )

    sample_image = find_sample_image()

    print(
        f"\nSample test image selected:\n"
        f"{sample_image}"
    )

    predicted_name, confidence = generate_gradcam(
        sample_image,
        model,
        feature_model
    )

    print("\n" + "=" * 70)
    print("GRAD-CAM SUMMARY")
    print("=" * 70)

    print(
        f"Predicted class : {predicted_name}"
    )

    print(
        f"Confidence      : {confidence * 100:.2f}%"
    )

    print(
        f"Output folder   : {OUTPUT_DIR}"
    )

    print("=" * 70)

    print(
        "\n✓ GRAD-CAM COMPLETED"
    )