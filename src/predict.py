# ============================================================
# SINGLE MRI IMAGE PREDICTION
# Brain Tumor MRI Classification
# ============================================================

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_best.keras"

TEST_DIR = PROJECT_ROOT / "dataset" / "test"


# ============================================================
# CONFIGURATION
# ============================================================

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

def load_prediction_model():

    print("=" * 70)
    print("BRAIN TUMOR MRI - SINGLE IMAGE PREDICTION")
    print("=" * 70)

    print("\nLoading MobileNetV2 model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    model = keras.models.load_model(MODEL_PATH)

    print("✓ Model loaded successfully")

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load image as RGB
    image = keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    # Convert PIL image to NumPy array
    image_array = keras.utils.img_to_array(image)

    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Keep the same input scale used during training.
    #
    # IMPORTANT:
    # Do NOT automatically divide by 255 here unless your
    # MobileNetV2 training pipeline also divided by 255.
    #
    # Your saved model may already contain preprocessing.
    image_array = image_array.astype("float32")

    return image_array


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(model, image_path):

    image_array = preprocess_image(image_path)

    print("\nAnalyzing MRI image...")

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    image_path,
    predicted_class,
    confidence,
    probabilities
):

    print("\n" + "=" * 70)

    print("PREDICTION RESULT")

    print("=" * 70)

    print(
        f"\nImage       : {Path(image_path).name}"
    )

    print(
        "Model       : MobileNetV2"
    )

    print(
        f"Prediction  : {predicted_class}"
    )

    print(
        f"Confidence  : {confidence * 100:.2f}%"
    )

    print("\nClass probabilities:")

    print("-" * 45)

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        print(
            f"{class_name:<15} "
            f": {probability * 100:>7.2f}%"
        )

    print("-" * 45)


# ============================================================
# FIND SAMPLE IMAGE
# ============================================================

def find_sample_image():

    """
    Used when predict.py is executed without an image path.

    The function searches the test dataset and selects
    one MRI image automatically.
    """

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    images = []

    for extension in extensions:

        images.extend(
            TEST_DIR.rglob(extension)
        )

    if not images:

        raise FileNotFoundError(
            f"No MRI images found inside:\n{TEST_DIR}"
        )

    # Sort so the selected sample is reproducible
    images = sorted(images)

    return images[0]


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        model = load_prediction_model()

        # ----------------------------------------------------
        # Determine image
        # ----------------------------------------------------

        if len(sys.argv) > 1:

            image_path = Path(sys.argv[1])

            print(
                "\nUsing image provided from command line:"
            )

        else:

            print(
                "\nNo image path supplied."
            )

            print(
                "Selecting one image from the test dataset..."
            )

            image_path = find_sample_image()

        print(
            f"\nImage:\n{image_path}"
        )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        (
            predicted_class,
            confidence,
            probabilities

        ) = predict_image(
            model,
            image_path
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        display_results(
            image_path,
            predicted_class,
            confidence,
            probabilities
        )

        print("\n" + "=" * 70)

        print(
            "✓ PREDICTION COMPLETED"
        )

        print("=" * 70)

        return (
            predicted_class,
            confidence,
            probabilities
        )

    except Exception as error:

        print("\n" + "=" * 70)

        print("PREDICTION FAILED")

        print("=" * 70)

        print(
            f"\nError: {error}"
        )

        print(
            "\nCheck the model path and image path."
        )

        raise


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()