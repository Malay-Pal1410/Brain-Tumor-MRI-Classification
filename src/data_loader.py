from pathlib import Path

import tensorflow as tf
from tensorflow import keras


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

IMG_HEIGHT = 224
IMG_WIDTH = 224

IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 32

SEED = 42


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),

        keras.layers.RandomRotation(0.08),

        keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10)
        ),

        keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    print("=" * 70)
    print("LOADING BRAIN MRI DATASET")
    print("=" * 70)

    print("\nLoading training dataset...")

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED
    )


    print("\nLoading validation dataset...")

    valid_ds = keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )


    print("\nLoading test dataset...")

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )


    class_names = train_ds.class_names


    print("\nClasses detected:")

    for index, class_name in enumerate(class_names):

        print(
            f"{index} -> "
            f"{class_name.replace('_', ' ').title()}"
        )


    # --------------------------------------------------------
    # PERFORMANCE OPTIMIZATION
    # --------------------------------------------------------

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(
        buffer_size=autotune
    )

    valid_ds = valid_ds.prefetch(
        buffer_size=autotune
    )

    test_ds = test_ds.prefetch(
        buffer_size=autotune
    )


    print("\nDataset loading completed.")

    return (
        train_ds,
        valid_ds,
        test_ds,
        class_names
    )


# ============================================================
# TEST DATA LOADER
# ============================================================

if __name__ == "__main__":

    train_ds, valid_ds, test_ds, class_names = (
        load_datasets()
    )

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Image size : {IMAGE_SIZE}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Classes    : {class_names}")

    for images, labels in train_ds.take(1):

        print("\nFirst training batch:")

        print(
            "Image batch shape :",
            images.shape
        )

        print(
            "Label batch shape :",
            labels.shape
        )

        print(
            "Pixel minimum     :",
            float(tf.reduce_min(images))
        )

        print(
            "Pixel maximum     :",
            float(tf.reduce_max(images))
        )

    print("\n✓ DATA LOADER TEST PASSED")