from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_data_augmentation():

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation"
    )


# ============================================================
# BUILD MOBILENETV2 MODEL
# ============================================================

def build_mobilenet_model():

    print("=" * 70)
    print("BUILDING MOBILENETV2 TRANSFER LEARNING MODEL")
    print("=" * 70)

    inputs = keras.Input(
        shape=(224, 224, 3),
        name="input_image"
    )

    # Data augmentation
    x = create_data_augmentation()(inputs)

    # MobileNetV2 requires pixels scaled approximately to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    # --------------------------------------------------------
    # PRETRAINED FEATURE EXTRACTOR
    # --------------------------------------------------------

    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained network for first training stage
    base_model.trainable = False

    x = base_model(
        x,
        training=False
    )

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout_1"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="dense_features"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout_2"
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="Brain_Tumor_MobileNetV2"
    )

    return model


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_mobilenet_model(model):

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = build_mobilenet_model()

    model = compile_mobilenet_model(model)

    print("\nMOBILENETV2 ARCHITECTURE\n")

    model.summary()

    print("\n" + "=" * 70)

    print(
        f"Total parameters     : "
        f"{model.count_params():,}"
    )

    print(
        f"Trainable parameters : "
        f"{sum(
            int(keras.backend.count_params(w))
            for w in model.trainable_weights
        ):,}"
    )

    print(
        f"Frozen parameters    : "
        f"{sum(
            int(keras.backend.count_params(w))
            for w in model.non_trainable_weights
        ):,}"
    )

    print("=" * 70)

    print(
        "\n✓ MOBILENETV2 MODEL BUILT SUCCESSFULLY"
    )