from tensorflow import keras
from tensorflow.keras import layers

from data_loader import (
    IMAGE_SIZE,
    data_augmentation
)


# ============================================================
# CUSTOM CNN CONFIGURATION
# ============================================================

NUM_CLASSES = 4


# ============================================================
# BUILD CUSTOM CNN
# ============================================================

def build_custom_cnn():

    inputs = keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        name="input_image"
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    x = data_augmentation(inputs)


    # --------------------------------------------------------
    # NORMALIZATION
    # Convert pixels from 0-255 to 0-1
    # --------------------------------------------------------

    x = layers.Rescaling(
        1.0 / 255,
        name="normalization"
    )(x)


    # --------------------------------------------------------
    # CONVOLUTION BLOCK 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)


    # --------------------------------------------------------
    # CONVOLUTION BLOCK 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)


    # --------------------------------------------------------
    # CONVOLUTION BLOCK 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.25)(x)


    # --------------------------------------------------------
    # CONVOLUTION BLOCK 4
    # --------------------------------------------------------

    x = layers.Conv2D(
        256,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)


    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.40)(x)


    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Brain_Tumor_Custom_CNN"
    )


    return model


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_custom_cnn(model):

    optimizer = keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# TEST MODEL ARCHITECTURE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BUILDING CUSTOM CNN")
    print("=" * 70)

    model = build_custom_cnn()

    model = compile_custom_cnn(model)

    print("\nCUSTOM CNN ARCHITECTURE\n")

    model.summary()

    print("\n" + "=" * 70)

    print(
        f"Total parameters: "
        f"{model.count_params():,}"
    )

    print(
        f"Input shape     : "
        f"{model.input_shape}"
    )

    print(
        f"Output shape    : "
        f"{model.output_shape}"
    )

    print("=" * 70)

    print("\n✓ CUSTOM CNN BUILT SUCCESSFULLY")