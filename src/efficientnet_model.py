# ============================================================
# EFFICIENTNETB0 MODEL
# Brain Tumor MRI Classification
# ============================================================

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

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation"
    )

    return data_augmentation


# ============================================================
# BUILD EFFICIENTNETB0 MODEL
# ============================================================

def build_efficientnet_model(
    input_shape=(224, 224, 3),
    num_classes=4
):

    print("=" * 70)
    print("BUILDING EFFICIENTNETB0 TRANSFER LEARNING MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Input layer
    # --------------------------------------------------------

    inputs = keras.Input(
        shape=input_shape,
        name="input_image"
    )

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    data_augmentation = create_data_augmentation()

    x = data_augmentation(inputs)

    # --------------------------------------------------------
    # EfficientNetB0 base model
    # --------------------------------------------------------
    #
    # EfficientNet models in Keras already include their
    # required input rescaling/preprocessing internally.
    #
    # Therefore, DO NOT add Rescaling(1./255) here.
    # --------------------------------------------------------

    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained ImageNet layers
    base_model.trainable = False

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    x = base_model(
        x,
        training=False
    )

    # Convert feature maps into feature vector
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.Dropout(
        0.30,
        name="dropout_1"
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="dense_features"
    )(x)

    x = layers.BatchNormalization(
        name="batch_normalization"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout_2"
    )(x)

    # --------------------------------------------------------
    # Output layer
    # --------------------------------------------------------

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Brain_Tumor_EfficientNetB0"
    )

    return model


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_efficientnet_model(
    model,
    learning_rate=0.001
):

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_information(model):

    print("\n")
    print("=" * 70)
    print("EFFICIENTNETB0 ARCHITECTURE")
    print("=" * 70)

    model.summary()

    print("\n")
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    total_parameters = model.count_params()

    trainable_parameters = sum(
        int(layer.count_params())
        for layer in model.layers
        if layer.trainable
    )

    frozen_parameters = (
        total_parameters - trainable_parameters
    )

    print(f"Total parameters     : {total_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")
    print(f"Frozen parameters    : {frozen_parameters:,}")

    print(f"Input shape          : {model.input_shape}")
    print(f"Output shape         : {model.output_shape}")

    print("=" * 70)


# ============================================================
# FINE-TUNING FUNCTION
# ============================================================

def enable_fine_tuning(
    model,
    num_layers=20
):

    print("\n")
    print("=" * 70)
    print("ENABLING EFFICIENTNETB0 FINE-TUNING")
    print("=" * 70)

    # Find EfficientNet base model
    base_model = None

    for layer in model.layers:

        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:

        print("EfficientNet base model not found.")
        return model

    # Allow base model to become trainable
    base_model.trainable = True

    # Freeze all layers except last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    # Keep BatchNormalization frozen for stable fine-tuning
    for layer in base_model.layers:

        if isinstance(
            layer,
            layers.BatchNormalization
        ):
            layer.trainable = False

    print(
        f"Last {num_layers} EfficientNet layers "
        "enabled for fine-tuning."
    )

    return model


# ============================================================
# FINE-TUNING COMPILER
# ============================================================

def compile_for_fine_tuning(
    model,
    learning_rate=0.00001
):

    model.compile(

        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )

    print(
        f"Model compiled for fine-tuning "
        f"with learning rate = {learning_rate}"
    )

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("EFFICIENTNETB0 MODEL TEST")
    print("=" * 70)

    # Build model
    model = build_efficientnet_model(
        input_shape=(224, 224, 3),
        num_classes=NUM_CLASSES
    )

    # Compile model
    model = compile_efficientnet_model(
        model,
        learning_rate=0.001
    )

    # Display architecture
    print_model_information(model)

    print("\n")
    print("✓ EFFICIENTNETB0 MODEL BUILT SUCCESSFULLY")
    print("=" * 70)