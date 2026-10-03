# ============================================================
# TRAIN EFFICIENTNETB0
# Brain Tumor MRI Classification
# ============================================================

from pathlib import Path
import json

import matplotlib.pyplot as plt
from tensorflow import keras

from data_loader import load_datasets
from efficientnet_model import (
    build_efficientnet_model,
    compile_efficientnet_model
)


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 20
LEARNING_RATE = 0.001


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


BEST_MODEL_PATH = MODEL_DIR / "efficientnet_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "efficientnet_final.keras"

HISTORY_PATH = RESULTS_DIR / "efficientnet_history.json"

ACCURACY_FIGURE_PATH = (
    FIGURES_DIR / "efficientnet_accuracy.png"
)

LOSS_FIGURE_PATH = (
    FIGURES_DIR / "efficientnet_loss.png"
)


# ============================================================
# TRAINING CALLBACKS
# ============================================================

def create_callbacks():

    # Save model whenever validation accuracy improves
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Stop training if validation accuracy stops improving
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when validation accuracy plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        mode="max",
        verbose=1
    )

    return [
        checkpoint,
        early_stopping,
        reduce_lr
    ]


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_training_history(history):

    history_dictionary = {}

    for key, values in history.history.items():

        history_dictionary[key] = [
            float(value)
            for value in values
        ]

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_dictionary,
            file,
            indent=4
        )

    print("\n✓ Training history saved to:")
    print(HISTORY_PATH)


# ============================================================
# PLOT ACCURACY
# ============================================================

def plot_accuracy(history):

    training_accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]

    epochs = range(
        1,
        len(training_accuracy) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        training_accuracy,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        validation_accuracy,
        marker="o",
        label="Validation Accuracy"
    )

    plt.title(
        "EfficientNetB0 Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        ACCURACY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n✓ Accuracy graph saved:")
    print(ACCURACY_FIGURE_PATH)


# ============================================================
# PLOT LOSS
# ============================================================

def plot_loss(history):

    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    epochs = range(
        1,
        len(training_loss) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        training_loss,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation Loss"
    )

    plt.title(
        "EfficientNetB0 Training and Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        LOSS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n✓ Loss graph saved:")
    print(LOSS_FIGURE_PATH)


# ============================================================
# TRAINING SUMMARY
# ============================================================

def print_training_summary(history):

    validation_accuracy = history.history[
        "val_accuracy"
    ]

    best_validation_accuracy = max(
        validation_accuracy
    )

    best_epoch = (
        validation_accuracy.index(
            best_validation_accuracy
        )
        + 1
    )

    print("\n")
    print("=" * 70)
    print("EFFICIENTNETB0 TRAINING SUMMARY")
    print("=" * 70)

    print(
        f"Epochs completed         : "
        f"{len(validation_accuracy)}"
    )

    print(
        f"Best validation epoch   : "
        f"{best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print("=" * 70)


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("EFFICIENTNETB0 TRANSFER LEARNING - TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        train_ds,
        valid_ds,
        test_ds,
        class_names
    ) = load_datasets()

    print("\nClass names:")

    for index, class_name in enumerate(class_names):

        print(
            f"{index} -> {class_name}"
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding EfficientNetB0 model...")

    model = build_efficientnet_model(
        input_shape=(224, 224, 3),
        num_classes=len(class_names)
    )

    # --------------------------------------------------------
    # Compile model
    # --------------------------------------------------------

    model = compile_efficientnet_model(
        model,
        learning_rate=LEARNING_RATE
    )

    print("\nModel ready for training.")

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = create_callbacks()

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nStarting EfficientNetB0 training...\n")

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.save(
        FINAL_MODEL_PATH
    )

    print("\n✓ Final model saved to:")
    print(FINAL_MODEL_PATH)

    print("\n✓ Best model saved to:")
    print(BEST_MODEL_PATH)

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_training_history(
        history
    )

    # --------------------------------------------------------
    # Save graphs
    # --------------------------------------------------------

    plot_accuracy(
        history
    )

    plot_loss(
        history
    )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    print_training_summary(
        history
    )

    print("\n✓ EFFICIENTNETB0 TRAINING COMPLETED")
    print("=" * 70)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()