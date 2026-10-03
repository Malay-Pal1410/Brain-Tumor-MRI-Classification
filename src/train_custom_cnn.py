from pathlib import Path
import json

import matplotlib.pyplot as plt
from tensorflow import keras

from data_loader import load_datasets
from custom_cnn import build_custom_cnn, compile_custom_cnn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "custom_cnn_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "custom_cnn_final.keras"

HISTORY_PATH = RESULTS_DIR / "custom_cnn_history.json"

EPOCHS = 30


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOM CNN TRAINING")
print("=" * 70)

train_ds, valid_ds, test_ds, class_names = load_datasets()

print("\nClass order:")

for index, class_name in enumerate(class_names):
    print(f"{index} -> {class_name}")


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding Custom CNN...")

model = build_custom_cnn()
model = compile_custom_cnn(model)

print("\nModel successfully created.")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # Save only the model with the best validation accuracy
    keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    # Stop training if validation loss stops improving
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate if validation loss plateaus
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(FINAL_MODEL_PATH)

print(f"\n✓ Final model saved to:")
print(FINAL_MODEL_PATH)

print(f"\n✓ Best model saved to:")
print(BEST_MODEL_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_data = {}

for key, values in history.history.items():
    history_data[key] = [
        float(value) for value in values
    ]

with open(HISTORY_PATH, "w") as file:
    json.dump(
        history_data,
        file,
        indent=4
    )

print(f"\n✓ Training history saved to:")
print(HISTORY_PATH)


# ============================================================
# TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Custom CNN - Training and Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_path = (
    FIGURES_DIR /
    "custom_cnn_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\n✓ Accuracy graph saved:")
print(accuracy_path)


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Custom CNN - Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = (
    FIGURES_DIR /
    "custom_cnn_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\n✓ Loss graph saved:")
print(loss_path)


# ============================================================
# BEST TRAINING RESULTS
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    history.history["val_accuracy"].index(
        best_val_accuracy
    ) + 1
)

print("\n" + "=" * 70)
print("CUSTOM CNN TRAINING SUMMARY")
print("=" * 70)

print(
    f"Epochs completed       : "
    f"{len(history.history['loss'])}"
)

print(
    f"Best validation epoch  : "
    f"{best_epoch}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print("\n✓ CUSTOM CNN TRAINING COMPLETED")
print("=" * 70)