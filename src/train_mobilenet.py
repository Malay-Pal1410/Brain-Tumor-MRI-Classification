from pathlib import Path
import json

import matplotlib.pyplot as plt
from tensorflow import keras

from data_loader import load_datasets
from mobilenet_model import build_mobilenet_model, compile_mobilenet_model


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MOBILENETV2 TRANSFER LEARNING - TRAINING")
print("=" * 70)

train_ds, valid_ds, test_ds, class_names = load_datasets()


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MobileNetV2 model...")

model = build_mobilenet_model()
model = compile_mobilenet_model(model)

print("\nModel ready for training.")


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = MODEL_DIR / "mobilenet_best.keras"

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting MobileNetV2 training...\n")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "mobilenet_final.keras"

model.save(final_model_path)

print(f"\n✓ Final model saved to:\n{final_model_path}")
print(f"\n✓ Best model saved to:\n{best_model_path}")


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = RESULTS_DIR / "mobilenet_history.json"

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(history_path, "w") as file:
    json.dump(history_data, file, indent=4)

print(f"\n✓ Training history saved to:\n{history_path}")


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Training and Validation Accuracy")

plt.legend()
plt.grid(True)

accuracy_path = FIGURES_DIR / "mobilenet_accuracy.png"

plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✓ Accuracy graph saved:\n{accuracy_path}")


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Training and Validation Loss")

plt.legend()
plt.grid(True)

loss_path = FIGURES_DIR / "mobilenet_loss.png"

plt.savefig(loss_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✓ Loss graph saved:\n{loss_path}")


# ============================================================
# TRAINING SUMMARY
# ============================================================

best_val_accuracy = max(history.history["val_accuracy"])
best_epoch = history.history["val_accuracy"].index(best_val_accuracy) + 1

print("\n" + "=" * 70)
print("MOBILENETV2 TRAINING SUMMARY")
print("=" * 70)

print(f"Epochs completed          : {len(history.history['loss'])}")
print(f"Best validation epoch     : {best_epoch}")
print(f"Best validation accuracy  : {best_val_accuracy * 100:.2f}%")

print("\n✓ MOBILENETV2 TRAINING COMPLETED")