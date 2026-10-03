from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

from data_loader import load_datasets


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_best.keras"

RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
REPORT_DIR = RESULTS_DIR / "classification_reports"

CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MOBILENETV2 - TEST SET EVALUATION")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

train_ds, valid_ds, test_ds, class_names = load_datasets()

print("\nClass names:")

for i, class_name in enumerate(class_names):
    print(f"{i} -> {class_name}")


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best MobileNetV2 model...")

model = keras.models.load_model(MODEL_PATH)

print("✓ Best model loaded successfully")


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING TEST DATASET")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:
    
    predictions = model.predict(images, verbose=0)

    # Convert model probabilities to class indices
    predicted_classes = np.argmax(predictions, axis=1)

    # Convert one-hot encoded true labels to class indices
    true_classes = np.argmax(labels.numpy(), axis=1)

    y_pred.extend(predicted_classes.tolist())
    y_true.extend(true_classes.tolist())


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("OVERALL TEST METRICS")
print("=" * 70)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

report_path = (
    REPORT_DIR /
    "mobilenet_classification_report.txt"
)

with open(report_path, "w") as file:
    file.write(report)

print(f"\n✓ Classification report saved:\n{report_path}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 7))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("MobileNetV2 - Test Confusion Matrix")
plt.xticks(rotation=30)
plt.tight_layout()

confusion_path = (
    CONFUSION_DIR /
    "mobilenet_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\n✓ Confusion matrix saved:\n{confusion_path}")


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "test_loss": float(test_loss),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

metrics_path = (
    RESULTS_DIR /
    "mobilenet_test_metrics.json"
)

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

print(f"\n✓ Test metrics saved:\n{metrics_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MOBILENETV2 FINAL TEST SUMMARY")
print("=" * 70)

print(f"Test images : {len(y_true)}")
print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Precision   : {precision * 100:.2f}%")
print(f"Recall      : {recall * 100:.2f}%")
print(f"F1-score    : {f1 * 100:.2f}%")

print("=" * 70)

print("\n✓ MOBILENETV2 TEST EVALUATION COMPLETED")