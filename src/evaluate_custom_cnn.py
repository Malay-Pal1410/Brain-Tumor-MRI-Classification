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

MODEL_PATH = PROJECT_ROOT / "models" / "custom_cnn_best.keras"

RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
REPORT_DIR = RESULTS_DIR / "classification_reports"

CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("CUSTOM CNN - TEST SET EVALUATION")
print("=" * 70)

train_ds, valid_ds, test_ds, class_names = load_datasets()

print("\nClass names:")

for index, name in enumerate(class_names):
    print(f"{index} -> {name}")


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best Custom CNN model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = keras.models.load_model(MODEL_PATH)

print("✓ Best model loaded successfully")


# ============================================================
# KERAS TEST EVALUATION
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

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_classes = np.argmax(
        labels.numpy(),
        axis=1
    )

    y_pred.extend(predicted_classes)
    y_true.extend(true_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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

report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report_text)


report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

text_report_path = (
    REPORT_DIR /
    "custom_cnn_classification_report.txt"
)

with open(text_report_path, "w") as file:

    file.write("CUSTOM CNN - TEST SET EVALUATION\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy:.4f}\n\n"
    )

    file.write(report_text)


json_report_path = (
    REPORT_DIR /
    "custom_cnn_classification_report.json"
)

with open(json_report_path, "w") as file:
    json.dump(
        report_dict,
        file,
        indent=4
    )


print(
    f"\n✓ Classification report saved:\n"
    f"{text_report_path}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(9, 8)
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title(
    "Custom CNN - Test Set Confusion Matrix"
)

plt.xticks(rotation=30)

plt.tight_layout()

confusion_path = (
    CONFUSION_DIR /
    "custom_cnn_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\n✓ Confusion matrix saved:\n"
    f"{confusion_path}"
)


# ============================================================
# SAVE SUMMARY METRICS
# ============================================================

metrics = {
    "model": "Custom CNN",
    "test_loss": float(test_loss),
    "accuracy": float(accuracy),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1)
}

metrics_path = (
    RESULTS_DIR /
    "custom_cnn_test_metrics.json"
)

with open(metrics_path, "w") as file:
    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    f"\n✓ Test metrics saved:\n"
    f"{metrics_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CUSTOM CNN FINAL TEST SUMMARY")
print("=" * 70)

print(
    f"Test images : {len(y_true)}"
)

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Precision   : {precision * 100:.2f}%"
)

print(
    f"Recall      : {recall * 100:.2f}%"
)

print(
    f"F1-score    : {f1 * 100:.2f}%"
)

print("=" * 70)

print(
    "\n✓ CUSTOM CNN TEST EVALUATION COMPLETED"
)