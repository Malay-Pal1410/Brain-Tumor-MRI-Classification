# ============================================================
# EVALUATE EFFICIENTNETB0
# Brain Tumor MRI Classification
# ============================================================

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data_loader import load_datasets


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_best.keras"

RESULTS_DIR = PROJECT_ROOT / "results"

CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
REPORT_DIR = RESULTS_DIR / "classification_reports"
FIGURES_DIR = RESULTS_DIR / "figures"

CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("EFFICIENTNETB0 - TEST SET EVALUATION")
print("=" * 70)

train_ds, val_ds, test_ds, class_names = load_datasets()

print("\nClass names:")

for index, name in enumerate(class_names):
    print(f"{index} -> {name}")


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best EfficientNetB0 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n"
        "Run train_efficientnet.py first."
    )

model = keras.models.load_model(MODEL_PATH)

print("✓ Best model loaded successfully.")


# ============================================================
# EVALUATE TEST DATASET
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
y_probabilities = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(predictions, axis=1)

    # Handles both one-hot and integer labels
    labels_numpy = labels.numpy()

    if labels_numpy.ndim > 1:
        true_classes = np.argmax(labels_numpy, axis=1)
    else:
        true_classes = labels_numpy.astype(int)

    y_true.extend(true_classes.tolist())
    y_pred.extend(predicted_classes.tolist())
    y_probabilities.extend(predictions.tolist())


y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probabilities = np.array(y_probabilities)


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

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    REPORT_DIR /
    "efficientnet_classification_report.txt"
)

with open(report_path, "w") as file:

    file.write(
        "EFFICIENTNETB0 - BRAIN TUMOR MRI CLASSIFICATION\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(report)


print(
    f"\n✓ Classification report saved:\n"
    f"{report_path}"
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


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    cmap="Blues",
    values_format="d",
    ax=ax,
    colorbar=False
)

plt.title(
    "EfficientNetB0 - Confusion Matrix",
    fontsize=14
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

confusion_path = (
    CONFUSION_DIR /
    "efficientnet_confusion_matrix.png"
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
# SAVE TEST METRICS AS JSON
# ============================================================

metrics = {

    "model": "EfficientNetB0",

    "test_images": int(len(y_true)),

    "test_loss": float(test_loss),

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "accuracy_percent": float(accuracy * 100),

    "precision_percent": float(precision * 100),

    "recall_percent": float(recall * 100),

    "f1_score_percent": float(f1 * 100)
}


metrics_path = (
    RESULTS_DIR /
    "efficientnet_test_metrics.json"
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
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for class_index, class_name in enumerate(class_names):

    mask = y_true == class_index

    class_total = np.sum(mask)

    class_correct = np.sum(
        y_pred[mask] == class_index
    )

    if class_total > 0:

        class_accuracy = (
            class_correct /
            class_total
        )

    else:

        class_accuracy = 0


    print(
        f"{class_name:15s} : "
        f"{class_correct}/{class_total} "
        f"({class_accuracy * 100:.2f}%)"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\nCreating sample prediction figure...")

sample_images = []
sample_true = []

for images, labels in test_ds.take(1):

    sample_images = images.numpy()

    labels_numpy = labels.numpy()

    if labels_numpy.ndim > 1:

        sample_true = np.argmax(
            labels_numpy,
            axis=1
        )

    else:

        sample_true = labels_numpy.astype(int)


sample_predictions = model.predict(
    sample_images,
    verbose=0
)

sample_predicted = np.argmax(
    sample_predictions,
    axis=1
)


# ============================================================
# DISPLAY 12 SAMPLE PREDICTIONS
# ============================================================

number_of_samples = min(
    12,
    len(sample_images)
)

plt.figure(
    figsize=(14, 10)
)

for i in range(number_of_samples):

    plt.subplot(
        3,
        4,
        i + 1
    )

    image = sample_images[i]

    # Convert image safely for visualization
    image = np.clip(
        image,
        0,
        255
    ).astype("uint8")

    plt.imshow(image)

    true_name = class_names[
        sample_true[i]
    ]

    predicted_name = class_names[
        sample_predicted[i]
    ]

    confidence = np.max(
        sample_predictions[i]
    ) * 100

    title = (
        f"True: {true_name}\n"
        f"Pred: {predicted_name}\n"
        f"Conf: {confidence:.1f}%"
    )

    plt.title(
        title,
        fontsize=9
    )

    plt.axis("off")


plt.suptitle(
    "EfficientNetB0 - Sample Predictions",
    fontsize=16
)

plt.tight_layout()


prediction_figure_path = (
    FIGURES_DIR /
    "efficientnet_sample_predictions.png"
)


plt.savefig(
    prediction_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\n✓ Sample predictions saved:\n"
    f"{prediction_figure_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EFFICIENTNETB0 FINAL TEST SUMMARY")
print("=" * 70)

print(
    f"Test images : {len(y_true)}"
)

print(
    f"Test loss   : {test_loss:.4f}"
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
    "\n✓ EFFICIENTNETB0 TEST EVALUATION COMPLETED"
)