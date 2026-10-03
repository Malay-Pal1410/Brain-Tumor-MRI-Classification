# ============================================================
# MODEL COMPARISON
# Brain Tumor MRI Classification
# Custom CNN vs MobileNetV2 vs EfficientNetB0
# ============================================================

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# METRIC FILE PATHS
# ============================================================

MODEL_FILES = {
    "Custom CNN": RESULTS_DIR / "custom_cnn_test_metrics.json",
    "MobileNetV2": RESULTS_DIR / "mobilenet_test_metrics.json",
    "EfficientNetB0": RESULTS_DIR / "efficientnet_test_metrics.json",
}


# ============================================================
# LOAD JSON METRICS
# ============================================================

def load_metrics(file_path):
    """
    Load test metrics from JSON file.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nMetric file not found:\n{file_path}\n"
        )

    with open(file_path, "r") as file:
        metrics = json.load(file)

    return metrics


# ============================================================
# EXTRACT METRIC SAFELY
# ============================================================

def get_metric(metrics, name):
    """
    Get a metric from different model JSON formats
    and convert decimal values to percentages.
    """

    possible_keys = {
        "accuracy": [
            "accuracy",
            "test_accuracy",
            "accuracy_percent"
        ],

        "precision": [
            "precision",
            "precision_weighted",
            "weighted_precision",
            "precision_percent"
        ],

        "recall": [
            "recall",
            "recall_weighted",
            "weighted_recall",
            "recall_percent"
        ],

        "f1_score": [
            "f1_score",
            "f1_weighted",
            "weighted_f1",
            "weighted_f1_score",
            "f1_score_percent"
        ]
    }

    for key in possible_keys[name]:

        if key in metrics:

            value = float(metrics[key])

            # Convert decimal value to percentage
            # Example: 0.8496 -> 84.96
            if value <= 1.0:
                value *= 100

            return value

    raise KeyError(
        f"Metric '{name}' not found. "
        f"Available keys: {list(metrics.keys())}"
    )


# ============================================================
# LOAD ALL MODEL RESULTS
# ============================================================

print("=" * 75)
print("BRAIN TUMOR MRI CLASSIFICATION - MODEL COMPARISON")
print("=" * 75)

print("\nLoading saved test metrics...\n")

all_results = {}

for model_name, file_path in MODEL_FILES.items():

    print(f"Loading {model_name}...")

    metrics = load_metrics(file_path)

    all_results[model_name] = {
        "accuracy": get_metric(metrics, "accuracy"),
        "precision": get_metric(metrics, "precision"),
        "recall": get_metric(metrics, "recall"),
        "f1_score": get_metric(metrics, "f1_score"),
    }

    print("✓ Loaded successfully")


# ============================================================
# DISPLAY COMPARISON TABLE
# ============================================================

print("\n")
print("=" * 75)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 75)

header = (
    f"{'Model':<20}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1-score':>12}"
)

print(header)
print("-" * 75)

for model_name, metrics in all_results.items():

    print(
        f"{model_name:<20}"
        f"{metrics['accuracy']:>11.2f}%"
        f"{metrics['precision']:>11.2f}%"
        f"{metrics['recall']:>11.2f}%"
        f"{metrics['f1_score']:>11.2f}%"
    )

print("=" * 75)


# ============================================================
# FIND BEST MODEL FOR EACH METRIC
# ============================================================

metrics_names = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]

print("\nBEST RESULT FOR EACH METRIC")
print("=" * 75)

for metric in metrics_names:

    best_model = max(
        all_results,
        key=lambda model: all_results[model][metric]
    )

    best_value = all_results[best_model][metric]

    display_name = metric.replace("_", " ").title()

    print(
        f"{display_name:<15}: "
        f"{best_model:<15} "
        f"({best_value:.2f}%)"
    )


# ============================================================
# OVERALL BEST MODEL BASED ON TEST ACCURACY
# ============================================================

best_model = max(
    all_results,
    key=lambda model: all_results[model]["accuracy"]
)

best_accuracy = all_results[best_model]["accuracy"]

print("\n")
print("=" * 75)
print("OVERALL BEST MODEL")
print("=" * 75)

print(f"Model         : {best_model}")
print(f"Test Accuracy : {best_accuracy:.2f}%")

print("=" * 75)


# ============================================================
# PREPARE DATA FOR GRAPHS
# ============================================================

model_names = list(all_results.keys())

accuracies = [
    all_results[model]["accuracy"]
    for model in model_names
]

precisions = [
    all_results[model]["precision"]
    for model in model_names
]

recalls = [
    all_results[model]["recall"]
    for model in model_names
]

f1_scores = [
    all_results[model]["f1_score"]
    for model in model_names
]


# ============================================================
# GRAPH 1 - ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    model_names,
    accuracies
)

plt.title(
    "Brain Tumor Classification - Model Accuracy Comparison",
    fontsize=14
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")

plt.ylim(0, 100)

for bar, value in zip(bars, accuracies):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=11
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

accuracy_graph_path = (
    FIGURES_DIR /
    "model_accuracy_comparison.png"
)

plt.savefig(
    accuracy_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\n✓ Accuracy comparison graph saved:\n"
    f"{accuracy_graph_path}"
)


# ============================================================
# GRAPH 2 - ALL METRICS COMPARISON
# ============================================================

x = np.arange(len(model_names))

width = 0.20

plt.figure(figsize=(12, 7))

plt.bar(
    x - 1.5 * width,
    accuracies,
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    precisions,
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    recalls,
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    f1_scores,
    width,
    label="F1-score"
)

plt.title(
    "Brain Tumor MRI Classification - Performance Comparison",
    fontsize=14
)

plt.ylabel("Score (%)")
plt.xlabel("Model")

plt.xticks(
    x,
    model_names
)

plt.ylim(0, 100)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

metrics_graph_path = (
    FIGURES_DIR /
    "model_metrics_comparison.png"
)

plt.savefig(
    metrics_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\n✓ Metrics comparison graph saved:\n"
    f"{metrics_graph_path}"
)


# ============================================================
# SAVE COMPARISON RESULTS AS JSON
# ============================================================

comparison_output = {
    "models": all_results,
    "best_model": best_model,
    "best_test_accuracy": best_accuracy,
}

comparison_json_path = (
    RESULTS_DIR /
    "model_comparison.json"
)

with open(
    comparison_json_path,
    "w"
) as file:

    json.dump(
        comparison_output,
        file,
        indent=4
    )

print(
    f"\n✓ Model comparison JSON saved:\n"
    f"{comparison_json_path}"
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = (
    RESULTS_DIR /
    "model_comparison_report.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(
        "BRAIN TUMOR MRI CLASSIFICATION\n"
    )

    file.write(
        "MODEL COMPARISON REPORT\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    for model_name, metrics in all_results.items():

        file.write(
            f"{model_name}\n"
        )

        file.write(
            f"Accuracy  : "
            f"{metrics['accuracy']:.2f}%\n"
        )

        file.write(
            f"Precision : "
            f"{metrics['precision']:.2f}%\n"
        )

        file.write(
            f"Recall    : "
            f"{metrics['recall']:.2f}%\n"
        )

        file.write(
            f"F1-score  : "
            f"{metrics['f1_score']:.2f}%\n\n"
        )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        f"Highest test accuracy: "
        f"{best_model} "
        f"({best_accuracy:.2f}%)\n"
    )

print(
    f"\n✓ Comparison report saved:\n"
    f"{report_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("MODEL COMPARISON SUMMARY")
print("=" * 75)

for model_name, metrics in all_results.items():

    print(
        f"{model_name:<20}: "
        f"{metrics['accuracy']:.2f}% accuracy"
    )

print("-" * 75)

print(
    f"Highest test accuracy : "
    f"{best_model} ({best_accuracy:.2f}%)"
)

print("=" * 75)

print(
    "\n✓ MODEL COMPARISON COMPLETED"
)