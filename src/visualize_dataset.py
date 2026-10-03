from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================

print("=" * 70)
print("BRAIN TUMOR MRI DATASET VISUALIZATION")
print("=" * 70)

class_counts = defaultdict(int)

for split in SPLITS:

    for class_name in CLASSES:

        class_dir = DATASET_DIR / split / class_name

        images = [
            file for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        class_counts[class_name] += len(images)


print("\nOverall class distribution:")

for class_name in CLASSES:
    print(f"{class_name:<12}: {class_counts[class_name]}")


# ============================================================
# 2. CLASS DISTRIBUTION BAR GRAPH
# ============================================================

labels = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

values = [class_counts[class_name] for class_name in CLASSES]

plt.figure(figsize=(9, 6))

bars = plt.bar(labels, values)

plt.title(
    "Brain Tumor MRI Dataset - Class Distribution",
    fontsize=14
)

plt.xlabel("Tumor Class")
plt.ylabel("Number of Images")

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 10,
        str(value),
        ha="center"
    )

plt.tight_layout()

bar_path = RESULTS_DIR / "class_distribution.png"

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\n✓ Saved: {bar_path}")


# ============================================================
# 3. DISPLAY SAMPLE MRI IMAGES
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(14, 7)
)

fig.suptitle(
    "Sample Brain MRI Images by Class",
    fontsize=16
)

for column, class_name in enumerate(CLASSES):

    class_dir = DATASET_DIR / "train" / class_name

    images = sorted([
        file for file in class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ])

    selected_images = images[:2]

    for row in range(2):

        ax = axes[row, column]

        if row < len(selected_images):

            image_path = selected_images[row]

            with Image.open(image_path) as image:

                image = image.convert("RGB")

                ax.imshow(image)

            ax.set_title(
                class_name.replace("_", " ").title()
            )

        ax.axis("off")


plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

samples_path = RESULTS_DIR / "sample_mri_images.png"

plt.savefig(
    samples_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"✓ Saved: {samples_path}")


# ============================================================
# 4. TRAIN / VALID / TEST DISTRIBUTION
# ============================================================

split_counts = {}

for split in SPLITS:

    split_counts[split] = []

    for class_name in CLASSES:

        class_dir = DATASET_DIR / split / class_name

        count = len([
            file for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ])

        split_counts[split].append(count)


x = range(len(CLASSES))

width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width for i in x],
    split_counts["train"],
    width,
    label="Train"
)

plt.bar(
    x,
    split_counts["valid"],
    width,
    label="Validation"
)

plt.bar(
    [i + width for i in x],
    split_counts["test"],
    width,
    label="Test"
)

plt.xticks(
    x,
    labels
)

plt.xlabel("Tumor Class")
plt.ylabel("Number of Images")

plt.title(
    "Train / Validation / Test Class Distribution"
)

plt.legend()

plt.tight_layout()

split_path = RESULTS_DIR / "dataset_split_distribution.png"

plt.savefig(
    split_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"✓ Saved: {split_path}")


print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)