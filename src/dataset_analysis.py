from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, UnidentifiedImageError
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

SPLITS = ["train", "valid", "test"]
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

print("=" * 70)
print("BRAIN TUMOR MRI DATASET ANALYSIS")
print("=" * 70)

print(f"\nDataset location: {DATASET_DIR}")

# ============================================================
# 1. CHECK DATASET STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET STRUCTURE CHECK")
print("=" * 70)

structure_ok = True

for split in SPLITS:

    split_path = DATASET_DIR / split

    if split_path.exists():
        print(f"\n✓ Found split: {split}")
    else:
        print(f"\n✗ Missing split: {split}")
        structure_ok = False
        continue

    csv_path = split_path / "_classes.csv"

    if csv_path.exists():
        print("  ✓ _classes.csv found")
    else:
        print("  ✗ _classes.csv missing")
        structure_ok = False

    for class_name in CLASSES:

        class_path = split_path / class_name

        if class_path.exists():
            print(f"  ✓ {class_name}")
        else:
            print(f"  ✗ Missing class folder: {class_name}")
            structure_ok = False


# ============================================================
# 2. COUNT IMAGES PER CLASS
# ============================================================

print("\n" + "=" * 70)
print("2. IMAGE COUNT PER CLASS")
print("=" * 70)

all_counts = {}

for split in SPLITS:

    print(f"\n{split.upper()} SET")

    split_counts = {}

    for class_name in CLASSES:

        class_path = DATASET_DIR / split / class_name

        images = [
            file for file in class_path.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        count = len(images)

        split_counts[class_name] = count

        print(f"  {class_name:<12}: {count}")

    total = sum(split_counts.values())

    print(f"  {'TOTAL':<12}: {total}")

    all_counts[split] = split_counts


# ============================================================
# 3. READ CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("3. CSV ANALYSIS")
print("=" * 70)

for split in SPLITS:

    csv_path = DATASET_DIR / split / "_classes.csv"

    print(f"\n{split.upper()} CSV")

    try:

        df = pd.read_csv(csv_path)

        # Clean column names
        df.columns = [column.strip() for column in df.columns]

        print(f"  Rows    : {len(df)}")
        print(f"  Columns : {list(df.columns)}")

        # Show label totals
        label_columns = [
            column for column in df.columns
            if column.lower() != "filename"
        ]

        print("  Label totals:")

        for column in label_columns:

            if pd.api.types.is_numeric_dtype(df[column]):
                print(
                    f"    {column:<15}: "
                    f"{int(df[column].fillna(0).sum())}"
                )

    except Exception as error:

        print(f"  ✗ Could not read CSV: {error}")


# ============================================================
# 4. IMAGE INTEGRITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("4. IMAGE INTEGRITY CHECK")
print("=" * 70)

corrupted_images = []
total_checked = 0

for split in SPLITS:

    print(f"\nChecking {split} images...")

    for class_name in CLASSES:

        class_path = DATASET_DIR / split / class_name

        for image_path in class_path.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):

                total_checked += 1

                try:

                    with Image.open(image_path) as image:
                        image.verify()

                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError
                ) as error:

                    corrupted_images.append(
                        (str(image_path), str(error))
                    )

print(f"\nImages checked   : {total_checked}")
print(f"Corrupted images : {len(corrupted_images)}")

if corrupted_images:

    print("\nCorrupted files:")

    for path, error in corrupted_images:
        print(f"  ✗ {path}")
        print(f"    {error}")

else:

    print("✓ No corrupted images detected.")


# ============================================================
# 5. IMAGE RESOLUTION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. IMAGE RESOLUTION ANALYSIS")
print("=" * 70)

resolution_counts = Counter()
mode_counts = Counter()

widths = []
heights = []

for split in SPLITS:

    for class_name in CLASSES:

        class_path = DATASET_DIR / split / class_name

        for image_path in class_path.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):

                try:

                    with Image.open(image_path) as image:

                        width, height = image.size

                        resolution_counts[(width, height)] += 1
                        mode_counts[image.mode] += 1

                        widths.append(width)
                        heights.append(height)

                except Exception:
                    pass

print(f"\nUnique resolutions : {len(resolution_counts)}")

if widths and heights:

    print(f"Minimum width       : {min(widths)}")
    print(f"Maximum width       : {max(widths)}")
    print(f"Minimum height      : {min(heights)}")
    print(f"Maximum height      : {max(heights)}")

print("\nMost common resolutions:")

for resolution, count in resolution_counts.most_common(10):

    print(
        f"  {resolution[0]} x {resolution[1]}"
        f" : {count} images"
    )

print("\nImage color modes:")

for mode, count in mode_counts.items():
    print(f"  {mode}: {count}")


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("6. CLASS DISTRIBUTION")
print("=" * 70)

overall_counts = defaultdict(int)

for split in SPLITS:

    for class_name, count in all_counts[split].items():
        overall_counts[class_name] += count

total_images = sum(overall_counts.values())

for class_name in CLASSES:

    count = overall_counts[class_name]

    percentage = (
        (count / total_images) * 100
        if total_images > 0
        else 0
    )

    print(
        f"{class_name:<12}: "
        f"{count:<5} "
        f"({percentage:.2f}%)"
    )

print(f"\nTOTAL MRI IMAGES: {total_images}")


# ============================================================
# 7. EXPECTED DATASET CHECK
# ============================================================

print("\n" + "=" * 70)
print("7. EXPECTED DATASET SIZE CHECK")
print("=" * 70)

expected_counts = {
    "train": 1695,
    "valid": 502,
    "test": 246
}

dataset_valid = True

for split in SPLITS:

    actual = sum(all_counts[split].values())
    expected = expected_counts[split]

    if actual == expected:

        print(
            f"✓ {split:<5}: "
            f"{actual} images "
            f"(expected {expected})"
        )

    else:

        print(
            f"✗ {split:<5}: "
            f"{actual} images "
            f"(expected {expected})"
        )

        dataset_valid = False


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET VALIDATION SUMMARY")
print("=" * 70)

print(f"Folder structure valid : {structure_ok}")
print(f"Dataset size valid      : {dataset_valid}")
print(f"Corrupted images        : {len(corrupted_images)}")
print(f"Total MRI images        : {total_images}")

if (
    structure_ok
    and dataset_valid
    and len(corrupted_images) == 0
):

    print("\n✓ DATASET VALIDATION PASSED")
    print("Dataset is ready for preprocessing and model training.")

else:

    print("\n⚠ DATASET VALIDATION REQUIRES ATTENTION")

print("=" * 70)