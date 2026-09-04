import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_ANNOTATIONS_DIR = SCRIPT_DIR.parent / "annotations"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "reports" / "csv_50pct"


def export_split(split, annotations_dir, output_dir, ratio, seed):
    annotation_path = annotations_dir / f"{split}.pkl"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open("rb") as file:
        data = pickle.load(file)

    image_names = np.asarray(data["image_name"])
    labels = np.asarray(data["label"])
    dataset_ids = np.asarray(data["dataset_ids"])
    dataset_names = np.asarray(data["dataset_names"])
    attribute_names = list(data["attr_name"])

    sample_count = int(len(image_names) * ratio)
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(image_names), sample_count, replace=False))

    fieldnames = ["image_name", "dataset_id", "dataset_name"] + attribute_names
    output_path = output_dir / f"{split}_50pct.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        for index in selected_indices:
            writer.writerow(
                [image_names[index], dataset_ids[index], dataset_names[index]]
                + labels[index].tolist()
            )

    print(f"{split}: {sample_count:,}/{len(image_names):,} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export 50% of each UPAR split to CSV.")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.ratio <= 1:
        parser.error("--ratio must be greater than 0 and at most 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        export_split(split, args.annotations_dir, args.output_dir, args.ratio, args.seed)


if __name__ == "__main__":
    main()