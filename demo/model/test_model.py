"""
Loads trained_model.h5 and runs it against the "test" split from manifest.csv
(samples the model has never seen during training or validation).

Reports:
  - Top-1 accuracy (exact match)
  - Top-5 accuracy (correct answer was in the model's top 5 guesses) -
    worth tracking separately since with many classes, top-1 is a much
    harder bar than a typical classification problem.
  - A handful of example predictions so you can eyeball what it's
    getting right/wrong, not just stare at a percentage.

IMPORTANT: preprocessing here is imported directly from train.py
(process_sequence, MAX_FRAMES, FEATURE_DIM) instead of being redefined in
this file. Train/test must see data preprocessed IDENTICALLY - copy-pasting
the transform into a second file is exactly how this broke last time (this
script was still building the old 126-dim raw/unnormalized features while
trained_model.h5 expected the new 128-dim normalized+presence-flagged
ones, which throws a matrix-shape error at model.predict()). Importing the
one true implementation makes that class of bug impossible going forward.
"""

import os
import csv
import json
import sys

import numpy as np
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(__file__))
from train import process_sequence, MAX_FRAMES, FEATURE_DIM  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "weights", "trained_model.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels",  "label_classes.json")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_test_split():
    rows = []
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test":
                rows.append(row)
    return rows


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH} - run train.py first.")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    # train.py now saves label_classes.json as {"labels": [...], "temperature": T}
    # instead of a bare list, so the calibrated confidence temperature can
    # travel alongside the label order. Handle both formats here so this
    # script keeps working whether you're pointed at a freshly retrained
    # file or an older bare-list one still lying around.
    if isinstance(label_data, dict):
        classes = label_data["labels"]
    else:
        classes = label_data  # old format: plain list, index -> label

    rows = load_test_split()
    if len(rows) == 0:
        print(
            "No test samples in manifest.csv - this happens if most/all of "
            "your classes only had 1-2 samples (everything went to train/val "
            "instead). That's expected right now given the sample counts; it "
            "means you can't get a trustworthy accuracy number yet, only "
            "spot-check individual predictions by hand (see below)."
        )
        return

    print(f"Evaluating on {len(rows)} held-out test samples...")

    X = []
    y_true_labels = []
    for row in rows:
        path = os.path.join(PROJECT_ROOT, row["filepath"])
        if not os.path.exists(path):
            continue
        seq = np.load(path).astype(np.float32)
        features = process_sequence(seq)  # same resample+normalize+presence pipeline as training
        X.append(features)
        y_true_labels.append(row["label"])

    if len(X) == 0:
        print("None of the test-split files were found on disk - nothing to evaluate.")
        return

    X = np.stack(X)
    predictions = model.predict(X, verbose=0)  # shape: (num_samples, num_classes)

    top1_correct = 0
    top5_correct = 0
    examples = []

    for i, probs in enumerate(predictions):
        true_label = y_true_labels[i]
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_labels = [classes[idx] for idx in top5_idx]

        if top5_labels[0] == true_label:
            top1_correct += 1
        if true_label in top5_labels:
            top5_correct += 1

        if i < 15:  # keep a few examples to print
            examples.append((true_label, top5_labels))

    n = len(y_true_labels)
    print(f"\nTop-1 accuracy: {top1_correct}/{n} = {top1_correct/n:.1%}")
    print(f"Top-5 accuracy: {top5_correct}/{n} = {top5_correct/n:.1%}")

    print("\nExample predictions (true -> model's top 5 guesses):")
    for true_label, top5_labels in examples:
        marker = "OK" if top5_labels[0] == true_label else "  "
        print(f"  [{marker}] {true_label!r:30s} -> {top5_labels}")


if __name__ == "__main__":
    main()