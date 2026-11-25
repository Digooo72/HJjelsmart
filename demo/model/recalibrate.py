"""
recalibrate.py

Refits ONLY the calibration temperature for every already-trained category
model, without retraining any weights. Use this after fixing the logits
bug in analyzer.py/train.py - the trained models are fine (per-category
accuracy was already 100% for most categories), only their temperature
values were wrong because they were fit on reconstructed-from-softmax
"logits" instead of the model's real pre-softmax output.

This is much faster than finomhangolas.py's full retrain, and isolates
the fix: if the ensemble suddenly works after just recalibrating (no
retraining), that confirms the logits bug was the real root cause.

NOTE: fit_temperature() minimizes NLL on the validation split. For
categories with a small, near-perfectly-classified val set (e.g. 6-9
samples with 0 mistakes), there's no error signal pushing back against
shrinking T -> 0, since a lower temperature just makes every already-
correct prediction's probability approach 1 with no penalty. Left
unchecked this collapses to a degenerate temperature and the model
reports ~100% confidence on everything, right or wrong - which silently
breaks any ensemble/argmax-over-confidence strategy across categories.
MIN_TEMPERATURE below guards against that degenerate case.

Usage:
    python recalibrate.py                  # recalibrate every category
    python recalibrate.py --category food  # just one category
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from categories import CATEGORY_LABELS, WORD_TO_CATEGORY
from train import (
    load_manifest,
    load_split,
    build_logits_model,
    fit_temperature,
    model_paths,
)

# Floor for the fitted temperature. Prevents small/clean val splits from
# collapsing NLL-minimization to T -> 0, which would saturate confidence
# to ~100% on every prediction (right or wrong) and break cross-category
# confidence comparisons in the ensemble.
MIN_TEMPERATURE = 1.0


def recalibrate_one(category):
    print(f"\n=== Recalibrating: {category} ===")

    rows = load_manifest()
    rows = [r for r in rows if WORD_TO_CATEGORY.get(r["label"]) == category]
    if not rows:
        print(f"  No manifest rows for '{category}', skipping.")
        return

    model_path, _best_path, labels_path = model_paths(category)
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print(f"  No trained model/labels for '{category}' yet, skipping.")
        return

    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data["labels"] if isinstance(data, dict) else data
    old_temperature = data.get("temperature", 1.0) if isinstance(data, dict) else 1.0

    encoder = LabelEncoder()
    encoder.classes_ = np.array(labels)

    print("  Loading val split...")
    X_val, y_val_raw = load_split(rows, "val")
    if len(X_val) == 0:
        print(f"  No val samples for '{category}' - can't recalibrate, skipping.")
        return
    y_val = encoder.transform(y_val_raw)

    if len(labels) < 2:
        print("  Single-class model - calibration is meaningless, leaving temperature=1.0.")
        new_temperature = 1.0
    else:
        print("  Loading trained model...")
        model = tf.keras.models.load_model(model_path)
        logits_model = build_logits_model(model)

        print("  Computing real logits on val split...")
        val_logits = logits_model.predict(X_val, verbose=0)

        # --- Diagnostic: are these actually pre-softmax logits? ---------
        row_sums = val_logits.sum(axis=1)
        row_min = val_logits.min()
        row_max = val_logits.max()
        looks_like_probs = (
            row_min >= -1e-4
            and row_max <= 1.0 + 1e-4
            and np.allclose(row_sums, 1.0, atol=1e-3)
        )
        print(f"  [diag] logits range: [{row_min:.4f}, {row_max:.4f}], "
              f"row-sum mean: {row_sums.mean():.4f} (should be large/unbounded "
              f"for real logits, NOT ~1.0)")
        if looks_like_probs:
            print("  [diag] !! These look like SOFTMAX PROBABILITIES, not logits "
                  "(bounded [0,1], rows sum to ~1). build_logits_model is probably "
                  "not stripping the final softmax activation. Temperature scaling "
                  "on these values cannot produce well-calibrated confidences - "
                  "fix build_logits_model before trusting this recalibration.")
        # ------------------------------------------------------------------

        fitted_temperature = fit_temperature(val_logits, y_val)

        if fitted_temperature < MIN_TEMPERATURE:
            print(f"  Fitted temperature {fitted_temperature:.3f} is below the floor "
                  f"({MIN_TEMPERATURE:.3f}) - val split likely has too few samples/mistakes "
                  f"to constrain it. Clamping to the floor.")
            new_temperature = MIN_TEMPERATURE
        else:
            new_temperature = fitted_temperature

    print(f"  Old (broken) temperature: {old_temperature:.3f}")
    print(f"  New (recalibrated) temperature: {new_temperature:.3f}")

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {"labels": labels, "temperature": new_temperature},
            f, ensure_ascii=False, indent=2,
        )
    print(f"  Saved -> {labels_path}")


def recalibrate_all():
    for cat_id in CATEGORY_LABELS:
        recalibrate_one(cat_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refit calibration temperature only, no retraining.")
    parser.add_argument("--category", default=None,
                         help="recalibrate only this category; omit to do all categories")
    args = parser.parse_args()

    if args.category:
        recalibrate_one(args.category)
    else:
        recalibrate_all()