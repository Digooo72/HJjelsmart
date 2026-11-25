"""
Scans dataset/ for valid (non-empty) .npy files and builds a clean manifest
(dataset/manifest.csv) ready to feed into a training script.

Handles the reality of this dataset:
  - Skips any .npy that's empty/broken (still_broken.json from the repair
    script, or anything with .size == 0).
  - Classes with only 1 usable sample go entirely into the TRAIN split -
    there's nothing to validate against, so they're excluded from val/test
    but still contribute to training rather than being thrown away.
  - Classes with >= 2 samples get a proper split.
  - Augmented files ("name__aug0.npy", "name__aug1.npy", ...) are SKIPPED
    entirely here - they're augment_dataset.py's territory, which appends
    them to manifest.csv itself (always split="train") after this script
    runs. Re-running this script never touches them, so a near-duplicate
    augmented sample can never get randomly shuffled into val/test.

Output columns: filepath, label, num_frames, split
"""

import os
import json
import csv
import random
import re
import argparse

DATASET_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
LABELS_JSON  = os.path.join(DATASET_DIR, "labels.json")
MANIFEST_OUT = os.path.join(DATASET_DIR, "manifest.csv")

random.seed(42)  # reproducible splits

_cli_parser = argparse.ArgumentParser(add_help=True)
_cli_parser.add_argument(
    "--min-samples", type=int, default=None,
    help="Override MIN_SAMPLES_PER_CLASS for this run without editing the file. "
         "Pass a number, or 0 to disable filtering entirely."
)
_cli_args, _ = _cli_parser.parse_known_args()

# MIN_SAMPLES_PER_CLASS: classes with fewer than this many original (non-
# augmented) samples are EXCLUDED from manifest.csv entirely. Their .npy
# files are left untouched on disk - they just won't be trained on until
# more recordings push them over the threshold.
#
# Why a threshold instead of a fixed class count (the old
# TOP_N_CLASSES_FOR_DEBUG approach): picking "top 20 classes" can still let
# a thin class in if there simply aren't 20 well-stocked ones, and it also
# arbitrarily excludes a well-stocked 21st class. A threshold keeps every
# class that's actually learnable, however many that turns out to be.
#
# Set to None to disable filtering and include every class regardless of
# sample count (this is what caused the accuracy collapse from ~190 classes
# with 4-6 samples each swamping the well-stocked ones - avoid unless you
# know what you're doing).
MIN_SAMPLES_PER_CLASS = 5
if _cli_args.min_samples is not None:
    MIN_SAMPLES_PER_CLASS = None if _cli_args.min_samples == 0 else _cli_args.min_samples
    print(f"*** --min-samples passed on CLI: using MIN_SAMPLES_PER_CLASS={MIN_SAMPLES_PER_CLASS} "
          f"(overriding the hardcoded default) ***")

# Matches the "__aug{i}" suffix that augment_dataset.py appends, e.g.
# "ablak_2__aug0.npy" -> stripping this gives back "ablak_2.npy", the
# original file that labels.json actually has an entry for.
AUG_SUFFIX_RE = re.compile(r'__aug\d+$')


def load_label_map():
    """
    Supports both label.json formats:
      old: {out_name: "label"}
      new: {out_name: {"label": "...", "source_video": "..."}}
    Falls back to deriving the label from the filename if labels.json
    is missing or a file isn't listed in it.
    """
    label_map = {}
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for out_name, value in raw.items():
            if isinstance(value, dict):
                label_map[out_name] = value["label"]
            else:
                label_map[out_name] = value
    return label_map


def derive_label_from_filename(fname):
    # Strip an augmented-file suffix first ("ablak_2__aug0" -> "ablak_2"),
    # so it doesn't get treated as part of the label below.
    stem = fname[:-4]  # remove ".npy"
    stem = AUG_SUFFIX_RE.sub('', stem)
    # strip "_<N>" take/recording number suffix
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def is_broken(path):
    try:
        import numpy as np
        arr = np.load(path, mmap_mode="r")  # mmap = fast, doesn't load full array
        return arr.size == 0
    except Exception:
        return True


def build_manifest():
    import numpy as np

    label_map = load_label_map()
    all_files = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith(".npy"))

    # Augmented files are augment_dataset.py's territory: it appends them to
    # manifest.csv itself with split="train" once this manifest exists. If we
    # included them here, re-running this script after augmenting would fold
    # them back into the normal train/val/test split logic and could shuffle
    # a near-duplicate augmented sample into val/test - exactly the leakage
    # augment_dataset.py's docstring says to avoid. So: originals only.
    aug_files = [f for f in all_files if AUG_SUFFIX_RE.search(os.path.splitext(f)[0])]
    all_files = [f for f in all_files if f not in set(aug_files)]

    print(f"Found {len(all_files)} original .npy files in {DATASET_DIR} "
          f"({len(aug_files)} augmented files skipped - handled by augment_dataset.py)")

    if aug_files:
        print(
            "\n*** WARNING: found augmented ('__aug') files already in dataset/. ***\n"
            "*** This script intentionally does NOT include them - see the   ***\n"
            "*** module docstring - so they will NOT appear in the manifest  ***\n"
            "*** this run produces. If augment_dataset.py already ran before ***\n"
            "*** this script in your pipeline, that augmented data is now    ***\n"
            "*** effectively orphaned. Correct order is:                    ***\n"
            "***   dataset_builder.py -> build_manifest.py -> augment_dataset.py -> train.py\n"
            "*** Run augment_dataset.py again AFTER this script finishes to  ***\n"
            "*** regenerate them against the fresh manifest.                ***\n"
        )

    rows = []
    skipped_broken = 0

    for fname in all_files:
        path = os.path.join(DATASET_DIR, fname)

        if is_broken(path):
            skipped_broken += 1
            continue

        label = label_map.get(fname) or derive_label_from_filename(fname)

        arr = np.load(path, mmap_mode="r")
        num_frames = arr.shape[0]

        rows.append({"filepath": os.path.join("dataset", fname),
                      "label": label,
                      "num_frames": num_frames})

    print(f"Skipped {skipped_broken} broken/empty files")
    print(f"{len(rows)} usable samples remain")

    # Always show the real (pre-augmentation) distribution before filtering,
    # so MIN_SAMPLES_PER_CLASS is picked based on actual data, not a guess.
    raw_counts = {}
    for row in rows:
        raw_counts[row["label"]] = raw_counts.get(row["label"], 0) + 1
    sorted_counts = sorted(raw_counts.items(), key=lambda kv: -kv[1])
    print(f"\n*** Real (non-augmented) sample counts, top 30: ***")
    for lbl, c in sorted_counts[:30]:
        print(f"    {lbl:30s} {c}")
    counts_list = sorted(raw_counts.values())
    print(f"*** Median: {counts_list[len(counts_list)//2]}, "
          f"Max: {sorted_counts[0][1]}, "
          f"Classes with >=8: {sum(1 for c in raw_counts.values() if c>=8)}, "
          f"Classes with >=6: {sum(1 for c in raw_counts.values() if c>=6)} ***\n")

    if MIN_SAMPLES_PER_CLASS is not None:
        counts = {}
        for row in rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1

        kept_labels = {lbl for lbl, c in counts.items() if c >= MIN_SAMPLES_PER_CLASS}
        dropped_labels = sorted(
            (lbl for lbl, c in counts.items() if c < MIN_SAMPLES_PER_CLASS),
            key=lambda lbl: (-counts[lbl], lbl)
        )

        rows = [r for r in rows if r["label"] in kept_labels]

        print(
            f"\n*** MIN_SAMPLES_PER_CLASS={MIN_SAMPLES_PER_CLASS} ***\n"
            f"*** Keeping {len(kept_labels)} classes with enough data ({len(rows)} samples). ***\n"
            f"*** Excluding {len(dropped_labels)} classes below the threshold - their .npy   ***\n"
            f"*** files are untouched on disk, they just need more recordings before        ***\n"
            f"*** they'll be included in training.                                          ***\n"
        )

        # Write the dropped list to disk so it's a usable "go record more of
        # these" checklist rather than scrollback you have to copy by hand.
        dropped_path = os.path.join(DATASET_DIR, "excluded_low_sample_classes.txt")
        with open(dropped_path, "w", encoding="utf-8") as f:
            f.write(f"# Classes excluded from manifest.csv (fewer than "
                     f"{MIN_SAMPLES_PER_CLASS} original samples)\n")
            f.write(f"# label\tcurrent_sample_count\tneed_at_least\n")
            for lbl in dropped_labels:
                f.write(f"{lbl}\t{counts[lbl]}\t{MIN_SAMPLES_PER_CLASS}\n")
        print(f"*** Full list written to: {dropped_path} ***\n")

    # Group by label to decide splits
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    num_classes = len(by_label)
    singleton_classes = sum(1 for v in by_label.values() if len(v) == 1)
    print(f"{num_classes} classes total, {singleton_classes} with only 1 sample "
          f"(these go to train only, no val)")

    for label, samples in by_label.items():
        random.shuffle(samples)
        if len(samples) == 1:
            samples[0]["split"] = "train"
        elif len(samples) == 2:
            # not enough for train/val/test - do train/val, skip test
            samples[0]["split"] = "train"
            samples[1]["split"] = "val"
        else:
            n = len(samples)
            n_val = max(1, round(n * 0.15))
            n_test = max(1, round(n * 0.15))
            n_train = n - n_val - n_test
            for i, s in enumerate(samples):
                if i < n_train:
                    s["split"] = "train"
                elif i < n_train + n_val:
                    s["split"] = "val"
                else:
                    s["split"] = "test"

    split_counts = {"train": 0, "val": 0, "test": 0}
    for samples in by_label.values():
        for s in samples:
            split_counts[s["split"]] += 1

    print(f"Split sizes: {split_counts}")

    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "num_frames", "split"])
        writer.writeheader()
        for samples in by_label.values():
            for s in samples:
                writer.writerow(s)

    print(f"\nManifest written to {MANIFEST_OUT}")


if __name__ == "__main__":
    build_manifest()