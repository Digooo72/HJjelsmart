"""
Generates augmented copies of TRAIN-split samples only (never val/test - that
would leak near-duplicates across splits and give you a fake accuracy number).

For each ORIGINAL train sample (never an already-augmented one - see note
below), creates N augmented variants combining:
  - small rotation (around the depth/z axis, simulating camera angle)
  - small scale jitter (simulating distance from camera)
  - gaussian coordinate noise (simulating landmark-detection imprecision)
  - small translation (simulating hand position in frame)
  - time warp (speeding up/slowing down the sequence slightly)

Deliberately NOT doing left/right mirroring by default - many signs are
directionally/asymmetrically meaningful, so mirroring could silently turn a
valid sign into a different (or nonsense) one.

Augmented files are saved as: original_name__aug{i}.npy
and appended to manifest.csv with split="train".

IDEMPOTENCY: this script is safe to re-run. Any previously-generated
"__aug*" rows/files (from an earlier run) are detected, deleted, and
regenerated fresh from the originals. Without this, re-running would find
last run's augmented files still marked split="train" in the manifest and
augment THOSE too - rotating a rotation, adding noise to noise, warping an
already-warped sequence. That compounding distortion (with every generation
still carrying the original label) is a real way to quietly wreck training.

ZERO-LANDMARK SAFETY: dataset_builder.py writes [0,0,0] for any hand that
MediaPipe didn't detect in a given frame - that's a "no hand here" sentinel,
not a real coordinate. translate() and add_noise() would otherwise nudge
those zeros to some small nonzero value, fabricating a hand where none was
detected. We track which (frame, hand-landmark) entries were genuinely
absent and force them back to exact zero after augmenting.
"""

import os
import csv

import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

AUGMENTATIONS_PER_SAMPLE = 3  # tune this - more for classes with very few samples if you want
RNG = np.random.default_rng(42)

AUG_MARKER = "__aug"  # substring identifying an already-augmented file


def is_augmented_path(filepath):
    return AUG_MARKER in os.path.basename(filepath)


def rotate_z(sequence, angle_degrees):
    """Rotate all landmarks around the depth axis (z) - simulates camera angle."""
    theta = np.radians(angle_degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1],
    ])
    return sequence @ rot_matrix.T


def scale(sequence, factor):
    return sequence * factor


def add_noise(sequence, std):
    noise = RNG.normal(0, std, size=sequence.shape)
    return sequence + noise


def translate(sequence, shift):
    return sequence + shift  # shift: array of shape (3,)


def time_warp_with_presence(sequence, speed_factor):
    """
    Resample the sequence to speed_factor * original length via linear
    interpolation along the time axis, simulating faster/slower signing.

    Also resamples a per-landmark "was this hand actually detected here"
    presence mask using NEAREST-neighbor lookup (not interpolation) so it
    stays exactly binary - linearly interpolating across a "hand appears
    partway through the clip" boundary would otherwise produce fractional
    ghost-hand values that are neither a real zero nor a real detection.

    Returns (warped_sequence, presence_mask) where presence_mask has shape
    (new_num_frames, num_landmarks) and is True where a hand was detected.
    """
    num_frames = sequence.shape[0]
    new_len = max(2, int(round(num_frames * speed_factor)))
    old_idx = np.linspace(0, num_frames - 1, num_frames)
    new_idx = np.linspace(0, num_frames - 1, new_len)

    # interpolate each (landmark, coord) independently
    flat = sequence.reshape(num_frames, -1)
    warped_flat = np.stack([
        np.interp(new_idx, old_idx, flat[:, i]) for i in range(flat.shape[1])
    ], axis=1)
    warped = warped_flat.reshape(new_len, *sequence.shape[1:])

    presence = ~np.all(sequence == 0, axis=-1)  # (num_frames, num_landmarks)
    nearest_src_idx = np.clip(np.round(new_idx).astype(int), 0, num_frames - 1)
    warped_presence = presence[nearest_src_idx]  # (new_len, num_landmarks)

    return warped, warped_presence


def augment_once(sequence):
    seq = sequence.copy()

    seq, presence = time_warp_with_presence(seq, speed_factor=RNG.uniform(0.85, 1.15))
    seq = rotate_z(seq, angle_degrees=RNG.uniform(-10, 10))
    seq = scale(seq, factor=RNG.uniform(0.9, 1.1))
    seq = translate(seq, shift=RNG.normal(0, 0.02, size=3))
    seq = add_noise(seq, std=0.005)

    # Never fabricate a hand in a frame/landmark where one wasn't detected.
    seq = np.where(presence[..., None], seq, 0.0)

    return seq.astype(np.float32)


def main():
    rows = []
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # ── purge any stale augmented rows/files from a previous run ─────────────
    original_rows = [r for r in rows if not is_augmented_path(r["filepath"])]
    stale_aug_rows = [r for r in rows if is_augmented_path(r["filepath"])]

    if stale_aug_rows:
        print(f"Found {len(stale_aug_rows)} augmented rows from a previous run — "
              f"purging them so they don't get re-augmented (aug-of-aug).")
        for r in stale_aug_rows:
            stale_path = os.path.join(PROJECT_ROOT, r["filepath"])
            if os.path.exists(stale_path):
                os.remove(stale_path)

    train_rows = [r for r in original_rows if r["split"] == "train"]
    print(f"Augmenting {len(train_rows)} original train samples x{AUGMENTATIONS_PER_SAMPLE} each...")

    new_rows = []
    for row in train_rows:
        src_path = os.path.join(PROJECT_ROOT, row["filepath"])
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} missing, skipping.")
            continue

        sequence = np.load(src_path).astype(np.float32)
        base_name = os.path.splitext(os.path.basename(row["filepath"]))[0]

        for i in range(AUGMENTATIONS_PER_SAMPLE):
            augmented = augment_once(sequence)
            out_name = f"{base_name}{AUG_MARKER}{i}.npy"
            out_path = os.path.join(DATASET_DIR, out_name)
            np.save(out_path, augmented)

            new_rows.append({
                "filepath": os.path.join("dataset", out_name),
                "label": row["label"],
                "num_frames": augmented.shape[0],
                "split": "train",
            })

    print(f"Created {len(new_rows)} augmented files.")

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in original_rows + new_rows:
            writer.writerow(row)

    print(f"Manifest updated: {MANIFEST_PATH}")
    print(f"Total rows now: {len(original_rows) + len(new_rows)}")


if __name__ == "__main__":
    main()