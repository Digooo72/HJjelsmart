import argparse
import os
import json
import csv

import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from categories import WORD_TO_CATEGORY, CATEGORY_LABELS

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

MAX_FRAMES = 90  # target length after resampling - since we now interpolate
                  # instead of truncating, this doesn't need to be as large as
                  # your longest clips; it just needs to preserve enough
                  # temporal resolution for the gesture's shape.
NUM_LANDMARKS = 42  # 21 per hand x 2 hands (matches dataset_builder.py output)
NUM_COORDS = 3       # x, y, z per landmark
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS + 2  # +2: left/right hand presence flags


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"'{MANIFEST_PATH}' not found. Run model/build_manifest.py first."
        )
    rows = []
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_hand_presence(raw_sequence):
    """
    Returns (left_presence, right_presence), each a bool array of shape
    (frames,), True where that hand's 21 landmarks weren't the
    dataset_builder.py [0,0,0] "not detected" sentinel.
    """
    left_presence = ~np.all(raw_sequence[:, 0:21, :] == 0, axis=(1, 2))
    right_presence = ~np.all(raw_sequence[:, 21:42, :] == 0, axis=(1, 2))
    return left_presence, right_presence


def resample_sequence(sequence, target_frames=MAX_FRAMES):
    """
    Resamples a (frames, 42, 3) sequence to exactly target_frames, returning
    (coords, left_presence, right_presence).

    Coordinates are linearly interpolated along the time axis - this keeps
    the WHOLE gesture regardless of original length, instead of a long clip
    losing its ending to truncation.

    Presence is resampled separately via NEAREST-neighbor lookup rather than
    interpolation, so it stays exactly True/False. Linearly interpolating
    across a "hand appears partway through the clip" boundary would
    otherwise blend a real position with the [0,0,0] sentinel, fabricating
    a fractional ghost-hand that was never actually detected. The resampled
    presence mask is what lets normalize_sequence() zero those frames back
    out properly instead of leaving fabricated coordinates in the data.
    """
    num_frames = sequence.shape[0]
    left_presence_raw, right_presence_raw = compute_hand_presence(sequence)

    if num_frames < 2:
        # degenerate case - just repeat the single frame
        coords = np.repeat(sequence, target_frames, axis=0).astype(np.float32)
        left_presence = np.repeat(left_presence_raw, target_frames, axis=0)
        right_presence = np.repeat(right_presence_raw, target_frames, axis=0)
        return coords, left_presence, right_presence

    old_idx = np.linspace(0, num_frames - 1, num_frames)
    new_idx = np.linspace(0, num_frames - 1, target_frames)

    if num_frames == target_frames:
        coords = sequence.astype(np.float32)
    else:
        flat = sequence.reshape(num_frames, -1)
        resampled_flat = np.stack([
            np.interp(new_idx, old_idx, flat[:, i]) for i in range(flat.shape[1])
        ], axis=1)
        coords = resampled_flat.reshape(target_frames, *sequence.shape[1:]).astype(np.float32)

    nearest_src_idx = np.clip(np.round(new_idx).astype(int), 0, num_frames - 1)
    left_presence = left_presence_raw[nearest_src_idx]
    right_presence = right_presence_raw[nearest_src_idx]

    return coords, left_presence, right_presence


def normalize_sequence(coords, left_presence, right_presence):
    """
    Re-expresses each hand's landmarks relative to that hand's OWN wrist
    position, scaled by that hand's own size (wrist-to-middle-finger-MCP
    distance), instead of raw frame-relative coordinates.

    Without this, "the same sign performed a bit closer to the camera, or
    slightly left of center" looks like different input to the model - it
    has to separately relearn the hand shape for every distance/position a
    signer happens to use. This is usually the single biggest accuracy
    lever available for a landmark-based dataset like this one.

    Frames/hands with no detected hand are forced back to exact zero using
    the presence mask (not inferred from the coordinates, since after
    normalization [0,0,0] no longer uniquely means "absent" the way the raw
    sentinel did).
    """
    coords = coords.copy()
    for start, presence in [(0, left_presence), (21, right_presence)]:
        end = start + 21
        wrist = coords[:, start:start + 1, :]      # (frames, 1, 3)
        mcp = coords[:, start + 9:start + 10, :]    # middle-finger MCP landmark
        scale = np.linalg.norm(mcp - wrist, axis=-1, keepdims=True)  # (frames, 1, 1)
        scale = np.where(scale < 1e-6, 1.0, scale)  # guard divide-by-zero (absent hand -> scale 0)

        normalized = (coords[:, start:end, :] - wrist) / scale
        presence_mask = presence[:, None, None]
        coords[:, start:end, :] = np.where(presence_mask, normalized, 0.0)

    return coords.astype(np.float32)


def process_sequence(raw_sequence):
    """
    Full per-sample pipeline: resample (frame-count-safe, presence-safe) ->
    normalize (position/scale-invariant) -> flatten with explicit presence
    flags appended. Returns shape (MAX_FRAMES, FEATURE_DIM).
    """
    coords, left_presence, right_presence = resample_sequence(raw_sequence)
    coords = normalize_sequence(coords, left_presence, right_presence)

    flat = coords.reshape(MAX_FRAMES, NUM_LANDMARKS * NUM_COORDS)
    presence_features = np.stack([left_presence, right_presence], axis=1).astype(np.float32)
    return np.concatenate([flat, presence_features], axis=1)


def load_rows(rows, project_root=None):
    """
    Loads an arbitrary list of manifest rows (no split filtering) into
    (X, y). Used both by load_split() (which pre-filters by split name)
    and by kfold_evaluate() (which builds its own fold assignments and
    needs to load whatever rows it's given regardless of the manifest's
    original split column).
    """
    if project_root is None:
        project_root = os.path.join(os.path.dirname(__file__), "..")

    X, y = [], []
    for row in rows:
        path = os.path.join(project_root, row["filepath"])
        if not os.path.exists(path):
            print(f"Warning: {path} missing, skipping.")
            continue

        sequence = np.load(path).astype(np.float32)  # float32, not float64 - halves RAM
        features = process_sequence(sequence)
        X.append(features)
        y.append(row["label"])

    if len(X) == 0:
        return np.zeros((0, MAX_FRAMES, FEATURE_DIM), dtype=np.float32), []

    X = np.stack(X).astype(np.float32)
    return X, y


def load_split(rows, split_name):
    """
    Loads only the rows for one split (train/val/test) so we're not holding
    all three in memory at once longer than necessary.
    """
    filtered = [row for row in rows if row["split"] == split_name]
    return load_rows(filtered)


def build_model(num_classes, l2=1e-3):
    """
    Deliberately small relative to earlier versions of this file.

    At MIN_SAMPLES_PER_CLASS=6 with a 4/1/1 train/val/test split, you are
    training on 4 samples per class. The previous architecture
    (LSTM(128) -> LSTM(64) -> Dense(64)) has well over 100k parameters -
    with 4 examples per class it doesn't generalize, it memorizes, and
    which noise it memorizes is highly sensitive to random init/shuffle
    order. That's the actual source of the run-to-run instability, not a
    data pipeline bug.

    Shrinking the model AND adding L2 weight decay (on top of the existing
    dropout) reduces the model's capacity to overfit 4-sample classes.
    This will not make a 4-sample class learn brilliantly - no amount of
    architecture tuning fixes too little data - but it should make results
    much more stable/reproducible across runs, and generalize somewhat
    better on the samples you do have.
    """
    reg = regularizers.l2(l2)
    model = models.Sequential([
        layers.Input(shape=(MAX_FRAMES, FEATURE_DIM)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(32, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg),
        layers.LSTM(32, kernel_regularizer=reg, recurrent_regularizer=reg),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_logits_model(trained_model):
    """
    Returns a model identical to trained_model but with the final
    activation forced to linear, so it outputs true pre-softmax logits.
    Uses clone_model (works for Sequential and Functional models, and
    doesn't require the original model to have a defined .input tensor)
    instead of hand-rebuilding layers.
    """
    last_layer = trained_model.layers[-1]

    def clone_fn(layer):
        if layer is last_layer:
            cfg = layer.get_config()
            if cfg.get("activation") not in (None, "linear"):
                # Activation lives on this layer itself, e.g.
                # Dense(..., activation="softmax") - rebuild as linear.
                cfg = dict(cfg)
                cfg["activation"] = "linear"
                return layer.__class__.from_config(cfg)
            if layer.__class__.__name__ in ("Softmax", "Activation"):
                # Standalone softmax layer, no weights - swap for a linear
                # passthrough so pre-softmax values flow through unchanged.
                # FIX: previously referenced `tf.keras.layers.Activation`
                # but this module never imports `tf` directly (only
                # `from tensorflow.keras import layers, models`) - that
                # was dead code that would NameError the moment a model
                # architecture ended in a standalone Softmax layer instead
                # of Dense(..., activation="softmax"). Uses the already-
                # imported `layers` module instead.
                return layers.Activation("linear", name=layer.name)
        return layer.__class__.from_config(layer.get_config())

    logits_model = models.clone_model(trained_model, clone_function=clone_fn)
    logits_model.set_weights(trained_model.get_weights())
    return logits_model


def get_class_weights(y_train, max_ratio=10.0):
    """
    Returns a {class_index: weight} dict so rare classes (some of your
    words only have 1-2 train samples) don't get drowned out by classes
    with 10x as many.

    Weights are clipped to at most `max_ratio` times the smallest weight.
    Uncapped "balanced" weighting can give a 1-sample class a weight an
    order of magnitude larger than everyone else's - that single sample
    then dominates every batch it's in and can destabilize training
    entirely (large gradient spikes, oscillating loss), rather than
    just giving rare classes a fair say.
    """
    classes = np.unique(y_train)
    if len(classes) < 2:
        return None
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights = np.clip(weights, weights.min(), weights.min() * max_ratio)
    return dict(zip(classes, weights))


def fit_temperature(val_logits, val_label_indices, t_grid=None):
    """
    Finds the temperature T that best calibrates val_logits (TRUE
    pre-softmax logits, from build_logits_model - not reconstructed from
    softmax output) against the true val_label_indices, so a stated
    confidence of e.g. 0.9 actually means "about 90% likely correct" for
    THIS model - which is what makes it valid to later compare this
    model's confidence against a different category model's confidence
    (see analyzer.py's "all" ensemble).

    Grid goes up to 20.0 (not 5.0) because real logit gaps from a
    confident model can be large enough that even T=5 barely softens
    them - the old reconstructed-from-softmax "logits" artificially
    capped how large that gap could ever appear to be, so the old grid
    only needed to go to 5.0 to look like it covered the space.
    """
    if t_grid is None:
        t_grid = np.linspace(0.1, 20.0, 100)
    eps = 1e-8
    n = len(val_label_indices)
    best_t, best_nll = 1.0, float("inf")

    for t in t_grid:
        scaled = val_logits / t
        scaled = scaled - scaled.max(axis=1, keepdims=True)  # numerical stability
        exp_scaled = np.exp(scaled)
        probs_t = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        true_probs = probs_t[np.arange(n), val_label_indices]
        nll = -np.mean(np.log(true_probs + eps))
        if nll < best_nll:
            best_nll = nll
            best_t = t

    return float(best_t)


def model_paths(category):
    """category=None -> old unsuffixed generalist filenames (kept as a
    manual fallback/dev option). category="animals" etc -> per-category
    filenames, which is what analyzer.py actually loads at runtime."""
    suffix = f"_{category}" if category else ""
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    labels_dir  = os.path.join(os.path.dirname(__file__), "labels")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(labels_dir,  exist_ok=True)
    model_path  = os.path.join(weights_dir, f"trained_model{suffix}.h5")
    best_path   = os.path.join(weights_dir, f"best_model{suffix}.h5")
    labels_path = os.path.join(labels_dir,  f"label_classes{suffix}.json")
    return model_path, best_path, labels_path


def train_one(category=None):
    """
    Trains one model. category=None trains the old-style generalist
    model over the whole manifest. category="animals" etc filters the
    manifest down to just that category's words first, so the model
    only ever has to choose among that smaller vocabulary - and gets its
    own calibrated confidence temperature fitted on its own val split.
    """
    label = category or "all (generalist)"
    print(f"\n=== Training: {label} ===")

    rows = load_manifest()
    if category:
        rows = [r for r in rows if WORD_TO_CATEGORY.get(r["label"]) == category]
        if not rows:
            print(f"  No manifest rows found for category '{category}', skipping.")
            return

    model_path, best_path, labels_path = model_paths(category)

    all_labels = [row["label"] for row in rows]
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    num_classes = len(encoder.classes_)

    print("Loading train split...")
    X_train, y_train_raw = load_split(rows, "train")
    print("Loading val split...")
    X_val, y_val_raw = load_split(rows, "val")

    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples | Classes: {num_classes}")

    if len(X_train) == 0:
        print(f"  No train samples for '{label}', skipping.")
        return

    y_train = encoder.transform(y_train_raw)
    class_weight = get_class_weights(y_train)
    model = build_model(num_classes=num_classes)
    model.summary()

    temperature = 1.0

    if len(X_val) == 0:
        print(
            f"\n*** WARNING: no validation samples available for '{label}'. ***\n"
            "*** Training will proceed, but there's no way to measure   ***\n"
            "*** real accuracy, and confidence can't be calibrated      ***\n"
            "*** (temperature stays at 1.0, uncalibrated).              ***\n"
        )
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=min(8, len(X_train)),
            class_weight=class_weight,
        )
    else:
        y_val = encoder.transform(y_val_raw)
        callbacks = [
            # Stop once val_loss stops improving instead of grinding to
            # epoch 50 and overfitting the whole way there.
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            # Belt-and-suspenders: also save the best epoch to disk as we
            # go, in case a run gets interrupted before EarlyStopping fires.
            ModelCheckpoint(
                best_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=min(8, len(X_train)),
            callbacks=callbacks,
            class_weight=class_weight,
        )

        # Thanks to restore_best_weights=True, `model` now holds the best
        # epoch's weights, not epoch 50's - so this save is already the
        # good one, but best_*.h5 is kept too as a backup/checkpoint.
        loss, accuracy = model.evaluate(X_val, y_val)
        print(f"Validation accuracy (best epoch, restored): {accuracy:.3f}")
        print(
            "NOTE: with this manifest's split sizes, val is likely ~1 sample "
            "per class - treat this number as a rough sanity check, not a "
            "reliable estimate. Run --kfold for a trustworthy accuracy number."
        )

        # Calibrate this model's confidence against its own val split,
        # using TRUE pre-softmax logits (not reconstructed from softmax
        # output), so later ensemble comparisons across category models
        # (analyzer.py) compare like with like instead of one model's
        # raw overconfidence - and instead of comparing calibration
        # numbers that were never valid to begin with.
        if num_classes < 2:
            print("Single-class model - calibration is meaningless, leaving temperature=1.0.")
        else:
            logits_model = build_logits_model(model)
            val_logits = logits_model.predict(X_val, verbose=0)
            temperature = fit_temperature(val_logits, y_val)
            print(f"Calibrated temperature: {temperature:.3f}")

    model.save(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {"labels": list(encoder.classes_), "temperature": temperature},
            f, ensure_ascii=False, indent=2,
        )

    print(f"Saved model to {model_path}")
    print(f"Saved label classes (+ temperature) to {labels_path}")


def kfold_evaluate(category=None, k=3, epochs=50, patience=8):
    """
    Gives you an HONEST accuracy number, instead of trusting a val split
    that (at MIN_SAMPLES_PER_CLASS=6, with a 4/1/1 split) has only ~1
    sample per class. A single sample per class means val accuracy is
    close to a coin flip on which sample happened to land there - this
    is why results have felt unstable across runs, independent of any
    code bug.

    Ignores the manifest's train/val/test split column entirely and
    instead pools ALL rows for the category (train+val+test) and runs
    stratified k-fold cross-validation from scratch: k models are
    trained, each holding out a different 1/k slice, and accuracy is
    averaged across folds. This uses your data far more efficiently than
    a single fixed split and gives a much more trustworthy number.

    This does NOT replace train_one() for producing your deployed model
    - it's purely a diagnostic to tell you how good the model actually
    is, and which classes are the weak points. Classes with fewer than
    `k` total samples can't be stratified into k folds and are reported
    separately as "too small to fold".
    """
    label = category or "all (generalist)"
    print(f"\n=== K-fold evaluation (k={k}): {label} ===")

    rows = load_manifest()
    if category:
        rows = [r for r in rows if WORD_TO_CATEGORY.get(r["label"]) == category]

    if not rows:
        print(f"  No manifest rows found for '{label}'.")
        return

    all_labels = [row["label"] for row in rows]
    label_counts = {}
    for lbl in all_labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    too_small = sorted(lbl for lbl, c in label_counts.items() if c < k)
    if too_small:
        print(
            f"  Skipping {len(too_small)} classes with fewer than k={k} total "
            f"samples (can't stratify): {', '.join(too_small)}"
        )

    fold_rows = [r for r in rows if label_counts[r["label"]] >= k]
    if not fold_rows:
        print("  No classes have enough samples for k-fold. Lower k or add data.")
        return

    fold_labels = [r["label"] for r in fold_rows]
    encoder = LabelEncoder()
    encoder.fit(fold_labels)
    y_all = encoder.transform(fold_labels)
    num_classes = len(encoder.classes_)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_accuracies = []
    all_true, all_pred = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(fold_rows, y_all), start=1):
        train_rows = [fold_rows[i] for i in train_idx]
        val_rows = [fold_rows[i] for i in val_idx]

        X_train, y_train_raw = load_rows(train_rows)
        X_val, y_val_raw = load_rows(val_rows)

        y_train = encoder.transform(y_train_raw)
        y_val = encoder.transform(y_val_raw)

        class_weight = get_class_weights(y_train)
        model = build_model(num_classes=num_classes)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=min(8, len(X_train)),
            class_weight=class_weight,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=0)],
            verbose=0,
        )

        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(accuracy)
        print(f"  Fold {fold_idx}/{k}: val accuracy = {accuracy:.3f}  "
              f"(train={len(X_train)}, val={len(X_val)})")

        preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
        all_true.extend(y_val.tolist())
        all_pred.extend(preds.tolist())

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    print(f"\n  Mean accuracy across {k} folds: {mean_acc:.3f} (+/- {std_acc:.3f})")

    print("\n  Per-class report (pooled across all folds):")
    report = classification_report(
        all_true, all_pred,
        labels=list(range(num_classes)),
        target_names=list(encoder.classes_),
        zero_division=0,
    )
    print(report)

    return mean_acc, std_acc


def train():
    train_one(category=None)


def train_all_categories():
    for cat_id in CATEGORY_LABELS:
        train_one(category=cat_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the sign-recognition model.")
    parser.add_argument("--category", default=None,
                         help="train only on this category's words (see categories.py)")
    parser.add_argument("--all-categories", action="store_true",
                         help="train one separate model per category, looping over "
                              "every category in categories.py")
    parser.add_argument("--kfold", action="store_true",
                         help="run stratified k-fold cross-validation instead of a "
                              "normal train (diagnostic only - does not save a model). "
                              "Use this to get an honest accuracy number instead of "
                              "trusting a ~1-sample val split.")
    parser.add_argument("--k", type=int, default=3,
                         help="number of folds for --kfold (default 3)")
    args = parser.parse_args()

    if args.kfold:
        if args.all_categories:
            for cat_id in CATEGORY_LABELS:
                kfold_evaluate(category=cat_id, k=args.k)
        else:
            kfold_evaluate(category=args.category, k=args.k)
    elif args.all_categories:
        train_all_categories()
    else:
        train_one(category=args.category)

# --------------------------------------------------------------------------
# If you hit an actual MemoryError/OOM (not just "training is slow"):
# swap load_split()'s all-in-memory arrays for a tf.data.Dataset that reads
# one .npy per step instead of stacking everything into one giant array:
#
#   def make_dataset(rows, split_name, encoder, batch_size=8):
#       filtered = [r for r in rows if r["split"] == split_name]
#       paths = [os.path.join(project_root, r["filepath"]) for r in filtered]
#       labels = encoder.transform([r["label"] for r in filtered])
#
#       def gen():
#           for path, label in zip(paths, labels):
#               feats = process_sequence(np.load(path).astype(np.float32))
#               yield feats, label
#
#       ds = tf.data.Dataset.from_generator(
#           gen,
#           output_signature=(
#               tf.TensorSpec(shape=(MAX_FRAMES, FEATURE_DIM), dtype=tf.float32),
#               tf.TensorSpec(shape=(), dtype=tf.int32),
#           ),
#       )
#       return ds.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# --------------------------------------------------------------------------