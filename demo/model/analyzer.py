import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

MODEL_DIR = os.path.dirname(__file__)

# IMPORTANT: import the exact same preprocessing used during training
# (resample -> per-hand normalize -> presence flags) instead of redefining
# it here. A second hand-written copy of this transform is exactly what
# caused this file to send the model 60 frames of raw 126-dim landmarks
# when the model was actually trained on 90 frames of normalized 128-dim
# features - a shape mismatch that either crashes or silently corrupts
# every live prediction.
sys.path.insert(0, MODEL_DIR)
from train import process_sequence, MAX_FRAMES, FEATURE_DIM, build_logits_model  # noqa: E402
from categories import CATEGORY_LABELS, WORD_TO_CATEGORY  # noqa: E402

NUM_LANDMARKS = 42
NUM_COORDS = 3

WEIGHTS_DIR = os.path.join(MODEL_DIR, "weights")
LABELS_DIR  = os.path.join(MODEL_DIR, "labels")

HAND_LANDMARKER_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_landmarker = None
_frame_buffer = deque(maxlen=MAX_FRAMES)  # matches train.py's MAX_FRAMES, not a locally-guessed number

# HandLandmarker in VIDEO running mode requires every timestamp passed to
# detect_for_video() to be strictly increasing FOR THE LIFETIME OF THE
# LANDMARKER INSTANCE - not just within one call. Since _landmarker below is
# a single shared instance used by both the live webcam path
# (analyze_frame) and clip processing (extract_landmark_sequence), neither
# caller can safely restart its own timestamp counter at 0 - the second
# caller to run would hand the landmarker a timestamp lower than one it's
# already seen, which raises a hard error ("timestamp must be monotonically
# increasing"). Using wall-clock time instead of a synthetic per-session
# frame counter sidesteps this: it's naturally monotonic across every call,
# every session, and every reset, with no shared counter to get out of sync.
_last_timestamp_ms = -1


def _next_timestamp_ms():
    global _last_timestamp_ms
    ts = int(time.time() * 1000)
    if ts <= _last_timestamp_ms:
        ts = _last_timestamp_ms + 1  # guards against sub-millisecond bursts
    _last_timestamp_ms = ts
    return ts

# How many top categories (by generalist-rolled-up probability) to actually
# consult specialists for, in the "all" router. Bumped from 2 -> 3: several
# misses happened because the generalist's OWN top category guess was
# wrong (e.g. routing "ember" toward "szia"'s category), so widening the
# candidate net gives the true category more chances to still be considered
# even when the generalist's single best guess is off.
ROUTER_TOP_K = 3

# category_id -> (logits_model, labels, temperature). Loaded lazily per
# category so import doesn't require every category to already be
# trained, and so we're not loading models nobody ends up asking for.
#
# NOTE: what's cached here is a LOGITS model (softmax stripped off the
# final layer, see train.build_logits_model), not the original softmax
# model. Every .predict() call on it below returns true pre-softmax
# logits, which is required for _apply_temperature to work correctly -
# calibrating against a softmax output reconstructed back into "logits"
# via log(p) silently breaks the moment any class saturates to exactly
# 0.0 or 1.0 in float32.
_model_cache = {}

# Separate single slot for the old-style generalist model (trained_model.h5,
# no category suffix - what you get from running `python train.py` with no
# --category flag). Kept distinct from _model_cache since it's not keyed by
# category id. Also now doubles as the ROUTER for the "all" ensemble below,
# not just the --compare-all baseline.
_generalist_cache = None


def _model_paths(category_id):
    model_path  = os.path.join(WEIGHTS_DIR, f"trained_model_{category_id}.h5")
    labels_path = os.path.join(LABELS_DIR,  f"label_classes_{category_id}.json")
    return model_path, labels_path


def _load_model_files(model_path, labels_path):
    """
    Shared loader for a (model.h5, label_classes.json) pair, used by both
    the per-category cache and the generalist model. Returns
    (logits_model, labels, temperature) or None if either file is missing.

    The returned model has had its final softmax stripped to linear
    (see train.build_logits_model) so every .predict() call downstream
    returns true logits, not probabilities.
    """
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None

    model = tf.keras.models.load_model(model_path)
    logits_model = build_logits_model(model)

    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Backwards compatible with the old plain-list label_classes.json
    # format (no calibration info) - assumes uncalibrated (T=1.0).
    if isinstance(data, list):
        labels, temperature = data, 1.0
    else:
        labels, temperature = data["labels"], data.get("temperature", 1.0)

    return logits_model, labels, temperature


def _load_category_model(category_id):
    """
    Loads (and caches) one category's logits model + labels + calibration
    temperature. Returns None if that category hasn't been trained yet
    (e.g. run train.py --all-categories first).
    """
    if category_id in _model_cache:
        return _model_cache[category_id]

    model_path, labels_path = _model_paths(category_id)
    loaded = _load_model_files(model_path, labels_path)
    if loaded is not None:
        _model_cache[category_id] = loaded
    return loaded


def _load_generalist_model():
    """
    Loads (and caches) the old-style generalist model - trained_model.h5 /
    label_classes.json with no category suffix, from running plain
    `python train.py`. Returns None if that hasn't been trained/saved.
    """
    global _generalist_cache
    if _generalist_cache is not None:
        return _generalist_cache

    model_path  = os.path.join(WEIGHTS_DIR, "trained_model.h5")
    labels_path = os.path.join(LABELS_DIR,  "label_classes.json")
    loaded = _load_model_files(model_path, labels_path)
    if loaded is not None:
        _generalist_cache = loaded
    return loaded


def _ensure_landmarker_loaded():
    global _landmarker
    if _landmarker is None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
        )
        _landmarker = HandLandmarker.create_from_options(options)


def extract_landmark_sequence(frame_bgrs):
    """
    Runs MediaPipe landmark extraction on a full list of BGR frames (e.g.
    from a recorded clip) and returns a raw landmark sequence of shape
    (N, 42, 3) - same format as dataset .npy files and what
    predict_from_sequence() expects.

    Uses the same shared _landmarker instance and _next_timestamp_ms()
    clock as the live webcam path in analyze_frame(), so clip processing
    can never hand the VIDEO-mode landmarker a timestamp that goes
    backwards relative to a previous call (which MediaPipe rejects).
    """
    _ensure_landmarker_loaded()
    landmarks_seq = []

    for frame_bgr in frame_bgrs:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = _next_timestamp_ms()
        result = _landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_map = {"Left": None, "Right": None}
        for j, handedness_list in enumerate(result.handedness):
            label = handedness_list[0].category_name
            hand_map[label] = result.hand_landmarks[j]

        frame_landmarks = []
        for side in ["Left", "Right"]:
            hand = hand_map[side]
            if hand:
                frame_landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand])
            else:
                frame_landmarks.extend([[0.0, 0.0, 0.0]] * 21)

        landmarks_seq.append(np.array(frame_landmarks))

    return np.array(landmarks_seq)  # (N, 42, 3)


def get_top_candidate_words(raw_sequence, n=4):
    """
    Returns the top-N word candidates from the generalist-routed ensemble,
    as a list of (word, score) tuples sorted by score descending. Used by
    the vision fallback to know which reference videos to compare against.

    Shares the same generalist forward pass + category rollup logic as
    predict_from_sequence()'s "all" branch, rather than the caller
    re-deriving it separately.
    """
    features = process_sequence(raw_sequence.astype(np.float32))
    sequence = features.reshape(1, MAX_FRAMES, FEATURE_DIM)

    gen = _generalist_word_probs(sequence)
    if gen is None:
        return []

    gen_labels, gen_probs = gen
    cat_scores = _rollup_to_category_probs(gen_labels, gen_probs)
    if not cat_scores:
        return []

    word_scores = {}
    top_cats = sorted(cat_scores.items(), key=lambda kv: kv[1], reverse=True)[:ROUTER_TOP_K + 2]
    for cat_id, cat_prob in top_cats:
        loaded = _load_category_model(cat_id)
        if loaded is None:
            continue
        model, labels, temperature = loaded
        if len(labels) < 2:
            continue
        logits = model.predict(sequence, verbose=0)[0]
        calibrated = _apply_temperature(logits, temperature)
        for label, prob in zip(labels, calibrated):
            combined = (cat_prob * float(prob)) ** 0.5
            if label not in word_scores or combined > word_scores[label]:
                word_scores[label] = combined

    sorted_words = sorted(word_scores.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_words[:n]


def reset_buffer():
    _frame_buffer.clear()


def _apply_temperature(logits, temperature):
    """
    Applies calibration temperature T directly to TRUE pre-softmax
    logits, then softmaxes. T=1.0 is a no-op, T>1 softens an
    overconfident model, T<1 sharpens an underconfident one. This is
    what makes comparing confidence ACROSS category models meaningful,
    instead of just picking whichever model happens to run hottest.
    """
    if temperature == 1.0:
        scaled = logits
    else:
        scaled = logits / temperature
    scaled = scaled - scaled.max()
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum()


def get_available_targets(category=None):
    """
    Returns the list of words the practice-mode target picker can choose
    from. category=None/"all" returns every word across every trained
    category model; otherwise just that one category's words.
    """
    if category and category != "all":
        loaded = _load_category_model(category)
        return loaded[1] if loaded else []

    words = []
    for cat_id in CATEGORY_LABELS:
        loaded = _load_category_model(cat_id)
        if loaded:
            words.extend(loaded[1])
    return words


def get_categories():
    """
    Returns [{id, label, count}] for every category that has a trained
    model, plus an "all" option covering everything. `count` lets the
    UI show how well-stocked each category is.
    """
    options = []
    total = 0
    for cat_id, label in CATEGORY_LABELS.items():
        loaded = _load_category_model(cat_id)
        if loaded:
            count = len(loaded[1])
            options.append({"id": cat_id, "label": label, "count": count})
            total += count

    return [{"id": "all", "label": "Összes", "count": total}] + options


def _generalist_word_probs(sequence):
    """
    Returns (labels, calibrated_probs) from the generalist model, or None
    if no generalist has been trained. `sequence` is already the
    (1, MAX_FRAMES, FEATURE_DIM) reshaped array predict_from_sequence uses.
    """
    loaded = _load_generalist_model()
    if loaded is None:
        return None
    model, labels, temperature = loaded
    logits = model.predict(sequence, verbose=0)[0]
    calibrated = _apply_temperature(logits, temperature)
    return labels, calibrated


def _rollup_to_category_probs(labels, probs):
    """
    Sums the generalist's per-word probabilities into per-category
    probabilities, using WORD_TO_CATEGORY. Words not present in that
    mapping are skipped (shouldn't happen for anything actually trained,
    but avoids a KeyError if the manifest ever drifts from categories.py).
    """
    cat_scores = {}
    for word, p in zip(labels, probs):
        cat = WORD_TO_CATEGORY.get(word)
        if cat is None:
            continue
        cat_scores[cat] = cat_scores.get(cat, 0.0) + float(p)
    return cat_scores


def _predict_all_bruteforce(sequence, confidence_threshold):
    """
    Old "all" behavior, kept as a fallback for when no generalist model
    exists yet: runs every trained category specialist and returns
    whichever one reports the highest calibrated confidence.

    Known weakness: specialists have never seen negative examples, so a
    specialist's calibrated confidence isn't comparable across models -
    every specialist's top-1 looks like a slam dunk even on inputs that
    don't belong to it at all. Only used when the router (see
    predict_from_sequence) can't run because no generalist is trained.
    """
    best_word, best_conf = None, -1.0
    for cat_id in CATEGORY_LABELS:
        loaded = _load_category_model(cat_id)
        if loaded is None:
            continue
        model, labels, temperature = loaded
        if len(labels) < 2:
            continue
        logits = model.predict(sequence, verbose=0)[0]
        calibrated = _apply_temperature(logits, temperature)
        local_idx = int(np.argmax(calibrated))
        local_conf = float(calibrated[local_idx])
        if local_conf >= best_conf:
            best_conf = local_conf
            best_word = labels[local_idx]

    if best_word is not None and best_conf >= confidence_threshold:
        return best_word, round(best_conf, 3)
    return None, round(max(best_conf, 0.0), 3)


def predict_from_sequence(raw_sequence, confidence_threshold=0.5, category=None):
    """
    Runs the actual prediction pipeline on an already-extracted raw
    landmark sequence of shape (frames, 42, 3) - the same shape as any
    dataset/*.npy file, or a sequence you built by hand from a recorded
    video. This is independent of the live webcam buffer, so it can be
    called directly for manual testing with no camera involved.

    category=<id>: loads that category's dedicated model and predicts
    directly among just its words - no masking, the model itself only
    ever learned that vocabulary.

    category=None/"all": uses the generalist model as a ROUTER instead of
    running every specialist blind. Specialists were only ever trained on
    their own category's words, so a specialist has no way to say "this
    isn't one of my classes" - every specialist's top-1 confidence looks
    like a slam dunk even on inputs that don't belong to it at all, which
    made a naive "highest confidence wins across all models" ensemble
    unreliable (it kept picking whichever unrelated specialist happened
    to saturate hardest).

    Instead: the generalist's per-word probabilities are rolled up into
    per-category probabilities, the top ROUTER_TOP_K categories are taken
    as candidates, and only those specialists are consulted. Final score
    is sqrt(P_generalist(category) * P_specialist(word | category)), so a
    specialist can only win if the generalist also believes its category
    is plausible - not just because that specialist happened to be
    overconfident. Falls back to the old brute-force ensemble if no
    generalist model has been trained yet.

    FIX: best_conf used to start at 0.0 and only update on strictly-greater
    (`>`), which meant that if EVERY candidate's combined score rounded to
    0.0 (very possible for a small category with a tiny generalist-rollup
    probability), nothing ever "won" and the function returned (None, 0.0)
    even though candidates were genuinely tried. Starting at -1.0 and using
    `>=` guarantees the first candidate is always recorded.
    """
    features = process_sequence(raw_sequence.astype(np.float32))  # resample+normalize+presence
    sequence = features.reshape(1, MAX_FRAMES, FEATURE_DIM)

    if category and category != "all":
        loaded = _load_category_model(category)
        if loaded is None:
            return None, 0.0
        model, labels, temperature = loaded
        logits = model.predict(sequence, verbose=0)[0]
        calibrated = _apply_temperature(logits, temperature)
        best_idx = int(np.argmax(calibrated))
        best_conf = float(calibrated[best_idx])
        if best_conf >= confidence_threshold:
            return labels[best_idx], round(best_conf, 3)
        return None, round(best_conf, 3)

    # "all" - generalist-routed ensemble
    gen = _generalist_word_probs(sequence)
    if gen is None:
        return _predict_all_bruteforce(sequence, confidence_threshold)

    gen_labels, gen_probs = gen
    cat_scores = _rollup_to_category_probs(gen_labels, gen_probs)
    if not cat_scores:
        return _predict_all_bruteforce(sequence, confidence_threshold)

    top_categories = sorted(cat_scores.items(), key=lambda kv: kv[1], reverse=True)[:ROUTER_TOP_K]

    best_word, best_conf = None, -1.0
    for cat_id, cat_prob in top_categories:
        loaded = _load_category_model(cat_id)
        if loaded is None:
            continue
        model, labels, temperature = loaded
        if len(labels) < 2:
            continue

        logits = model.predict(sequence, verbose=0)[0]
        calibrated = _apply_temperature(logits, temperature)

        if np.isnan(calibrated).any():
            print(f"  [WARN] NaN in calibrated output for category '{cat_id}' (temp={temperature})")
            continue

        local_idx = int(np.argmax(calibrated))
        local_conf = float(calibrated[local_idx])

        # Combined score: the specialist can only win if the generalist
        # also trusts its category. Geometric mean keeps the final number
        # on a similar 0-1 "confidence" scale to what confidence_threshold
        # and the UI already expect, rather than two stacked probabilities
        # silently shrinking it (as a plain product would).
        combined = (cat_prob * local_conf) ** 0.5
        if combined >= best_conf:
            best_conf = combined
            best_word = labels[local_idx]

    # Debug visibility - remove once you're happy with the "all" behavior.
    print(f"  [debug] all: top_categories={top_categories} "
          f"best_word={best_word!r} best_conf={best_conf:.4f}")

    if best_word is not None and best_conf >= confidence_threshold:
        return best_word, round(best_conf, 3)
    return None, round(max(best_conf, 0.0), 3)


def predict_generalist(raw_sequence, confidence_threshold=0.5):
    """
    Predicts using the OLD-STYLE single generalist model (trained_model.h5
    / label_classes.json, no category suffix) instead of the per-category
    calibrated ensemble. Not used by app.py or the live webcam path - this
    exists purely so manual_test.py's --compare-all can measure the
    generalist model against the router-based "all" ensemble on the same
    files, instead of assuming one is better without checking.

    Returns (None, 0.0) if no generalist model has been trained/saved
    (i.e. `python train.py` with no --category flag was never run).
    """
    loaded = _load_generalist_model()
    if loaded is None:
        return None, 0.0
    model, labels, temperature = loaded

    features = process_sequence(raw_sequence.astype(np.float32))
    sequence = features.reshape(1, MAX_FRAMES, FEATURE_DIM)
    logits = model.predict(sequence, verbose=0)[0]
    calibrated = _apply_temperature(logits, temperature)

    best_idx = int(np.argmax(calibrated))
    best_conf = float(calibrated[best_idx])
    if best_conf >= confidence_threshold:
        return labels[best_idx], round(best_conf, 3)
    return None, round(best_conf, 3)


def analyze_frame(frame_rgb, confidence_threshold=0.5, category=None):
    """
    Webcam path: extracts hand landmarks from one live frame, buffers
    frames until MAX_FRAMES are collected, then delegates the actual
    prediction to predict_from_sequence() - the same function manual
    tests call directly, so the live app and manual testing can never
    silently diverge in how a sequence gets turned into a prediction.
    """
    _ensure_landmarker_loaded()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = _next_timestamp_ms()

    # Synchronous call, no callback needed
    result = _landmarker.detect_for_video(mp_image, timestamp_ms)

    hand_map = {"Left": None, "Right": None}
    for i, handedness_list in enumerate(result.handedness):
        label = handedness_list[0].category_name
        hand_map[label] = result.hand_landmarks[i]

    frame_landmarks = []
    for side in ["Left", "Right"]:
        hand = hand_map[side]
        if hand:
            frame_landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand])
        else:
            frame_landmarks.extend([[0.0, 0.0, 0.0]] * 21)

    _frame_buffer.append(np.array(frame_landmarks))

    if len(_frame_buffer) < MAX_FRAMES:
        return None, 0.0

    raw_sequence = np.array(_frame_buffer)  # (MAX_FRAMES, 42, 3)
    return predict_from_sequence(raw_sequence, confidence_threshold=confidence_threshold, category=category)