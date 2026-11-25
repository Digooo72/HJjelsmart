"""
manual_test.py — test predictions against existing dataset/*.npy files,
no camera and no running Flask server needed.

Usage examples (run from the model/ folder):

  # Test one specific file, unrestricted (category="all")
  python manual_test.py kutya_2.npy

  # Test the same file, but restricted to the "animals" category
  python manual_test.py kutya_2.npy --category animals

  # Test a random sample of files from the test split, unrestricted
  python manual_test.py --random 10

  # Test a random sample, restricted to one category (good for checking
  # whether category filtering actually helps vs "all")
  python manual_test.py --random 10 --category animals

  # Loop through EVERY category automatically, print a per-category
  # accuracy summary (good for automated/pipeline use)
  python manual_test.py --random 10 --all-categories

  # Compare the generalist model against the calibrated per-category
  # ensemble on the same random files, side by side
  python manual_test.py --random 30 --compare-all

  # Compare landmark-only vs landmark+vision-fallback accuracy on the same
  # random files - i.e. does the vision model actually help? Requires
  # Ollama reachable and PROJECT_ROOT (dataset/ + videos/) resolvable, and
  # is slow (one Ollama call per file that falls below the fallback
  # threshold), so start with a small N.
  python manual_test.py --random 15 --compare-vision
"""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import analyzer  # noqa: E402
from categories import CATEGORY_LABELS  # noqa: E402

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

# model/ sits one level below the project root, alongside dataset/, videos/,
# and vision_verification/ - same layout endpoints.py assumes for its own
# PROJECT_ROOT default. Override with the PROJECT_ROOT env var if yours
# differs.
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
sys.path.insert(0, PROJECT_ROOT)


def true_label_for(fname):
    """Reads manifest.csv to find the real label for a given filename,
    so we can print whether the prediction was actually correct."""
    if not os.path.exists(MANIFEST_PATH):
        return None
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if os.path.basename(row["filepath"]) == fname:
                return row["label"]
    return None


def test_one_file(fname, category, quiet=False):
    """Returns True/False/None (correct/incorrect/unknown-true-label).
    Prints a line unless quiet=True (used when we only want the summary)."""
    path = os.path.join(DATASET_DIR, fname)
    if not os.path.exists(path):
        if not quiet:
            print(f"  !! File not found: {path}")
        return None

    raw_sequence = np.load(path).astype(np.float32)
    predicted, confidence = analyzer.predict_from_sequence(
        raw_sequence, confidence_threshold=0.0, category=category
    )
    true_label = true_label_for(fname)

    correct = (predicted == true_label) if true_label is not None else None
    if not quiet:
        marker = "?" if true_label is None else ("OK" if correct else "  ")
        print(f"  [{marker}] {fname:30s} true={true_label!r:30s} -> "
              f"predicted={predicted!r:30s} confidence={confidence:.1%}")
    return correct


def run_all_categories(n_per_category):
    """
    Loops through every category, tests n_per_category random files from
    each (restricted to that category), and prints a per-category
    accuracy summary. Meant for automated/pipeline use - a quick smoke
    test that category filtering behaves sensibly across the board, not
    a substitute for test_model.py's proper held-out accuracy.
    """
    print(f"Testing {n_per_category} random files per category...\n")
    overall_correct = 0
    overall_total = 0

    for cat_id, cat_label in CATEGORY_LABELS.items():
        files = pick_random_files(n_per_category, cat_id)
        if not files:
            print(f"  {cat_label:25s} ({cat_id}): no trained files in this category, skipping")
            continue

        correct = 0
        total = 0
        for fname in files:
            result = test_one_file(fname, cat_id, quiet=True)
            if result is not None:
                total += 1
                if result:
                    correct += 1

        overall_correct += correct
        overall_total += total
        pct = (correct / total * 100) if total else 0.0
        print(f"  {cat_label:25s} ({cat_id:12s}): {correct}/{total} = {pct:.1f}%")

    if overall_total:
        print(f"\nOverall across all categories: {overall_correct}/{overall_total} = "
              f"{overall_correct/overall_total*100:.1f}%")


def pick_random_files(n, category):
    """
    Picks n random original (non-augmented) .npy files from dataset/,
    restricted to files whose true label is one the CURRENTLY LOADED
    model(s) actually know about (analyzer.get_available_targets()).

    Without this restriction, if a model was trained on a filtered
    subset, random picks from the full dataset/ folder would mostly be
    words the model was never trained on and literally cannot predict
    correctly - an unfair, misleading test rather than a real sanity
    check.

    If `category` is given, further restricts to that category's words
    (still only ones the model actually knows). category=None behaves
    like "all" - every word across every trained category model.
    """
    known_words = set(analyzer.get_available_targets(category))
    if not known_words:
        return []

    all_files = [f for f in os.listdir(DATASET_DIR)
                 if f.endswith(".npy") and "__aug" not in f]
    all_files = [f for f in all_files if true_label_for(f) in known_words]

    random.shuffle(all_files)
    return all_files[:n]


def compare_all_strategies(n):
    """
    Runs the same N random files through BOTH "all" strategies -
    analyzer.predict_generalist() (the old single shared model) and
    analyzer.predict_from_sequence(..., category="all") (the calibrated
    per-category ensemble) - and prints accuracy side by side.

    This exists because the calibrated ensemble is a reasonable bet, not
    a proven win: temperature scaling only helps as much as the
    validation data it was fit on allows, and cross-model confidence
    comparison is inherently softer than comparing within one model. This
    checks which one actually wins on your data instead of assuming.
    """
    print(f"Comparing \"all\" strategies on {n} random files...\n")

    files = pick_random_files(n, category=None)
    if not files:
        print("No files found to test with.")
        return

    generalist_correct = generalist_total = 0
    ensemble_correct = ensemble_total = 0

    for fname in files:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            continue
        raw_sequence = np.load(path).astype(np.float32)
        true_label = true_label_for(fname)
        if true_label is None:
            continue

        gen_pred, gen_conf = analyzer.predict_generalist(raw_sequence, confidence_threshold=0.0)
        ens_pred, ens_conf = analyzer.predict_from_sequence(
            raw_sequence, confidence_threshold=0.0, category="all"
        )

        generalist_total += 1
        if gen_pred == true_label:
            generalist_correct += 1

        ensemble_total += 1
        if ens_pred == true_label:
            ensemble_correct += 1

        print(f"  {fname:30s} true={true_label!r:20s} "
              f"generalist={gen_pred!r:20s}({gen_conf:.1%})  "
              f"ensemble={ens_pred!r:20s}({ens_conf:.1%})")

    gen_pct = (generalist_correct / generalist_total * 100) if generalist_total else 0.0
    ens_pct = (ensemble_correct / ensemble_total * 100) if ensemble_total else 0.0

    print(f"\n  Generalist model:      {generalist_correct}/{generalist_total} = {gen_pct:.1f}%")
    print(f"  Calibrated ensemble:   {ensemble_correct}/{ensemble_total} = {ens_pct:.1f}%")

    if generalist_total == 0:
        print(
            "\n  Note: generalist model produced no predictions - has it been "
            "trained? Run `python train.py` (no --category flag) to produce "
            "trained_model.h5 / label_classes.json first."
        )


def compare_vision_strategies(n, category):
    """
    Runs N random files through the SAME hybrid pipeline endpoints.py's
    /analyze_clip uses in production: landmark ensemble first, vision
    model fallback only when landmark confidence < FALLBACK_THRESHOLD.
    Prints landmark-only vs hybrid accuracy side by side, so you can see
    whether the vision fallback is actually helping instead of assuming.

    Uses the SAME threshold, candidate count, prompt, and response
    parsing as the live app (all imported from
    vision_verification.ollama_fallback) - a separately-tuned offline
    check wouldn't tell you anything about what the app will actually do.

    Needs each test file's OWN source video (to get real query frames,
    not just landmarks) - resolved via find_video_for_entry(). Files
    whose .npy has no traceable source_video, or whose vision candidates
    have no reference videos, are skipped and reported separately rather
    than silently excluded, since a shrunken N changes what the accuracy
    number means.

    This is slow and hits the network - one Ollama call per file that
    falls below the confidence threshold - so start with a small N.
    """
    try:
        from vision_verification.find_reference_videos import find_videos_for_word, find_video_for_entry
        from vision_verification.frame_utils import extract_frames
        from vision_verification.ollama_fallback import (
            classify as vision_classify,
            FALLBACK_THRESHOLD,
            TOP_N_CANDIDATES,
        )
    except ImportError as e:
        print(f"Can't run --compare-vision: {e}\n"
              "This needs the vision_verification package (and its deps: "
              "opencv-python, requests) importable from PROJECT_ROOT "
              f"({PROJECT_ROOT}). Set the PROJECT_ROOT env var if that's wrong.")
        return

    print(f"Comparing landmark-only vs landmark+vision on {n} random files "
          f"(category={category})...")
    print(f"Vision triggers below confidence {FALLBACK_THRESHOLD:.1f}, using top "
          f"{TOP_N_CANDIDATES} candidates - same as production. This calls "
          f"Ollama and will be slow.\n")

    files = pick_random_files(n, None if category == "all" else category)
    if not files:
        print("No matching files found for that category.")
        return

    landmark_correct = hybrid_correct = total = 0
    vision_calls = vision_helped = vision_hurt = 0
    skipped_no_query_video = skipped_no_candidates = 0

    for fname in files:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            continue
        raw_sequence = np.load(path).astype(np.float32)
        true_label = true_label_for(fname)
        if true_label is None:
            continue

        lm_predicted, lm_confidence = analyzer.predict_from_sequence(
            raw_sequence, confidence_threshold=0.0, category=category
        )
        final_predicted, final_confidence, method = lm_predicted, lm_confidence, "landmark"

        if lm_confidence < FALLBACK_THRESHOLD:
            out_name = os.path.splitext(fname)[0]
            query_path, reason = find_video_for_entry(PROJECT_ROOT, out_name)

            if query_path is None:
                skipped_no_query_video += 1
                print(f"  {fname:30s} skip vision ({reason}) - can't trace this "
                      f"entry back to its own source video")
            else:
                top_candidates = analyzer.get_top_candidate_words(raw_sequence, n=TOP_N_CANDIDATES)
                candidate_frames = {}
                for word, _ in top_candidates:
                    resolved, no_source, missing = find_videos_for_word(PROJECT_ROOT, word, n=2)
                    if no_source or missing:
                        print(f"  [warn] candidate word {word!r} had coverage gaps: "
                              f"{len(no_source)} old-format, {len(missing)} missing on disk")
                    # Exclude the query's own video from its own candidate
                    # set - otherwise this word's reference row can literally
                    # BE the query, which would make the accuracy number
                    # measure nothing.
                    resolved = [p for p in resolved
                                if os.path.normpath(p) != os.path.normpath(query_path)]
                    if not resolved:
                        continue
                    frames = []
                    for vpath in resolved:
                        try:
                            frames.extend(extract_frames(vpath, n_frames=4))
                        except Exception:
                            continue
                    if frames:
                        candidate_frames[word] = frames

                if not candidate_frames:
                    skipped_no_candidates += 1
                    print(f"  {fname:30s} skip vision (no usable reference "
                          f"videos for any candidate)")
                else:
                    try:
                        query_frames = extract_frames(query_path, n_frames=6)
                    except Exception as e:
                        query_frames = None
                        print(f"  {fname:30s} skip vision (couldn't read own "
                              f"video: {e})")

                    if query_frames:
                        vision_calls += 1
                        t0 = time.time()
                        vis_predicted, vis_confidence = vision_classify(candidate_frames, query_frames)
                        elapsed = time.time() - t0
                        print(f"  {fname:30s} vision call ({elapsed:.1f}s) -> "
                              f"{vis_predicted!r} "
                              f"({vis_confidence:.1%})" if vis_confidence is not None
                              else f"  {fname:30s} vision call ({elapsed:.1f}s) -> unavailable")
                        if vis_predicted is not None:
                            final_predicted, final_confidence, method = vis_predicted, vis_confidence, "vision"

        total += 1
        lm_ok = (lm_predicted == true_label)
        final_ok = (final_predicted == true_label)
        if lm_ok:
            landmark_correct += 1
        if final_ok:
            hybrid_correct += 1
        if method == "vision":
            if final_ok and not lm_ok:
                vision_helped += 1
            elif lm_ok and not final_ok:
                vision_hurt += 1

        marker = "OK" if final_ok else "  "
        print(f"  [{marker}] {fname:30s} true={true_label!r:20s} "
              f"landmark={lm_predicted!r:20s}({lm_confidence:.1%})  "
              f"final={final_predicted!r:20s}({method})")

    if total == 0:
        print("\nNo files could be scored (missing manifest labels?).")
        return

    lm_pct = landmark_correct / total * 100
    hy_pct = hybrid_correct / total * 100

    print(f"\n  Landmark-only:         {landmark_correct}/{total} = {lm_pct:.1f}%")
    print(f"  Landmark + vision:     {hybrid_correct}/{total} = {hy_pct:.1f}%")
    print(f"  Vision actually called: {vision_calls}/{total} files "
          f"(confidence was < {FALLBACK_THRESHOLD:.1f})")
    if vision_calls:
        print(f"    -> fixed a wrong landmark guess: {vision_helped}")
        print(f"    -> broke a guess landmark had right: {vision_hurt}")
    skipped = skipped_no_query_video + skipped_no_candidates
    if skipped:
        print(f"  Skipped vision on {skipped} low-confidence file(s): "
              f"{skipped_no_query_video} with no traceable source video, "
              f"{skipped_no_candidates} with no usable reference videos "
              f"for their candidates.")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="a specific .npy file in dataset/ to test")
    parser.add_argument("--category", default="all",
                         help="restrict prediction to one category id (see categories.py), or 'all'")
    parser.add_argument("--random", type=int, metavar="N",
                         help="instead of one file, test N random dataset files")
    parser.add_argument("--all-categories", action="store_true",
                         help="loop through every category, testing N random files "
                              "from each (N from --random, default 10), and print a "
                              "per-category accuracy summary")
    parser.add_argument("--compare-all", action="store_true",
                         help="compare the generalist model against the calibrated "
                              "per-category ensemble on N random files (N from "
                              "--random, default 10), side by side")
    parser.add_argument("--compare-vision", action="store_true",
                         help="compare landmark-only vs landmark+vision-fallback "
                              "accuracy on N random files (N from --random, default "
                              "5) - does the vision model actually help? Requires "
                              "Ollama reachable and source videos resolvable; slow.")
    args = parser.parse_args()

    if args.category != "all" and args.category not in CATEGORY_LABELS:
        print(f"Unknown category '{args.category}'. Valid options: "
              f"{', '.join(CATEGORY_LABELS.keys())}, or 'all'")
        return

    if args.compare_vision:
        compare_vision_strategies(args.random or 5, args.category)
        return

    if args.compare_all:
        compare_all_strategies(args.random or 10)
        return

    if args.all_categories:
        run_all_categories(args.random or 10)
        return

    print(f"Category filter: {args.category}\n")

    if args.random:
        files = pick_random_files(args.random, args.category)
        if not files:
            print("No matching files found for that category.")
            return
        for fname in files:
            test_one_file(fname, args.category)
    elif args.filename:
        test_one_file(args.filename, args.category)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()