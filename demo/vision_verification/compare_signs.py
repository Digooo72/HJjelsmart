"""
compare_signs.py — sends a query video + reference videos for 2+ candidate
words to a local Ollama vision model and asks it to identify which candidate
the query matches.

All frames are composited into ONE grid image (reference rows on top, query
row at the bottom) because llama3.2-vision only accepts a single image per
request. Labels are burned into the first frame of each row.

USAGE EXAMPLE:
    python compare_signs.py \
        --query "path/to/macska_5.mp4" \
        --query-label macska \
        --candidate macska "ref1.mp4" "ref2.mp4" \
        --candidate ehes "ref1.mp4" "ref2.mp4" \
        --model llama3.2-vision
"""

import argparse
import json
import os
import time
from datetime import datetime

import requests

from frame_utils import extract_frames, build_comparison_grid
import ollama_fallback
from ollama_fallback import build_prompt, call_ollama

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", required=True, help="path to the query .mp4")
    parser.add_argument("--query-label", default=None, help="true label of query (for scoring only, never shown to model)")
    parser.add_argument("--candidate", action="append", nargs="+", metavar=("LABEL", "VIDEO"),
                        required=True,
                        help="label followed by one or more reference video paths. Repeat for each candidate.")
    parser.add_argument("--model", default="llama3.2-vision:latest")
    parser.add_argument("--n-frames", type=int, default=6, help="frames per video")
    parser.add_argument("--save-grid", action="store_true", help="save the grid image to results/ for inspection")
    args = parser.parse_args()

    candidates = {}
    for group in args.candidate:
        label, videos = group[0], group[1:]
        if not videos:
            raise ValueError(f"--candidate {label} needs at least one video path")
        candidates[label] = videos

    candidate_labels = list(candidates.keys())
    print(f"Candidates: {candidate_labels}")
    print(f"Query: {args.query}")
    print(f"Model: {args.model}\n")

    # --- extract frames ---
    print("Extracting frames...")
    candidate_frames = {}
    for label, videos in candidates.items():
        all_frames = []
        for v in videos:
            all_frames.extend(extract_frames(v, n_frames=args.n_frames))
        candidate_frames[label] = all_frames
        print(f"  '{label}': {len(videos)} video(s) -> {len(all_frames)} frames")

    query_frames = extract_frames(args.query, n_frames=args.n_frames)
    print(f"  QUERY: {len(query_frames)} frames")

    # --- build single grid image ---
    print("\nBuilding comparison grid...")
    grid_b64 = build_comparison_grid(candidate_frames, query_frames)
    print(f"  Grid image: {len(grid_b64)} b64 chars ({len(grid_b64)*3//4//1024} KB)")

    # --- optionally save grid for inspection ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_name = os.path.splitext(os.path.basename(args.query))[0]

    if args.save_grid:
        import base64
        grid_path = os.path.join(RESULTS_DIR, f"{timestamp}_{query_name}_grid.jpg")
        with open(grid_path, "wb") as f:
            f.write(base64.b64decode(grid_b64))
        print(f"  Saved grid to: {grid_path}")

    # --- call model ---
    prompt = build_prompt(candidate_labels)
    print(f"\nCalling Ollama ({args.model})... this may take a while on CPU.")
    t0 = time.time()
    try:
        response_text = call_ollama(grid_b64, prompt, model=args.model, timeout=300)
    except requests.exceptions.ConnectionError:
        print(f"\n!! Could not connect to Ollama at {ollama_fallback.OLLAMA_URL}\n"
              "Run `ollama serve` in another terminal, or set OLLAMA_URL.\n")
        return
    except requests.exceptions.HTTPError as e:
        print(f"\n!! Ollama error: {e}\nResponse: {e.response.text if e.response else '(none)'}\n")
        return
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RESPONSE (took {elapsed:.1f}s):")
    print(f"{'='*60}")
    print(response_text)
    print(f"{'='*60}\n")

    # parse prediction
    predicted = None
    for line in response_text.splitlines():
        if line.strip().upper().startswith("BEST_MATCH:"):
            predicted = line.split(":", 1)[1].strip()
            break

    result = {
        "timestamp": timestamp,
        "model": args.model,
        "query_video": args.query,
        "query_true_label": args.query_label,
        "candidates": {label: videos for label, videos in candidates.items()},
        "n_frames_per_video": args.n_frames,
        "elapsed_seconds": round(elapsed, 2),
        "raw_response": response_text,
        "parsed_prediction": predicted,
    }

    if args.query_label and predicted:
        correct = predicted.strip().lower() == args.query_label.strip().lower()
        result["correct"] = correct
        print(f"Predicted: '{predicted}'  |  True: '{args.query_label}'  |  {'✓ CORRECT' if correct else '✗ WRONG'}")

    result_path = os.path.join(RESULTS_DIR, f"{timestamp}_{query_name}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved result to: {result_path}")


if __name__ == "__main__":
    main()