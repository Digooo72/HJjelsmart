"""
find_reference_videos.py — resolves a word (label) to its real source .mp4
paths on disk, using the SAME path logic dataset_builder.py used when it
wrote labels.json in the first place.

Why this exists instead of just eyeballing labels.json yourself: given how
much reorganizing this dataset has been through, a few things can go wrong
if you resolve paths by hand:

  1. source_video is stored RELATIVE TO videos/, not relative to dataset/
     or the project root directly (see dataset_builder.py: `rel =
     os.path.relpath(video_path, VIDEOS_DIR)`). Joining it against the
     wrong base directory will silently point at a path that doesn't
     exist, or worse, a different file that happens to exist there.

  2. labels.json may have a MIX of old-format entries ({out_name: "label"})
     and new-format entries ({out_name: {"label":..., "source_video":...}})
     if it was built/updated across different versions of
     dataset_builder.py - old entries have no source_video to resolve at
     all.

  3. Files can genuinely be missing now even with a correct relative path,
     if videos/ was reorganized/cleaned up since extraction ran.

This script surfaces all three cases explicitly instead of failing quietly.

USAGE:
    python find_reference_videos.py --project-root "C:\\Users\\krist\\sign_language" --word macska
    python find_reference_videos.py --project-root "C:\\Users\\krist\\sign_language" --word macska --n 3
"""

import argparse
import json
import os


def load_labels_map(dataset_dir):
    labels_path = os.path.join(dataset_dir, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found at {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_videos_for_word(project_root, word, n=None):
    """
    Returns three lists: (resolved_paths, entries_without_source_video,
    entries_with_missing_file) - so the caller can see exactly what's
    usable vs. what's broken, rather than just getting a possibly-short
    list with no explanation.
    """
    dataset_dir = os.path.join(project_root, "dataset")
    videos_dir = os.path.join(project_root, "videos")

    labels_map = load_labels_map(dataset_dir)

    resolved = []
    no_source_video = []
    missing_on_disk = []

    for out_name, value in labels_map.items():
        if isinstance(value, dict):
            label = value.get("label")
            source_video = value.get("source_video")
        else:
            # old format: {out_name: "label"} - no source_video at all
            label = value
            source_video = None

        if label != word:
            continue

        if not source_video:
            no_source_video.append(out_name)
            continue

        # source_video is relative to videos/, per dataset_builder.py's
        # `os.path.relpath(video_path, VIDEOS_DIR)` - NOT relative to
        # dataset/ or project_root directly.
        abs_path = os.path.normpath(os.path.join(videos_dir, source_video))

        if os.path.exists(abs_path):
            resolved.append(abs_path)
        else:
            missing_on_disk.append((out_name, abs_path))

    if n is not None:
        resolved = resolved[:n]

    return resolved, no_source_video, missing_on_disk


def find_video_for_entry(project_root, out_name):
    """
    Resolves ONE specific dataset entry (an .npy out_name, e.g. "kutya_2",
    no extension) back to its own source .mp4 - not "any video for this
    word" like find_videos_for_word, but the exact clip that entry's
    landmarks were extracted from.

    This matters for testing: if you want to know how well the vision
    model does on the query represented by a given .npy file, you need
    that file's OWN video, not some other recording of the same word
    (and specifically NOT a recording that also ends up in the candidate
    reference set - see the caller in manual_test.py, which excludes it).

    Returns (path, None) on success, or (None, reason) where reason is
    one of "not_found" / "no_source_video" / "missing_on_disk" - mirrors
    the three failure modes find_videos_for_word already distinguishes,
    for the same reasons (silent failure here just looks like "the vision
    model wasn't tested on this file" with no way to tell why).
    """
    dataset_dir = os.path.join(project_root, "dataset")
    videos_dir = os.path.join(project_root, "videos")

    labels_map = load_labels_map(dataset_dir)

    # labels.json's keys have been observed both WITH the .npy extension
    # (e.g. "kutya_1.npy" - what dataset_builder.py actually writes today)
    # and, in older/hand-edited files, WITHOUT it (e.g. "kutya_1"). Callers
    # may reasonably pass either form of out_name too. Try the exact key
    # first, then the other form of the SAME name, instead of assuming one
    # convention and silently returning "not_found" for every entry when
    # the assumption doesn't match what's actually on disk.
    value = labels_map.get(out_name)
    if value is None:
        if out_name.endswith(".npy"):
            value = labels_map.get(out_name[: -len(".npy")])
        else:
            value = labels_map.get(out_name + ".npy")

    if value is None:
        return None, "not_found"

    if isinstance(value, dict):
        source_video = value.get("source_video")
    else:
        source_video = None  # old-format entry: {out_name: "label"}, no path recorded

    if not source_video:
        return None, "no_source_video"

    abs_path = os.path.normpath(os.path.join(videos_dir, source_video))
    if not os.path.exists(abs_path):
        return None, "missing_on_disk"

    return abs_path, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", required=True,
                         help=r'e.g. "C:\Users\krist\sign_language" - the folder containing both dataset/ and videos/')
    parser.add_argument("--word", required=True, help="the Hungarian label to look up, e.g. macska")
    parser.add_argument("--n", type=int, default=None, help="max number of resolved paths to return (default: all)")
    args = parser.parse_args()

    resolved, no_source_video, missing = find_videos_for_word(args.project_root, args.word, args.n)

    print(f"Word: '{args.word}'\n")

    if resolved:
        print(f"Resolved, usable video paths ({len(resolved)}):")
        for p in resolved:
            print(f"  {p}")
    else:
        print("No usable video paths found.")

    if no_source_video:
        print(f"\n{len(no_source_video)} .npy entries for this word have NO source_video "
              f"recorded (old-format labels.json entry) - can't trace these back to an "
              f"mp4 at all:")
        for out_name in no_source_video:
            print(f"  {out_name}")

    if missing:
        print(f"\n{len(missing)} entries had a source_video recorded, but the file "
              f"isn't at that path anymore (moved/deleted since extraction?):")
        for out_name, path in missing:
            print(f"  {out_name} -> expected at: {path}")

    if not resolved:
        print(
            "\nNothing usable found for this word. If this word genuinely has real "
            "recordings, check whether videos/ has been moved or reorganized since "
            "dataset_builder.py last ran - you may need to re-run it, or manually "
            "locate the clips."
        )


if __name__ == "__main__":
    main()