"""
frame_utils.py — extracts frames from .mp4 files and composites them into
a single labeled grid image for sending to vision models.
"""

import base64
import os
import cv2
import numpy as np

def extract_frames(video_path, n_frames=6):
    """Returns a list of n_frames evenly-spaced frames as OpenCV BGR numpy arrays."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count from {video_path}")

    if n_frames == 1:
        indices = [total // 2]
    elif total <= n_frames:
        indices = list(range(total))
    else:
        indices = [round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"Extracted 0 usable frames from {video_path}")

    return frames

def _resize_frame(frame, max_dim):
    """Resizes frame so its longest side is max_dim, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame

def _fit_frame(frame, target_w, target_h):
    """
    Resizes frame to fit within target_w x target_h preserving aspect ratio,
    then pads with black (letterbox) to hit the exact target size.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def _add_label(frame, text, font_scale=0.5, thickness=1):
    """Burns a text label into the top-left of a frame (black outline for legibility)."""
    out = frame.copy()
    # Bottom layer (black shadow)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Top layer (white text)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out

def build_comparison_grid(candidate_frames_dict, query_frames, max_dim=224, cols=4, quality=85):
    """
    Builds a single JPEG grid image for vision models.
    
    Layout: 
    - Row 1: QUERY (the sign to identify)
    - Rows 2+: REF (reference examples)
    
    Each row has up to `cols` frames. default is 4 to keep hand size large.
    """
    def make_row(frames, label, thumb_w, thumb_h):
        """Pad/trim to exactly `cols` thumbs, add label to first frame."""
        row_frames = []
        for i, f in enumerate(frames[:cols]):
            f = _fit_frame(f, thumb_w, thumb_h)
            if i == 0:
                f = _add_label(f, label)
            row_frames.append(f)
        
        # pad with black if fewer than cols frames
        blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        while len(row_frames) < cols:
            row_frames.append(blank.copy())
        return np.hstack(row_frames)

    # 1. Determine uniform thumbnail size based on the largest resized frame
    all_raw = []
    for frames in candidate_frames_dict.values():
        all_raw.extend(frames)
    all_raw.extend(query_frames)

    if not all_raw:
        raise ValueError("No frames provided for grid.")

    resized_samples = [_resize_frame(f, max_dim) for f in all_raw]
    thumb_h = max(f.shape[0] for f in resized_samples)
    thumb_w = max(f.shape[1] for f in resized_samples)

    rows = []

    # 2. Row 1: The QUERY (Priority #1 for the model)
    resized_query = [_resize_frame(f, max_dim) for f in query_frames]
    rows.append(make_row(resized_query, "QUERY:?", thumb_w, thumb_h))

    # 3. Subsequent Rows: The REFERENCES
    for label, frames in candidate_frames_dict.items():
        resized_ref = [_resize_frame(f, max_dim) for f in frames]
        rows.append(make_row(resized_ref, f"REF:{label}", thumb_w, thumb_h))

    # 4. Stack and Encode
    grid = np.vstack(rows)
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode grid image")

    return base64.b64encode(buf.tobytes()).decode("utf-8")

def save_frames_to_disk(frames, out_dir, prefix="frame"):
    """Debug helper: dump frames as .jpg files."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames and save to disk for visual inspection.")
    parser.add_argument("video", help="path to an .mp4 file")
    parser.add_argument("--out", default="./frame_preview")
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()
    frames = extract_frames(args.video, n_frames=args.n)
    paths = save_frames_to_disk(frames, args.out, prefix=os.path.splitext(os.path.basename(args.video))[0])
    print(f"Saved {len(paths)} frames to {args.out}:")
    for p in paths:
        print(f"  {p}")