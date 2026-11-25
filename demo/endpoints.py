import io
import os
import random

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

from model import analyzer

# ---------------------------------------------------------------------------
# Konfiguráció – minden szerver-specifikus beállítás .env-ből jön,
# kódban nem kell piszkálni semmit.
# ---------------------------------------------------------------------------

# URL prefix, amivel a reverse proxy kiszolgálja az appot.
# Ha Nginx-en: /test  →  Flask Gunicorn (ez az érték)
# Ha közvetlenül porton fut: hagyd üresen ("")
URL_PREFIX = os.environ.get("URL_PREFIX", "/test")

# Melyik Flask/Gunicorn host:port-ot hívjuk meg, ha ez a szerver
# egy másik backend-nek akar küldeni kérést (pl. load balance, node2)
# Helyi hálózaton: http://192.168.1.42:5000  vagy  http://jelsmart-node2:5000
PEER_BACKEND_URL = os.environ.get("PEER_BACKEND_URL", "")

# Ollama szerver URL – ha másik gépen fut az Ollama
# pl. http://192.168.1.10:11434
# A vision_verification csomag az OLLAMA_URL env var-t olvassa közvetlenül,
# itt csak dokumentáció céljából jelezzük
# OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Projekt gyökér (dataset/ és videos/ mappák szülője)
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.join(os.path.dirname(__file__), "..")
)

# ---------------------------------------------------------------------------
# Flask app – APPLICATION_ROOT mondja meg a Flasknak, hogy mi a prefix,
# így url_for() és redirect() is helyes URL-t generál
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["APPLICATION_ROOT"] = URL_PREFIX
app.config["PREFERRED_URL_SCHEME"] = "https"

# ---------------------------------------------------------------------------
# Vision fallback import (változatlan logika)
# ---------------------------------------------------------------------------

from vision_verification.ollama_fallback import (
    FALLBACK_THRESHOLD as VISION_FALLBACK_THRESHOLD,
    TOP_N_CANDIDATES as VISION_TOP_N_CANDIDATES,
    classify as vision_classify,
)
from vision_verification.frame_utils import extract_frames
from vision_verification.find_reference_videos import find_videos_for_word


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def _decode_frame(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame_bgr


def _vision_fallback(frame_bgrs, top_candidates, category):
    """
    Sends frame_bgrs + reference videos for top_candidates to the vision
    model and returns (predicted_word, confidence_str).
    Returns (None, None) if vision model is unavailable or errors.
    """
    if not top_candidates:
        return None, None

    candidate_frames = {}
    for word, _ in top_candidates:
        resolved, no_source, missing = find_videos_for_word(PROJECT_ROOT, word, n=2)
        if not resolved:
            reason = []
            if no_source:
                reason.append(f"{len(no_source)} entries with no source_video")
            if missing:
                reason.append(f"{len(missing)} entries with a missing file")
            print(
                f"[vision fallback] no usable reference video for '{word}'"
                + (f" ({', '.join(reason)})" if reason else "")
            )
            continue

        word_frames = []
        for vpath in resolved:
            try:
                word_frames.extend(extract_frames(vpath, n_frames=4))
            except Exception as e:
                print(f"[vision fallback] failed to extract frames from {vpath}: {e}")
        if word_frames:
            candidate_frames[word] = word_frames

    if not candidate_frames:
        print("[vision fallback] no reference videos found for candidates")
        return None, None

    query_frames = [frame_bgr for frame_bgr in frame_bgrs[::3]]
    return vision_classify(candidate_frames, query_frames)


# ---------------------------------------------------------------------------
# Routes – a Blueprint-es megközelítés a legtisztább módja annak, hogy
# Flask tudja a /test prefixet, miközben a route dekorátorok rövidek maradnak
# ---------------------------------------------------------------------------

from flask import Blueprint

bp = Blueprint("demo", __name__, template_folder="/home/ihne/jelsmart-server/demo/template")

@bp.route("/")
def index():
    return render_template("main.html")


@bp.route("demo/categories", methods=["GET"])
def get_categories():
    return jsonify({"categories": analyzer.get_categories()})


@bp.route("demo/labels", methods=["GET"])
def get_labels():
    """Visszaadja az adott kategóriában elérhető jelek (szavak) listáját."""
    category = request.args.get("category")
    labels   = analyzer.get_available_targets(category)
    return jsonify({"labels": sorted(labels)})


@bp.route("demo/reset", methods=["POST"])
def reset():
    analyzer.reset_buffer()
    return jsonify({"status": "ok"})


@bp.route("demo/analyze", methods=["POST"])
def analyze():
    """Eredeti single-frame streaming endpoint – visszafelé kompatibilis."""
    category   = request.form.get("category")
    image_file = request.files.get("image")

    if image_file is None:
        return jsonify({"error": "missing image"}), 400

    frame_bgr = _decode_frame(image_file.read())
    if frame_bgr is None:
        return jsonify({"error": "could not decode image"}), 400

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predicted, confidence = analyzer.analyze_frame(
        frame_rgb, confidence_threshold=0.5, category=category
    )

    return jsonify({"predicted": predicted, "confidence": confidence})


@bp.route("demo/analyze_clip", methods=["POST"])
def analyze_clip():
    """
    Receives a sequence of JPEG frames (key='frames') captured from the
    browser, runs the landmark ensemble on the full sequence, and falls back
    to the vision model if confidence is below VISION_FALLBACK_THRESHOLD.

    Returns:
        predicted   - recognized word, or null
        confidence  - 0.0-1.0
        method      - 'landmark' or 'vision'
    """
    category = request.form.get("category")
    files    = request.files.getlist("frames")

    if not files:
        return jsonify({"error": "no frames received"}), 400

    frame_bgrs = []
    for f in files:
        bgr = _decode_frame(f.read())
        if bgr is not None:
            frame_bgrs.append(bgr)

    if not frame_bgrs:
        return jsonify({"error": "could not decode any frames"}), 400

    print(f"[analyze_clip] received {len(frame_bgrs)} frames, category={category}")

    # --- landmark pass ---
    try:
        raw_sequence = analyzer.extract_landmark_sequence(frame_bgrs)
    except Exception as e:
        return jsonify({"error": f"landmark extraction failed: {e}"}), 500

    predicted, confidence = analyzer.predict_from_sequence(
        raw_sequence, confidence_threshold=0.5, category=category
    )
    method = "landmark"

    print(f"[analyze_clip] landmark: predicted={predicted!r} confidence={confidence:.3f}")

    # --- vision fallback ---
    if confidence < VISION_FALLBACK_THRESHOLD:
        print(
            f"[analyze_clip] confidence {confidence:.3f} < {VISION_FALLBACK_THRESHOLD}, "
            "trying vision fallback"
        )
        top_candidates = analyzer.get_top_candidate_words(raw_sequence, n=VISION_TOP_N_CANDIDATES)
        print(f"[analyze_clip] top candidates: {top_candidates}")

        if top_candidates:
            vis_predicted, vis_confidence = _vision_fallback(frame_bgrs, top_candidates, category)
            if vis_predicted is not None:
                predicted  = vis_predicted
                confidence = vis_confidence
                method     = "vision"
                print(f"[analyze_clip] vision: predicted={predicted!r} confidence={confidence:.3f}")

    return jsonify({
        "predicted":  predicted,
        "confidence": confidence,
        "method":     method,
    })


# ---------------------------------------------------------------------------
# Opcionális: peer backend health proxy
# Ha PEER_BACKEND_URL be van állítva, ez az endpoint továbbítja a kérést
# a másik szerverre – hasznos debuggoláshoz vagy simple load-sharinghez
# ---------------------------------------------------------------------------

@bp.route("demo/peer/health", methods=["GET"])
def peer_health():
    """Ellenőrzi, hogy a peer backend elérhető-e."""
    if not PEER_BACKEND_URL:
        return jsonify({"peer": None, "status": "not configured"}), 200

    import urllib.request
    import urllib.error
    try:
        url = PEER_BACKEND_URL.rstrip("/") + URL_PREFIX + "/categories"
        with urllib.request.urlopen(url, timeout=3) as resp:
            return jsonify({"peer": PEER_BACKEND_URL, "status": "ok", "code": resp.status})
    except urllib.error.URLError as e:
        return jsonify({"peer": PEER_BACKEND_URL, "status": "unreachable", "error": str(e)}), 502


# ---------------------------------------------------------------------------
# Blueprint regisztráció a prefix-szel
# ---------------------------------------------------------------------------

app.register_blueprint(bp, url_prefix=URL_PREFIX)


# ---------------------------------------------------------------------------
# Lokális fejlesztéshez: python app.py
# Élesben: gunicorn -w 4 -b 0.0.0.0:5000 app:app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"[app] running on http://0.0.0.0:{port}{URL_PREFIX}")
    app.run(host="0.0.0.0", port=port, debug=debug)