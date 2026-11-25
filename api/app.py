"""
JelSmart - API app (port 6700).

Handles only sign-language recognition:
  POST /api/tukor  - classify a webcam frame
  GET  /api/test   - health check

Runs as a separate process from the site app (app.py on port 6710) so that
the heavy ML stack (transformers, torch) is isolated and doesn't affect the
site process startup time or memory footprint.

CORS is wide open here because this endpoint is intentionally public and is
meant to be called cross-origin from jelsmart.hu (and potentially other
frontends).
"""

import io
import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jelsmart.api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG = os.environ.get("API_FLASK_DEBUG", "false").lower() == "true"
HOST = os.environ.get("API_FLASK_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_FLASK_PORT", "6700"))

SIGN_MODEL_NAME = os.environ.get(
    "SIGN_MODEL_NAME", "prithivMLmods/Alphabet-Sign-Language-Detection"
)
MIN_CONFIDENCE = float(os.environ.get("SIGN_MODEL_MIN_CONFIDENCE", "0.5"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Wide-open CORS: this API is designed to be called cross-origin.
CORS(app)


# ---------------------------------------------------------------------------
# Sign-language classifier
#
# Loaded once, eagerly, at process startup so the first real request never
# eats the model-load latency. Run warm_model.py (one directory up) before
# first boot to pre-populate the HF cache so this doesn't hit the network
# at startup either.
# ---------------------------------------------------------------------------
_classifier = None


def load_classifier():
    global _classifier
    from transformers import pipeline

    logger.info("Loading sign-language model: %s", SIGN_MODEL_NAME)
    _classifier = pipeline("image-classification", model=SIGN_MODEL_NAME)
    logger.info("Model loaded.")


def get_classifier():
    if _classifier is None:
        load_classifier()
    return _classifier


# Load at import time so gunicorn/uwsgi workers also warm up on boot.
load_classifier()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/test", methods=["GET"])
def test_endpoint():
    return jsonify({"message": "Python server is running!"})


@app.route("/api/tukor", methods=["POST"])
def tukor_api():
    if "frame" not in request.files:
        return jsonify({"error": "Nincs kep"}), 400

    file = request.files["frame"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Nem sikerult beolvasni a kepet"}), 400

    try:
        classifier = get_classifier()
        predictions = classifier(image)
    except Exception as exc:  # noqa: BLE001
        logger.error("Modell hiba: %s", exc)
        return jsonify({"error": "A felismeres pillanatnyilag nem elerheto"}), 503

    if not predictions:
        return jsonify({"message": "Nincs kezmozdulat felismerve"})

    top = predictions[0]
    letter = top["label"]
    confidence = float(top["score"])

    if confidence < MIN_CONFIDENCE:
        return jsonify({"message": "Nincs kezmozdulat felismerve", "confidence": round(confidence, 3)})

    return jsonify(
        {
            "message": f"Felismert: {letter}",
            "letter": letter,
            "confidence": round(confidence, 3),
            "top_predictions": [
                {"letter": p["label"], "confidence": round(float(p["score"]), 3)}
                for p in predictions[:3]
            ],
        }
    )


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
