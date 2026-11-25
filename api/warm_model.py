"""
Pre-downloads the sign-language model into the local Hugging Face cache so
the Flask app never has to fetch it over the network at request time.

Run this once during deploy, e.g.:

    python warm_model.py

Or as a Docker build step, before COPY-ing the rest of the app in, so the
weights are baked into the image layer and cached across builds.

Respects HF_HOME / SIGN_MODEL_NAME env vars if set, same as app.py, so the
cache location and model always agree with what the running app expects.
"""

import os

from huggingface_hub import snapshot_download

MODEL_NAME = os.environ.get("SIGN_MODEL_NAME", "prithivMLmods/Alphabet-Sign-Language-Detection")


def main():
    print(f"Downloading {MODEL_NAME} ...")
    path = snapshot_download(repo_id=MODEL_NAME)
    print(f"Model cached at: {path}")


if __name__ == "__main__":
    main()
