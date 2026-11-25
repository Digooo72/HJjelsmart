"""
ollama_fallback.py — the actual "ask the vision model which candidate this
matches" step: builds the prompt, calls Ollama, parses the response.

This used to exist twice, slightly differently, in endpoints.py's
_vision_fallback() and in compare_signs.py's build_prompt()/call_ollama().
The two prompts had already drifted apart (different reasoning-length
instructions, one missing an explanatory line) with no way to know which
one reflects what's actually running in production. Centralizing it here
means:
  - the live app (endpoints.py) and offline testing (manual_test.py's
    --compare-vision) always exercise the identical prompt/parsing, so
    a manual_test accuracy number actually predicts what the app will do
  - compare_signs.py (ad-hoc single-comparison CLI tool) can point at the
    same logic instead of maintaining its own copy
"""

import os

import requests

from .frame_utils import build_comparison_grid

OLLAMA_URL = os.environ.get("OLLAMA_URL", "https://llm.ihne.hu")
#optionaly if you want stonger model you can use llama3.2-vision:latest but it is slower and more expensive
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava:7b")

# Maps the model's self-reported low/medium/high confidence word to a
# numeric score comparable against the landmark ensemble's confidence
# scale, so VISION_FALLBACK_THRESHOLD-style comparisons and the UI's
# confidence bar mean roughly the same thing regardless of which method
# produced the number.
CONFIDENCE_MAP = {"low": 0.4, "medium": 0.65, "high": 0.85}
DEFAULT_CONFIDENCE = 0.6  # used if the model's reply doesn't parse cleanly

# Below this landmark confidence, the app (and manual_test.py's
# --compare-vision) tries the vision model instead. Canonical here so a
# manual_test accuracy run and the live app always trigger vision at the
# same point - if they used separately-defined thresholds, an accuracy
# comparison run offline wouldn't actually predict what production does.
FALLBACK_THRESHOLD = float(os.environ.get("VISION_FALLBACK_THRESHOLD", 0.5))
TOP_N_CANDIDATES = int(os.environ.get("VISION_TOP_N_CANDIDATES", 4))

# How long to wait for Ollama's response. This is a single call to an
# 11B vision model (llama3.2-vision) over a grid image built from up to
# ~30 frames (query + candidates) - on CPU-only hardware (no CUDA/DirectML
# on native Windows for TF either, per the training logs, so this box is
# almost certainly CPU-bound for Ollama too) that can genuinely take
# several minutes, especially on a cold-loaded model. 120s was too tight
# and was timing out on every real call rather than reflecting an actual
# error - canonical + env-overridable here so both endpoints.py and
# manual_test.py's --compare-vision use the same, realistic budget instead
# of drifting or guessing per-caller.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", 300))

# How long Ollama should keep the model loaded in memory/VRAM after this
# call finishes. Ollama's default (a few minutes, version-dependent) means
# a batch of calls spaced more than that apart - e.g. manual_test.py's
# --compare-vision looping over N files with landmark inference in
# between each - can each pay a full model-load cost on top of inference,
# which for an 11B model is a large chunk of total latency. Set generously
# high here since these are short-lived test/serving processes, not a
# long-running multi-model server where holding VRAM matters; tune down
# via env var if this box is genuinely memory constrained.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

# Hard cap on generated tokens. The prompt asks for a few short lines
# (BEST_MATCH / CONFIDENCE / REASONING in 1-2 sentences), which smaller/
# more compliant models answer in well under 150 tokens - but observed
# behavior from llama3.2-vision:latest is to write a long free-form
# analysis BEFORE ever stating BEST_MATCH, so a 150-token cap can cut the
# response off mid-reasoning, before the model ever reaches its actual
# answer (predicted then wrongly looks like "model couldn't decide"
# rather than "model ran out of room to say what it decided"). Raised to
# 400 to give room to finish even when a model doesn't follow the
# requested compact format; tune down via env var if latency matters more
# than completeness for your model.
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", 400))


def build_prompt(candidate_labels):
    labels_list = ", ".join(f"'{c}'" for c in candidate_labels)
    return (
        "You are an expert in Hungarian sign language gesture recognition.\n\n"
        "The image below is a grid of video frames. Row 1 is the 'QUERY' (the sign we need to identify).\n"
        "The following rows are 'REF' (reference examples of known signs).\n\n"
        "Instructions:\n"
        "1. Observe the hand shapes and movement in the QUERY (Row 1).\n"
        "2. Compare them to the REF rows.\n"
        f"3. Identify which REF label best matches the QUERY from this list: {labels_list}\n\n"
        "Your reply MUST start with the BEST_MATCH line below as the very "
        "first thing you write - do not write any analysis, comparison, or "
        "explanation before it. Put your reasoning after, not before:\n"
        "BEST_MATCH: <label>\n"
        "CONFIDENCE: <low|medium|high>\n"
        "REASONING: <1-2 sentences>\n"
    )


def call_ollama(grid_b64, prompt, model=VISION_MODEL, timeout=OLLAMA_TIMEOUT):
    """Raises on any request/HTTP failure - callers decide how to handle that."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [grid_b64]}],
            "stream": False,
            # Keeps the model resident between calls instead of paying a
            # full reload for every single classify() invocation - see
            # OLLAMA_KEEP_ALIVE comment above.
            "keep_alive": OLLAMA_KEEP_ALIVE,
            # num_predict bounds worst-case generation time; the other
            # options here are left at Ollama's defaults deliberately -
            # only touching what actually affects latency without
            # affecting the model's actual comparison/reasoning quality.
            "options": {"num_predict": OLLAMA_NUM_PREDICT},
        },
        timeout=timeout,
    )
    if not resp.ok:
        # raise_for_status() alone only gives "400 Client Error: Bad Request
        # for url: ...", discarding Ollama's actual explanation (e.g. "model
        # does not support images", a bad model name, malformed payload). A
        # FAST 400 (order-of-seconds, not a timeout) almost always means the
        # request was rejected before inference even started - the body is
        # the only way to tell why instead of guessing.
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} from Ollama - body: {resp.text[:500]}",
            response=resp,
        )
    return resp.json()["message"]["content"]


def parse_response(text):
    """Returns (predicted_word_or_None, confidence_float)."""
    predicted = None
    confidence_str = "medium"
    for line in text.splitlines():
        clean = line.strip().lstrip("*").rstrip("*").strip()
        # Some smaller models (observed with llava:7b) emit markdown-escaped
        # underscores/asterisks, e.g. "BEST\_MATCH:" instead of "BEST_MATCH:".
        # Without unescaping this, startswith("BEST_MATCH:") never matches
        # and predicted silently stays None even though the model answered
        # correctly - looks identical to "model couldn't compare the images"
        # in the caller, which it isn't.
        clean = clean.replace("\\_", "_").replace("\\*", "*")
        if clean.upper().startswith("BEST_MATCH:"):
            predicted = clean.split(":", 1)[1].strip()
            # Strip markdown emphasis and any wrapping quote characters the
            # model might add around the label (e.g. "BEST_MATCH: 'szia'")
            # - otherwise the stored prediction is "'szia'" (with quotes),
            # which will never string-equal the true label "szia" even when
            # the model picked the exactly correct word.
            predicted = predicted.strip("*").strip("'\"").strip()
            if predicted.upper().startswith("REF:"):
                predicted = predicted[4:].strip()
        elif clean.upper().startswith("CONFIDENCE:"):
            confidence_str = clean.split(":", 1)[1].strip().lower()

    confidence = CONFIDENCE_MAP.get(confidence_str, DEFAULT_CONFIDENCE)
    return predicted, confidence


def find_candidate_in_text(text, candidate_labels):
    """
    Fallback for when the model ignores the requested BEST_MATCH/CONFIDENCE
    format entirely and just answers in free prose (observed with
    llama3.2-vision:latest, e.g. "The best match for the QUERY sign is
    'könyv' with high confidence...") - parse_response finds no BEST_MATCH:
    line at all in that case, even though the answer is right there in the
    text.

    Returns the candidate label that appears earliest in the text
    (case-insensitive substring match), or None if none of them appear at
    all. This is deliberately simple - with only TOP_N_CANDIDATES (a
    handful) of short, fairly distinct labels, "which one is mentioned
    first" is a reasonable proxy for "which one the model concluded was
    the match", since the models overwhelmingly lead with their answer
    (e.g. "the best match is X") rather than mentioning the winning label
    only deep in a rejection of other candidates.
    """
    lowered = text.lower()
    best_label, best_pos = None, None
    for label in candidate_labels:
        pos = lowered.find(label.lower())
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_label, best_pos = label, pos
    return best_label


def classify(candidate_frames, query_frames, model=VISION_MODEL, timeout=OLLAMA_TIMEOUT):
    """
    candidate_frames: {label: [frame, frame, ...]} (raw BGR numpy arrays)
    query_frames: [frame, frame, ...]

    Returns (predicted_word, confidence) on success, or (None, None) if the
    grid couldn't be built or Ollama couldn't be reached/parsed - callers
    treat that as "vision fallback unavailable, keep the landmark result".
    """
    if not candidate_frames or not query_frames:
        return None, None

    try:
        grid_b64 = build_comparison_grid(candidate_frames, query_frames)
    except Exception as e:
        print(f"[ollama_fallback] grid build error: {e}")
        return None, None

    prompt = build_prompt(list(candidate_frames.keys()))

    try:
        text = call_ollama(grid_b64, prompt, model=model, timeout=timeout)
    except Exception as e:
        print(f"[ollama_fallback] ollama error: {e}")
        return None, None

    predicted, confidence = parse_response(text)
    if predicted is None:
        # The request succeeded and Ollama returned text, but nothing in
        # it matched "BEST_MATCH: ..." - either a markdown-escaped variant
        # (handled above) or, seen with llama3.2-vision:latest, the model
        # ignoring the structured format entirely and just answering in
        # prose ("The best match ... is 'könyv' ..."). Printing the raw
        # text here is the only way to tell "model ignored the format"
        # apart from "model genuinely refused/couldn't compare the
        # images" without guessing.
        print(f"[ollama_fallback] unparseable response from {model!r}:\n{text}\n")
        predicted = find_candidate_in_text(text, list(candidate_frames.keys()))
        if predicted is not None:
            confidence = DEFAULT_CONFIDENCE
            print(f"[ollama_fallback] recovered via free-text label search: {predicted!r}")
    return predicted, confidence