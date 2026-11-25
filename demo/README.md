# JelSmart DEMO
First things first you need to replace the content off app.py with new_app.py inorder to activate it

A Hungarian sign language practice app. Shows you a sign to perform, watches you via webcam, and tells you what it thinks you're showing.

## Structure

```
jelsmart/
├── endpoints.py              # Flask routes: /analyze, /target, /reset
├── train.py                  # trains the LSTM model on the built dataset
├── model/
│   ├── dataset_builder.py    # MP4s → dataset/*.npy + labels.json (run manually)
│   └── analyzer.py           # frame analyzer used by the live app
├── templates/
│   └── main.html             # frontend: mode switch, webcam, prediction display
├── videos/                   # source MP4s, organized as videos/<source>/<sign>/clip.mp4
├── dataset/                  # dataset_builder.py output (*.npy + labels.json)
├── hand_landmarker.task      # MediaPipe hand landmark model (download separately)
├── trained_model.h5          # output of train.py (generated, not in repo)
└── label_classes.json        # output of train.py (generated, not in repo)
```

## Pipeline

```
videos/<source>/<sign>/*.mp4
        ↓  dataset_builder.py
dataset/*.npy + labels.json
        ↓  train.py
trained_model.h5 + label_classes.json
        ↓  analyzer.py
endpoints.py + main.html
```

## Setup

### 1. Download the MediaPipe hand landmark model

```bash
# PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"

# or curl
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 2. Install dependencies and run

```bash
python -m venv venv
venv\Scripts\activate        # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

python endpoints.py
```

Open http://127.0.0.1:5000 and allow camera access.

## Building the dataset

Videos should be organized as:

```
videos/
└── jelnyelv/
    ├── ablak/
    │   ├── ablak_1.mp4
    │   ├── ablak_2.mp4
    │   └── ablak_3.mp4
    ├── adag/
    │   └── ...
```

The sign label is always taken from the **folder name**, not the filename. The source folder is ignored — it only exists to group videos by origin for potential train/test splitting later.

Run:

```bash
python model/dataset_builder.py
```

Produces `dataset/<label>_<n>.npy` (hand landmark sequences, shape `(frames, 42, 3)`) and `dataset/labels.json`.

## Training the model

Once the dataset is built:

```bash
python train.py
```

Produces `trained_model.h5` and `label_classes.json`. The model is a 2-layer LSTM that takes sequences of `(60, 126)` — 60 frames × 42 landmarks × 3 coordinates — and outputs a sign class.

Requires at least 2 distinct signs with multiple clips each for a meaningful train/test split.

## Using the app

- **Szabad gyakorlás (free practice)** — no target, just shows what sign it currently detects in real time
- **Gyakorlás (practice)** — shows a random target sign, listens for your sign, advances automatically on a correct match

## Notes

- The app sends a JPEG frame to `/analyze` every 700ms
- Predictions only start after 60 frames have accumulated in the buffer (`MAX_FRAMES` in `analyzer.py` and `train.py`)
- `reset_buffer()` is called when switching modes or fetching a new target, so old frames don't bleed into the next prediction
- `hand_landmarker.task` and generated files (`trained_model.h5`, `label_classes.json`, `dataset/`) should be added to `.gitignore`


# vision_verification/

Standalone experiment, separate from the main training pipeline. Tests
whether a local vision LLM (via Ollama) can tell apart confusable signs
by comparing video frames directly — no relation to your `dataset/`,
`manifest.csv`, or trained models. Safe to run in a separate terminal
while `finomhangolas.py` is running.

## Setup (one-time)

```
pip install -r requirements.txt --break-system-packages
```

Install Ollama from https://ollama.com, then pull a vision-capable model:

```
ollama pull llama3.2-vision
```

(If that's too slow/large for your machine, try `ollama pull llava` instead
and pass `--model llava`.)

Make sure the Ollama server is running — it usually auto-starts after
install; otherwise run `ollama serve` in a terminal, or just open the
Ollama desktop app.

## Usage

Pick a confusable pair (or more) from your ensemble output — e.g. your
`macska` vs `éhes` confusion from the earlier `manual_test.py --compare-all`
run — and point this at real source .mp4 files for each:

```
python compare_signs.py ^
    --query "C:\path\to\macska_5.mp4" ^
    --query-label macska ^
    --candidate macska "C:\path\to\macska_1.mp4" "C:\path\to\macska_2.mp4" ^
    --candidate éhes "C:\path\to_ehes_1.mp4" "C:\path\to\ehes_2.mp4" ^
    --model llama3.2-vision
```

(`^` is the Windows cmd line-continuation character; use `` ` `` in
PowerShell, or just put it all on one line.)

- `--query-label` is optional — pass it if you know the true answer, so
  the script can tell you correct/wrong automatically. It's never shown
  to the model.
- `--candidate` can be repeated for 3+ candidates, e.g. your model's real
  top-5 guesses for a given clip, not just a pair.
- Results (full response + parsed prediction + correct/wrong) are saved
  to `results/<timestamp>_<query_name>.json`, so you can run several of
  these in parallel terminals without clobbering each other's output.

## Finding real .mp4 paths

Your `labels.json` (in `dataset/`) maps each `.npy` filename to a
`source_video` field in its newer format — that's your path back to the
original clip for a given sample.

## What to actually test first

Pull 2-3 pairs your ensemble already confused for real
(`macska`/`éhes`, `medve`/`füzet`, `rák`/`kesztyű` from your last
`--compare-all` run), and see whether the model reliably tells them apart
from real video frames alone. If it can't do that manually on a handful
of known-confusable pairs, that's a real signal this approach won't help —
much cheaper to find out now with a few manual runs than to build a whole
reranking pipeline around it first.

## Known limitations of this first version

- Frame count is fixed per video (`--n-frames`, default 6) — very short or
  very long clips both get squeezed to the same count. Fine for a first
  test; revisit if results seem frame-count-sensitive.
- No batching — one query per invocation. That's intentional for this
  exploratory phase (keeps each run's output easy to inspect), but means
  testing many pairs means many manual invocations. Worth scripting a
  batch runner once you know the single-comparison prompt actually works.
- CPU-only Ollama inference will likely be slow (expect it to take a
  while per call, possibly a minute or more) — this is why it's built as
  a separate, parallel-runnable tool rather than something wired into the
  live pipeline.
