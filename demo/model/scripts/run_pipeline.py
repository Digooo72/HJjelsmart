"""
run_pipeline.py  —  full end-to-end training pipeline.

Runs every step needed to go from raw videos to trained, tested models:

  1. dataset_builder.py     extract landmarks from videos -> dataset/*.npy
  2. build_manifest.py      assign train/val/test splits  -> dataset/manifest.csv
  3. argument_database.py   augment TRAIN split in-place  -> dataset/*__aug*.npy
  4. train.py --all-categories   train one model per category -> weights/ + labels/
  5. test_model.py          evaluate generalist on held-out test split
  6. manual_test.py --all-categories   per-category accuracy sweep
  7. manual_test.py --random 30        sanity check: "all" ensemble
  8. manual_test.py --random 30 --compare-all   ensemble vs generalist

Run from model/scripts/ or anywhere with the venv active:

    python scripts/run_pipeline.py

Skip steps you've already done with flags:
    --skip-extract      skip dataset_builder  (landmarks already extracted)
    --skip-manifest     skip build_manifest   (manifest.csv already exists)
    --skip-augment      skip augmentation     (augmented files already exist)
    --skip-train        skip training         (models already trained)
    --skip-test         skip test_model.py    (just want the manual checks)
    --only-train        only re-run training + tests (shortcut for the above 3)

Stops immediately if any step fails. All output is logged to
scripts/pipeline.log AND printed to the console.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))   # model/scripts/
MODEL_DIR   = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))  # model/
LOG_FILE    = os.path.join(SCRIPTS_DIR, "pipeline.log")


# ---------------------------------------------------------------------------
# Step definitions
# Each entry: (label, base_dir, script_filename, args_list_or_None)
# base_dir tells run_step both where to find the script and where to set cwd.
# ---------------------------------------------------------------------------
ALL_STEPS = [
    # tag            label                              base_dir     script                   args
    ("extract",  "1. Extract landmarks",               MODEL_DIR,   "dataset_builder.py",    None),
    ("manifest", "2. Build manifest",                  SCRIPTS_DIR, "build_manifest.py",     None),
    ("augment",  "3. Augment train split",             MODEL_DIR,   "augment_database.py",  None),
    ("train",    "4. Train all category models",       MODEL_DIR,   "train.py",              ["--all-categories", "--kfold"]),
    ("test",     "5. Evaluate on held-out test split", MODEL_DIR,   "test_model.py",         None),
    ("check_all","6. Per-category accuracy sweep",     MODEL_DIR,   "manual_test.py",        ["--random", "30", "--all-categories"]),
    ("sanity",   "7. Ensemble sanity check (all)",     MODEL_DIR,   "manual_test.py",        ["--random", "30"]),
    ("compare",  "8. Ensemble vs generalist",          MODEL_DIR,   "manual_test.py",        ["--random", "30", "--compare-all"]),
]


# ---------------------------------------------------------------------------
# Tee: writes to log file + stdout simultaneously
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, log_path):
        self.log = open(log_path, "a", encoding="utf-8")
        self._step_results = []  # (label, success, duration_seconds)

    def write(self, msg):
        try:
            print(msg, end="", flush=True)
        except UnicodeEncodeError:
            safe = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            print(safe, end="", flush=True)
        self.log.write(msg)
        self.log.flush()

    def record(self, label, success, elapsed):
        self._step_results.append((label, success, elapsed))

    def print_summary(self):
        self.write(f"\n{'#'*60}\n")
        self.write("PIPELINE SUMMARY\n")
        self.write(f"{'#'*60}\n")
        for label, success, elapsed in self._step_results:
            icon = "✓" if success else "✗"
            mins, secs = divmod(int(elapsed), 60)
            duration = f"{mins}m {secs}s" if mins else f"{secs}s"
            skipped = " (skipped)" if elapsed == 0 and success else ""
            self.write(f"  {icon}  {label:<45s} {duration}{skipped}\n")
        self.write(f"{'#'*60}\n")

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------
def run_step(label, base_dir, script, tee, args=None):
    script_path = os.path.join(base_dir, script)

    tee.write(f"\n{'='*60}\n")
    tee.write(f"STEP: {label}\n")
    tee.write(f"Script: {script_path}\n")
    tee.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    tee.write(f"{'='*60}\n")

    if not os.path.exists(script_path):
        tee.write(f"\n!! ERROR: Script not found: {script_path}\n")
        return False

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, script_path] + (args or [])
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=base_dir,
        env=child_env,
    )

    for line in process.stdout:
        tee.write(line)

    process.wait()
    tee.write(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if process.returncode != 0:
        tee.write(f"\n!! STEP FAILED (exit code {process.returncode})\n")
        tee.write(f"!! Pipeline stopped at: {label}\n")
        return False

    tee.write(f"\n✓ {label} completed successfully.\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full sign-language training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-extract",  action="store_true", help="skip landmark extraction")
    parser.add_argument("--skip-manifest", action="store_true", help="skip manifest rebuild")
    parser.add_argument("--skip-augment",  action="store_true", help="skip augmentation")
    parser.add_argument("--skip-train",    action="store_true", help="skip training")
    parser.add_argument("--skip-test",     action="store_true", help="skip test_model.py evaluation")
    parser.add_argument("--only-train",    action="store_true",
                        help="shortcut: skip extract + manifest + augment (just train + test)")
    args = parser.parse_args()

    if args.only_train:
        args.skip_extract  = True
        args.skip_manifest = True
        args.skip_augment  = True

    skip_map = {
        "extract":  args.skip_extract,
        "manifest": args.skip_manifest,
        "augment":  args.skip_augment,
        "train":    args.skip_train,
        "test":     args.skip_test,
    }

    tee = Tee(LOG_FILE)
    tee.write(f"\n{'#'*60}\n")
    tee.write(f"PIPELINE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    tee.write(f"Log: {LOG_FILE}\n")
    tee.write(f"{'#'*60}\n")

    for tag, label, base_dir, script, step_args in ALL_STEPS:
        if skip_map.get(tag, False):
            tee.write(f"\n-- Skipping: {label}\n")
            tee.record(label, True, 0)
            continue

        t_start = datetime.now()
        success = run_step(label, base_dir, script, tee, args=step_args)
        elapsed = (datetime.now() - t_start).total_seconds()
        tee.record(label, success, elapsed)

        if not success:
            tee.print_summary()
            tee.write(f"\nPIPELINE FAILED at: {label}\n")
            tee.close()
            sys.exit(1)

    tee.print_summary()
    tee.write(f"\nPIPELINE COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    tee.close()


if __name__ == "__main__":
    main()
