"""
finomhangolas.py  —  manifest, augmentáció, újratanítás, kalibráció és tesztelés.

Lefuttatja a teljes finomhangolási pipeline-t: manifest újraépítése,
augmentáció, tanítás, kalibráció-újraillesztés, majd az utána következő
tesztek. Az adatkinyerést (a nyers .npy fájlok generálását videóból) NEM
futtatja - azt csak akkor kell külön futtatni, ha új videókat vettél fel.

  1. build_manifest.py                manifest.csv újraépítése a dataset/-ből
  2. augment_database.py             augmentált minták generálása + hozzáadása a manifesthez
  3. train.py --all-categories        minden kategória modelljének újratanítása
  4. recalibrate.py                   kalibrációs hőmérséklet újraillesztése (súlyok érintetlenek)
  5. test_model.py                    kiértékelés a held-out teszthalmazon
  6. manual_test.py --all-categories  kategóriánkénti pontossági összefoglaló
  7. manual_test.py --random 30       ensemble ellenőrzés ("all" mód)
  8. manual_test.py --random 30 --compare-all  ensemble vs. generalista összehasonlítás
  9. [opt-in --compare-vision] manual_test.py --random 30 --compare-vision
                                 landmark-only vs. landmark->vision-fallback összehasonlítás

FONTOS: a build_manifest.py és argument_database.py lépések alapértelmezetten
FUTNAK. Korábban ez a szkript kihagyta őket, ami azt eredményezte, hogy a
tanítás a nyers, augmentálatlan mintákon futott (kevesebb adat, mint
amennyi ténylegesen rendelkezésre áll) - ez adta a gyanúsan alacsony
flat-model pontosságot. Ha biztosan tudod, hogy a manifest és az
augmentált adatok már frissek és nem változott semmi a dataset/-ben,
használd a --skip-manifest / --skip-augment kapcsolókat a gyorsabb
futtatáshoz.

Használat (a venv aktiválása után, bárhonnan):

    python scripts/finomhangolas.py

Opciók:
    --skip-manifest       kihagyja a manifest.csv újraépítését
    --skip-augment        kihagyja az augmentált minták generálását
    --skip-train           kihagyja az újratanítást (csak a kalibrációt/teszteket futtatja)
    --skip-recalibrate     kihagyja a kalibráció újraillesztését
    --kfold               stratifikált k-fold cross-validation futtatása a tanítás helyett
                            (diagnosztikus csak - nem menti a modellt)
    --category KATEG       csak egy kategória modelljét tanítja/kalibrálja újra
                            pl: --category animals
                            (a manifest és augmentáció ilyenkor is a teljes
                            dataset-en fut, mert azok nem kategóriánkéntiek)
    --random N              ennyi véletlenszerű fájlt tesztel (alapértelmezett: 30)
    --compare-vision        hozzáadja a manual_test.py --compare-vision lépést a pipeline
                            végéhez (landmark-only vs. landmark->vision-fallback pontosság).
                            Opt-in: Ollamát hív hálózaton keresztül és lassú (egy hívás
                            minden 0.6 alatti konfidenciájú fájlra), ezért alapértelmezetten
                            NEM fut le.

    --sweep                MIN_SAMPLES_PER_CLASS küszöb-sweep: lefuttatja a teljes
                            pipeline-t (build_manifest.py --min-samples N-nel) minden
                            N értékre a --sweep-thresholds listából, MINDKÉT móddal
                            (--kfold ÉS anélkül), és az összes kulcsfontosságú
                            eredményt (manifest statisztikák, held-out accuracy,
                            kategóriánkénti sweep, ensemble vs. generalista) egyetlen,
                            könnyen áttekinthető fájlba menti:
                                scripts/sweep_results.md
                            A teljes nyers kimenet is bekerül a fájl végére,
                            ha részletesebben meg akarod nézni egy adott futást.
    --sweep-thresholds     vesszővel elválasztott MIN_SAMPLES_PER_CLASS értékek a
                            sweephez (alapértelmezett: "4,5,6")

A kimenet a scripts/finomhangolas.log fájlba és a konzolra is kerül.
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
LOG_FILE    = os.path.join(SCRIPTS_DIR, "finomhangolas.log")
SWEEP_RESULTS_FILE = os.path.join(SCRIPTS_DIR, "sweep_results.md")

# Name of the augmentation script.
AUGMENT_SCRIPT = "augment_database.py"


# ---------------------------------------------------------------------------
# Tee: egyszerre ír a logfájlba és a konzolra
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, log_path):
        self.log = open(log_path, "a", encoding="utf-8")
        self._results = []

    def write(self, msg):
        try:
            print(msg, end="", flush=True)
        except UnicodeEncodeError:
            safe = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            print(safe, end="", flush=True)
        self.log.write(msg)
        self.log.flush()

    def record(self, label, success, elapsed):
        self._results.append((label, success, elapsed))

    def print_summary(self):
        self.write(f"\n{'#'*60}\n")
        self.write("ÖSSZEFOGLALÓ\n")
        self.write(f"{'#'*60}\n")
        for label, success, elapsed in self._results:
            icon = "✓" if success else "✗"
            mins, secs = divmod(int(elapsed), 60)
            duration = f"{mins}m {secs}s" if mins else f"{secs}s"
            skipped = " (kihagyva)" if elapsed == 0 and success else ""
            self.write(f"  {icon}  {label:<45s} {duration}{skipped}\n")
        self.write(f"{'#'*60}\n")

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Lépésfuttató
# ---------------------------------------------------------------------------
def run_step(label, base_dir, script, tee, args=None, capture=False):
    """
    Runs one pipeline step. Always streams output to the tee (console + log).
    If capture=True, also returns the full captured stdout text as the second
    return value (used by --sweep to parse accuracy numbers out of it) -
    otherwise returns None for that slot to keep normal-mode runs cheap.
    """
    script_path = os.path.join(base_dir, script)

    tee.write(f"\n{'='*60}\n")
    tee.write(f"LÉPÉS: {label}\n")
    tee.write(f"Szkript: {script_path}\n")
    tee.write(f"Kezdés: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    tee.write(f"{'='*60}\n")

    if not os.path.exists(script_path):
        tee.write(f"\n!! HIBA: A szkript nem található: {script_path}\n")
        return False, (None if not capture else "")

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

    captured_lines = [] if capture else None
    for line in process.stdout:
        tee.write(line)
        if capture:
            captured_lines.append(line)

    process.wait()
    tee.write(f"\nBefejezés: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    captured_text = "".join(captured_lines) if capture else None

    if process.returncode != 0:
        tee.write(f"\n!! LÉPÉS SIKERTELEN (kilépési kód: {process.returncode})\n")
        tee.write(f"!! A pipeline megállt itt: {label}\n")
        return False, captured_text

    tee.write(f"\n✓ {label} sikeresen befejezve.\n")
    return True, captured_text


# ---------------------------------------------------------------------------
# Metrika-kinyerés a szkriptek kimenetéből (a --sweep módhoz)
# ---------------------------------------------------------------------------
def extract_metrics(text):
    """
    Best-effort regex extraction of the numeric results we know these scripts
    print (based on observed output from test_model.py / manual_test.py /
    build_manifest.py). Returns a dict; missing keys just mean that pattern
    wasn't found in this particular run's output (e.g. train.py --kfold's
    format isn't known, so its raw text is kept separately and shown in full
    rather than parsed).
    """
    m = {}

    manifest_keep = re.search(r"Keeping (\d+) classes with enough data \((\d+) samples\)", text)
    if manifest_keep:
        m["manifest_classes_kept"] = int(manifest_keep.group(1))
        m["manifest_samples_kept"] = int(manifest_keep.group(2))

    manifest_excl = re.search(r"Excluding (\d+) classes below the threshold", text)
    if manifest_excl:
        m["manifest_classes_excluded"] = int(manifest_excl.group(1))

    top1 = re.search(r"Top-1 accuracy:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", text)
    if top1:
        m["top1"] = f"{top1.group(1)}/{top1.group(2)} = {top1.group(3)}%"

    top5 = re.search(r"Top-5 accuracy:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", text)
    if top5:
        m["top5"] = f"{top5.group(1)}/{top5.group(2)} = {top5.group(3)}%"

    overall = re.search(r"Overall across all categories:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", text)
    if overall:
        m["overall_category_sweep"] = f"{overall.group(1)}/{overall.group(2)} = {overall.group(3)}%"

    generalist = re.search(r"Generalist model:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", text)
    if generalist:
        m["generalist_vs_ensemble_generalist"] = f"{generalist.group(1)}/{generalist.group(2)} = {generalist.group(3)}%"

    ensemble = re.search(r"Calibrated ensemble:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", text)
    if ensemble:
        m["generalist_vs_ensemble_ensemble"] = f"{ensemble.group(1)}/{ensemble.group(2)} = {ensemble.group(3)}%"

    # Per-category sweep lines, e.g.:
    #   Köszönés és alapok        (greetings   ): 26/27 = 96.3%
    per_cat = re.findall(
        r"^\s*(\S.*?)\s+\((\w+)\s*\):\s*(\d+)/(\d+)\s*=\s*([\d.]+)%",
        text, re.MULTILINE
    )
    if per_cat:
        m["per_category"] = [
            {"name_hu": name.strip(), "key": key, "correct": int(c), "total": int(t), "pct": float(pct)}
            for name, key, c, t, pct in per_cat
        ]

    return m


def format_metrics_table(metrics_by_threshold):
    """Builds the top-of-file markdown summary table across all thresholds."""
    lines = []
    lines.append("| Threshold | Classes kept | Excluded | Held-out Top-1 | Held-out Top-5 | "
                  "Per-category overall | Ensemble | Generalist |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for th, m in metrics_by_threshold.items():
        lines.append(
            f"| {th} | {m.get('manifest_classes_kept', '?')} | {m.get('manifest_classes_excluded', '?')} | "
            f"{m.get('top1', '?')} | {m.get('top5', '?')} | "
            f"{m.get('overall_category_sweep', '?')} | "
            f"{m.get('generalist_vs_ensemble_ensemble', '?')} | "
            f"{m.get('generalist_vs_ensemble_generalist', '?')} |"
        )
    return "\n".join(lines)


def format_per_category_table(metrics_by_threshold):
    """Side-by-side per-category accuracy across thresholds, so you can see
    exactly which categories improve or degrade as the threshold changes."""
    all_keys = []
    for m in metrics_by_threshold.values():
        for row in m.get("per_category", []):
            if row["key"] not in all_keys:
                all_keys.append(row["key"])

    if not all_keys:
        return "(nincs kategóriánkénti adat egyik futtatásból sem)"

    thresholds = list(metrics_by_threshold.keys())
    header = "| Category | " + " | ".join(f"th={th}" for th in thresholds) + " |"
    sep = "|---|" + "---|" * len(thresholds)
    lines = [header, sep]
    for key in all_keys:
        row_cells = []
        for th in thresholds:
            per_cat = metrics_by_threshold[th].get("per_category", [])
            match = next((r for r in per_cat if r["key"] == key), None)
            row_cells.append(f"{match['correct']}/{match['total']} = {match['pct']}%" if match else "?")
        lines.append(f"| {key} | " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sweep mód
# ---------------------------------------------------------------------------
def run_sweep(thresholds, random_n, tee, skip_kfold=False):
    """
    For each threshold in `thresholds`, runs:
      (a) build_manifest.py --min-samples <threshold>
      (b) argument_database.py
      (c) train.py --kfold --all-categories   (diagnostic, raw output kept, no model saved)
      (d) the full real pipeline: train.py --all-categories -> recalibrate.py ->
          test_model.py -> manual_test.py (x3) - this DOES save a real model,
          so the threshold you ran LAST is what ends up as your live model
          when the sweep finishes.
    All captured output is parsed where possible and written to
    scripts/sweep_results.md - a summary table first, then full raw logs
    per threshold/mode below it for anything the regexes didn't catch.
    """
    n = str(random_n)
    metrics_by_threshold = {}
    raw_logs = []  # list of (heading, text) to dump at the end of the file

    for th in thresholds:
        tee.write(f"\n{'#'*60}\n")
        tee.write(f"SWEEP: MIN_SAMPLES_PER_CLASS = {th}\n")
        tee.write(f"{'#'*60}\n")

        combined_text_parts = []

        # --- shared data prep for this threshold ---
        ok, out = run_step(
            f"[th={th}] Manifest újraépítése", MODEL_DIR,
            os.path.join("scripts", "build_manifest.py"),
            tee, args=["--min-samples", str(th)], capture=True
        )
        combined_text_parts.append(out or "")
        raw_logs.append((f"th={th} — build_manifest.py --min-samples {th}", out or ""))
        if not ok:
            tee.write(f"\n!! Sweep megállt itt (th={th}, manifest újraépítés sikertelen)\n")
            continue

        ok, out = run_step(
            f"[th={th}] Augmentált minták generálása", MODEL_DIR,
            AUGMENT_SCRIPT, tee, args=None, capture=True
        )
        raw_logs.append((f"th={th} — {AUGMENT_SCRIPT}", out or ""))
        if not ok:
            tee.write(f"\n!! Sweep megállt itt (th={th}, augmentáció sikertelen)\n")
            continue

        # --- (c) kfold diagnostic run (optional) ---
        if skip_kfold:
            tee.write(f"\n-- [th={th}] Kihagyva: K-fold cross-validation (--skip-kfold)\n")
        else:
            ok, out = run_step(
                f"[th={th}] K-fold cross-validation (diagnosztikus)", MODEL_DIR,
                "train.py", tee, args=["--kfold", "--all-categories"], capture=True
            )
            raw_logs.append((f"th={th} — train.py --kfold --all-categories", out or ""))
            if not ok:
                tee.write(f"\n!! [th={th}] K-fold lépés sikertelen, folytatás a valódi tanítással\n")

        # --- (d) full real pipeline: train -> recalibrate -> evaluate ---
        ok, out = run_step(
            f"[th={th}] Minden kategória újratanítása (valódi)", MODEL_DIR,
            "train.py", tee, args=["--all-categories"], capture=True
        )
        raw_logs.append((f"th={th} — train.py --all-categories", out or ""))
        if not ok:
            tee.write(f"\n!! Sweep megállt itt (th={th}, valódi tanítás sikertelen)\n")
            continue

        ok, out = run_step(
            f"[th={th}] Kalibráció újraillesztése", MODEL_DIR,
            "recalibrate.py", tee, args=None, capture=True
        )
        raw_logs.append((f"th={th} — recalibrate.py", out or ""))

        ok, out = run_step(
            f"[th={th}] Kiértékelés (held-out teszt)", MODEL_DIR,
            "test_model.py", tee, args=None, capture=True
        )
        combined_text_parts.append(out or "")
        raw_logs.append((f"th={th} — test_model.py", out or ""))

        ok, out = run_step(
            f"[th={th}] Kategóriánkénti pontossági sweep", MODEL_DIR,
            "manual_test.py", tee, args=["--random", n, "--all-categories"], capture=True
        )
        combined_text_parts.append(out or "")
        raw_logs.append((f"th={th} — manual_test.py --all-categories", out or ""))

        ok, out = run_step(
            f"[th={th}] Ensemble vs. generalista", MODEL_DIR,
            "manual_test.py", tee, args=["--random", n, "--compare-all"], capture=True
        )
        combined_text_parts.append(out or "")
        raw_logs.append((f"th={th} — manual_test.py --compare-all", out or ""))

        # Parse everything gathered for this threshold in one pass
        metrics_by_threshold[th] = extract_metrics("\n".join(combined_text_parts))

    # --- write the results file ---
    with open(SWEEP_RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Sweep eredmények\n\n")
        f.write(f"Futtatva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Küszöbök: {thresholds}\n\n")
        f.write("## Összefoglaló\n\n")
        f.write(format_metrics_table(metrics_by_threshold))
        f.write("\n\n## Kategóriánkénti bontás küszöbönként\n\n")
        f.write(format_per_category_table(metrics_by_threshold))
        f.write("\n\n## Teljes nyers log minden lépésről (részletes vizsgálathoz)\n\n")
        for heading, text in raw_logs:
            f.write(f"### {heading}\n\n```\n{text}\n```\n\n")

    tee.write(f"\n{'#'*60}\n")
    tee.write(f"SWEEP KÉSZ. Eredmények: {SWEEP_RESULTS_FILE}\n")
    tee.write(f"MEGJEGYZÉS: az utoljára futtatott küszöb ({thresholds[-1]}) modellje maradt\n")
    tee.write(f"betöltve/mentve - ha egy korábbi küszöböt akarsz élesben, futtasd újra\n")
    tee.write(f"azzal a --min-samples értékkel külön (nem sweep módban).\n")
    tee.write(f"{'#'*60}\n")


# ---------------------------------------------------------------------------
# Fő logika
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Manifest, augmentáció, újratanítás, kalibráció és tesztelés pipeline (finomhangoláshoz).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-manifest", action="store_true",
                        help="kihagyja a manifest.csv újraépítését")
    parser.add_argument("--skip-augment", action="store_true",
                        help="kihagyja az augmentált minták generálását")
    parser.add_argument("--skip-train", action="store_true",
                        help="kihagyja a tanítást, csak a kalibrációt/teszteket futtatja")
    parser.add_argument("--skip-recalibrate", action="store_true",
                        help="kihagyja a kalibrációs hőmérséklet újraillesztését")
    parser.add_argument("--kfold", action="store_true",
                        help="stratifikált k-fold cross-validation futtatása (diagnosztikus - "
                             "nem menti a modellt, megkerüli a kalibráció és manuális teszteket)")
    parser.add_argument("--category", default=None,
                        help="csak ezt a kategóriát tanítja/kalibrálja újra (pl: animals)")
    parser.add_argument("--random", type=int, default=30, metavar="N",
                        help="ennyi véletlenszerű fájlt tesztel (alapértelmezett: 30)")
    parser.add_argument("--sweep", action="store_true",
                        help="MIN_SAMPLES_PER_CLASS küszöb-sweep --sweep-thresholds értékekre, "
                             "mindkét móddal (kfold és valódi tanítás), eredmények "
                             "scripts/sweep_results.md-be mentve")
    parser.add_argument("--sweep-thresholds", default="4,5,6", metavar="N,N,N",
                        help="vesszővel elválasztott küszöbértékek a --sweep-hez (alapértelmezett: 4,5,6)")
    parser.add_argument("--skip-kfold", action="store_true",
                        help="a --sweep módban kihagyja a k-fold diagnosztikai lépést minden "
                             "küszöbnél, és csak a valódi (mentett modellt eredményező) tanítást "
                             "futtatja - lényegesen gyorsabb, de az eredmény kevésbé megbízható "
                             "kis val-splitek esetén")
    parser.add_argument("--compare-vision", action="store_true",
                        help="hozzáadja a manual_test.py --compare-vision lépést a pipeline "
                             "végéhez, ami a landmark-only és a landmark->vision-fallback "
                             "pontosságot hasonlítja össze. Opt-in, mert Ollamát hív hálózaton "
                             "keresztül és lassú (egy hívás minden 0.6 alatti konfidenciájú "
                             "fájlra) - alapértelmezetten NEM fut.")
    args = parser.parse_args()

    if args.sweep:
        thresholds = [int(x.strip()) for x in args.sweep_thresholds.split(",") if x.strip()]
        tee = Tee(LOG_FILE)
        tee.write(f"\n{'#'*60}\n")
        tee.write(f"SWEEP INDÍTVA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        tee.write(f"Küszöbök: {thresholds}\n")
        tee.write(f"Log: {LOG_FILE}\n")
        tee.write(f"Eredmények: {SWEEP_RESULTS_FILE}\n")
        tee.write(f"{'#'*60}\n")
        run_sweep(thresholds, args.random, tee, skip_kfold=args.skip_kfold)
        tee.close()
        return

    # K-fold módban más a pipeline
    if args.kfold:
        tee = Tee(LOG_FILE)
        tee.write(f"\n{'#'*60}\n")
        tee.write(f"K-FOLD CROSS-VALIDATION INDÍTVA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        tee.write(f"Log: {LOG_FILE}\n")
        tee.write(f"{'#'*60}\n")

        # --kfold and --category are separate flags for train.py, not one
        # value passed as the other's argument. Previously this was built as
        # ["--category", "--kfold", args.category], which fed "--kfold" as
        # the value of --category and left the real category name dangling
        # as an unconsumed extra arg - train.py would either error out or
        # silently ignore/misparse it.
        kfold_train_args = ["--kfold"]
        if args.category:
            kfold_train_args += ["--category", args.category]

        KFOLD_STEPS = [
            ("Manifest újraépítése",               MODEL_DIR,   os.path.join("scripts", "build_manifest.py"), None,           args.skip_manifest),
            ("Augmentált minták generálása",       MODEL_DIR,   AUGMENT_SCRIPT, None,                                          args.skip_augment),
            ("K-fold cross-validation",            MODEL_DIR,   "train.py",     kfold_train_args,                             False),
        ]

        for label, base_dir, script, step_args, skip in KFOLD_STEPS:
            if skip:
                tee.write(f"\n-- Kihagyva: {label}\n")
                tee.record(label, True, 0)
                continue

            t_start = datetime.now()
            success, _ = run_step(label, base_dir, script, tee, args=step_args)
            elapsed = (datetime.now() - t_start).total_seconds()
            tee.record(label, success, elapsed)

            if not success:
                tee.print_summary()
                tee.write(f"\nK-FOLD CROSS-VALIDATION SIKERTELEN ITT: {label}\n")
                tee.close()
                sys.exit(1)

        tee.print_summary()
        tee.write(f"\nK-FOLD CROSS-VALIDATION KÉSZ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        tee.write(f"\nMegjegyzés: Ez diagnosztikus futtatás volt - nem mentett modellt.\n")
        tee.write(f"Az eredmények fent találhatók. Ha jó az accuracy, futtasd újra ezzel:\n")
        tee.write(f"  python scripts/finomhangolas.py\n")
        tee.write(f"hogy egy teljes modellt betaníts és tesztelsz.\n")
        tee.close()
        return

    # Normális mód (nem k-fold, nem sweep)
    # Tanítási argumentumok összeállítása
    if args.category:
        # Was previously ["--category", "--kfold", args.category] here too -
        # same bug as the k-fold branch, just without --kfold even being
        # requested. Fixed to just pass --category <name>.
        train_args = ["--category", args.category]
        train_label = f"Újratanítás: {args.category}"
        recalib_args = ["--category", args.category]
        recalib_label = f"Kalibráció újraillesztése: {args.category}"
    else:
        train_args = ["--all-categories"]
        train_label = "Minden kategória újratanítása"
        recalib_args = []
        recalib_label = "Kalibráció újraillesztése (minden kategória)"

    n = str(args.random)

    STEPS = [
        ("Manifest újraépítése",               MODEL_DIR,   os.path.join("scripts", "build_manifest.py"), None,           args.skip_manifest),
        ("Augmentált minták generálása",       MODEL_DIR,   AUGMENT_SCRIPT, None,                                          args.skip_augment),
        (train_label,                          MODEL_DIR,   "train.py",        train_args,                            args.skip_train),
        (recalib_label,                        MODEL_DIR,   "recalibrate.py",  recalib_args,                          args.skip_recalibrate),
        ("Kiértékelés (held-out teszt)",       MODEL_DIR,   "test_model.py",   None,                                  False),
        ("Kategóriánkénti pontossági sweep",   MODEL_DIR,   "manual_test.py",  ["--random", n, "--all-categories"],   False),
        (f"Ensemble ellenőrzés (all, n={n})",  MODEL_DIR,   "manual_test.py",  ["--random", n],                       False),
        ("Ensemble vs. generalista",           MODEL_DIR,   "manual_test.py",  ["--random", n, "--compare-all"],       False),
    ]

    if args.compare_vision:
        STEPS.append(
            (f"Landmark vs. vision-fallback (n={n})", MODEL_DIR, "manual_test.py",
             ["--random", n, "--compare-vision"], False)
        )

    tee = Tee(LOG_FILE)
    tee.write(f"\n{'#'*60}\n")
    tee.write(f"FINOMHANGOLÁS INDÍTVA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    tee.write(f"Log: {LOG_FILE}\n")
    if args.category:
        tee.write(f"Kategória: {args.category}\n")
    if args.skip_manifest:
        tee.write(f"Manifest újraépítése: kihagyva\n")
    if args.skip_augment:
        tee.write(f"Augmentáció: kihagyva\n")
    if args.skip_train:
        tee.write(f"Tanítás: kihagyva\n")
    if args.skip_recalibrate:
        tee.write(f"Kalibráció: kihagyva\n")
    tee.write(f"{'#'*60}\n")

    for label, base_dir, script, step_args, skip in STEPS:
        if skip:
            tee.write(f"\n-- Kihagyva: {label}\n")
            tee.record(label, True, 0)
            continue

        t_start = datetime.now()
        success, _ = run_step(label, base_dir, script, tee, args=step_args)
        elapsed = (datetime.now() - t_start).total_seconds()
        tee.record(label, success, elapsed)

        if not success:
            tee.print_summary()
            tee.write(f"\nFINOMHANGOLÁS SIKERTELEN ITT: {label}\n")
            tee.close()
            sys.exit(1)

    tee.print_summary()
    tee.write(f"\nFINOMHANGOLÁS KÉSZ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    tee.close()


if __name__ == "__main__":
    main()