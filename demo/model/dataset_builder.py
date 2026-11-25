
"""
dataset_builder.py  —  builds dataset/ from videos/ (and oktatoanyagok/)

Label extraction rules (filename always wins over folder name):
  - Strip leading number prefix:  "001 - "  or  "001 - Bagoly - "
  - Strip trailing format suffix: ".f134.mp4", ".f135.mp4", " - magyar jelnyelv"
    (dash can be "-", "–", "—", "_", or just whitespace — all handled)

For jelnyelv/ (folder-name-as-label), ALL trailing take/recording numbers
are stripped and merged into a single label, regardless of separator style
— these are multiple recordings of the SAME word, not distinct signs:
  - "ablak_1_1"  -> "ablak"
  - "ablak2_1"   -> "ablak"
  - "ablak-1"    -> "ablak"
  - "ablak-2"    -> "ablak"
  - "ablak"      -> "ablak"

JELNYELV_FILTER_MODE controls how jelnyelv/ words are filtered:
  "allowlist" (default) — ONLY keep words in JELNYELV_ALLOWLIST, a small
      curated set (~190 words) of beginner/child-friendly essentials:
      greetings, family, numbers, colors, animals, school, food, common
      verbs, body parts, clothes, weather, basic adjectives, house
      objects, common places. Everything else is skipped. Good for a
      first training pass where you want the model to reliably learn a
      manageable number of classes.
  "blocklist" — keep everything EXCEPT JELNYELV_BLOCKLIST (religion/
      weapons/war/political/tiny cities) and, if ESSENTIAL_MODE is True,
      ESSENTIAL_EXTRA_SKIP (medical/legal/banking/alcohol/trades/niche
      cities/niche sports). This gives hundreds of classes — broader
      vocabulary, but harder for the model to learn well with limited
      takes per word.

Folder skip rules:
  youtube_signlanguagetreasury  →  skip: vallás, földrajz, országok, politika,
                                          bankban, hivatal-munka, egészség
  youtube_extra_playlists       →  keep ONLY: állatok, iskola
  anything else under videos/   →  process normally (flat sign folders)

NON-JELNYELV FOLDERS ARE NOW UNRESTRICTED:
  jelnyelv/ is processed first. Non-jelnyelv folders (youtube_signlanguagetreasury,
  youtube_extra_playlists, etc.) are processed afterwards. Videos from those
  folders are kept if their label translates to valid Hungarian (contains
  Hungarian characters) — this way we capture treasury/playlist content
  without restricting to only jelnyelv-matched words. This massively improves
  dataset size without introducing garbage labels.

TREASURY LABEL RESOLUTION (filename -> folder-name fallback):
  youtube_signlanguagetreasury/ and other non-jelnyelv folders use
  English filenames like "007 - White.f134.mp4" inside Hungarian-named
  category folders like "színek". The label is normally derived from the
  filename ("white" -> translated -> "fehér"). If the filename yields
  nothing usable at all (empty string after stripping), the category
  folder name itself is translated and used instead, so a video never
  gets silently discarded just for having an unparseable filename.

OKTATOANYAGOK SOURCE (outside videos/, in the sibling jelsmart-server/
project folder):
  In addition to everything under videos/, we also walk:

      <2 levels above this script>/jelsmart-server/oktatoanyagok/

  This folder has the same shape as jelnyelv/: a flat list of category
  folders, one per word, each containing several .mp4 recordings of that
  word. The catch is that the "second version" recording inside a
  category folder is marked on the FILENAME, and inconsistently so — most
  commonly a trailing "o" (e.g. "alma.mp4" / "almao.mp4"), but some files
  instead use a trailing "A" or "Ao", or a "(A)"/"(B)" suffix. None of
  these actually indicate a different word — they're just alternate
  takes/recordings of the same sign.

  Rather than trying to pattern-match every inconsistent filename variant
  (which is fragile and easy to get wrong — e.g. stripping a trailing "a"
  or "o" off a filename would mangle plenty of legitimately Hungarian
  words that end in a plain "a", like "alma"), the label for every video
  in a category folder is simply the (normalized) CATEGORY FOLDER NAME,
  exactly like jelnyelv/. The filename itself is never parsed for the
  label here — every .mp4 inside a category folder, regardless of its
  own suffix, becomes just another take of that folder's word. This is
  the "fix" for the label: it can't fragment into "alma" / "almao" /
  "almaa" / "alma-a" etc., because the filename never enters the
  labelling decision in the first place.

  The same normalize_jelnyelv_label() take-number stripping and the same
  JELNYELV_FILTER_MODE / JELNYELV_ALLOWLIST filtering used for jelnyelv/
  is reused for oktatoanyagok/ category names, so the two sources stay
  consistent.

AUTO-TRANSLATION:
  Some source folders/filenames (e.g. certain YouTube playlists) are
  titled in English rather than Hungarian. Every label is run through
  the translator with automatic source-language detection (via
  deep-translator's own detection, which is far more reliable on short
  single-word labels than langdetect) and reformatted to match the
  lowercase-hyphenated label convention. If the translation comes back
  identical to the input (i.e. it was already Hungarian, or untranslatable),
  the original label is kept as-is to avoid needless reformatting.
  Requires `deep-translator` (pip install deep-translator); if it's
  missing, this step is skipped with a warning and labels are left as-is.
  Results are cached in dataset/translation_cache.json so repeated runs
  don't re-hit the translation API for labels already seen.

  NOTE: translation is only attempted for jelnyelv/ and oktatoanyagok/
  labels that are NOT already in JELNYELV_ALLOWLIST (see
  normalize_jelnyelv_label callers below). Translating an already-correct
  Hungarian label risks the translator's auto-detection guessing the
  wrong source language and silently corrupting a perfectly valid word -
  which would then fail the allowlist check and vanish from the dataset
  with no visible error.

UNICODE NORMALIZATION:
  All labels are normalized with NFC (Normalization Form Composed) to fix
  invisible character differences. Example: "péntek" and "péntek" may look
  identical but have different Unicode representations. NFC makes them
  truly identical so comparisons work correctly.
"""

import os
import json
import re
import sys
import unicodedata

import cv2
import mediapipe as mp
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
VIDEOS_DIR   = os.path.abspath(os.path.join(PROJECT_ROOT, "videos"))
DATASET_DIR  = os.path.abspath(os.path.join(PROJECT_ROOT, "dataset"))

# Sibling project folder, two levels above this script's own directory:
#   .../jelsmart-server/model/dataset_builder.py   (this file)
#   .../jelsmart-server/                            (2 up from this file's dir)
#   .../jelsmart-server/oktatoanyagok/              (the folder we want)
OKTATOANYAGOK_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "oktatoanyagok"
))

# "allowlist" -> small curated essentials-only set (~190 words)
# "blocklist" -> broad vocab, only cutting niche/adult categories
JELNYELV_FILTER_MODE = "allowlist"

# Only used when JELNYELV_FILTER_MODE == "blocklist"
ESSENTIAL_MODE = True

# Auto-translate labels into Hungarian before they're used/filtered.
# Set to False to disable entirely even if the translation library is installed.
AUTO_TRANSLATE_ENGLISH_LABELS = True

# ── folder filter rules ───────────────────────────────────────────────────────
# Folders to skip entirely from youtube_signlanguagetreasury.
SIGNLANGUAGETREASURY_SKIP = {
    "vallás", "földrajz", "országok", "politika",
    "bankban", "hivatal-munka", "egészség",
}

# From youtube_extra_playlists we ONLY keep these
EXTRA_PLAYLISTS_KEEP = {
    "állatok", "iskola",
}

# ── label aliases (map treasury labels to jelnyelv equivalents) ────────────────
LABEL_ALIASES = {
    # school
    "hátitáska": "hátizsák",
    "órarend": "óra",

    # animals
    "cica": "macska",
    "csibe": "csirke",
    "csirke-madár": "csirke",
    "malac": "disznó",
    "delfin": "hal",
    "cápa": "hal",

    # colors (treasury sometimes adds "-szín" suffix)
    "fehér-szín": "fehér",
    "fekete-szín": "fekete",
    "piros-szín": "piros",
    "kék-szín": "kék",
    "zöld-szín": "zöld",
    "sárga-szín": "sárga",
    "barna-szín": "barna",
    "szürke-szín": "szürke",
    "lila-szín": "lila",
    "rózsaszín-szín": "rózsaszín",
    "narancssárga-szín": "narancssárga",
}

# ── curated beginner/essentials allowlist for jelnyelv/ ──────────────────────
JELNYELV_ALLOWLIST = {
    # greetings / basic social phrases
    "szia", "köszönöm", "kösz", "köszönöm-szépen", "csókolom",
    "jó-reggelt", "jó-napot", "jó-estét", "jó-éjszakát", "viszontlátásra",
    "hogy-vagy", "hogy-érzed-magad", "hogy-hívnak", "igen", "nem",
    "persze", "oké", "kér-kérés",
    "bocsánat-megbocsát-megbocsátás-megbocsátó",

    # family / people
    "anya", "apa", "testvér", "bácsi", "néni", "barát", "barátnő",
    "gyerek", "fiú", "lány", "ember", "család", "én", "te", "ő",

    # numbers
    "számok-0-tól-9-ig", "számok-11-től-19-ig", "számok-20-20-3090",
    "hatvan",

    # colors
    "fehér", "fekete", "kék", "piros", "sárga", "zöld", "barna",
    "szürke", "lila", "rózsaszín", "narancssárga",

    # animals
    "kutya", "macska", "ló", "tehén", "birka", "disznó", "kacsa", "liba",
    "csirke", "nyúl-állat", "egér", "béka", "hal", "madár", "medve",
    "majom", "zsiráf", "tigris", "kígyó", "lepke", "méh", "méhecske",
    "pók", "bogár", "csiga", "teve", "kecske", "ökör", "bagoly", "sas",
    "kenguru", "krokodil", "gorilla", "csimpánz", "leopárd", "párduc",
    "rák", "pingvin", "gólya", "fecske", "sün", "mókus", "róka", "farkas",
    "szarvas", "vaddisznó", "patkány", "tyúk", "kakas", "pulyka",
    "papagáj", "gyík", "állat", "vadállat",

    # school
    "iskola", "iskolatáska", "tanár", "tanul-tanulás-megtanul", "tanuló",
    "osztály", "füzet", "toll", "ceruza", "könyv", "hátizsák", "rajz",
    "rajzol", "számol-számolás-megszámol", "olvasás-elolvasás",
    "betű", "szám", "matematika",
    "testnevelés", "óra",

    # food
    "alma", "banán", "kenyér", "tej", "sajt", "hús", "tojás", "kávé",
    "tea", "víz-vizes", "cukor", "só", "leves", "saláta", "gyümölcs",
    "zöldség", "narancs", "citrom", "körte", "szilva", "cseresznye",
    "eper", "dinnye", "sütemény", "csokoládé", "fagylalt", "palacsinta",
    "pizza", "hamburger",

    # common verbs
    "eszik-evés-evő", "iszik-ivás-ivó-megiszik", "alszik-alvó-elalszik",
    "jár", "megy", "fut-futás-futó", "ugrik-ugrás-beugró", "nevet",
    "sír-sírás", "játszik-játék-játszó", "néz-nézés-megnéz", "beszél",
    "szeret-szeretet", "tud-tudás", "akar", "van", "nincs",

    # body parts
    "fej", "szem", "száj", "orr", "fül", "kéz", "láb", "haj", "has",
    "hát", "váll", "ujj",

    # clothes
    "ruha", "nadrág", "ing", "cipő", "zokni", "kabát", "sapka", "sál",
    "kesztyű", "szoknya", "blúz", "pulóver", "kalap",

    # weather
    "eső", "hó", "nap", "szél", "felhő", "vihar", "meleg", "hideg",
    "hűvös",

    # basic adjectives
    "nagy", "kicsi", "jó", "rossz", "szép", "csúnya", "gyors", "lassú",
    "hosszú", "rövid", "magas", "alacsony", "boldog",
    "szomorú-szomorúság", "fáradt", "éhes", "szomjas-szomjazik",

    # house objects
    "asztal", "szék", "ágy", "ajtó", "ablak", "lámpa", "tükör",
    "szekrény", "tv", "telefon",

    # common places
    "ház", "park", "bolt", "mozi", "állatkert", "kórház", "étterem",
}

# ── blocklist mode (used only if JELNYELV_FILTER_MODE == "blocklist") ───────
# Labels to skip from jelnyelv/ (niche, religious, weapons, political etc.)
# NOTE: these are already normalized base words (no trailing take-numbers),
# since normalize_jelnyelv_label() strips take-numbers before this is checked.
JELNYELV_BLOCKLIST = {
    # religion
    "amen", "atya", "biblia", "bizonyságtétel", "bűnbocsánat",
    "első-áldozás", "afrika", "adomány-adományoz-adományozó", "eucharisztia",
    "hitvallás", "húsvét", "ima", "imádkozik",
    "jézus-krisztus", "karácsony", "karácsonyfa", "kegyelem", "keresztény",
    "misszió", "oltár", "pápa", "pünkösd", "szentlélek",
    "advent", "adventi-koszorú", "bérmálkozás", "bíboros",
    "dicsőség", "egyház", "felekezet", "gyónás", "halleluja", "hit",
    "hozsanna", "irgalom", "isten", "isten-igéje", "katolikus", "konvoj",
    "korona", "követ-személy", "légvédelem", "lélek", "lőszer",
    "megváltó", "megtérés", "ószövetség",
    "pap", "püspök", "rakéta", "sír-nyughely",
    "szent", "szenteste", "szűz-mária", "tanítvány", "templom",
    "törvény", "újszövetség", "üdvösség", "vallás", "zsoltár",
    "áldozás", "ágyú", "golyóálló-mellény",
    "fegyver", "hatalom",
    # weapons / war
    "atombomba", "atomháború",
    "puska", "katona", "katonaság",
    "hadsereg", "háború", "tank", "vadászgép",
    "béketárgyalás",
    # political / legal niche
    "bizottság", "biztonságpolitika", "fellebbez",
    "kihallgatás-kihallgat", "köztársaság", "köztársasági-elnök",
    "miniszterelnök", "szocializmus", "kommunizmus", "kommunista",
    "elnök", "elnökség", "ensz", "eu", "európai-tanács", "európai-unio",
    "forradalom", "jogszabály", "jog", "király", "királynő",
    "kormány", "kormányzat", "közgyűlés", "lakosság", "párt", "politika",
    "politikus", "politizál", "puccs", "reform", "rendelet",
    "társadalom", "tüntetés", "tüntet-tüntető",
    "szavazás", "szavaz", "sztrájk-sztrájkol", "uralkodik-uralkodás",
    "uralkodó", "demokrácia", "egyezmény", "civil", "képviselő",
    # tiny/niche cities
    "balatonalmádi", "balatonfüred", "balmazújváros", "berettyóújfalu",
    "csongrád", "komló", "mosonmagyaróvár", "nagykanizsa",
    "ózd", "várpalota", "kazincbarcika", "kiskunfélegyháza",
    "hajdúböszörmény", "hajdúszoboszló", "hódmezővásárhely", "dombóvár",
    "gyöngyös", "püspökladány", "salgótarján",
}

# Extra cuts for ESSENTIAL_MODE (only used in "blocklist" mode): everyday +
# animals + school + kid-friendly vocab is kept; everything below is
# filtered as too specialized/adult.
ESSENTIAL_EXTRA_SKIP = {
    # detailed medical / illness / assistive-tech jargon
    "agyrázkodás", "agyvérzes", "allergia-allergiás", "alkoholizmus-alkoholista",
    "bőrkiütés", "depresszió", "elájul-elájulás-ájul-ájulás",
    "fertőzés-fertőz-fertőző", "foghúzás", "fogtömés", "gyomorrontás",
    "hasmenés", "hallókészülék", "himlő", "injekció", "kezelés-kezel-kezelő",
    "kivizsgál-kivizsgálás-vizsgál-vizsgálat-vizsgáló", "migrén",
    "mozgássérült", "műt-műtés-műtét", "műtő", "nagyothalló", "orrfolyás",
    "pulzus", "sérülés-sérül", "siket", "siketvak", "siketlimpia",
    "szívinfarktus", "szédül-szédülés", "vérzik", "vérvétel", "védőnő",
    "taj-kártya", "indukciós-hurok", "szájról-olvasás", "fonomimika",
    "fonomimikai-ábécé",

    # banking / legal / government paperwork / insurance
    "alapítvány", "aláírás-aláír", "aláíró-pad", "bank", "bankbetét",
    "bankkártya", "bankszámlaszám", "benyújt-benyújtás", "bérlakás",
    "bérlés", "biztosítás", "biztosító", "blokk", "csekk",
    "elektronikus-aláírás", "elektronikus-ügyintézés", "online-ügyintézés",
    "engedély", "engedélyez", "erkölcsi-bizonyítvány", "hitel",
    "jogosítvány", "kölcsön-kölcsönöz-kölcsönzés-kölcsönző", "kölcsönöz",
    "nyomtatvány-elektronikusan-tölthető", "nyomtatvány-kézzel-tölthető",
    "szerződés", "ügyfél", "ügyfélfogadási-rend", "ügyfélpult",
    "ügyintézési-idő", "ügyintéző", "vízum", "zöldkártya", "bíró",
    "tanúsítvány", "üzembentartó", "forgalomból-kivonás", "kresz",
    "forgalmi", "jegyzőkönyv",

    # alcohol / gambling / adult-only venues
    "alkohol", "bor", "sör", "pálinka", "vodka", "koktél", "pezsgő",
    "lottó", "diszkó", "kocsma",

    # specialized adult trades/professions
    "ápoló", "ápolónő", "autószerelő", "bőrdíszműves", "buszvezető",
    "cukrász", "díszlet", "dokumentumfilm", "hegesztő",
    "hegeszt-hegesztés-hegesztett", "informatikus", "kőműves", "könyvelő",
    "kozmetikus", "nyomdász", "pedikűrös", "manikűrös", "recepciós",
    "szabó", "varrónő", "varroda", "üzem", "üzemanyag", "üzletközpont",

    # small/niche Hungarian town & landmark names (beyond the base blocklist)
    "debrecen", "eger", "esztegrom", "gyula", "győr", "kaposvár",
    "kecskemét", "keszthely", "makó", "miskolc", "mohács", "nyíregyháza",
    "pécs", "siófok", "sopron", "szeged", "székesfehérvár", "szekszárd",
    "szolnok", "szombathely", "tapolca", "tata", "tatabánya", "veszprém",
    "zalaegerszeg", "vác", "szigetvár", "hortobágy", "alföld", "fertő-tó",
    "velencei-tó", "tisza", "hősök-tere", "lánchíd", "budai-vár",

    # niche precision sports
    "curling", "darts", "dartsozik",

    # obsolete brands / overly technical transport jargon
    "malév", "máv", "ic",
}

# ── auto-translation of labels into Hungarian ─────────────────────────────────
try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("  [WARN] 'deep-translator' not installed — labels will NOT be "
          "auto-translated. Run:\n"
          "         pip install deep-translator\n")

TRANSLATION_CACHE_PATH = os.path.join(DATASET_DIR, "translation_cache.json")
_translation_cache = {}


def _load_translation_cache():
    global _translation_cache
    if os.path.exists(TRANSLATION_CACHE_PATH):
        try:
            with open(TRANSLATION_CACHE_PATH, "r", encoding="utf-8") as f:
                _translation_cache = json.load(f)
        except Exception:
            _translation_cache = {}


def _save_translation_cache():
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)
        with open(TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_translation_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [WARN] could not save translation cache: {e}")


def normalize_unicode_label(label):
    """
    Makes visually identical Unicode strings identical.

    Example:
        péntek (combining accent: e + combining accent)
        péntek (precomposed: single character)

    become the same string after NFC normalization.

    Also enforces the dataset naming format:
        lowercase-hyphen-separated
    """
    if not label:
        return label

    # NFC normalization fixes invisible Unicode differences
    label = unicodedata.normalize("NFC", label)

    label = label.strip().lower()
    label = re.sub(r'[\s_]+', '-', label)
    label = re.sub(r'-+', '-', label)
    label = label.strip('-')

    return label


# Kézi szótár a makacs angol szavakhoz
FORCED_TRANSLATIONS = {
    "cousin": "unokatestvér",
    "grandfather": "nagyapa",
    "grandmother": "nagymama",
    "family": "család",
}

def maybe_translate_to_hungarian(label, force_en=False):
    """
    Megpróbálja lefordítani a címkét. Ha force_en=True, akkor tudjuk, hogy angol.
    """

    #if not AUTO_TRANSLATE_ENGLISH_LABELS or not TRANSLATION_AVAILABLE:
    #   return label
    #if not label or len(label) < 2:
    #   return label

    #if label in _translation_cache:
    #    return _translation_cache[label]

    # Szóközökre cseréljük a kötőjeleket a fordításhoz
    search_term = label.replace("-", " ").strip()
    
    try:
        # Ha a szó benne van a kézi szótárban, ne is hívjuk az API-t
        manual = {"coffee": "kávé", "milk": "tej", "bread": "kenyér", "water": "víz", "cheese": "sajt"}
        if search_term.lower() in manual:
            res = manual[search_term.lower()]
        else:
            # Szigorúbb fordítás: ha force_en, akkor en->hu, egyébként auto->hu
            src = "en" if force_en else "auto"
            res = _GoogleTranslator(source=src, target="hu").translate(search_term)
        
        if res:
            new_label = normalize_unicode_label(res)
            # Ha a fordító ugyanazt adta vissza, az gyanús (pl. bread -> bread)
            #if new_label.lower() == label.lower() and not looks_like_hungarian(new_label):
            #    return label 
            
            _translation_cache[label] = new_label
            return new_label
    except Exception as e:
        print(f"      [!] Translation error for '{label}': {e}")
    
    return label

def looks_like_hungarian(label):
    """
    Quick heuristic: if a label contains Hungarian-specific characters
    (áéíóöőúüűcs), it's probably valid Hungarian and should be kept.
    """
    if not label:
        return False
    return any(c in label for c in 'áéíóöőúüűcsdzs')


# ── mediapipe setup ───────────────────────────────────────────────────────────
BaseOptions         = mp.tasks.BaseOptions
HandLandmarker      = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode


def extract_landmarks_from_video(video_path, verbose=True):
    """
    Returns a NumPy array of shape (num_frames, 42, 3) — both hands per frame.
    Returns None if extraction failed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print(f"    !! FAILED TO OPEN: {video_path}")
        cap.release()
        return None

    total_frames_reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    sequence       = []
    frame_index    = 0
    last_ts_ms     = -1

    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ts_ms = int(frame_index * 1000 / fps)
                if ts_ms <= last_ts_ms:
                    ts_ms = last_ts_ms + 1
                last_ts_ms = ts_ms

                results  = landmarker.detect_for_video(mp_image, ts_ms)
                hand_map = {"Left": None, "Right": None}
                for i, handedness_list in enumerate(results.handedness):
                    label = handedness_list[0].category_name
                    hand_map[label] = results.hand_landmarks[i]

                frame_landmarks = []
                for side in ["Left", "Right"]:
                    hand = hand_map[side]
                    if hand:
                        frame_landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand])
                    else:
                        frame_landmarks.extend([[0.0, 0.0, 0.0]] * 21)

                sequence.append(frame_landmarks)
                frame_index += 1

    except Exception as e:
        if verbose:
            print(f"    !! ERROR on frame {frame_index} of {video_path}: {e}")
        cap.release()
        return np.array(sequence) if sequence else None

    cap.release()

    if not sequence:
        if verbose:
            print(f"    !! ZERO FRAMES: {video_path}")
        return None

    if total_frames_reported and len(sequence) < total_frames_reported * 0.5:
        if verbose:
            print(f"    ~~ WARNING: only {len(sequence)}/{int(total_frames_reported)} frames")

    return np.array(sequence)


# ── label extraction from filename ───────────────────────────────────────────
def label_from_filename(filename):
    """
    Always derive the label from the video filename, not the folder.

    Patterns handled:
      "001 - Persze.f135.mp4"                  -> "persze"
      "002 - Bagoly - magyar jelnyelv.mp4"      -> "bagoly"
      "031 - Medve (2) - magyar jelnyelv.mp4"   -> "medve"  (drops recording number)
      "007 - White.f134.mp4"                    -> "white"
      "017 - Red.f134.mp4"                      -> "red"
      "some_sign_1.mp4"                         -> "some_sign"  (flat layout)

    The "magyar jelnyelv" suffix strip tolerates any dash style (hyphen,
    en-dash, em-dash, underscore) or just whitespace before it, since
    source filenames aren't consistent about which dash character they use.

    FIX: the yt-dlp format-code suffix (".f134", ".f251", etc.) is now
    stripped as its own standalone step BEFORE the combined
    format-suffix regex runs, and it tolerates the format code appearing
    with or without a following container extension (".f134.mp4",
    ".f134.webm", or a bare ".f134" with nothing after it). Previously
    only the combined ".f<digits>.mp4"-shaped pattern was handled, which
    happened to work for this dataset's actual filenames but was fragile
    to any variation (e.g. ".f134.MP4", a missing extension in a raw
    directory listing, etc.).

    FIX 2: Recording numbers in parentheses like "Medve (2)" now drop
    the number entirely instead of becoming "medve-2", since these are
    just multiple recordings of the same word, not distinct signs.
    """
    name = filename

    # Strip yt-dlp format-code suffixes first, tolerating an optional
    # container extension after the code, or none at all.
    name = re.sub(r'\.f\d+(\.(mp4|m4a|webm))?$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\.webm$', '', name, flags=re.IGNORECASE)

    # Belt-and-suspenders: original combined-suffix pattern, in case the
    # above didn't match some other ordering.
    name = re.sub(r'\.(f\d+|webm)\.(mp4|m4a|webm)$', '', name, flags=re.IGNORECASE)
    # Plain .mp4
    name = re.sub(r'\.mp4$', '', name, flags=re.IGNORECASE)

    # Strip leading number prefix "001 - " (hyphen, en-dash, or em-dash)
    name = re.sub(r'^\d+\s*[-–—]\s*', '', name)

    # Strip trailing " - magyar jelnyelv" and anything after it. Tolerates
    # hyphen/en-dash/em-dash/underscore as the separator, or no separator
    # at all beyond whitespace (e.g. "Bagoly — Magyar Jelnyelv (HD).mp4").
    name = re.sub(r'\s*[-–—_]*\s*magyar\s+jelnyelv.*$', '', name, flags=re.IGNORECASE)

    # Drop parenthesised recording numbers: "Medve (2)" -> "medve"
    # (not "medve-2", because these are just multiple recordings, not distinct signs)
    name = re.sub(r'\s*\(\d+\)', '', name)

    return normalize_unicode_label(name)


def normalize_jelnyelv_label(category):
    """
    Normalizes jelnyelv/ (and oktatoanyagok/) folder names so multiple
    takes of the SAME word merge into a single label instead of
    fragmenting the dataset.

    These are all just different recordings of "ablak" (window), not
    distinct signs, so they must all collapse to "ablak":
      "ablak_1_1"  -> "ablak"
      "ablak2_1"   -> "ablak"
      "ablak-1"    -> "ablak"
      "ablak-2"    -> "ablak"
      "ablak"      -> "ablak"

    FIX: now lowercases and strips whitespace FIRST, before any of the
    take-number stripping. Previously this function preserved whatever
    case the folder name happened to be in (e.g. "Fekete" or "FEKETE_1"),
    while JELNYELV_ALLOWLIST is entirely lowercase - so a folder that
    wasn't already perfectly lowercase would fail the `label not in
    JELNYELV_ALLOWLIST` check and get silently skipped, even though the
    word was clearly meant to be included. label_from_filename() already
    did this correctly; this brings the jelnyelv path in line with it.

    Also reused for oktatoanyagok/ category folders — see the module
    docstring's "OKTATOANYAGOK SOURCE" section for why the label there is
    always taken from the folder, never the filename.
    """
    label = category.strip().lower()

    # Strip trailing "_<digits>" groups, possibly repeated: "_1_1", "_2", etc.
    label = re.sub(r'(_\d+)+$', '', label)

    # Strip a trailing "-<digits>" take/recording number: "-1", "-2", etc.
    label = re.sub(r'-\d+$', '', label)

    # Strip a bare trailing digit glued directly to the word, e.g. "ablak2".
    label = re.sub(r'\d+$', '', label)

    # Tidy up any leftover/doubled separators from the stripping above.
    label = re.sub(r'[\s_]+', '-', label)
    label = re.sub(r'-+', '-', label).strip('-')

    return normalize_unicode_label(label)


def category_folder_label(category):
    """
    Turns a category folder name (e.g. "színek", "állatok") into a
    label-formatted string, mirroring the lowercase-hyphenated convention
    used everywhere else. Used only as a fallback in collect_videos() when
    a per-file label can't be derived from the filename at all.
    """
    label = category.strip().lower()
    label = re.sub(r'[\s_]+', '-', label)
    label = re.sub(r'-+', '-', label).strip('-')
    return normalize_unicode_label(label)


def collect_videos(videos_dir):
    """
    Yields (video_path, label) pairs.
    FORCED PRIORITY: Processes youtube_signlanguagetreasury FIRST for quick debugging.

    Sources, in yield order:
      1. non-jelnyelv folders under videos/ (treasury, extra playlists, ...)
      2. oktatoanyagok/  (sibling jelsmart-server/ project, see module docstring)
      3. jelnyelv/       (under videos/)
    """
    jelnyelv_videos = []
    oktatoanyagok_videos = []
    other_videos_temp = []
    collection_log = {
        "jelnyelv_kept": [], "jelnyelv_skipped": [],
        "treasury_kept": [], "treasury_dropped": [],
        "oktatoanyagok_kept": [], "oktatoanyagok_skipped": [],
    }

    # 1. Get all folders and FORCE youtube_signlanguagetreasury to the front
    all_sources = sorted(os.listdir(videos_dir))
    priority_folder = "youtube_signlanguagetreasury"
    
    if priority_folder in all_sources:
        all_sources.remove(priority_folder)
        all_sources.insert(0, priority_folder) # Put it at index 0
        print(f"  [DEBUG] Priority mode: Processing {priority_folder} first!")

    # ── PASS 1: Collection (videos/) ────────────────────────────────────────
    for source in all_sources:
        source_path = os.path.join(videos_dir, source)
        if not os.path.isdir(source_path): continue

        source_lower = source.lower()
        is_jelnyelv = "jelnyelv" in source_lower
        is_treasury = "signlanguagetreasury" in source_lower

        for category in sorted(os.listdir(source_path)):
            category_path = os.path.join(source_path, category)
            if not os.path.isdir(category_path): continue

            # Folder skips
            if is_treasury and category in SIGNLANGUAGETREASURY_SKIP: continue
            if "extra_playlists" in source_lower and category not in EXTRA_PLAYLISTS_KEEP: continue

            for filename in sorted(os.listdir(category_path)):
                if not filename.lower().endswith(".mp4"): continue
                
                video_path = os.path.join(category_path, filename)
                rel_path = os.path.relpath(video_path, videos_dir)

                if is_jelnyelv:
                    label = normalize_jelnyelv_label(category)
                    if JELNYELV_FILTER_MODE == "allowlist":
                        if label not in JELNYELV_ALLOWLIST:
                            label = maybe_translate_to_hungarian(label)
                        if label not in JELNYELV_ALLOWLIST:
                            collection_log["jelnyelv_skipped"].append({"video": rel_path, "label": label})
                            continue
                    jelnyelv_videos.append((video_path, label))
                else:
                    # Treasury / Others
                    label = label_from_filename(filename)
                    label = maybe_translate_to_hungarian(label)
                    label = normalize_unicode_label(label)
                    label = LABEL_ALIASES.get(label, label)

                    if not label:
                        label = category_folder_label(category)
                        label = maybe_translate_to_hungarian(label)

                    # --- IMMEDIATE VERIFICATION PRINT ---
                    if is_treasury:
                        # Check if it would be kept
                        status = "KEEP" if (label in JELNYELV_ALLOWLIST) else "DROP"
                        print(f"    [PREVIEW] Treasury File: {filename} | Label: '{label}' | Action: {status}")
                        if status == "DROP" and "cousin" in label.lower():
                            print(f"    !!! SUCCESS: 'cousin' was caught and will be dropped.")
                    # ------------------------------------

                    other_videos_temp.append({"video_path": video_path, "label": label, "rel_path": rel_path})

    # ── PASS 1b: Collection (oktatoanyagok/, outside videos_dir) ────────────
    # Same shape as jelnyelv/: category folder = word, every .mp4 inside is
    # just another take/recording of that word. The label ALWAYS comes from
    # the category folder name — filename suffixes like a trailing "o",
    # "A", "Ao", "(A)" or "(B)" (used inconsistently across files to mark a
    # "second version" recording) are never parsed for the label, so they
    # can't fragment a word into multiple labels. See module docstring.
    if os.path.isdir(OKTATOANYAGOK_DIR):
        for category in sorted(os.listdir(OKTATOANYAGOK_DIR)):
            category_path = os.path.join(OKTATOANYAGOK_DIR, category)
            if not os.path.isdir(category_path):
                continue

            label = normalize_jelnyelv_label(category)
            if JELNYELV_FILTER_MODE == "allowlist":
                if label not in JELNYELV_ALLOWLIST:
                    label = maybe_translate_to_hungarian(label)
                if label not in JELNYELV_ALLOWLIST:
                    collection_log["oktatoanyagok_skipped"].append(
                        {"video": category, "label": label}
                    )
                    continue

            for filename in sorted(os.listdir(category_path)):
                if not filename.lower().endswith(".mp4"):
                    continue

                video_path = os.path.join(category_path, filename)
                rel_path = os.path.relpath(video_path, OKTATOANYAGOK_DIR)

                oktatoanyagok_videos.append((video_path, label))
                collection_log["oktatoanyagok_kept"].append(
                    {"video": rel_path, "label": label}
                )
    else:
        print(f"  [WARN] oktatoanyagok folder not found at: {OKTATOANYAGOK_DIR}")

    # ── PASS 2: Final Filter (videos/ non-jelnyelv sources) ────────────────
    final_other_videos = []
    for item in other_videos_temp:
        label = item["label"]
        if JELNYELV_FILTER_MODE == "allowlist":
            if label in JELNYELV_ALLOWLIST:
                final_other_videos.append((item["video_path"], label))
                collection_log["treasury_kept"].append({"video": item["rel_path"], "label": label})
            else:
                collection_log["treasury_dropped"].append({"video": item["rel_path"], "label": label})
        else:
            # Blocklist mode logic...
            if looks_like_hungarian(label) and label not in JELNYELV_BLOCKLIST:
                final_other_videos.append((item["video_path"], label))

    # Yield Treasury/other first, then oktatoanyagok, then jelnyelv
    for v_path, v_label in final_other_videos: yield v_path, v_label
    for v_path, v_label in oktatoanyagok_videos: yield v_path, v_label
    for v_path, v_label in jelnyelv_videos: yield v_path, v_label


# ── main build ────────────────────────────────────────────────────────────────
def build_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)

    # ── step 1: wipe existing dataset (originals + augmented) ────────────────
    print("=" * 60)
    print("STEP 1 — clearing existing dataset/")
    print("=" * 60)
    removed = 0
    for fname in os.listdir(DATASET_DIR):
        if fname.endswith(".npy"):
            os.remove(os.path.join(DATASET_DIR, fname))
            removed += 1
    print(f"  Removed {removed} .npy files.\n")

    # ── step 2: extract ───────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2 — extracting landmarks from videos/ and oktatoanyagok/")
    print("=" * 60)
    print(f"  jelnyelv filter mode: {JELNYELV_FILTER_MODE}")
    print(f"  auto-translate labels: "
          f"{AUTO_TRANSLATE_ENGLISH_LABELS and TRANSLATION_AVAILABLE}")
    print(f"  oktatoanyagok dir: {OKTATOANYAGOK_DIR}"
          f"{'' if os.path.isdir(OKTATOANYAGOK_DIR) else '  [NOT FOUND]'}")

    _load_translation_cache()

    if not os.path.isdir(VIDEOS_DIR):
        print(f"'{VIDEOS_DIR}' not found — nothing to process.")
        return

    videos = list(collect_videos(VIDEOS_DIR))
    _save_translation_cache()

    if not videos:
        print("No .mp4 files found.")
        return

    labels_map     = {}
    label_counters = {}
    failed         = []

    for video_path, label in videos:
        # video_path may live outside VIDEOS_DIR (oktatoanyagok/), so
        # os.path.relpath just produces a "../"-prefixed path for those —
        # still a valid, readable relative path for the labels/failed logs.
        rel = os.path.relpath(video_path, VIDEOS_DIR)
        print(f"\n  {rel}")
        print(f"    -> label: '{label}'")

        sequence = extract_landmarks_from_video(video_path)

        label_counters[label] = label_counters.get(label, 0) + 1
        out_name = f"{label}_{label_counters[label]}.npy"

        if sequence is None:
            failed.append({"video": rel, "out_name": out_name, "label": label})
            print(f"    -> SKIPPED (extraction failed)")
            continue

        np.save(os.path.join(DATASET_DIR, out_name), sequence)
        labels_map[out_name] = {"label": label, "source_video": rel}
        print(f"    -> saved {out_name}  shape={sequence.shape}")

    # ── step 3: write labels.json ─────────────────────────────────────────────
    with open(os.path.join(DATASET_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_map, f, ensure_ascii=False, indent=2)

    if failed:
        with open(os.path.join(DATASET_DIR, "failed.json"), "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"\n{len(failed)} videos failed — see dataset/failed.json")

    unique_labels = len(label_counters)
    print("\n" + "=" * 60)
    print(f"Done. {len(labels_map)} videos saved to dataset/ across {unique_labels} labels")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. python model/build_manifest.py")
    print("  2. python model/augment_dataset.py")
    print("  3. python model/train.py")


if __name__ == "__main__":
    build_dataset()