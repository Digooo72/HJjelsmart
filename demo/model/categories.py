"""
categories.py — shared word -> category mapping.

Mirrors the section groupings already used as comments in
dataset_builder.py's JELNYELV_ALLOWLIST, so the category boundaries match
how the vocabulary was originally curated (greetings, family, numbers,
colors, animals, school, food, verbs, body parts, clothes, weather,
adjectives, house objects, places).

Used today by the live app (model/analyzer.py) to restrict which of the
model's labels are eligible when the user picks a category. Reusable later
if/when per-category models get trained (build_manifest.py could tag rows
with WORD_TO_CATEGORY[label] the same way).
"""

CATEGORY_LABELS = {
    "greetings":  "Köszönés és alapok",
    "family":     "Család és emberek",
    "numbers":    "Számok",
    "colors":     "Színek",
    "animals":    "Állatok",
    "school":     "Iskola",
    "food":       "Étel és ital",
    "verbs":      "Igék",
    "body":       "Testrészek",
    "clothes":    "Ruházat",
    "weather":    "Időjárás",
    "adjectives": "Melléknevek",
    "house":      "Otthoni tárgyak",
    "places":     "Helyek",
}

WORD_TO_CATEGORY = {
    # greetings / basic social phrases
    "szia": "greetings", "köszönöm": "greetings", "kösz": "greetings",
    "köszönöm-szépen": "greetings", "csókolom": "greetings",
    "jó-reggelt": "greetings", "jó-napot": "greetings", "jó-estét": "greetings",
    "jó-éjszakát": "greetings", "viszontlátásra": "greetings",
    "hogy-vagy": "greetings", "hogy-érzed-magad": "greetings",
    "hogy-hívnak": "greetings", "igen": "greetings", "nem": "greetings",
    "persze": "greetings", "oké": "greetings", "kér-kérés": "greetings",
    "bocsánat-megbocsát-megbocsátás-megbocsátó": "greetings",

    # family / people
    "anya": "family", "apa": "family", "testvér": "family", "bácsi": "family",
    "néni": "family", "barát": "family", "barátnő": "family", "gyerek": "family",
    "fiú": "family", "lány": "family", "ember": "family", "család": "family",
    "én": "family", "te": "family", "ő": "family",

    # numbers
    "számok-0-tól-9-ig": "numbers", "számok-11-től-19-ig": "numbers",
    "számok-20-20-3090": "numbers", "hatvan": "numbers",

    # colors
    "fehér": "colors", "fekete": "colors", "kék": "colors", "piros": "colors",
    "sárga": "colors", "zöld": "colors", "barna": "colors", "szürke": "colors",
    "lila": "colors", "rózsaszín": "colors", "narancssárga": "colors",

    # animals
    "kutya": "animals", "macska": "animals", "ló": "animals", "tehén": "animals",
    "birka": "animals", "disznó": "animals", "kacsa": "animals", "liba": "animals",
    "csirke": "animals", "nyúl-állat": "animals", "egér": "animals", "béka": "animals",
    "hal": "animals", "madár": "animals", "medve": "animals", "majom": "animals",
    "zsiráf": "animals", "tigris": "animals", "kígyó": "animals", "lepke": "animals",
    "méh": "animals", "méhecske": "animals", "pók": "animals", "bogár": "animals",
    "csiga": "animals", "teve": "animals", "kecske": "animals", "ökör": "animals",
    "bagoly": "animals", "sas": "animals", "kenguru": "animals", "krokodil": "animals",
    "gorilla": "animals", "csimpánz": "animals", "leopárd": "animals", "párduc": "animals",
    "rák": "animals", "pingvin": "animals", "gólya": "animals", "fecske": "animals",
    "sün": "animals", "mókus": "animals", "róka": "animals", "farkas": "animals",
    "szarvas": "animals", "vaddisznó": "animals", "patkány": "animals", "tyúk": "animals",
    "kakas": "animals", "pulyka": "animals", "papagáj": "animals", "gyík": "animals",
    "állat": "animals", "vadállat": "animals",

    # school
    "iskola": "school", "iskolatáska": "school", "tanár": "school",
    "tanul-tanulás-megtanul": "school", "tanuló": "school", "osztály": "school",
    "füzet": "school", "toll": "school", "ceruza": "school", "könyv": "school",
    "hátizsák": "school", "rajz": "school", "rajzol": "school",
    "számol-számolás-megszámol": "school", "olvasás-elolvasás": "school",
    "betű": "school", "szám": "school", "matematika": "school",
    "testnevelés": "school", "óra": "school",

    # food
    "alma": "food", "banán": "food", "kenyér": "food", "tej": "food", "sajt": "food",
    "hús": "food", "tojás": "food", "kávé": "food", "tea": "food", "víz-vizes": "food",
    "cukor": "food", "só": "food", "leves": "food", "saláta": "food", "gyümölcs": "food",
    "zöldség": "food", "narancs": "food", "citrom": "food", "körte": "food",
    "szilva": "food", "cseresznye": "food", "eper": "food", "dinnye": "food",
    "sütemény": "food", "csokoládé": "food", "fagylalt": "food", "palacsinta": "food",
    "pizza": "food", "hamburger": "food",

    # common verbs
    "eszik-evés-evő": "verbs", "iszik-ivás-ivó-megiszik": "verbs",
    "alszik-alvó-elalszik": "verbs", "jár": "verbs", "megy": "verbs",
    "fut-futás-futó": "verbs", "ugrik-ugrás-beugró": "verbs", "nevet": "verbs",
    "sír-sírás": "verbs", "játszik-játék-játszó": "verbs", "néz-nézés-megnéz": "verbs",
    "beszél": "verbs", "szeret-szeretet": "verbs", "tud-tudás": "verbs",
    "akar": "verbs", "van": "verbs", "nincs": "verbs",

    # body parts
    "fej": "body", "szem": "body", "száj": "body", "orr": "body", "fül": "body",
    "kéz": "body", "láb": "body", "haj": "body", "has": "body", "hát": "body",
    "váll": "body", "ujj": "body",

    # clothes
    "ruha": "clothes", "nadrág": "clothes", "ing": "clothes", "cipő": "clothes",
    "zokni": "clothes", "kabát": "clothes", "sapka": "clothes", "sál": "clothes",
    "kesztyű": "clothes", "szoknya": "clothes", "blúz": "clothes", "pulóver": "clothes",
    "kalap": "clothes",

    # weather
    "eső": "weather", "hó": "weather", "nap": "weather", "szél": "weather",
    "felhő": "weather", "vihar": "weather", "meleg": "weather", "hideg": "weather",
    "hűvös": "weather",

    # basic adjectives
    "nagy": "adjectives", "kicsi": "adjectives", "jó": "adjectives",
    "rossz": "adjectives", "szép": "adjectives", "csúnya": "adjectives",
    "gyors": "adjectives", "lassú": "adjectives", "hosszú": "adjectives",
    "rövid": "adjectives", "magas": "adjectives", "alacsony": "adjectives",
    "boldog": "adjectives", "szomorú-szomorúság": "adjectives", "fáradt": "adjectives",
    "éhes": "adjectives", "szomjas-szomjazik": "adjectives",

    # house objects
    "asztal": "house", "szék": "house", "ágy": "house", "ajtó": "house",
    "ablak": "house", "lámpa": "house", "tükör": "house", "szekrény": "house",
    "tv": "house", "telefon": "house",

    # common places
    "ház": "places", "park": "places", "bolt": "places", "mozi": "places",
    "állatkert": "places", "kórház": "places", "étterem": "places",
}


def category_of(word):
    """Returns the category id for a word, or None if it isn't mapped
    (e.g. a trained label that wasn't part of the curated allowlist this
    mapping was built from)."""
    return WORD_TO_CATEGORY.get(word)