# JelSmart – egyesített rendszer (monorepo)

Forrás: a jelsmart.hu éles szerveren futó kód (`/home/ihne/jelsmart-server`), 2026.11.

| Mappa | Mi ez |
|---|---|
| `app.py` | Fő oldal (Leckék, Tükör) + a Demo V2 felcsatolása `/demo` alá |
| `api/app.py` | **Tükör V2** API – `POST /api/tukor`, HuggingFace ujjábécé-modell |
| `static/js/tukor.js`, `templates/tukor.html` | Tükör V2 frontend |
| `demo/` | **Demo V2** – LSTM kategória-modellek + Ollama vision fallback |
| `programs/JelSmartApp/signlang-learn/` | Tanuló oldal (`app.py`) + **Admin felület** (`admin_app.py`, :6750) |
| `programs/mirror/` | Tükör V1 (régi pickle modell, `model.p`) |
| `deploy/` | systemd service fájlok a szerverekről |

## Ami NINCS a Gitben (a szerveren van)
`oktatoanyagok/` (~4 GB), `programs/media/`, `demo/videos/`, `demo/dataset/`,
`demo/model/weights/*.h5`, `*.task` (MediaPipe), `programs/mirror/data/`.
Az üres mappák `.gitkeep`-pel jelzik a helyüket.

## Előzmények
- A `signlang-learn` korábbi története (GitHub: ihneAi/signlang-learn) az
  `archiv/signlang-learn-2025` ágon, a szerver nem commitolt módosításai az
  `archiv/szerver-stash-2025-09` ágon.
