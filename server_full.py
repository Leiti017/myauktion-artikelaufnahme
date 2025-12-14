# server_full.py – MyAuktion KI-Artikelaufnahme-System
# Komplette, modernisierte Version mit:
# - Hintergrund-KI (OpenAI Vision)
# - Titel + Beschreibung + Preis
# - CSV-Export ohne Einheit
# - Admin + Frontend kompatibel
# - Sehr robust und schnell


from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
from PIL import Image, ImageOps
import io, json, time, csv

app = FastAPI()

# CORS (für Handy/Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Upload-Ordner
RAW_DIR = BASE_DIR / "uploads" / "raw"
PROCESSED_DIR = BASE_DIR / "uploads" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Export
EXPORT_DIR = BASE_DIR / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_CSV = EXPORT_DIR / "artikel_export.csv"

# Static
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


# ------------------------------------------------------------
# META JSON
# ------------------------------------------------------------

def _default_meta(artikelnr: str) -> dict:
    return {
        "artikelnr": artikelnr,
        "titel": f"Artikel {artikelnr}",
        "beschreibung": "",
        "retail_price": 0.0,
        "rufpreis": 0.0,
        "lagerort": "",
        "einlieferer": "",
        "mitarbeiter": "",
        "lagerstand": 1,
        "reviewed": False,
        "ki_source": "",
        "batch_done": False,
        "last_image": "",
        "ki_runtime_ms": 0,
        "ki_last_error": "",
    }


def _load_meta_json(artikelnr: str) -> dict:
    meta_path = RAW_DIR / f"{artikelnr}.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text("utf-8"))
        except:
            data = {}
    else:
        data = {}

    merged = _default_meta(artikelnr)
    merged.update(data)
    return merged


def _save_meta_json(artikelnr: str, data: dict):
    path = RAW_DIR / f"{artikelnr}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    return path


# ------------------------------------------------------------
# KI (OpenAI Vision) – Hintergrundmodus
# ------------------------------------------------------------

def _run_meta_with_retry(artikelnr: str, img_path: Path, max_attempts=2):
    """
    Ruft ki_engine_openai.generate_meta auf.
    Erkennt Default-Fälle und versucht erneut.
    """
    try:
        import ki_engine_openai as ki_engine
    except Exception as e:
        print("[KI] Importfehler ki_engine_openai:", e)
        return _default_meta(artikelnr), False, str(e), 0

    total_start = time.time()
    last_error = None

    for attempt in range(1, max_attempts + 1):
        print(f"[KI] Versuch {attempt} für Artikel {artikelnr} …")
        try:
            start = time.time()
            meta = ki_engine.generate_meta(str(img_path), str(artikelnr))
            dur = int((time.time() - start) * 1000)

            print(f"[KI] Antwortzeit {dur} ms")

            # Wenn nur defaults → KI-Ausfall
            if (
                meta["title"].startswith("Artikel")
                and meta["description"].startswith("Artikel wie abgebildet")
            ):
                last_error = "ki_default"
                continue

            return meta, True, "", int((time.time() - total_start) * 1000)

        except Exception as e:
            last_error = str(e)
            print("[KI-FEHLER] Exception:", e)

    return _default_meta(artikelnr), False, last_error or "unknown", int(
        (time.time() - total_start) * 1000
    )


def _run_meta_background(artikelnr: str, img_path: Path):
    """
    Läuft im Hintergrund.
    Speichert Titel, Beschreibung, retail_price und rufpreis=20%.
    """
    print(f"[BG-KI] Starte Hintergrund-KI für Artikel {artikelnr}")

    meta, ki_ok, ki_error, runtime_ms = _run_meta_with_retry(artikelnr, img_path)

    mj = _load_meta_json(artikelnr)

    mj["ki_runtime_ms"] = runtime_ms
    mj["ki_last_error"] = "" if ki_ok else ki_error
    mj["last_image"] = img_path.name

    if ki_ok:
        mj["titel"] = meta["title"]
        mj["beschreibung"] = meta["description"]

        retail = float(meta.get("retail_price", 0.0))
        mj["retail_price"] = round(retail, 2)
        mj["rufpreis"] = round(retail * 0.20, 2)  # 20 % Startpreis

        mj["ki_source"] = "realtime"
        mj["batch_done"] = True
        mj["reviewed"] = False

    _save_meta_json(artikelnr, mj)
    _rebuild_csv_export()

    print(f"[BG-KI] Fertig für Artikel {artikelnr} – ki_ok={ki_ok}")


# ------------------------------------------------------------
# CSV EXPORT
# ------------------------------------------------------------

def _rebuild_csv_export():
    rows = []

    for meta_path in RAW_DIR.glob("*.json"):
        try:
            d = json.loads(meta_path.read_text("utf-8"))
        except:
            continue

        nr = d.get("artikelnr") or meta_path.stem

        rows.append({
            "ArtikelNr": nr,
            "Menge": 1,
            "Bezeichnung": d.get("titel", f"Artikel {nr}"),
            "Beschreibung": d.get("beschreibung", ""),
            "Rufpreis": d.get("rufpreis", 0.0),
            "Lagerstand": d.get("lagerstand", 1),
            "Lagerort": d.get("lagerort", ""),
            "Einlieferer": d.get("einlieferer", ""),
            "Mitarbeiter": d.get("mitarbeiter", ""),
            "RetailPreis": d.get("retail_price", 0.0),  # NEU
        })

    # Sort
    rows.sort(key=lambda r: int(r["ArtikelNr"]))

    EXPORT_CSV.write_text("", "utf-8")
    with EXPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ArtikelNr",
                "Menge",
                "Bezeichnung",
                "Beschreibung",
                "Rufpreis",
                "Lagerstand",
                "Lagerort",
                "Einlieferer",
                "Mitarbeiter",
                "RetailPreis"
            ],
            delimiter=";"
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("[CSV] Export aktualisiert:", EXPORT_CSV)


# ------------------------------------------------------------
# API ROUTES
# ------------------------------------------------------------

@app.get("/")
def root():
    path = BASE_DIR / "index.html"
    return FileResponse(str(path))


@app.get("/admin")
def admin_root():
    path = BASE_DIR / "admin.html"
    return FileResponse(str(path))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/next_artikelnr")
def next_artikelnr():
    max_nr = 0
    for p in RAW_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
            nr = int(d.get("artikelnr", 0))
            max_nr = max(max_nr, nr)
        except:
            pass

    return {"ok": True, "next": max_nr + 1}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    artikelnr: str = Form(...),
    background_tasks: BackgroundTasks = None,
):
    artikelnr = artikelnr.strip()
    data = await file.read()

    # Bild öffnen
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
    except Exception as e:
        return {"ok": False, "error": f"Ungültiges Bild: {e}"}

    # Bild speichern
    existing = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    idx = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0

    out = RAW_DIR / f"{artikelnr}_{idx}.jpg"
    img.save(out, "JPEG", quality=85)

    # Meta minimal
    mj = _load_meta_json(artikelnr)
    mj["last_image"] = out.name
    _save_meta_json(artikelnr, mj)

    _rebuild_csv_export()

    # Hintergrund-KI
    if background_tasks:
        background_tasks.add_task(_run_meta_background, artikelnr, out)

    return {"ok": True}


@app.get("/api/images")
def images(artikelnr: str):
    artikelnr = artikelnr.strip()
    files = []
    for f in RAW_DIR.glob(f"{artikelnr}_*.jpg"):
        rel = f.relative_to(BASE_DIR)
        files.append("/static/" + str(rel).replace("\\", "/"))
    return {"ok": True, "files": files}


@app.get("/api/meta")
def meta(artikelnr: str):
    artikelnr = artikelnr.strip()
    mj = _load_meta_json(artikelnr)

    # Lade letztes Bild
    pics = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    img_url = ""
    if pics:
        rel = pics[-1].relative_to(BASE_DIR)
        img_url = "/static/" + str(rel).replace("\\", "/")

    return {
        "ok": True,
        "artikelnr": artikelnr,
        "titel": mj["titel"],
        "beschreibung": mj["beschreibung"],
        "retail_price": mj["retail_price"],
        "rufpreis": mj["rufpreis"],
        "lagerort": mj["lagerort"],
        "einlieferer": mj["einlieferer"],
        "mitarbeiter": mj["mitarbeiter"],
        "reviewed": mj["reviewed"],
        "ki_last_error": mj["ki_last_error"],
        "ki_source": mj["ki_source"],
        "image": img_url,
    }


@app.post("/api/save")
def save(data: dict):
    artikelnr = str(data.get("artikelnr"))
    if not artikelnr:
        return {"ok": False, "error": "Artikelnummer fehlt"}

    mj = _load_meta_json(artikelnr)

    for k in ["titel", "beschreibung", "lagerort", "einlieferer", "mitarbeiter"]:
        if k in data:
            mj[k] = data[k]

    mj["reviewed"] = data.get("reviewed", False)

    _save_meta_json(artikelnr, mj)
    _rebuild_csv_export()
    return {"ok": True}


# ---- Admin API ----

@app.get("/api/admin/articles")
def admin_articles():
    items = []

    for p in RAW_DIR.glob("*.json"):
        mj = json.loads(p.read_text("utf-8"))
        nr = mj.get("artikelnr")
        titel = mj.get("titel")
        desc = mj.get("beschreibung")

        # Bild
        pics = sorted(RAW_DIR.glob(f"{nr}_*.jpg"))
        img_url = ""
        if pics:
            rel = pics[-1].relative_to(BASE_DIR)
            img_url = "/static/" + str(rel).replace("\\", "/")

        items.append({
            "artikelnr": nr,
            "titel": titel,
            "beschreibung": desc,
            "retail_price": mj.get("retail_price", 0.0),
            "rufpreis": mj.get("rufpreis", 0.0),
            "reviewed": mj.get("reviewed", False),
            "ki_source": mj.get("ki_source", ""),
            "image": img_url
        })

    items.sort(key=lambda x: int(x["artikelnr"]))
    return {"ok": True, "items": items}

@app.get("/api/describe")
def describe(artikelnr: str):
    """
    Manuelle KI-Neuberechnung (Button in index.html).
    Nimmt das letzte Bild, ruft die KI auf, aktualisiert JSON & CSV.
    """
    artikelnr = artikelnr.strip()
    pics = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    if not pics:
        return {"ok": False, "error": "Kein Bild für diese Artikelnummer gefunden."}

    img_path = pics[-1]
    meta, ki_ok, ki_error, runtime_ms = _run_meta_with_retry(artikelnr, img_path)

    mj = _load_meta_json(artikelnr)
    mj["ki_runtime_ms"] = runtime_ms
    mj["ki_last_error"] = "" if ki_ok else ki_error

    if ki_ok:
        mj["titel"] = meta["title"]
        mj["beschreibung"] = meta["description"]

        retail = float(meta.get("retail_price", 0.0))
        mj["retail_price"] = round(retail, 2)
        mj["rufpreis"] = round(retail * 0.20, 2)

        mj["ki_source"] = "realtime"
        mj["batch_done"] = True
        mj["reviewed"] = False

    _save_meta_json(artikelnr, mj)
    _rebuild_csv_export()

    if not ki_ok:
        return {"ok": False, "error": ki_error or "KI konnte keinen Vorschlag liefern."}

    return {
        "ok": True,
        "title": mj["titel"],
        "description": mj["beschreibung"],
        "retail_price": mj["retail_price"],
        "rufpreis": mj["rufpreis"],
        "ki_runtime_ms": runtime_ms,
    }

@app.post("/api/admin/articles/update")
def admin_update(data: dict):
    artikelnr = str(data.get("artikelnr"))
    mj = _load_meta_json(artikelnr)

    mj["titel"] = data.get("titel", mj["titel"])
    mj["beschreibung"] = data.get("beschreibung", mj["beschreibung"])
    mj["reviewed"] = bool(data.get("reviewed", True))

    _save_meta_json(artikelnr, mj)
    _rebuild_csv_export()
    return {"ok": True}
