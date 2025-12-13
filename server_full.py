# server_full.py – MyAuktion KI-Artikelaufnahme-System
# FINAL (1x KI-Versuch, kein Fallback, kein Retry)
#
# Features:
# - Upload + sofortige Vorschau
# - Hintergrund-KI (OpenAI Vision) -> Titel, Beschreibung, retail_price
# - rufpreis = 20% von retail_price (nur wenn KI ok)
# - KEIN Fallback-Text, KEIN Dauerversuch
# - /api/describe für manuelle KI-Neuberechnung
# - Admin API + CSV Export (ohne Einheit)

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
from PIL import Image, ImageOps
import io, json, time, csv
import math

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

# Static (damit /static/... funktioniert)
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


# ------------------------------------------------------------
# META JSON
# ------------------------------------------------------------

def _default_meta(artikelnr: str) -> dict:
    # KEIN Fallback-Text. Leere Felder wenn KI nicht läuft.
    return {
        "artikelnr": artikelnr,
        "titel": "",
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
        except Exception:
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
# KI (OpenAI Vision) – GENAU 1 VERSUCH, KEIN FALLBACK, KEIN RETRY
# ------------------------------------------------------------

def _run_meta_once(artikelnr: str, img_path: Path):
    """
    Führt GENAU EINEN KI-Versuch aus.
    Kein Fallback, kein Retry.
    """
    try:
        import ki_engine_openai as ki_engine
    except Exception as e:
        print("[KI] Importfehler:", e)
        return None, False, str(e), 0

    start = time.time()
    try:
        meta = ki_engine.generate_meta(str(img_path), str(artikelnr))
        runtime_ms = int((time.time() - start) * 1000)

        # Wenn kein sinnvoller Output -> Fehler
        if not meta or not meta.get("title"):
            return None, False, "empty_result", runtime_ms

        return meta, True, "", runtime_ms

    except Exception as e:
        runtime_ms = int((time.time() - start) * 1000)
        print("[KI] Fehler:", e)
        return None, False, str(e), runtime_ms


def _run_meta_background(artikelnr: str, img_path: Path):
    """
    Läuft im Hintergrund.
    Speichert Titel, Beschreibung, retail_price und rufpreis=20% – NUR wenn KI ok.
    """
    print(f"[BG-KI] Starte Hintergrund-KI für Artikel {artikelnr}")

    meta, ki_ok, ki_error, runtime_ms = _run_meta_once(artikelnr, img_path)

    mj = _load_meta_json(artikelnr)
    mj["ki_runtime_ms"] = runtime_ms
    mj["last_image"] = img_path.name

    if ki_ok:
        mj["titel"] = (meta.get("title") or "").strip()
        mj["beschreibung"] = (meta.get("description") or "").strip()

        retail = meta.get("retail_price", 0.0)
        try:
            retail_f = float(retail)
        except (TypeError, ValueError):
            retail_f = 0.0

        mj["retail_price"] = round(retail_f, 2)
        mj["rufpreis"] = float(math.ceil(retail_f * 0.20))

        mj["ki_source"] = "realtime"
        mj["batch_done"] = True
        mj["reviewed"] = False
        mj["ki_last_error"] = ""

    else:
        # KEIN Fallback, KEIN Default. Felder bleiben wie sie sind (leer oder manuell).
        mj["ki_last_error"] = ki_error or "ki_failed"
        mj["ki_source"] = "failed"
        mj["batch_done"] = True  # fertig = ja, aber KI fehlgeschlagen

    _save_meta_json(artikelnr, mj)
    _rebuild_csv_export()

    print(f"[BG-KI] Fertig für Artikel {artikelnr} – ki_ok={ki_ok}")


# ------------------------------------------------------------
# CSV EXPORT (ohne Einheit)
# ------------------------------------------------------------

def _rebuild_csv_export():
    rows = []

    for meta_path in RAW_DIR.glob("*.json"):
        try:
            d = json.loads(meta_path.read_text("utf-8"))
        except Exception:
            continue

        nr = d.get("artikelnr") or meta_path.stem

        rows.append({
            "ArtikelNr": nr,
            "Menge": 1,
            "Bezeichnung": d.get("titel", "") or "",
            "Beschreibung": d.get("beschreibung", "") or "",
            "Rufpreis": d.get("rufpreis", 0.0) or 0.0,
            "Lagerstand": d.get("lagerstand", 1) or 1,
            "Lagerort": d.get("lagerort", "") or "",
            "Einlieferer": d.get("einlieferer", "") or "",
            "Mitarbeiter": d.get("mitarbeiter", "") or "",
            "RetailPreis": d.get("retail_price", 0.0) or 0.0,
        })

    # Sort
    try:
        rows.sort(key=lambda r: int(r["ArtikelNr"]))
    except Exception:
        rows.sort(key=lambda r: str(r["ArtikelNr"]))

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
                "RetailPreis",
            ],
            delimiter=";"
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("[CSV] Export aktualisiert:", EXPORT_CSV)


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------

@app.get("/")
def root():
    path = BASE_DIR / "index.html"
    return FileResponse(str(path))


@app.get("/admin")
def admin_root():
    path = BASE_DIR / "admin.html"
    return FileResponse(str(path))


@app.get("/api/export.csv")
def export_csv():
    # Für Admin-Button: CSV direkt downloaden
    if not EXPORT_CSV.exists():
        _rebuild_csv_export()
    return FileResponse(str(EXPORT_CSV), filename="artikel_export.csv", media_type="text/csv")


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
        except Exception:
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
        return JSONResponse({"ok": False, "error": f"Ungültiges Bild: {e}"}, status_code=400)

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

    # Hintergrund-KI (1 Versuch)
    if background_tasks:
        background_tasks.add_task(_run_meta_background, artikelnr, out)

    return {"ok": True}


@app.get("/api/images")
def images(artikelnr: str):
    artikelnr = artikelnr.strip()
    files = []
    for f in sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg")):
        rel = f.relative_to(BASE_DIR)
        files.append("/static/" + str(rel).replace("\\", "/"))
    return {"ok": True, "files": files}


@app.get("/api/meta")
def meta(artikelnr: str):
    artikelnr = artikelnr.strip()
    mj = _load_meta_json(artikelnr)

    pics = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    img_url = ""
    if pics:
        rel = pics[-1].relative_to(BASE_DIR)
        img_url = "/static/" + str(rel).replace("\\", "/")

    return {
        "ok": True,
        "artikelnr": artikelnr,
        "titel": mj.get("titel", "") or "",
        "beschreibung": mj.get("beschreibung", "") or "",
        "retail_price": mj.get("retail_price", 0.0) or 0.0,
        "rufpreis": mj.get("rufpreis", 0.0) or 0.0,
        "lagerort": mj.get("lagerort", "") or "",
        "einlieferer": mj.get("einlieferer", "") or "",
        "mitarbeiter": mj.get("mitarbeiter", "") or "",
        "reviewed": mj.get("reviewed", False) or False,
        "ki_last_error": mj.get("ki_last_error", "") or "",
        "ki_source": mj.get("ki_source", "") or "",
        "image": img_url,
    }


@app.post("/api/save")
def save(data: dict):
    artikelnr = str(data.get("artikelnr", "")).strip()
    if not artikelnr:
        return JSONResponse({"ok": False, "error": "Artikelnummer fehlt"}, status_code=400)

    mj = _load_meta_json(artikelnr)

    for k in ["titel", "beschreibung", "lagerort", "einlieferer", "mitarbeiter"]:
        if k in data and data[k] is not None:
            mj[k] = str(data[k])

    mj["reviewed"] = bool(data.get("reviewed", False))

    _save_meta_json(artikelnr, mj)
    _rebuild_csv_export()
    return {"ok": True}


@app.get("/api/describe")
def describe(artikelnr: str):
    """
    Manuelle KI-Neuberechnung (Button in index.html).
    GENAU 1 Versuch, kein Fallback, kein Retry.
    """
    artikelnr = artikelnr.strip()
    pics = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    if not pics:
        return JSONResponse({"ok": False, "error": "Kein Bild für diese Artikelnummer gefunden."}, status_code=400)

    img_path = pics[-1]
    meta, ki_ok, ki_error, runtime_ms = _run_meta_once(artikelnr, img_path)

    mj = _load_meta_json(artikelnr)
    mj["ki_runtime_ms"] = runtime_ms
    mj["last_image"] = img_path.name

    if ki_ok:
        mj["titel"] = (meta.get("title") or "").strip()
        mj["beschreibung"] = (meta.get("description") or "").strip()

        retail = meta.get("retail_price", 0.0)
        try:
            retail_f = float(retail)
        except (TypeError, ValueError):
            retail_f = 0.0

        mj["retail_price"] = round(retail_f, 2)
        mj["rufpreis"] = round(retail_f * 0.20, 2)

        mj["ki_source"] = "realtime"
        mj["batch_done"] = True
        mj["reviewed"] = False
        mj["ki_last_error"] = ""

        _save_meta_json(artikelnr, mj)
        _rebuild_csv_export()

        return {
            "ok": True,
            "title": mj["titel"],
            "description": mj["beschreibung"],
            "retail_price": mj["retail_price"],
            "rufpreis": mj["rufpreis"],
            "ki_runtime_ms": runtime_ms,
        }

    # KI fehlgeschlagen: nichts überschreiben, nur Fehler setzen
    mj["ki_last_error"] = ki_error or "ki_failed"
    mj["ki_source"] = "failed"
    mj["batch_done"] = True
    _save_meta_json(artikelnr, mj)
    _rebuild_csv_export()

    return JSONResponse({"ok": False, "error": mj["ki_last_error"]}, status_code=502)


# ---- Admin API ----

@app.get("/api/admin/articles")
def admin_articles():
    items = []

    for p in RAW_DIR.glob("*.json"):
        try:
            mj = json.loads(p.read_text("utf-8"))
        except Exception:
            continue

        nr = mj.get("artikelnr", p.stem)

        pics = sorted(RAW_DIR.glob(f"{nr}_*.jpg"))
        img_url = ""
        if pics:
            rel = pics[-1].relative_to(BASE_DIR)
            img_url = "/static/" + str(rel).replace("\\", "/")

        items.append({
            "artikelnr": nr,
            "titel": mj.get("titel", "") or "",
            "beschreibung": mj.get("beschreibung", "") or "",
            "retail_price": mj.get("retail_price", 0.0) or 0.0,
            "rufpreis": mj.get("rufpreis", 0.0) or 0.0,
            "reviewed": bool(mj.get("reviewed", False)),
            "ki_source": mj.get("ki_source", "") or "",
            "image": img_url
        })

    try:
        items.sort(key=lambda x: int(x["artikelnr"]))
    except Exception:
        items.sort(key=lambda x: str(x["artikelnr"]))

    return {"ok": True, "items": items}


@app.post("/api/admin/articles/update")
def admin_update(data: dict):
    artikelnr = str(data.get("artikelnr", "")).strip()
    if not artikelnr:
        return JSONResponse({"ok": False, "error": "Artikelnummer fehlt"}, status_code=400)

    mj = _load_meta_json(artikelnr)

    if "titel" in data and data["titel"] is not None:
        mj["titel"] = str(data["titel"])
    if "beschreibung" in data and data["beschreibung"] is not None:
        mj["beschreibung"] = str(data["beschreibung"])
    if "reviewed" in data:
        mj["reviewed"] = bool(data["reviewed"])

    _save_meta_json(artikelnr, mj)
    _rebuild_csv_export()
    return {"ok": True}
