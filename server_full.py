from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
from PIL import Image, ImageOps
import io, json

app = FastAPI()

# CORS für Browser-Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basis-Pfad = Ordner, in dem server_full.py liegt
BASE_DIR = Path(__file__).resolve().parent

# Upload-Ordner
RAW_DIR = BASE_DIR / "uploads" / "raw"
PROCESSED_DIR = BASE_DIR / "uploads" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Static-Files (für Bilder etc.)
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


def _default_meta(artikelnr: str) -> dict:
    """Fallback-Metadaten, falls (noch) keine KI-Texte vorhanden sind."""
    return {
        "title": f"Artikel {artikelnr}",
        "description": "Artikel wie abgebildet, genaue Details bitte den Fotos entnehmen.",
        "category": "Allgemein",
        "tags": "",
        "starting_price": 0,
    }


def _load_meta_json(artikelnr: str) -> dict:
    """Lädt die Metadaten-JSON zu einer Artikelnummer oder liefert Defaults."""
    meta_path = RAW_DIR / f"{artikelnr}.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("artikelnr", artikelnr)
    data.setdefault("titel", f"Artikel {artikelnr}")
    data.setdefault("beschreibung", "")
    data.setdefault("reviewed", False)
    data.setdefault("ki_source", "")
    data.setdefault("batch_done", False)
    data.setdefault("last_image", "")

    return data


def _run_meta(artikelnr: str, img_path: Path) -> dict:
    """
    Manuelle KI-Berechnung:
    - Wird vom /api/describe-Endpunkt verwendet (Button 'KI neu berechnen')
    - Nutzt ki_engine.generate_meta (Vision + Text über Ollama)
    """
    try:
        import ki_engine
    except ImportError:
        print("[WARNUNG] ki_engine.py nicht gefunden – nutze Fallback-Metadaten.")
        return _default_meta(artikelnr)

    try:
        existing = _load_meta_json(artikelnr)
        meta = ki_engine.generate_meta_multi([str(p) for p in img_paths], str(artikelnr), existing=existing)
    except Exception as e:
        print(f"[KI-FEHLER] generate_meta für {artikelnr} fehlgeschlagen: {e}")
        return _default_meta(artikelnr)

    if not isinstance(meta, dict):
        return _default_meta(artikelnr)

    title = meta.get("title") or f"Artikel {artikelnr}"
    description = meta.get("description") or _default_meta(artikelnr)["description"]
    meta["title"] = title
    meta["description"] = description
    return meta


@app.get("/")
def root():
    """Liefert index.html (Lager-Oberfläche)."""
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            {"ok": False, "error": "index.html nicht gefunden"}, status_code=500
        )
    return FileResponse(str(index_path))


@app.get("/admin")
def admin_root():
    """Liefert admin.html (Büro-Kontrolloberfläche)."""
    admin_path = BASE_DIR / "admin.html"
    if not admin_path.exists():
        return JSONResponse(
            {"ok": False, "error": "admin.html nicht gefunden"}, status_code=500
        )
    return FileResponse(str(admin_path))


@app.get("/api/health")
def health():
    return {"ok": True, "status": "running"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), artikelnr: str = Form(...)):
    """
    Nimmt ein Bild entgegen, dreht es korrekt, speichert als ARTNR_N.jpg
    (0,1,2,...) und liefert nur Standard-Metadaten zurück.
    KI läuft NICHT mehr hier, sondern im Batch-Worker.
    """
    artikelnr = str(artikelnr).strip()
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        # EXIF-Drehung korrigieren (Handy-Fotos)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Ungültiges Bildformat: {e}"}, 400)

    # nächsten Index für diese Artikelnummer finden (_0, _1, _2, ...)
    existing = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    if existing:
        last = existing[-1]
        try:
            last_idx = int(last.stem.split("_")[-1])
        except ValueError:
            last_idx = -1
        next_idx = last_idx + 1
    else:
        next_idx = 0

    out = RAW_DIR / f"{artikelnr}_{next_idx}.jpg"
    img.save(out, "JPEG", quality=80, optimize=True)

    # >>> NEU: KI direkt beim Upload ausführen (eine Modellanfrage)
    meta = _run_meta(artikelnr, out)

    return {
        "ok": True,
        "path": str(out),
        "meta": {
            "title": meta.get("title", f"Artikel {artikelnr}"),
            "description": meta.get("description", ""),
        },
    }


@app.get("/api/describe")
def describe(artikelnr: str):
    """
    Manuelle Neu-Berechnung der KI anhand des letzten Bildes für diese Artikelnummer.
    Wird nur vom 'KI neu berechnen'-Button verwendet.
    """
    artikelnr = str(artikelnr).strip()
    candidates = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    if not candidates:
        return {"ok": False, "error": "Bild nicht gefunden"}

    # immer das neueste Bild nehmen
    img_path = candidates[-1]
    meta = _run_meta(artikelnr, img_path)

    return {
        "ok": True,
        "title": meta.get("title", f"Artikel {artikelnr}"),
        "description": meta.get("description", ""),
    }


@app.get("/api/check_artnr")
def check_artnr(artikelnr: str):
    """Prüft, ob es bereits Dateien mit dieser Artikelnummer im raw-Ordner gibt."""
    artikelnr = str(artikelnr).strip()
    exists = any(f.name.startswith(artikelnr + "_") for f in RAW_DIR.glob("*"))
    return {"ok": True, "exists": exists}


@app.get("/api/images")
def images(artikelnr: str):
    """Gibt eine Liste von Bildpfaden zu dieser Artikelnummer zurück (RAW_DIR)."""
    artikelnr = str(artikelnr).strip()
    files = []
    for f in RAW_DIR.glob(f"{artikelnr}_*.jpg"):
        rel = f.relative_to(BASE_DIR)
        files.append("/static/" + str(rel).replace("\\", "/"))
    return {"ok": True, "files": files}


@app.get("/api/meta")
def get_meta(artikelnr: str):
    """Gibt Titel/Beschreibung/Status (+ Bild) zu einer Artikelnummer zurück."""
    artikelnr = str(artikelnr).strip()
    if not artikelnr:
        return {"ok": False, "error": "Artikelnummer fehlt"}

    data = _load_meta_json(artikelnr)

    # neuestes Bild bestimmen
    candidates = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    img_url = ""
    if candidates:
        rel = candidates[-1].relative_to(BASE_DIR)
        img_url = "/static/" + str(rel).replace("\\", "/")

    return {
        "ok": True,
        "artikelnr": artikelnr,
        "titel": data.get("titel") or f"Artikel {artikelnr}",
        "beschreibung": data.get("beschreibung") or "",
        "reviewed": bool(data.get("reviewed", False)),
        "ki_source": data.get("ki_source") or "",
        "image": img_url,
    }


@app.get("/api/next_artikelnr")
def next_artikelnr():
    """
    Ermittelt die nächste freie Artikelnummer anhand vorhandener JSONs/Bilder.
    Falls nichts vorhanden ist, beginnt bei 1.
    """
    max_nr = 0

    # JSON-Dateien
    for meta_path in RAW_DIR.glob("*.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            art_str = str(data.get("artikelnr") or meta_path.stem)
        except Exception:
            art_str = meta_path.stem
        if art_str.isdigit():
            n = int(art_str)
            max_nr = max(max_nr, n)

    # Fallback: falls nur Bilder existieren
    for img_path in RAW_DIR.glob("*_*.jpg"):
        art_str = img_path.stem.split("_")[0]
        if art_str.isdigit():
            n = int(art_str)
            max_nr = max(max_nr, n)

    next_nr = max_nr + 1 if max_nr > 0 else 1
    return {"ok": True, "next": next_nr}


@app.post("/api/save")
def save(data: dict):
    """Speichert Metadaten pro Artikelnummer als JSON (merge mit bestehender Datei)."""
    artikelnr = str(data.get("artikelnr") or "").strip()
    if not artikelnr:
        return {"ok": False, "error": "Artikelnummer fehlt"}

    meta_path = RAW_DIR / f"{artikelnr}.json"

    # bestehende Metadaten laden (z.B. von KI-Batch)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"artikelnr": artikelnr}
    else:
        meta = {"artikelnr": artikelnr}

    # neue Daten darüberlegen, KI-Felder bleiben erhalten, falls nicht überschrieben
    meta.update(data)

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "saved": str(meta_path)}


@app.get("/api/admin/articles")
def admin_articles(status: str = "unreviewed"):
    """
    Liefert eine Liste von Artikeln für die Büro-Kontrolle.
    status=unreviewed -> nur noch nicht geprüfte
    status=all        -> alle
    """
    items = []
    for meta_path in RAW_DIR.glob("*.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        artikelnr = str(data.get("artikelnr") or meta_path.stem)
        titel = data.get("titel") or f"Artikel {artikelnr}"
        beschreibung = data.get("beschreibung") or ""
        reviewed = bool(data.get("reviewed", False))
        ki_source = data.get("ki_source", "")

        if status == "unreviewed" and reviewed:
            continue

        # neuestes Bild bestimmen
        candidates = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
        img_url = ""
        if candidates:
            rel = candidates[-1].relative_to(BASE_DIR)
            img_url = "/static/" + str(rel).replace("\\", "/")

        items.append({
            "artikelnr": artikelnr,
            "titel": titel,
            "beschreibung": beschreibung,
            "image": img_url,
            "reviewed": reviewed,
            "ki_source": ki_source,
        })

    items.sort(key=lambda x: x["artikelnr"])

    return {"ok": True, "items": items}


@app.post("/api/admin/articles/update")
def admin_articles_update(data: dict):
    """
    Aktualisiert Titel/Beschreibung/Reviewed-Status für eine Artikelnummer.
    Wird vom Büro-UI aufgerufen.
    """
    artikelnr = str(data.get("artikelnr") or "").strip()
    if not artikelnr:
        return {"ok": False, "error": "Artikelnummer fehlt"}

    meta_path = RAW_DIR / f"{artikelnr}.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"artikelnr": artikelnr}
    else:
        meta = {"artikelnr": artikelnr}

    meta["titel"] = data.get("titel") or f"Artikel {artikelnr}"
    meta["beschreibung"] = data.get("beschreibung") or ""
    meta["reviewed"] = bool(data.get("reviewed", True))

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True}
