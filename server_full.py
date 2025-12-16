
# server_full.py – MyAuktion KI-Artikelaufnahme-System (FINAL – FIXED)

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
from PIL import Image, ImageOps
import io, json, time, csv, math, os, re
from typing import Any, Dict, Optional, Tuple

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# Middleware
# --------------------------------------------------
@app.middleware("http")
async def _no_cache_html(request, call_next):
    resp = await call_next(request)
    try:
        ct = resp.headers.get("content-type", "")
        if ct.startswith("text/html"):
            resp.headers["Cache-Control"] = "no-store"
    except Exception:
        pass
    return resp

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "uploads" / "raw"
EXPORT_DIR = BASE_DIR / "export"

RAW_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_CSV = EXPORT_DIR / "eartikel_export.csv"
DATA_JSON = EXPORT_DIR / "artikel_data.json"

app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _default_meta(artikelnr: str) -> Dict[str, Any]:
    return {
        "artikelnr": artikelnr,
        "titel": "",
        "beschreibung": "",
        "kategorie": "",
        "retail_price": 0.0,
        "rufpreis": 0.0,
        "lagerort": "",
        "menge": 1,
        "lagerstand": 1,
        "uebernehmen": 1,
        "sortiment": "",
        "einlieferer_id": "",
        "angeliefert": "",
        "betriebsmittel": "",
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }

def _meta_path(artikelnr: str) -> Path:
    return RAW_DIR / f"{artikelnr}.json"

def _load_meta_json(artikelnr: str) -> Dict[str, Any]:
    if _meta_path(artikelnr).exists():
        try:
            return json.loads(_meta_path(artikelnr).read_text("utf-8"))
        except Exception:
            pass
    return _default_meta(artikelnr)

def _save_meta_json(artikelnr: str, data: Dict[str, Any]):
    data["updated_at"] = int(time.time())
    _meta_path(artikelnr).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

# --------------------------------------------------
# Persistenz (Variante A – FIXED)
# --------------------------------------------------
def _rebuild_data_json():
    rows = []
    for p in RAW_DIR.glob("*.json"):
        try:
            rows.append(json.loads(p.read_text("utf-8")))
        except Exception:
            continue
    try:
        DATA_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        print("[DATA] artikel_data.json aktualisiert")
    except Exception as e:
        print("[DATA] Fehler:", e)

def _restore_from_data_json():
    if not DATA_JSON.exists():
        return
    if any(RAW_DIR.glob("*.json")):
        return
    try:
        rows = json.loads(DATA_JSON.read_text("utf-8"))
        for d in rows:
            if "artikelnr" in d:
                _save_meta_json(d["artikelnr"], d)
        print("[DATA] Wiederhergestellt aus artikel_data.json")
    except Exception:
        pass

# --------------------------------------------------
# CSV Export
# --------------------------------------------------
CSV_FIELDS = [
    "ArtikelNr","Menge","Preis","Ladenpreis","Lagerort","Lagerstand",
    "Uebernehmen","Sortiment","Kategorie","EinliefererID","Angeliefert","Betriebsmittel"
]

def _rebuild_csv_export():
    rows = []
    for p in RAW_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        rows.append({
            "ArtikelNr": d.get("artikelnr",""),
            "Menge": d.get("menge",1),
            "Preis": d.get("rufpreis",0),
            "Ladenpreis": d.get("retail_price",0),
            "Lagerort": d.get("lagerort",""),
            "Lagerstand": d.get("lagerstand",1),
            "Uebernehmen": d.get("uebernehmen",1),
            "Sortiment": d.get("sortiment",""),
            "Kategorie": d.get("kategorie",""),
            "EinliefererID": d.get("einlieferer_id",""),
            "Angeliefert": d.get("angeliefert",""),
            "Betriebsmittel": d.get("betriebsmittel",""),
        })

    with EXPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)

# --------------------------------------------------
# Startup
# --------------------------------------------------
@app.on_event("startup")
def startup():
    _restore_from_data_json()
    _rebuild_csv_export()
    _rebuild_data_json()

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/api/export.csv")
def export_csv():
    _rebuild_csv_export()
    _rebuild_data_json()
    return FileResponse(str(EXPORT_CSV), filename="eartikel_export.csv")

@app.post("/api/save")
def save(data: Dict[str, Any]):
    nr = str(data.get("artikelnr","")).strip()
    if not nr:
        return {"ok": False}

    mj = _load_meta_json(nr)

    for k in mj.keys():
        if k in data:
            mj[k] = data[k]

    # Reset defaults after save
    mj["lagerstand"] = 1
    mj["uebernehmen"] = 1

    _save_meta_json(nr, mj)
    _rebuild_csv_export()
    _rebuild_data_json()
    return {"ok": True}

@app.get("/api/health")
def health():
    return {"ok": True}

# --------------------------------------------------
# Render Start
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
