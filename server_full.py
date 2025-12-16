# server_full.py – MyAuktion KI-Artikelaufnahme-System (FINAL – FIXED + ADMIN)
#
# Adds:
# - /admin route to serve admin.html
# - Admin APIs expected by admin.html: /api/admin/budget and /api/admin/articles
# - Optional admin token guard via env ADMIN_TOKEN (empty => open)
# - CSV export supports optional range params from admin page: from_nr / to_nr
#
# Keeps:
# - Persistenz (Variante A): export/artikel_data.json
# - CSV export with requested column order
# - After /api/save: lagerstand and uebernehmen are reset to 1

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import io, json, time, csv, os
from typing import Any, Dict, Optional

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
        if ct.startswith("text/html") or request.url.path in ("/", "/admin"):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
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
USAGE_JSON = EXPORT_DIR / "ki_usage.json"

app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# --------------------------------------------------
# Admin Token / Budget settings
# --------------------------------------------------
ADMIN_TOKEN = (os.getenv("ADMIN_TOKEN", "") or "").strip()  # leer => offen (nur Tests)
DEFAULT_BUDGET_EUR = float((os.getenv("KI_BUDGET_EUR", "10") or "10").replace(",", "."))
COST_PER_SUCCESS_CALL_EUR = float((os.getenv("KI_COST_PER_SUCCESS_CALL_EUR", "0.003") or "0.003").replace(",", "."))

def _is_admin(request: Request) -> bool:
    if not ADMIN_TOKEN:
        return True
    token = (request.headers.get("X-Admin-Token", "") or "").strip()
    if not token:
        token = (request.query_params.get("token", "") or "").strip()
    return token == ADMIN_TOKEN

def _admin_guard(request: Request) -> Optional[JSONResponse]:
    if _is_admin(request):
        return None
    return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

# --------------------------------------------------
# Helpers (Meta JSON)
# --------------------------------------------------
def _default_meta(artikelnr: str) -> Dict[str, Any]:
    return {
        "artikelnr": str(artikelnr),
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
        # optional fields for admin overview
        "reviewed": False,
        "ki_source": "",
        "ki_last_error": "",
        "last_image": "",
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }

def _meta_path(artikelnr: str) -> Path:
    return RAW_DIR / f"{artikelnr}.json"

def _load_meta_json(artikelnr: str) -> Dict[str, Any]:
    p = _meta_path(artikelnr)
    data: Dict[str, Any] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text("utf-8"))
        except Exception:
            data = {}
    merged = _default_meta(artikelnr)
    if isinstance(data, dict):
        merged.update(data)
    return merged

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
            d = json.loads(p.read_text("utf-8"))
            if isinstance(d, dict):
                rows.append(d)
        except Exception:
            continue
    # sort by artikelnr if numeric
    def _k(d):
        v = str(d.get("artikelnr", ""))
        try:
            return int(v)
        except Exception:
            return v
    rows.sort(key=_k)
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
        if not isinstance(rows, list):
            return
        for d in rows:
            if isinstance(d, dict) and d.get("artikelnr"):
                _save_meta_json(str(d["artikelnr"]), d)
        print("[DATA] Wiederhergestellt aus artikel_data.json")
    except Exception:
        pass

# --------------------------------------------------
# KI Usage (Budget Anzeige – nur Schätzung)
# --------------------------------------------------
def _load_usage() -> Dict[str, Any]:
    if USAGE_JSON.exists():
        try:
            u = json.loads(USAGE_JSON.read_text("utf-8"))
        except Exception:
            u = {}
    else:
        u = {}
    u.setdefault("budget_eur", DEFAULT_BUDGET_EUR)
    u.setdefault("cost_per_success_call_eur", COST_PER_SUCCESS_CALL_EUR)
    u.setdefault("success_calls", 0)
    u.setdefault("failed_calls", 0)
    u.setdefault("last_error", "")
    u.setdefault("last_success_at", 0)
    u.setdefault("spent_est_eur", round(float(u.get("success_calls", 0)) * float(u.get("cost_per_success_call_eur", COST_PER_SUCCESS_CALL_EUR)), 4))
    return u

def _save_usage(u: Dict[str, Any]) -> None:
    USAGE_JSON.write_text(json.dumps(u, ensure_ascii=False, indent=2), "utf-8")

# --------------------------------------------------
# CSV Export
# --------------------------------------------------
CSV_FIELDS = [
    "ArtikelNr",
    "Menge",
    "Preis",
    "Ladenpreis",
    "Lagerort",
    "Lagerstand",
    "Uebernehmen",
    "Sortiment",
    "Kategorie",
    "EinliefererID",
    "Angeliefert",
    "Betriebsmittel",
]

def _to_int_safe(v: Any) -> Optional[int]:
    try:
        s = str(v).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None

def _rebuild_csv_export(from_nr: Optional[int] = None, to_nr: Optional[int] = None) -> None:
    rows = []
    for p in RAW_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue

        nr = str(d.get("artikelnr", "")).strip()
        nr_i = _to_int_safe(nr)

        if from_nr is not None and nr_i is not None and nr_i < from_nr:
            continue
        if to_nr is not None and nr_i is not None and nr_i > to_nr:
            continue

        rows.append({
            "ArtikelNr": nr,
            "Menge": d.get("menge", 1) or 1,
            # Preis = Rufpreis (Startpreis), Ladenpreis = Listenpreis
            "Preis": d.get("rufpreis", 0.0) or 0.0,
            "Ladenpreis": d.get("retail_price", 0.0) or 0.0,
            "Lagerort": d.get("lagerort", "") or "",
            "Lagerstand": d.get("lagerstand", 1) or 1,
            "Uebernehmen": d.get("uebernehmen", 1) or 1,
            "Sortiment": d.get("sortiment", "") or "",
            "Kategorie": d.get("kategorie", "") or "",
            "EinliefererID": d.get("einlieferer_id", "") or "",
            "Angeliefert": d.get("angeliefert", "") or "",
            "Betriebsmittel": d.get("betriebsmittel", "") or "",
        })

    def _sort_key(r):
        try:
            return int(r["ArtikelNr"])
        except Exception:
            return r["ArtikelNr"]

    rows.sort(key=_sort_key)

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
# Routes: Pages
# --------------------------------------------------
@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/admin")
def admin_root():
    return FileResponse(str(BASE_DIR / "admin.html"))

@app.head("/")
def head_root():
    return Response(status_code=200)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.head("/api/health")
def health_head():
    return {"ok": True}

# --------------------------------------------------
# Routes: Export + Save
# --------------------------------------------------
@app.get("/api/export.csv")
def export_csv(request: Request, from_nr: str = "", to_nr: str = ""):
    # from_nr/to_nr are optional (admin page sends these). Empty => all.
    a = _to_int_safe(from_nr)
    b = _to_int_safe(to_nr)
    _rebuild_csv_export(from_nr=a, to_nr=b)
    _rebuild_data_json()
    return FileResponse(str(EXPORT_CSV), filename="eartikel_export.csv", media_type="text/csv")

@app.post("/api/save")
def save(data: Dict[str, Any]):
    nr = str(data.get("artikelnr","")).strip()
    if not nr:
        return JSONResponse({"ok": False, "error": "Artikelnummer fehlt"}, status_code=400)

    mj = _load_meta_json(nr)

    # Only accept known keys (avoid writing random client keys)
    for k in mj.keys():
        if k in data and data[k] is not None:
            mj[k] = data[k]

    # Reset defaults after save (as requested)
    mj["lagerstand"] = 1
    mj["uebernehmen"] = 1

    _save_meta_json(nr, mj)
    _rebuild_csv_export()
    _rebuild_data_json()
    return {"ok": True}

# --------------------------------------------------
# Admin APIs expected by admin.html
# --------------------------------------------------
@app.get("/api/admin/budget")
def admin_budget(request: Request):
    guard = _admin_guard(request)
    if guard:
        return guard

    u = _load_usage()
    budget = float(u.get("budget_eur", DEFAULT_BUDGET_EUR))
    spent = float(u.get("spent_est_eur", 0.0))
    remaining = round(max(budget - spent, 0.0), 4)

    return {
        "ok": True,
        "budget_eur": budget,
        "spent_est_eur": spent,
        "remaining_est_eur": remaining,
        "success_calls": int(u.get("success_calls", 0)),
        "failed_calls": int(u.get("failed_calls", 0)),
        "cost_per_success_call_eur": float(u.get("cost_per_success_call_eur", COST_PER_SUCCESS_CALL_EUR)),
        "last_success_at": int(u.get("last_success_at", 0)),
        "last_error": u.get("last_error", "") or "",
    }

@app.get("/api/admin/articles")
def admin_articles(request: Request, category: str = "", only_failed: int = 0):
    guard = _admin_guard(request)
    if guard:
        return guard

    items = []
    for p in RAW_DIR.glob("*.json"):
        try:
            mj = json.loads(p.read_text("utf-8"))
            if not isinstance(mj, dict):
                continue
        except Exception:
            continue

        nr = str(mj.get("artikelnr", p.stem))
        cat = (mj.get("kategorie", "") or "").strip()

        if category and cat.lower() != category.strip().lower():
            continue
        if only_failed and (mj.get("ki_source") != "failed"):
            continue

        items.append({
            "artikelnr": nr,
            "titel": mj.get("titel", "") or "",
            "beschreibung": mj.get("beschreibung", "") or "",
            "kategorie": cat,
            "retail_price": mj.get("retail_price", 0.0) or 0.0,
            "rufpreis": mj.get("rufpreis", 0.0) or 0.0,
            "reviewed": bool(mj.get("reviewed", False)),
            "ki_source": mj.get("ki_source", "") or "",
            "ki_last_error": mj.get("ki_last_error", "") or "",
            "last_image": mj.get("last_image", "") or "",
            # image is optional in this slim server; keep empty so admin UI still works
            "image": "",
        })

    def _sort_key(x):
        try:
            return int(x["artikelnr"])
        except Exception:
            return x["artikelnr"]

    items.sort(key=_sort_key)
    return {"ok": True, "items": items}

# --------------------------------------------------
# Render Start
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
