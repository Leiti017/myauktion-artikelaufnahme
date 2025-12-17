# server_full.py – MyAuktion KI-Artikelaufnahme-System (FINAL)
# Features:
# - Upload + sofortige Vorschau (Frontend)
# - Mehrere Fotos pro Artikel
# - Hintergrund-KI (OpenAI Vision) -> Titel, Beschreibung, Kategorie, Listenpreis
# - Rufpreis = 20% vom Listenpreis, IMMER auf ganze € aufrunden
# - Bei Upload wird KI automatisch neu gestartet (auch bei Foto 2/3/…)
# - Polling kann auf "genau dieses Foto" warten (filename)
# - Kein KI-Fallback-Text: bei Fehler -> ki_source="failed"
# - Live-Check Artikelnummer: /api/check_artnr
# - Foto löschen: /api/delete_image
# - CSV Export (inkl. Kategorie)
# - Admin-Only Budget/Flags (Token geschützt): /api/admin/budget /api/admin/articles
#
# Start:
#   python -m uvicorn server_full:app --host 0.0.0.0 --port 5050
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles



def _normalize_title(title: str) -> str:
    t = (title or "").strip()
    bad = {"Typenbezeichnung","typenbezeichnung","Typ","typ","Type","type","Model","model","Modell","modell"}
    parts = [p for p in re.split(r"\s+", t) if p]
    while parts and parts[-1] in bad:
        parts.pop()
    t = " ".join(parts).strip()
    t = re.sub(r"\bTypenbezeichnung\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

from pathlib import Path
from PIL import Image, ImageOps
import io, json, time, csv, math, os
import re
import os
from typing import Any, Dict, Optional, Tuple

app = FastAPI()

@app.middleware("http")
async def _no_cache_html(request, call_next):
    resp = await call_next(request)
    try:
        ct = resp.headers.get("content-type", "")
        if ct.startswith("text/html") or request.url.path in ("/", "/admin", "/static/site.webmanifest", "/static/manifest.webmanifest"):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
    except Exception:
        pass
    return resp


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "uploads" / "raw"
PROCESSED_DIR = BASE_DIR / "uploads" / "processed"
EXPORT_DIR = BASE_DIR / "export"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_CSV = EXPORT_DIR / "artikel_export.csv"


# ----------------------------
# CSV Export helpers (UTF-8 BOM + schneller Update)
# ----------------------------
CSV_FIELDS = [
    "ArtikelNr",
    "Menge",
    "Titel",
    "Beschreibung",
    "Preis",          # Rufpreis (ohne . oder ,)
    "Lagerort",
    "Lagerstand",
    "Uebernehmen",
    "EinliefererID",
    "Angeliefert",
    "Betriebsmittel",
    "Mitarbeiter",
]

CSV_HEADERS = CSV_FIELDS

def _format_rufpreis(val: Any) -> str:
    """Rufpreis immer ohne . oder , (z.B. 40 statt 40.0)."""
    try:
        f = float(str(val).replace(",", "."))
    except Exception:
        f = 0.0
    # sauber runden und als int ausgeben
    return str(int(round(f)))


def _ensure_export_csv_exists() -> None:
    if not EXPORT_CSV.exists():
        _rebuild_csv_export()
        # rewrite once with BOM + correct header order if needed
        try:
            txt = EXPORT_CSV.read_text("utf-8")
            EXPORT_CSV.write_bytes(("\ufeff" + txt).encode("utf-8"))
        except Exception:
            pass

def _update_csv_row_for_art(artikelnr: str, meta: Dict[str, Any]) -> None:
    """Update one row in export CSV without rebuilding everything."""
    _ensure_export_csv_exists()

    # Load existing rows
    import csv, io
    raw = EXPORT_CSV.read_bytes()
    # strip BOM if present
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8", errors="replace")
    rdr = csv.reader(io.StringIO(text), delimiter=";")
    rows = list(rdr)

    # Ensure headers
    if not rows:
        rows = [CSV_HEADERS]
    else:
        rows[0] = CSV_HEADERS

    # Build new row
    art = str(artikelnr)
    menge = int(meta.get("menge") or 1)
    titel = str(meta.get("titel") or "")
    beschr = str(meta.get("beschreibung") or "")
    preis = _format_rufpreis(meta.get("rufpreis", 0))
    lagerort = str(meta.get("lagerort") or "")
    lagerstand = int(meta.get("lagerstand") or 1)
    uebernehmen = int(meta.get("uebernehmen") or 1)
    einl = str(meta.get("einlieferer_id") or meta.get("einlieferer") or "")
    angel = str(meta.get("angeliefert") or "")
    betr = str(meta.get("betriebsmittel") or "")
    mitarbeiter = str(meta.get("mitarbeiter") or meta.get("mitarbeiter_name") or "")

    new_row = [art, str(menge), titel, beschr, preis, lagerort, str(lagerstand), str(uebernehmen), einl, angel, betr, mitarbeiter]

    # Replace or append
    out_rows = [rows[0]]
    replaced = False
    for r in rows[1:]:
        if len(r) > 0 and r[0] == art:
            out_rows.append(new_row)
            replaced = True
        else:
            out_rows.append(r)
    if not replaced:
        out_rows.append(new_row)

    # Write back with BOM
    bio = io.StringIO()
    w = csv.writer(bio, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    for r in out_rows:
        w.writerow(r)
    EXPORT_CSV.write_bytes(("\ufeff" + bio.getvalue()).encode("utf-8"))
USAGE_JSON = EXPORT_DIR / "ki_usage.json"

# Admin-Schutz (frei wählbar, NICHT OpenAI-Key)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()  # leer => Admin offen (nur für Tests)
DEFAULT_BUDGET_EUR = float((os.getenv("KI_BUDGET_EUR", "10") or "10").replace(",", "."))
COST_PER_SUCCESS_CALL_EUR = float((os.getenv("KI_COST_PER_SUCCESS_CALL_EUR", "0.003") or "0.003").replace(",", "."))

app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


# ----------------------------
# Admin Auth
# ----------------------------
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


# ----------------------------
# Meta JSON
# ----------------------------
def _default_meta(artikelnr: str) -> Dict[str, Any]:
    return {
        "artikelnr": str(artikelnr),
        "titel": "",
        "beschreibung": "",
        "kategorie": "",
        "retail_price": 0.0,
        "rufpreis": 0.0,
        "lagerort": "",
        "einlieferer": "",
        "mitarbeiter": "",
        "lagerstand": 1,
        "reviewed": False,
        "ki_source": "",          # pending | realtime | failed | ""
        "ki_last_error": "",
        "ki_runtime_ms": 0,
        "batch_done": False,
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
    merged.update(data)
    return merged


def _save_meta_json(artikelnr: str, data: Dict[str, Any]) -> None:
    data["updated_at"] = int(time.time())
    _meta_path(artikelnr).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ----------------------------
# KI Usage (Budget Anzeige – Schätzung)
# ----------------------------
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
    u.setdefault("spent_est_eur", 0.0)
    return u


def _save_usage(u: Dict[str, Any]) -> None:
    USAGE_JSON.write_text(json.dumps(u, ensure_ascii=False, indent=2), "utf-8")


def _usage_success() -> None:
    u = _load_usage()
    u["success_calls"] = int(u.get("success_calls", 0)) + 1
    u["spent_est_eur"] = round(float(u.get("success_calls", 0)) * float(u.get("cost_per_success_call_eur", COST_PER_SUCCESS_CALL_EUR)), 4)
    u["last_success_at"] = int(time.time())
    u["last_error"] = ""
    _save_usage(u)


def _usage_fail(err: str) -> None:
    u = _load_usage()
    u["failed_calls"] = int(u.get("failed_calls", 0)) + 1
    u["last_error"] = (err or "")[:250]
    _save_usage(u)


# ----------------------------
# CSV Export
# ----------------------------
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
    "Mitarbeiter",
]


def _rebuild_csv_export() -> None:
    rows = []
    for meta_file in RAW_DIR.glob("*.json"):
        try:
            d = json.loads(meta_file.read_text("utf-8"))
        except Exception:
            continue

        nr = str(d.get("artikelnr") or meta_file.stem)
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
            "EinliefererID": d.get("einlieferer_id", d.get("einlieferer", "")) or "",
            "Angeliefert": d.get("angeliefert", "") or "",
            "Betriebsmittel": d.get("betriebsmittel", "") or "",
            "Mitarbeiter": d.get("mitarbeiter", d.get("mitarbeiter_name","")) or "",
        })

    def _sort_key(r):
        try:
            return int(r["ArtikelNr"])
        except Exception:
            return r["ArtikelNr"]

    rows.sort(key=_sort_key)

    with EXPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("[CSV] Export aktualisiert:", EXPORT_CSV)


# ----------------------------
# KI (1 Versuch, kein Fallback, mit Kontext fürs 2. Foto)
# ----------------------------
def _run_meta_once(artikelnr: str, img_path: Path) -> Tuple[Optional[Dict[str, Any]], bool, str, int]:
    """Return (meta, ok, error, runtime_ms)"""
    try:
        import ki_engine_openai as ki_engine
    except Exception as e:
        err = f"import_error: {e}"
        print("[KI] Importfehler:", e)
        return None, False, err, 0

    current = _load_meta_json(artikelnr)

    start = time.time()
    try:
        meta = ki_engine.generate_meta(str(img_path), str(artikelnr), context=current)
        runtime_ms = int((time.time() - start) * 1000)

        if not meta:
            return None, False, "ki_failed", runtime_ms

        title = (meta.get("title") or "").strip()
        desc = (meta.get("description") or "").strip()
        cat = (meta.get("category") or "").strip()

        try:
            retail = float(meta.get("retail_price", 0) or 0)
        except Exception:
            retail = 0.0

        # OK nur wenn Titel + (Beschreibung ODER Preis)
        if not title or (not desc and retail <= 0):
            return None, False, "invalid_ki_result", runtime_ms

        meta["title"] = title
        meta["description"] = desc
        meta["category"] = cat
        meta["retail_price"] = retail
        return meta, True, "", runtime_ms

    except Exception as e:
        runtime_ms = int((time.time() - start) * 1000)
        err = str(e)
        print("[KI] Fehler:", e)
        return None, False, err, runtime_ms


def _apply_ki_to_meta(mj: Dict[str, Any], meta: Dict[str, Any]) -> None:
    mj["titel"] = _normalize_title(meta.get("title", "")).strip()
    mj["beschreibung"] = meta.get("description", "").strip()
    mj["kategorie"] = meta.get("category", "").strip()

    retail_f = float(meta.get("retail_price", 0.0) or 0.0)
    mj["retail_price"] = round(retail_f, 2)

    # Rufpreis: 20% auf ganze € aufrunden
    mj["rufpreis"] = float(math.ceil(retail_f * 0.20)) if retail_f > 0 else 0.0


def _run_meta_background(artikelnr: str, img_path: Path) -> None:
    print(f"[BG-KI] Starte Hintergrund-KI für Artikel {artikelnr}")
    meta, ok, err, runtime_ms = _run_meta_once(artikelnr, img_path)

    mj = _load_meta_json(artikelnr)
    mj["ki_runtime_ms"] = runtime_ms
    mj["last_image"] = img_path.name
    mj["batch_done"] = True

    if ok and meta:
        _apply_ki_to_meta(mj, meta)
        mj["ki_source"] = "realtime"
        mj["ki_last_error"] = ""
        mj["reviewed"] = False
        _save_meta_json(artikelnr, mj)
        _usage_success()
    else:
        mj["ki_source"] = "failed"
        mj["ki_last_error"] = err or "ki_failed"
        _save_meta_json(artikelnr, mj)
        _usage_fail(mj["ki_last_error"])

    _rebuild_csv_export()
    print(f"[BG-KI] Fertig für Artikel {artikelnr} – ki_ok={ok}")


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "index.html"))


@app.get("/admin")
def admin_root():
    return FileResponse(str(BASE_DIR / "admin.html"))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/export.csv")
def export_csv(from_nr: str | None = None, to_nr: str | None = None):
    """
    Export CSV (UTF-8 mit BOM für Excel). Optionaler Bereich:
      /api/export.csv?from_nr=123458&to_nr=123480
    """
    _ensure_export_csv_exists()

    # Kein Filter -> Datei direkt ausliefern
    if not from_nr and not to_nr:
        return FileResponse(str(EXPORT_CSV), filename="artikel_export.csv", media_type="text/csv")

    # Filter -> in-memory CSV erzeugen
    def _in_range(n: str) -> bool:
        try:
            ni = int(n)
        except Exception:
            return False
        if from_nr:
            try:
                if ni < int(from_nr): return False
            except Exception:
                pass
        if to_nr:
            try:
                if ni > int(to_nr): return False
            except Exception:
                pass
        return True

    import io, csv
    rows = []
    for jf in sorted(RAW_DIR.glob("*.json")):
        art = jf.stem
        if not _in_range(art):
            continue
        meta = _load_meta_json(art)
        rows.append([
            art,
            str(int(meta.get("menge") or 1)),
            str(meta.get("titel") or ""),
            str(meta.get("beschreibung") or ""),
            _format_rufpreis(meta.get("rufpreis", 0)),
            str(meta.get("lagerort") or ""),
            str(int(meta.get("lagerstand") or 1)),
            str(int(meta.get("uebernehmen") or 1)),
            str(meta.get("einlieferer_id") or meta.get("einlieferer") or ""),
            str(meta.get("angeliefert") or ""),
            str(meta.get("betriebsmittel") or ""),
            str(meta.get("mitarbeiter") or meta.get("mitarbeiter_name") or ""),
        ])

    bio = io.StringIO()
    w = csv.writer(bio, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(CSV_HEADERS)
    for r in rows:
        w.writerow(r)
    content = ("\ufeff" + bio.getvalue()).encode("utf-8")
    return Response(content, media_type="text/csv", headers={"Content-Disposition":"attachment; filename=eartikel_export.csv"})


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
    return {"ok": True, "next": max_nr + 1 if max_nr else 1}


@app.get("/api/check_artnr")
def check_artnr(artikelnr: str):
    artikelnr = str(artikelnr).strip()
    exists = _meta_path(artikelnr).exists() or any(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    return {"ok": True, "exists": bool(exists)}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    artikelnr: str = Form(...),
    background_tasks: BackgroundTasks = None,
):
    artikelnr = str(artikelnr).strip()
    data = await file.read()

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        # SPEED: kleiner + schneller für Base64/Upload
        img.thumbnail((1024, 1024))
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Ungültiges Bild: {e}"}, status_code=400)

    existing = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    idx = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0
    out = RAW_DIR / f"{artikelnr}_{idx}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=78)

    # set pending, damit Polling NICHT bei altem realtime sofort stoppt
    mj = _load_meta_json(artikelnr)
    mj["last_image"] = out.name
    mj["ki_source"] = "pending"
    mj["ki_last_error"] = ""
    mj["batch_done"] = False
    _save_meta_json(artikelnr, mj)

    _rebuild_csv_export()

    if background_tasks:
        background_tasks.add_task(_run_meta_background, artikelnr, out)

    return {"ok": True, "filename": out.name}


@app.get("/api/images")
def images(artikelnr: str):
    artikelnr = str(artikelnr).strip()
    files = []
    for f in sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg")):
        rel = f.relative_to(BASE_DIR)
        files.append("/static/" + str(rel).replace(os.sep, "/"))
    return {"ok": True, "files": files}



@app.post("/api/delete_last_image")
def delete_last_image(data: Dict[str, Any]):
    artikelnr = str(data.get("artikelnr", "")).strip()
    if not artikelnr:
        return JSONResponse({"ok": False, "error": "Artikelnummer fehlt"}, status_code=400)

    files = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    if not files:
        return JSONResponse({"ok": False, "error": "Keine Bilder vorhanden"}, status_code=404)

    target = files[-1]
    try:
        target.unlink()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Konnte Bild nicht löschen: {e}"}, status_code=500)

    # meta last_image aktualisieren
    mj = _load_meta_json(artikelnr)
    mj["last_image"] = files[-2].name if len(files) >= 2 else ""
    _save_meta_json(artikelnr, mj)

    return {"ok": True, "deleted": target.name}

@app.post("/api/delete_image")
def delete_image(payload: Dict[str, Any]):
    artikelnr = str(payload.get("artikelnr", "")).strip()
    filename = str(payload.get("filename", "")).strip()

    if not artikelnr or not filename:
        return JSONResponse({"ok": False, "error": "artikelnr/filename fehlt"}, status_code=400)

    # Sicherheitscheck: nur Dateien dieses Artikels erlauben
    if not filename.startswith(f"{artikelnr}_") or not filename.lower().endswith(".jpg"):
        return JSONResponse({"ok": False, "error": "ungültiger Dateiname"}, status_code=400)

    path = RAW_DIR / filename
    if not path.exists():
        return JSONResponse({"ok": False, "error": "Datei nicht gefunden"}, status_code=404)

    try:
        path.unlink()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"löschen fehlgeschlagen: {e}"}, status_code=500)

    mj = _load_meta_json(artikelnr)
    pics = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    mj["last_image"] = pics[-1].name if pics else ""
    _save_meta_json(artikelnr, mj)

    _rebuild_csv_export()
    return {"ok": True}


@app.get("/api/meta")
def meta(artikelnr: str):
    artikelnr = str(artikelnr).strip()
    mj = _load_meta_json(artikelnr)

    pics = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    img_url = ""
    if pics:
        rel = pics[-1].relative_to(BASE_DIR)
        img_url = "/static/" + str(rel).replace(os.sep, "/")

    return {
        "ok": True,
        "artikelnr": artikelnr,
        "titel": mj.get("titel", "") or "",
        "beschreibung": mj.get("beschreibung", "") or "",
        "kategorie": mj.get("kategorie", "") or "",
        "retail_price": mj.get("retail_price", 0.0) or 0.0,
        "rufpreis": mj.get("rufpreis", 0.0) or 0.0,
        "lagerort": mj.get("lagerort", "") or "",
        "einlieferer": mj.get("einlieferer", "") or "",
        "mitarbeiter": mj.get("mitarbeiter", "") or "",
        "reviewed": bool(mj.get("reviewed", False)),
        "ki_source": mj.get("ki_source", "") or "",
        "ki_last_error": mj.get("ki_last_error", "") or "",
        "last_image": mj.get("last_image", "") or "",
        "image": img_url,
    }


@app.post("/api/save")
def save(data: Dict[str, Any]):
    artikelnr = str(data.get("artikelnr", "")).strip()
    if not artikelnr:
        return JSONResponse({"ok": False, "error": "Artikelnummer fehlt"}, status_code=400)

    mj = _load_meta_json(artikelnr)

    # Textfelder
    for k in ["titel", "beschreibung", "lagerort", "einlieferer", "mitarbeiter", "kategorie"]:
        if k in data and data[k] is not None:
            mj[k] = str(data[k])

    # Settings-Felder (für CSV Export)
    if "menge" in data: mj["menge"] = int(data.get("menge") or 1)
    if "lagerstand" in data: mj["lagerstand"] = int(data.get("lagerstand") or 1)
    if "uebernehmen" in data: mj["uebernehmen"] = int(data.get("uebernehmen") or 1)
    if "einlieferer_id" in data: mj["einlieferer_id"] = str(data.get("einlieferer_id") or "")
    if "angeliefert" in data: mj["angeliefert"] = str(data.get("angeliefert") or "")
    if "betriebsmittel" in data: mj["betriebsmittel"] = str(data.get("betriebsmittel") or "")

    # Preise (falls manuell angepasst)
    if "retail_price" in data:
        try: mj["retail_price"] = float(data.get("retail_price") or 0)
        except Exception: mj["retail_price"] = 0.0
    if "rufpreis" in data:
        try: mj["rufpreis"] = float(data.get("rufpreis") or 0)
        except Exception: mj["rufpreis"] = 0.0

    if "reviewed" in data:
        mj["reviewed"] = bool(data.get("reviewed", False))

    _save_meta_json(artikelnr, mj)

    # Schnell: nur diese eine Zeile in CSV updaten
    try:
        _update_csv_row_for_art(artikelnr, mj)
    except Exception:
        # Fallback: wenn irgendwas schiefgeht, einmal komplett rebuild
        _rebuild_csv_export()

    return {"ok": True}


@app.get("/api/describe")
def describe(artikelnr: str):
    artikelnr = str(artikelnr).strip()
    pics = sorted(RAW_DIR.glob(f"{artikelnr}_*.jpg"))
    if not pics:
        return JSONResponse({"ok": False, "error": "Kein Bild für diese Artikelnummer gefunden."}, status_code=400)

    img_path = pics[-1]

    # set pending for manual too
    mj = _load_meta_json(artikelnr)
    mj["ki_source"] = "pending"
    mj["ki_last_error"] = ""
    mj["batch_done"] = False
    mj["last_image"] = img_path.name
    _save_meta_json(artikelnr, mj)

    meta, ok, err, runtime_ms = _run_meta_once(artikelnr, img_path)

    mj = _load_meta_json(artikelnr)
    mj["ki_runtime_ms"] = runtime_ms
    mj["last_image"] = img_path.name
    mj["batch_done"] = True

    if ok and meta:
        _apply_ki_to_meta(mj, meta)
        mj["ki_source"] = "realtime"
        mj["ki_last_error"] = ""
        mj["reviewed"] = False
        _save_meta_json(artikelnr, mj)
        _usage_success()
        _rebuild_csv_export()
        return {
            "ok": True,
            "title": mj["titel"],
            "description": mj["beschreibung"],
            "category": mj.get("kategorie", ""),
            "retail_price": mj["retail_price"],
            "rufpreis": mj["rufpreis"],
            "ki_runtime_ms": runtime_ms,
        }

    mj["ki_source"] = "failed"
    mj["ki_last_error"] = err or "ki_failed"
    _save_meta_json(artikelnr, mj)
    _usage_fail(mj["ki_last_error"])
    _rebuild_csv_export()
    return JSONResponse({"ok": False, "error": mj["ki_last_error"]}, status_code=502)


# ----------------------------
# Admin API (Token geschützt)
# ----------------------------
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
        except Exception:
            continue

        nr = str(mj.get("artikelnr", p.stem))
        cat = (mj.get("kategorie", "") or "").strip()

        if category and cat.lower() != category.strip().lower():
            continue
        if only_failed and (mj.get("ki_source") != "failed"):
            continue

        pics = sorted(RAW_DIR.glob(f"{nr}_*.jpg"))
        img_url = ""
        if pics:
            rel = pics[-1].relative_to(BASE_DIR)
            img_url = "/static/" + str(rel).replace(os.sep, "/")

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
            "image": img_url,
        })

    def _sort_key(x):
        try:
            return int(x["artikelnr"])
        except Exception:
            return x["artikelnr"]

    items.sort(key=_sort_key)
    return {"ok": True, "items": items}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("server_full:app", host="0.0.0.0", port=port)