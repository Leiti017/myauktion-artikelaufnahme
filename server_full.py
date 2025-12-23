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
import requests



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
import io, json, time, csv, math, os, datetime, zipfile, hashlib, threading
import re
from typing import Any, Dict, Optional, Tuple

app = FastAPI()

@app.middleware("http")
async def _no_cache_html(request, call_next):
    resp = await call_next(request)
    try:
        ct = resp.headers.get("content-type", "")
        if ct.startswith("text/html") or request.url.path in ("/", "/index.html", "/admin", "/articles") or request.url.path.endswith(("manifest.webmanifest","manifest.json")) or request.url.path.startswith("/static/uploads/"):
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
# Image filename scheme + stable ordering
# Titelbild: <artikelnr>.jpg
# weitere:   <artikelnr>_1.jpg, <artikelnr>_2.jpg, ...
# ----------------------------
def _img_sort_key(p: Path, artikelnr: str) -> tuple[int, int, str]:
    """Stable sort: titelbild first, then numeric suffix."""
    name = p.name
    if name == f"{artikelnr}.jpg":
        return (0, 0, name)
    m = re.match(rf"^{re.escape(artikelnr)}_(\d+)\.jpg$", name, flags=re.I)
    if m:
        try:
            return (1, int(m.group(1)), name)
        except Exception:
            return (1, 0, name)
    return (2, 0, name)

def _list_image_paths(artikelnr: str, directory: Path = None) -> list[Path]:
    """List all jpg images for an article in stable order."""
    directory = directory or RAW_DIR
    paths = list(directory.glob(f"{artikelnr}*.jpg"))
    paths.sort(key=lambda p: _img_sort_key(p, str(artikelnr)))
    return paths


def _next_image_path(artikelnr: str) -> Path:
    """Return next image path using scheme:
    - if <art>.jpg missing -> that is next (titelbild)
    - else -> <art>_1.jpg, <art>_2.jpg ... next free index

    NOTE: Older versions had a bug where the next index was computed wrong and
    could overwrite <art>_1.jpg repeatedly. This implementation is correct and stable.
    """
    art = str(artikelnr).strip()
    _migrate_legacy_zero(art)
    _migrate_bad_suffixes(art)

    main = RAW_DIR / f"{art}.jpg"
    if not main.exists():
        return main

    pics = _list_image_paths(art)
    nums: list[int] = []
    for p in pics:
        if p.name == f"{art}.jpg":
            nums.append(0)
            continue
        m = re.match(rf"^{re.escape(art)}_(\d+)\.jpg$", p.name, flags=re.I)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass

    next_idx = (max(nums) + 1) if nums else 1
    return RAW_DIR / f"{art}_{next_idx}.jpg"


def _migrate_legacy_zero(artikelnr: str) -> None:
    """Legacy support:
    - if <artikelnr>_0.jpg exists and <artikelnr>.jpg not -> rename to titelbild
    - if both exist -> move _0 to next free suffix (_1, _2, ...)
    """
    try:
        art = str(artikelnr).strip()
        legacy = RAW_DIR / f"{art}_0.jpg"
        main = RAW_DIR / f"{art}.jpg"
        if not legacy.exists():
            return

        if not main.exists():
            legacy.rename(main)
            # processed best-effort
            try:
                legacy_stem = f"{art}_0"
                new_stem = art
                for pf in PROCESSED_DIR.glob(legacy_stem + ".*"):
                    try:
                        pf.rename(PROCESSED_DIR / (new_stem + pf.suffix))
                    except Exception:
                        pass
            except Exception:
                pass
            return

        # main exists -> move legacy to next free suffix
        n = 1
        while (RAW_DIR / f"{art}_{n}.jpg").exists():
            n += 1
        newp = RAW_DIR / f"{art}_{n}.jpg"
        legacy.rename(newp)

        # processed best-effort
        try:
            legacy_stem = f"{art}_0"
            new_stem = f"{art}_{n}"
            for pf in PROCESSED_DIR.glob(legacy_stem + ".*"):
                try:
                    pf.rename(PROCESSED_DIR / (new_stem + pf.suffix))
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        return


def _migrate_bad_suffixes(artikelnr: str) -> None:
    """Fix older bug where suffix accidentally became next article number (e.g. 123456_123457.jpg).
    If we find very large suffixes and no small suffixes, we renumber them to _1, _2 ... preserving order.
    """
    art = str(artikelnr).strip()
    try:
        art_i = int(art)
    except Exception:
        return

    pics = []
    for p in RAW_DIR.glob(f"{art}_*.jpg"):
        m = re.match(rf"^{re.escape(art)}_(\d+)\.jpg$", p.name)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            continue
        pics.append((n, p))

    if not pics:
        return

    small = [n for n,_ in pics if 1 <= n <= 9999]
    huge  = [(n,p) for n,p in pics if n >= 100000 or n >= art_i]  # suspicious

    # only act if we have suspicious huge suffixes and NO small ones (avoid destroying correct data)
    if huge and not small:
        huge.sort(key=lambda t: t[0])
        next_idx = 1
        for _, p in huge:
            # find next free small index
            while (RAW_DIR / f"{art}_{next_idx}.jpg").exists():
                next_idx += 1
            try:
                p.rename(RAW_DIR / f"{art}_{next_idx}.jpg")
            except Exception:
                pass
            next_idx += 1


def _is_valid_article_image_name(artikelnr: str, filename: str) -> bool:
    """Allow <artikelnr>.jpg and <artikelnr>_N.jpg only."""
    if filename == f"{artikelnr}.jpg":
        return True
    return bool(re.match(rf"^{re.escape(artikelnr)}_(\d+)\.jpg$", filename, flags=re.I))


# ----------------------------
# CSV Export helpers (UTF-8 BOM + schneller Update)
# ----------------------------
CSV_FIELDS = [
    "ArtikelNr",
    "Bezeichnung",
    "Beschreibung",
    "Menge",
    "Preis",
    "Ladenpreis",
    "Lagerort",
    "Lagerstand",
    "Uebernehmen",
    "Sortiment",
    "Einlieferer-ID",
    "Angeliefert",
    "Betriebsmittel",
    "Mitarbeiter",
]
CSV_HEADERS = CSV_FIELDS

def _format_rufpreis(val: Any) -> str:
    """Rufpreis immer ohne . oder , (z.B. 1 statt 1.0)."""
    try:
        f = float(str(val or 0).replace(",", "."))
    except Exception:
        f = 0.0
    return str(int(round(f)))


def _format_ladenpreis(val: Any) -> str:
    """Listenpreis (= Ladenpreis) im Export mit Komma, z.B. 199,99; 0 => leer."""
    try:
        f = float(str(val or 0).replace(",", "."))
    except Exception:
        f = 0.0
    return (f"{f:.2f}".replace(".", ",") if f else "")

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
    rdr = csv.reader(io.StringIO(text, delimiter=";"))
    rows = list(rdr)

    # Ensure headers
    if not rows:
        rows = [CSV_HEADERS]
    else:
        rows[0] = CSV_HEADERS

    art = str(artikelnr)

    bezeichnung = str(meta.get("titel") or "")
    beschr = str(meta.get("beschreibung") or "")
    menge = int(meta.get("menge") or 1)
    preis = _format_rufpreis(meta.get("rufpreis", 0))
    # Listenpreis (= Ladenpreis) im Export mit Komma
    try:
        lp = float(str(meta.get("retail_price") or 0).replace(",", "."))
    except Exception:
        lp = 0.0
    ladenpreis = _format_ladenpreis(lp)

    lagerort = str(meta.get("lagerort") or "")
    lagerstand = int(meta.get("lagerstand") or 1)
    uebernehmen = int(meta.get("uebernehmen") or 1)
    sortiment = str(meta.get("sortiment_id") or "")
    einl_id = str(meta.get("einlieferer_id") or meta.get("einlieferer") or "")
    angel = str(meta.get("angeliefert") or "")
    betr = str(meta.get("betriebsmittel") or "")
    mitarb = str(meta.get("mitarbeiter") or "")

    new_row = [
        art,
        bezeichnung,
        beschr,
        str(menge),
        preis,
        ladenpreis,
        lagerort,
        str(lagerstand),
        str(uebernehmen),
        sortiment,
        einl_id,
        angel,
        betr,
        mitarb,
    ]

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
# ----------------------------
# Save de-duplication (prevents accidental double-click saves)
# - In-memory per process (fast, no DB needed)
# - If the same payload for the same artikelnr arrives within 2 seconds, we short-circuit.
# ----------------------------
_SAVE_DEDUP: dict[str, tuple[float, str]] = {}
_SAVE_DEDUP_LOCK = threading.Lock()

def _payload_hash(data: Dict[str, Any]) -> str:
    try:
        # Stable hashing; ignore fields that can vary without meaning
        d = dict(data or {})
        d.pop("reviewed", None)
        s = json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    except Exception:
        return str(time.time())



app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


# ----------------------------
# Google Drive Backup (Render Free, ohne Disk)
# ----------------------------
GDRIVE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GDRIVE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GDRIVE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN", "").strip()
GDRIVE_FOLDER_NAME = (os.getenv("DRIVE_FOLDER_NAME", "MyAuktion-Backups") or "MyAuktion-Backups").strip()
GDRIVE_RETENTION_DAYS = int(os.getenv("DRIVE_RETENTION_DAYS", "30") or "30")

_gdrive_cache = {"folder_id": None, "name_to_id": {}, "last_cleanup": None}

def _gdrive_enabled() -> bool:
    return bool(GDRIVE_CLIENT_ID and GDRIVE_CLIENT_SECRET and GDRIVE_REFRESH_TOKEN)

def _gdrive_access_token() -> str:
    r = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": GDRIVE_CLIENT_ID,
            "client_secret": GDRIVE_CLIENT_SECRET,
            "refresh_token": GDRIVE_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def _gdrive_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

def _gdrive_find_folder(token: str, name: str) -> Optional[str]:
    # Find folder created by this app (drive.file scope)
    q = f"mimeType='application/vnd.google-apps.folder' and name='{name}' and trashed=false"
    r = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers=_gdrive_headers(token),
        params={"q": q, "fields": "files(id,name)"},
        timeout=30,
    )
    r.raise_for_status()
    files = r.json().get("files", [])
    return files[0]["id"] if files else None

def _gdrive_create_folder(token: str, name: str) -> str:
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    r = requests.post(
        "https://www.googleapis.com/drive/v3/files",
        headers={**_gdrive_headers(token), "Content-Type": "application/json"},
        json=meta,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["id"]

def _gdrive_get_folder_id(token: str) -> str:
    if _gdrive_cache["folder_id"]:
        return _gdrive_cache["folder_id"]
    fid = _gdrive_find_folder(token, GDRIVE_FOLDER_NAME)
    if not fid:
        fid = _gdrive_create_folder(token, GDRIVE_FOLDER_NAME)
    _gdrive_cache["folder_id"] = fid
    return fid

def _gdrive_find_file_in_folder(token: str, folder_id: str, filename: str) -> Optional[str]:
    # cache
    if filename in _gdrive_cache["name_to_id"]:
        return _gdrive_cache["name_to_id"][filename]
    q = f"'{folder_id}' in parents and name='{filename}' and trashed=false"
    r = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers=_gdrive_headers(token),
        params={"q": q, "fields": "files(id,name)"},
        timeout=30,
    )
    r.raise_for_status()
    files = r.json().get("files", [])
    if not files:
        return None
    fid = files[0]["id"]
    _gdrive_cache["name_to_id"][filename] = fid
    return fid

def _gdrive_upload_bytes(token: str, folder_id: str, filename: str, data: bytes, mimetype: str) -> str:
    import uuid
    boundary = "===============" + uuid.uuid4().hex
    metadata = {"name": filename, "parents": [folder_id]}
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
        + json.dumps(metadata)
        + "\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: {mimetype}\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    existing_id = _gdrive_find_file_in_folder(token, folder_id, filename)
    if existing_id:
        url = f"https://www.googleapis.com/upload/drive/v3/files/{existing_id}"
        r = requests.patch(
            url,
            headers={**_gdrive_headers(token), "Content-Type": f"multipart/related; boundary={boundary}"},
            params={"uploadType": "multipart"},
            data=body,
            timeout=60,
        )
        r.raise_for_status()
        fid = r.json()["id"]
    else:
        url = "https://www.googleapis.com/upload/drive/v3/files"
        r = requests.post(
            url,
            headers={**_gdrive_headers(token), "Content-Type": f"multipart/related; boundary={boundary}"},
            params={"uploadType": "multipart"},
            data=body,
            timeout=60,
        )
        r.raise_for_status()
        fid = r.json()["id"]
    _gdrive_cache["name_to_id"][filename] = fid
    return fid

def _gdrive_cleanup_old(token: str, folder_id: str) -> None:
    # run max 1x per day
    today = time.strftime("%Y-%m-%d")
    if _gdrive_cache["last_cleanup"] == today:
        return
    cutoff_ts = time.time() - (GDRIVE_RETENTION_DAYS * 86400)
    cutoff_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cutoff_ts))
    q = f"'{folder_id}' in parents and modifiedTime < '{cutoff_iso}' and trashed=false"
    page_token = None
    while True:
        params = {"q": q, "fields": "nextPageToken,files(id,name,modifiedTime)", "pageSize": 1000}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get("https://www.googleapis.com/drive/v3/files", headers=_gdrive_headers(token), params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        files = js.get("files", [])
        for f in files:
            try:
                requests.delete(f"https://www.googleapis.com/drive/v3/files/{f['id']}", headers=_gdrive_headers(token), timeout=30).raise_for_status()
            except Exception:
                pass
        page_token = js.get("nextPageToken")
        if not page_token:
            break
    _gdrive_cache["last_cleanup"] = today

def _gdrive_backup_article(artikelnr: str) -> None:
    if not _gdrive_enabled():
        return
    try:
        token = _gdrive_access_token()
        folder_id = _gdrive_get_folder_id(token)
        # cleanup old (daily)
        _gdrive_cleanup_old(token, folder_id)

        # Upload/Update daily CSV
        day = time.strftime("%Y-%m-%d")
        daily_name = f"artikel_export_{day}.csv"
        latest_name = "artikel_export_latest.csv"
        # ensure current CSV exists
        _ensure_export_csv_exists()
        csv_bytes = EXPORT_CSV.read_bytes()
        _gdrive_upload_bytes(token, folder_id, daily_name, csv_bytes, "text/csv")
        # always keep an easy-to-find latest snapshot
        _gdrive_upload_bytes(token, folder_id, latest_name, csv_bytes, "text/csv")

        # Upload meta JSON for article
        try:
            meta_name = f"{artikelnr}.json"
            meta_bytes = _meta_path(artikelnr).read_bytes() if _meta_path(artikelnr).exists() else b""
            if meta_bytes:
                _gdrive_upload_bytes(token, folder_id, meta_name, meta_bytes, "application/json")
        except Exception:
            pass

        # Upload images for article (processed preferred, else raw)
        pics = _list_image_paths(artikelnr, PROCESSED_DIR)
        if not pics:
            pics = _list_image_paths(artikelnr, RAW_DIR)
        for p in pics:
            gname = p.name
            _gdrive_upload_bytes(token, folder_id, gname, p.read_bytes(), "image/jpeg")
    except Exception:
        # Backup darf nie das Speichern blockieren
        return


# ----------------------------
# Google Drive Restore (bei Render-ReDeploy / ohne Persistent Disk)
# - Lädt *.json und *.jpg aus dem Backup-Ordner zurück in uploads/raw
# - Danach wird CSV neu gebaut
# - Läuft im Background-Thread (damit Startup schnell bleibt)
# ----------------------------
RESTORE_MARKER = EXPORT_DIR / "restore_done.marker"

def _gdrive_list_files_in_folder(token: str, folder_id: str):
    files = []
    page_token = None
    while True:
        params = {"q": f"'{folder_id}' in parents and trashed=false",
                  "fields": "nextPageToken,files(id,name,modifiedTime)",
                  "pageSize": 1000}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get("https://www.googleapis.com/drive/v3/files", headers=_gdrive_headers(token), params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        files.extend(js.get("files", []))
        page_token = js.get("nextPageToken")
        if not page_token:
            break
    return files

def _gdrive_download_file(token: str, file_id: str) -> bytes:
    r = requests.get(f"https://www.googleapis.com/drive/v3/files/{file_id}", headers=_gdrive_headers(token), params={"alt":"media"}, timeout=120)
    r.raise_for_status()
    return r.content


def _restore_jsons_from_csv(csv_bytes: bytes) -> int:
    """Fallback: wenn im Drive keine *.json existieren, rekonstruieren wir minimale Meta-JSONs aus CSV.
    Damit bleiben alte Artikel sichtbar und editierbar.
    """
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        txt = csv_bytes.decode("utf-8-sig", errors="ignore")
        f = io.StringIO(txt)
        reader = csv.DictReader(f, delimiter=";")
        restored = 0
        now = int(time.time())
        for row in reader:
            nr = str(row.get("ArtikelNr","") or row.get("artikelnr","") or "").strip()
            if not nr.isdigit():
                continue
            meta_path = RAW_DIR / f"{nr}.json"
            if meta_path.exists():
                continue
            # Map common columns
            titel = str(row.get("Bezeichnung","") or row.get("Titel","") or "").strip()
            lagerort = str(row.get("Lagerort","") or "").strip()
            mitarbeiter = str(row.get("Mitarbeiter","") or "").strip()
            sortiment_id = str(row.get("Sortiment","") or "").strip()
            # prices may be "12,34" -> float
            def _p(x):
                s = str(x or "").strip().replace("€","").replace(" ","")
                s = s.replace(".","").replace(",",".") if "," in s else s
                try: return float(s)
                except Exception: return 0.0
            rufpreis = _p(row.get("Rufpreis",""))
            retail = _p(row.get("Listenpreis",""))
            menge = row.get("Menge","")
            try:
                menge = int(str(menge).strip() or "1")
            except Exception:
                menge = 1

            meta = {
                "artikelnr": nr,
                "titel": titel,
                "beschreibung": "",
                "kategorie": "",
                "lagerort": lagerort,
                "einlieferer": "",
                "mitarbeiter": mitarbeiter,
                "menge": menge,
                "rufpreis": rufpreis,
                "retail_price": retail,
                "sortiment_id": sortiment_id,
                "sortiment_name": "",
                "sortiment": "",
                "created_at": now,
                "updated_at": now,
                "reviewed": False,
                "ki_source": "imported",
                "ki_last_error": ""
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
            restored += 1
        return restored
    except Exception:
        return 0

def _gdrive_restore_all() -> None:
    if not _gdrive_enabled():
        return
    # only once per deploy/container
    try:
        if RESTORE_MARKER.exists():
            return
    except Exception:
        pass

    try:
        token = _gdrive_access_token()
        folder_id = _gdrive_get_folder_id(token)

        # If we already have data (e.g. dev/local), skip restore
        try:
            has_any = any(RAW_DIR.glob("*.json")) or any(RAW_DIR.glob("*.jpg"))
            if has_any:
                RESTORE_MARKER.write_text("skip_existing", "utf-8")
                return
        except Exception:
            pass

        files = _gdrive_list_files_in_folder(token, folder_id)

        # pick a CSV snapshot (latest)
        csv_file = None
        for f in files:
            if str(f.get('name','')).strip() == 'artikel_export_latest.csv':
                csv_file = f
                break
        if not csv_file:
            # fallback: newest artikel_export_*.csv
            cands = [f for f in files if str(f.get('name','')).startswith('artikel_export_') and str(f.get('name','')).lower().endswith('.csv')]
            # Drive returns modifiedTime; sort desc
            cands.sort(key=lambda x: str(x.get('modifiedTime','')), reverse=True)
            if cands:
                csv_file = cands[0]

        # restore JSON first (so meta exists even before images)
        json_files = [f for f in files if str(f.get("name","")).lower().endswith(".json")]
        jpg_files  = [f for f in files if str(f.get("name","")).lower().endswith(".jpg")]

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        restored = 0
        json_restored = 0
        for f in json_files:
            name = str(f.get("name","") or "")
            fid = str(f.get("id","") or "")
            if not name or not fid:
                continue
            # Only accept pure article meta names like 123456.json
            if not re.match(r"^\d+\.json$", name):
                continue
            try:
                data = _gdrive_download_file(token, fid)
                (RAW_DIR / name).write_bytes(data)
                restored += 1
                json_restored += 1
            except Exception:
                pass

        # Fallback: wenn keine JSONs im Drive liegen, rekonstruiere Meta aus CSV
        if json_restored == 0 and csv_file and csv_file.get('id'):
            try:
                csv_bytes = _gdrive_download_file(token, str(csv_file.get('id')))
                restored += _restore_jsons_from_csv(csv_bytes)
            except Exception:
                pass

        # restore images
        for f in jpg_files:
            name = str(f.get("name","") or "")
            fid = str(f.get("id","") or "")
            if not name or not fid:
                continue
            # Only accept article image scheme
            if not re.match(r"^\d+(?:_\d+)?\.jpg$", name, flags=re.I):
                continue
            try:
                data = _gdrive_download_file(token, fid)
                (RAW_DIR / name).write_bytes(data)
                restored += 1
                json_restored += 1
            except Exception:
                pass

        # Rebuild CSV from restored JSONs
        try:
            _rebuild_csv_export()
        except Exception:
            pass

        try:
            RESTORE_MARKER.write_text(f"restored:{restored}", "utf-8")
        except Exception:
            pass

        print(f"[GDRIVE] Restore done. Files restored: {restored}")
    except Exception as e:
        try:
            RESTORE_MARKER.write_text("failed:" + str(e)[:200], "utf-8")
        except Exception:
            pass
        print("[GDRIVE] Restore failed:", e)

def _start_restore_thread():
    try:
        if not _gdrive_enabled():
            return
        th = threading.Thread(target=_gdrive_restore_all, daemon=True)
        th.start()
    except Exception:
        return
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
        "sortiment": "",
        "sortiment_id": "",
        "sortiment_name": "",
        "reviewed": False,
        "ki_source": "",          # pending | realtime | failed | ""
        "ki_last_error": "",
        "ki_runtime_ms": 0,
        "batch_done": False,
        "last_image": "",
        "cover": "",
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
# CSV Export (Rebuild nutzt dieselben CSV_FIELDS wie Live-Update)
# ----------------------------
def _rebuild_csv_export() -> None:
    """Rebuild komplette artikel_export.csv aus allen *.json Metadaten.

    - Trennzeichen: ;
    - Encoding: UTF-8 mit BOM (utf-8-sig) damit Excel sauber spaltet
    - Sortiment/Kategorie werden NICHT exportiert
    - Mitarbeiter ist letzte Spalte
    """
    rows: list[dict] = []
    for meta_file in RAW_DIR.glob("*.json"):
        try:
            d = json.loads(meta_file.read_text("utf-8"))
        except Exception:
            continue

        nr = str(d.get("artikelnr") or meta_file.stem)
        rows.append({
            "ArtikelNr": nr,
            "Bezeichnung": str(d.get("titel", "") or ""),
            "Beschreibung": str(d.get("beschreibung", "") or ""),
            "Menge": int(d.get("menge", 1) or 1),
            "Preis": _format_rufpreis(d.get("rufpreis", 0.0) or 0.0),
            "Ladenpreis": _format_ladenpreis(d.get("retail_price", 0) or 0),
            "Lagerort": str(d.get("lagerort", "") or ""),
            "Lagerstand": int(d.get("lagerstand", 1) or 1),
            "Uebernehmen": int(d.get("uebernehmen", 1) or 1),
            "Sortiment": str(d.get("sortiment_id", "") or ""),
            "Einlieferer-ID": str(d.get("einlieferer_id", d.get("einlieferer", "")) or ""),
            "Angeliefert": str(d.get("angeliefert", "") or ""),
            "Betriebsmittel": str(d.get("betriebsmittel", "") or ""),
            "Mitarbeiter": str(d.get("mitarbeiter", "") or ""),
        })

    def _sort_key(r: dict):
        try:
            return int(r.get("ArtikelNr", 0))
        except Exception:
            return str(r.get("ArtikelNr", ""))

    rows.sort(key=_sort_key)

    with EXPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})

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
# Sortimente (persistent, Admin verwaltbar)
# - Ab jetzt: jedes Sortiment hat ID + Name.
# - JSON Format (export/sortimente.json):
#     [{"id":2349,"name":"(22) Drogerie/Kosmetik 85"}, ...]
# - Backward compatible: falls altes Format ["(22) ...", ...] -> wird beim Laden in IDs umgewandelt.
# ----------------------------
SORTIMENTE_JSON = EXPORT_DIR / "sortimente.json"

def _default_sortimente() -> list[dict]:
    return [{"id": 2349, "name": "(22) Tiernahrung 45"}]

def _next_sortiment_id(items: list[dict]) -> int:
    mx = 0
    for it in (items or []):
        try:
            mx = max(mx, int(it.get("id", 0) or 0))
        except Exception:
            pass
    return mx + 1 if mx else 1000

def _normalize_sortimente(data) -> list[dict]:
    out: list[dict] = []
    if not isinstance(data, list):
        return out

    # old format: list[str]
    if data and all(isinstance(x, str) for x in data):
        nid = 1000
        for s in data:
            name = str(s).strip()
            if not name:
                continue
            out.append({"id": nid, "name": name})
            nid += 1
        return out

    # new format: list[dict]
    for x in data:
        if not isinstance(x, dict):
            continue
        try:
            sid = int(x.get("id", 0) or 0)
        except Exception:
            sid = 0
        name = str(x.get("name", "") or "").strip()
        if not sid or not name:
            continue
        out.append({"id": sid, "name": name})

    # de-dupe by id (first wins)
    seen = set()
    dedup = []
    for it in out:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        dedup.append(it)
    return dedup

def _load_sortimente() -> list[dict]:
    if SORTIMENTE_JSON.exists():
        try:
            data = json.loads(SORTIMENTE_JSON.read_text("utf-8"))
            items = _normalize_sortimente(data)
            if items:
                # persist legacy -> new format
                if data and all(isinstance(x, str) for x in data):
                    _save_sortimente(items)
                return items
        except Exception:
            pass
    items = _default_sortimente()
    try:
        if not SORTIMENTE_JSON.exists():
            _save_sortimente(items)
    except Exception:
        pass
    return items

def _save_sortimente(items: list[dict]) -> None:
    cleaned = []
    seen_ids = set()
    for x in (items or []):
        if not isinstance(x, dict):
            continue
        try:
            sid = int(x.get("id", 0) or 0)
        except Exception:
            continue
        name = str(x.get("name", "") or "").strip()
        if not sid or not name:
            continue
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        cleaned.append({"id": sid, "name": name})
    SORTIMENTE_JSON.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), "utf-8")

def _sortiment_name_by_id(sortiment_id: str | int) -> str:
    try:
        sid = int(str(sortiment_id).strip() or 0)
    except Exception:
        sid = 0
    if not sid:
        return ""
    for it in _load_sortimente():
        try:
            if int(it.get("id", 0) or 0) == sid:
                return str(it.get("name", "") or "").strip()
        except Exception:
            continue
    return ""


# ----------------------------
# Artikel-Liste (Gesamtartikel – nach Datum sortiert)
# ----------------------------
def _list_articles() -> list[dict]:
    items = []
    for p in RAW_DIR.glob("*.json"):
        try:
            mj = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        nr = str(mj.get("artikelnr", p.stem))
        created = int(mj.get("created_at", 0) or 0)
        updated = int(mj.get("updated_at", 0) or 0)

        pics = _list_image_paths(nr)
        img_url = ""
        if pics:
            rel = pics[-1].relative_to(BASE_DIR)
            img_url = "/static/" + str(rel).replace("\\", "/")

        items.append({
            "artikelnr": nr,
            "titel": mj.get("titel", "") or "",
            "beschreibung": mj.get("beschreibung", "") or "",
            "sortiment": (mj.get("sortiment_name", "") or mj.get("sortiment", "") or "").strip(),
            "sortiment_id": str(mj.get("sortiment_id", "") or ""),
            "lagerort": mj.get("lagerort", "") or "",
            "menge": int(mj.get("menge", 1) or 1),
            "rufpreis": mj.get("rufpreis", 0.0) or 0.0,
            "retail_price": mj.get("retail_price", 0.0) or 0.0,
            "mitarbeiter": mj.get("mitarbeiter", "") or "",
            "angeliefert": mj.get("angeliefert", "") or "",
            "created_at": created,
            "updated_at": updated,
            "image": img_url,
        })
    # neueste zuerst
    items.sort(key=lambda x: int(x.get("created_at", 0) or 0), reverse=True)
    return items


@app.on_event("startup")
def _on_startup():
    # Restore alte Artikel aus Google Drive, falls Render Container neu gestartet wurde
    _start_restore_thread()

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "index.html"))


@app.get("/admin")
def admin_root():
    return FileResponse(str(BASE_DIR / "admin.html"))


@app.get("/articles")
def articles_page():
    return FileResponse(str(BASE_DIR / "articles.html"))


@app.api_route("/api/health", methods=["GET","HEAD"])
def health():
    return {"ok": True}


@app.get("/api/sortimente")
def sortimente_public():
    return {"ok": True, "items": _load_sortimente()}


@app.get("/api/articles")
def articles_api(date_from: str | None = None, date_to: str | None = None):
    """
    Gesamtartikel (neueste zuerst).
    Optional:
      /api/articles?date_from=YYYY-MM-DD&date_to=YYYY-MM-DD
    """
    items = _list_articles()

    def _to_ts(s: str, end: bool = False) -> int:
        try:
            dt = datetime.datetime.strptime(s, "%Y-%m-%d")
            if end:
                dt = dt + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
            return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        except Exception:
            return 0

    if date_from:
        tsf = _to_ts(date_from, end=False)
        if tsf:
            items = [x for x in items if int(x.get("created_at", 0) or 0) >= tsf]
    if date_to:
        tst = _to_ts(date_to, end=True)
        if tst:
            items = [x for x in items if int(x.get("created_at", 0) or 0) <= tst]

    return {"ok": True, "items": items}

@app.get("/api/recent_articles")
def recent_articles(mitarbeiter: str = "", limit: int = 20):
    """Letzte aufgenommene Artikel (neueste zuerst).
    Optional Filter:
      /api/recent_articles?mitarbeiter=Jan&limit=20
    """
    items = _list_articles()  # already newest first
    m = (mitarbeiter or "").strip().lower()
    if m:
        items = [x for x in items if str(x.get("mitarbeiter","") or "").strip().lower() == m]

    try:
        lim = int(limit or 20)
    except Exception:
        lim = 20
    lim = max(1, min(lim, 100))
    items = items[:lim]

    # keep payload small
    out = []
    for it in items:
        out.append({
            "artikelnr": it.get("artikelnr",""),
            "titel": it.get("titel",""),
            "created_at": int(it.get("created_at",0) or 0),
            "sortiment": it.get("sortiment",""),
            "sortiment_id": it.get("sortiment_id",""),
            "lagerort": it.get("lagerort",""),
            "rufpreis": it.get("rufpreis",0),
        })
    return {"ok": True, "items": out}


@app.get("/api/export.csv")
def export_csv(from_nr: str | None = None, to_nr: str | None = None, sortiment_id: str | None = None):
    """
    Export CSV (UTF-8 mit BOM für Excel). Optional:
      - Bereich: from_nr/to_nr
      - Filter: sortiment_id
      /api/export.csv?from_nr=123458&to_nr=123480
    """
    _ensure_export_csv_exists()

    # Kein Filter -> Datei direkt ausliefern (nur wenn wirklich kein Filter aktiv)
    if not from_nr and not to_nr and not sortiment_id:
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
        if sortiment_id:
            if str(meta.get('sortiment_id','') or '').strip() != str(sortiment_id).strip():
                continue
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
            str(meta.get("mitarbeiter") or ""),
        ])

    bio = io.StringIO()
    w = csv.writer(bio, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(CSV_HEADERS)
    for r in rows:
        w.writerow(r)
    content = ("\ufeff" + bio.getvalue()).encode("utf-8")
    return Response(content, media_type="text/csv", headers={"Content-Disposition":"attachment; filename=artikel_export.csv"})


# ----------------------------
# Bilder ZIP Export (Admin, Von/Bis ArtikelNr)
# ----------------------------
def _images_for_art(artikelnr: str):
    """Return list of image Paths for article, ordered: ART.jpg, ART_1.jpg, ART_2.jpg..."""
    art = str(artikelnr).strip()
    pics = list(RAW_DIR.glob(f"{art}*.jpg"))
    def _k(p: Path):
        if p.name == f"{art}.jpg":
            return 0
        m = re.search(rf"^{re.escape(art)}_(\d+)\.jpg$", p.name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 999999
        return 999999
    pics.sort(key=_k)
    return pics

def _in_range_art(n: str, from_nr: str | None, to_nr: str | None) -> bool:
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

@app.get("/api/images.zip")
@app.get("/api/export_images.zip")
@app.get("/api/bilder.zip")
def export_images_zip(request: Request, from_nr: str | None = None, to_nr: str | None = None, sortiment_id: str | None = None):
    """ZIP mit allen Bildern (RAW) für Artikel (optional Von/Bis ArtikelNr).
    Admin-Token geschützt (X-Admin-Token oder ?token=...).
    """
    guard = _admin_guard(request)
    if guard:
        return guard

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # Artikel über JSONs bestimmen (damit nur echte Positionen drin sind)
        for jf in sorted(RAW_DIR.glob("*.json")):
            art = jf.stem
            if (from_nr or to_nr) and not _in_range_art(art, from_nr, to_nr):
                continue
            if sortiment_id:
                try:
                    meta = _load_meta_json(art)
                    if str(meta.get('sortiment_id','') or '').strip() != str(sortiment_id).strip():
                        continue
                except Exception:
                    pass
            for p in _images_for_art(art):
                try:
                    # in zip nur Dateiname, weil schon eindeutig (ART...jpg)
                    z.writestr(p.name, p.read_bytes())
                except Exception:
                    pass

    content = bio.getvalue()
    return Response(
        content,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=artikel_bilder.zip"},
    )


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
    exists = _meta_path(artikelnr).exists() or any(RAW_DIR.glob(f"{artikelnr}*.jpg"))
    return {"ok": True, "exists": bool(exists)}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    artikelnr: str = Form(...),
    background_tasks: BackgroundTasks = None,
):
    artikelnr = str(artikelnr).strip()
    _migrate_legacy_zero(artikelnr)
    _migrate_bad_suffixes(artikelnr)
    data = await file.read()

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        # SPEED: kleiner + schneller für Base64/Upload
        img.thumbnail((1024, 1024))
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Ungültiges Bild: {e}"}, status_code=400)

    out = _next_image_path(artikelnr)
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
    _migrate_legacy_zero(artikelnr)
    _migrate_bad_suffixes(artikelnr)
    meta = _load_meta_json(artikelnr)
    cover = str(meta.get("cover") or "").strip()

    files = []
    for f in _list_image_paths(artikelnr):
        rel = f.relative_to(BASE_DIR)
        files.append("/static/" + str(rel).replace("\\", "/"))

    # cover-first ordering
    if cover:
        files.sort(key=lambda u: 0 if u.endswith("/" + cover) or u.endswith(cover) else 1)

    return {"ok": True, "files": files, "cover": cover}

@app.post("/api/set_cover")
def set_cover(payload: Dict[str, Any]):
    artikelnr = str(payload.get("artikelnr", "")).strip()
    _migrate_legacy_zero(artikelnr)
    _migrate_bad_suffixes(artikelnr)
    filename = str(payload.get("filename", "") or payload.get("file", "") or payload.get("url", "")).strip()
    if "/" in filename:
        filename = filename.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]

    if not artikelnr or not filename:
        return JSONResponse({"ok": False, "error": "missing data"}, status_code=400)

    p = RAW_DIR / filename
    if not p.exists():
        return JSONResponse({"ok": False, "error": "file not found"}, status_code=404)

    meta = _load_meta_json(artikelnr)
    meta["cover"] = filename
    _save_meta_json(artikelnr, meta)
    return {"ok": True, "cover": filename}





@app.post("/api/delete_last_image")
def delete_last_image(data: Dict[str, Any]):
    artikelnr = str(data.get("artikelnr", "")).strip()
    _migrate_legacy_zero(artikelnr)
    _migrate_bad_suffixes(artikelnr)
    if not artikelnr:
        return JSONResponse({"ok": False, "error": "Artikelnummer fehlt"}, status_code=400)

    files = _list_image_paths(artikelnr)
    if not files:
        return JSONResponse({"ok": False, "error": "Keine Bilder vorhanden"}, status_code=404)

    target = files[-1]
    try:
        target.unlink()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Konnte Bild nicht löschen: {e}"}, status_code=500)

    # auch ggf. bearbeitete Version löschen (best-effort)
    try:
        stem = target.stem
        for pf in PROCESSED_DIR.glob(stem + ".*"):
            try: pf.unlink()
            except Exception: pass
    except Exception:
        pass

    # meta last_image + cover aktualisieren
    mj = _load_meta_json(artikelnr)
    remaining = _list_image_paths(artikelnr)
    mj["last_image"] = remaining[-1].name if remaining else ""
    if str(mj.get("cover", "")).strip() == target.name:
        mj["cover"] = remaining[0].name if remaining else ""
    _save_meta_json(artikelnr, mj)

    _rebuild_csv_export()
    return {"ok": True, "deleted": target.name}



@app.post("/api/delete_image")
def delete_image(payload: Dict[str, Any]):
    artikelnr = str(payload.get("artikelnr", "")).strip()
    _migrate_legacy_zero(artikelnr)
    _migrate_bad_suffixes(artikelnr)
    filename = str(payload.get("filename", "") or payload.get("name","") or payload.get("file","") or payload.get("url","")).strip()

    # falls URL: nur basename nehmen
    if "/" in filename:
        filename = filename.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?", 1)[0]

    if not artikelnr or not filename:
        return JSONResponse({"ok": False, "error": "artikelnr/filename fehlt"}, status_code=400)

    # Sicherheitscheck: nur Dateien dieses Artikels erlauben
    if not filename.lower().endswith(".jpg") or not _is_valid_article_image_name(artikelnr, filename):
        return JSONResponse({"ok": False, "error": "ungültiger Dateiname"}, status_code=400)


    path = RAW_DIR / filename
    if not path.exists():
        return JSONResponse({"ok": False, "error": "Datei nicht gefunden"}, status_code=404)

    # RAW löschen
    try:
        path.unlink()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"löschen fehlgeschlagen: {e}"}, status_code=500)

    # ggf. bearbeitete Version löschen (best-effort)
    try:
        stem = path.stem
        for pf in PROCESSED_DIR.glob(stem + ".*"):
            try:
                pf.unlink()
            except Exception:
                pass
    except Exception:
        pass

    # meta updaten: last_image + cover korrekt setzen
    mj = _load_meta_json(artikelnr)
    pics = _list_image_paths(artikelnr)

    mj["last_image"] = pics[-1].name if pics else ""

    # wenn gelöschtes Bild Titelbild war -> neues Titelbild setzen (erstes verbleibendes, sonst leer)
    if str(mj.get("cover", "")).strip() == filename:
        mj["cover"] = pics[0].name if pics else ""

    _save_meta_json(artikelnr, mj)

    _rebuild_csv_export()
    return {"ok": True, "deleted": filename}



@app.get("/api/meta")
def meta(artikelnr: str):
    artikelnr = str(artikelnr).strip()
    _migrate_legacy_zero(artikelnr)
    _migrate_bad_suffixes(artikelnr)
    mj = _load_meta_json(artikelnr)

    pics = _list_image_paths(artikelnr)
    img_url = ""
    if pics:
        rel = pics[-1].relative_to(BASE_DIR)
        img_url = "/static/" + str(rel).replace("\\", "/")

    return {
        "ok": True,
        "artikelnr": artikelnr,
        "titel": mj.get("titel", "") or "",
        "beschreibung": mj.get("beschreibung", "") or "",
        "kategorie": mj.get("kategorie", "") or "",
        "retail_price": mj.get("retail_price", 0.0) or 0.0,
        "rufpreis": mj.get("rufpreis", 0.0) or 0.0,
        "lagerort": mj.get("lagerort", "") or "",
        "sortiment": mj.get("sortiment", "") or "",
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
    _migrate_legacy_zero(artikelnr)
    _migrate_bad_suffixes(artikelnr)
    if not artikelnr:
        return JSONResponse({"ok": False, "error": "Artikelnummer fehlt"}, status_code=400)

    # De-dup: same payload within short window -> ignore (prevents double-click duplicates)
    try:
        h = _payload_hash(data)
        now = time.time()
        with _SAVE_DEDUP_LOCK:
            prev = _SAVE_DEDUP.get(artikelnr)
            if prev and prev[1] == h and (now - prev[0]) < 2.0:
                return {"ok": True, "dedup": True}
            _SAVE_DEDUP[artikelnr] = (now, h)
    except Exception:
        pass

    mj = _load_meta_json(artikelnr)

    # Textfelder
    for k in ["titel", "beschreibung", "lagerort", "einlieferer", "mitarbeiter", "kategorie"]:
        if k in data and data[k] is not None:
            mj[k] = str(data[k])
    # Sortiment: prefer sortiment_id (numeric). Name wird aus ID abgeleitet.
    # Backward compatible: falls noch 'sortiment' Name kommt, speichern wir es als sortiment_name und versuchen ID zu finden.
    if "sortiment_id" in data and data.get("sortiment_id") is not None:
        sid = str(data.get("sortiment_id") or "").strip()
        mj["sortiment_id"] = sid
        mj["sortiment_name"] = _sortiment_name_by_id(sid)
    elif "sortiment" in data and data.get("sortiment") is not None:
        mj["sortiment_name"] = str(data.get("sortiment") or "").strip()
        nm = mj["sortiment_name"]
        for it in _load_sortimente():
            if str(it.get("name","")).strip().lower() == nm.lower():
                mj["sortiment_id"] = str(it.get("id"))
                break


    # Settings-Felder (für CSV Export)
    if "menge" in data: mj["menge"] = int(data.get("menge") or 1)
    if "lagerstand" in data: mj["lagerstand"] = int(data.get("lagerstand") or 1)
    if "uebernehmen" in data: mj["uebernehmen"] = int(data.get("uebernehmen") or 1)
    if "einlieferer_id" in data: mj["einlieferer_id"] = str(data.get("einlieferer_id") or "")
    if "angeliefert" in data: mj["angeliefert"] = str(data.get("angeliefert") or "")
    if "betriebsmittel" in data: mj["betriebsmittel"] = str(data.get("betriebsmittel") or "")
    if "mitarbeiter" in data: mj["mitarbeiter"] = str(data.get("mitarbeiter") or "")

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

    # Google Drive Backup (best-effort)
    try:
        _gdrive_backup_article(artikelnr)
    except Exception:
        pass


    return {"ok": True}


@app.get("/api/describe")
def describe(artikelnr: str):
    artikelnr = str(artikelnr).strip()
    _migrate_legacy_zero(artikelnr)
    _migrate_bad_suffixes(artikelnr)
    pics = _list_image_paths(artikelnr)
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
# Admin Ping (für UI Lock-Screen)
# - Wenn ADMIN_TOKEN leer ist -> 200 ohne Token
# - Wenn ADMIN_TOKEN gesetzt ist -> 401 ohne Token
# ----------------------------
@app.get("/api/admin/ping")
def admin_ping(request: Request):
    guard = _admin_guard(request)
    if guard:
        return guard
    return {"ok": True}

# ----------------------------
# Admin: CSV Import (manuell) – falls Drive Scope Dateien nicht sieht
# ----------------------------
from fastapi import UploadFile as _UploadFile
from fastapi import File as _File
from fastapi import Form as _Form

def _parse_price_any(s: str) -> float:
    s = str(s or "").strip().replace("€","").replace(" ", "")
    if not s:
        return 0.0
    if "," in s and "." in s:
        s = s.replace(".","").replace(",",".")
    elif "," in s:
        s = s.replace(",",".")
    try:
        return float(s)
    except Exception:
        return 0.0

@app.post("/api/admin/import_csv")
async def admin_import_csv(request: Request, file: _UploadFile = _File(...), overwrite: int = _Form(0)):
    guard = _admin_guard(request)
    if guard:
        return guard

    data = await file.read()
    txt = data.decode("utf-8-sig", errors="ignore")
    import csv, io
    rdr = csv.reader(io.StringIO(txt), delimiter=";")
    rows = list(rdr)
    if not rows:
        return JSONResponse({"ok": False, "error": "CSV leer"}, status_code=400)

    header = [c.strip() for c in rows[0]]

    def idx(name):
        try:
            return header.index(name)
        except Exception:
            return -1

    i_nr = idx("ArtikelNr")
    if i_nr < 0:
        for k in ("artikelnr","ARTIKELNR","Artikelnummer","Artikelnummern"):
            if k in header:
                i_nr = header.index(k); break
    if i_nr < 0:
        return JSONResponse({"ok": False, "error": "Header 'ArtikelNr' nicht gefunden"}, status_code=400)

    i_title = idx("Bezeichnung")
    i_desc  = idx("Beschreibung")
    i_menge = idx("Menge")
    i_preis = idx("Preis")
    i_lp    = idx("Ladenpreis")
    i_lager = idx("Lagerort")
    i_ls    = idx("Lagerstand")
    i_ueb   = idx("Uebernehmen")
    i_sort  = idx("Sortiment")
    i_einl  = idx("Einlieferer-ID")
    i_ang   = idx("Angeliefert")
    i_bm    = idx("Betriebsmittel")
    i_ma    = idx("Mitarbeiter")

    created = 0
    skipped = 0
    updated = 0
    now = int(time.time())

    for r in rows[1:]:
        if not r or len(r) <= i_nr:
            continue
        nr = str(r[i_nr]).strip()
        if not nr.isdigit():
            continue

        mp = _meta_path(nr)
        exists = mp.exists()
        if exists and not overwrite:
            skipped += 1
            continue

        mj = _load_meta_json(nr) if exists else _default_meta(nr)

        if i_title >= 0 and i_title < len(r): mj["titel"] = str(r[i_title]).strip()
        if i_desc  >= 0 and i_desc  < len(r): mj["beschreibung"] = str(r[i_desc]).strip()
        if i_lager >= 0 and i_lager < len(r): mj["lagerort"] = str(r[i_lager]).strip()
        if i_ma    >= 0 and i_ma    < len(r): mj["mitarbeiter"] = str(r[i_ma]).strip()

        if i_sort  >= 0 and i_sort  < len(r):
            mj["sortiment_id"] = str(r[i_sort]).strip()
            mj["sortiment_name"] = _sortiment_name_by_id(mj["sortiment_id"])

        if i_einl >= 0 and i_einl < len(r):
            mj["einlieferer_id"] = str(r[i_einl]).strip()
            mj["einlieferer"] = mj["einlieferer_id"]

        if i_ang >= 0 and i_ang < len(r): mj["angeliefert"] = str(r[i_ang]).strip()
        if i_bm  >= 0 and i_bm  < len(r): mj["betriebsmittel"] = str(r[i_bm]).strip()

        if i_menge >= 0 and i_menge < len(r):
            try: mj["menge"] = int(str(r[i_menge]).strip() or "1")
            except Exception: mj["menge"] = 1
        if i_ls >= 0 and i_ls < len(r):
            try: mj["lagerstand"] = int(str(r[i_ls]).strip() or "1")
            except Exception: mj["lagerstand"] = 1
        if i_ueb >= 0 and i_ueb < len(r):
            try: mj["uebernehmen"] = int(str(r[i_ueb]).strip() or "1")
            except Exception: mj["uebernehmen"] = 1

        if i_preis >= 0 and i_preis < len(r):
            mj["rufpreis"] = _parse_price_any(r[i_preis])
        if i_lp >= 0 and i_lp < len(r):
            mj["retail_price"] = _parse_price_any(r[i_lp])

        if not exists:
            mj["created_at"] = now
            created += 1
        else:
            updated += 1

        mj["ki_source"] = mj.get("ki_source") or "imported"
        _save_meta_json(nr, mj)

    _rebuild_csv_export()
    return {"ok": True, "created": created, "updated": updated, "skipped": skipped, "overwrite": int(overwrite)}

@app.post("/api/admin/restore_now")
def admin_restore_now(request: Request):
    guard = _admin_guard(request)
    if guard:
        return guard
    try:
        try:
            if RESTORE_MARKER.exists():
                RESTORE_MARKER.unlink()
        except Exception:
            pass
        _gdrive_restore_all()
        status = ""
        try:
            status = RESTORE_MARKER.read_text("utf-8") if RESTORE_MARKER.exists() else "no_marker"
        except Exception:
            status = "unknown"
        return {"ok": True, "status": status}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ----------------------------
# Admin API (Token geschützt)
# ----------------------------


@app.get("/api/admin/sortimente")
def admin_sortimente(request: Request):
    guard = _admin_guard(request)
    if guard:
        return guard
    return {"ok": True, "items": _load_sortimente()}


@app.post("/api/admin/sortimente/add")
def admin_sortimente_add(request: Request, data: Dict[str, Any]):
    guard = _admin_guard(request)
    if guard:
        return guard
    name = str(data.get("name", "") or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "name fehlt"}, status_code=400)

    items = _load_sortimente()
    sid_raw = str(data.get("id", "") or "").strip()
    if sid_raw:
        try:
            sid = int(sid_raw)
        except Exception:
            return JSONResponse({"ok": False, "error": "id ungültig"}, status_code=400)
    else:
        sid = _next_sortiment_id(items)

    for it in items:
        if int(it.get("id", 0) or 0) == sid:
            return JSONResponse({"ok": False, "error": "id existiert bereits"}, status_code=400)
        if str(it.get("name","")).strip().lower() == name.lower():
            return JSONResponse({"ok": False, "error": "name existiert bereits"}, status_code=400)

    items.append({"id": sid, "name": name})
    _save_sortimente(items)
    return {"ok": True, "items": _load_sortimente()}


@app.post("/api/admin/sortimente/rename")
def admin_sortimente_rename(request: Request, data: Dict[str, Any]):
    guard = _admin_guard(request)
    if guard:
        return guard
    sid_raw = str(data.get("id", "") or "").strip()
    new = str(data.get("new", "") or "").strip()
    if not sid_raw or not new:
        return JSONResponse({"ok": False, "error": "id/new fehlt"}, status_code=400)
    try:
        sid = int(sid_raw)
    except Exception:
        return JSONResponse({"ok": False, "error": "id ungültig"}, status_code=400)

    items = _load_sortimente()
    found = False
    for it in items:
        if int(it.get("id", 0) or 0) == sid:
            it["name"] = new
            found = True
            break
    if not found:
        return JSONResponse({"ok": False, "error": "id nicht gefunden"}, status_code=404)

    _save_sortimente(items)
    return {"ok": True, "items": _load_sortimente()}


@app.post("/api/admin/sortimente/delete")
def admin_sortimente_delete(request: Request, data: Dict[str, Any]):
    guard = _admin_guard(request)
    if guard:
        return guard
    sid_raw = str(data.get("id", "") or data.get("sortiment_id","") or "").strip()
    if not sid_raw:
        return JSONResponse({"ok": False, "error": "id fehlt"}, status_code=400)
    try:
        sid = int(sid_raw)
    except Exception:
        return JSONResponse({"ok": False, "error": "id ungültig"}, status_code=400)

    items = [x for x in _load_sortimente() if int(x.get("id",0) or 0) != sid]
    _save_sortimente(items)
    return {"ok": True, "items": _load_sortimente()}

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
def admin_articles(request: Request, sortiment: str = "", only_failed: int = 0):
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
        sorti = (mj.get("sortiment_name", "") or mj.get("sortiment", "") or "").strip()
        sorti_id = str(mj.get("sortiment_id", "") or "")

        if sortiment and sorti.lower() != sortiment.strip().lower():
            continue
        if only_failed and (mj.get("ki_source") != "failed"):
            continue

        pics = _list_image_paths(nr)
        img_url = ""
        if pics:
            rel = pics[-1].relative_to(BASE_DIR)
            img_url = "/static/" + str(rel).replace("\\", "/")

        items.append({
            "artikelnr": nr,
            "titel": mj.get("titel", "") or "",
            "beschreibung": mj.get("beschreibung", "") or "",
            "kategorie": cat,
            "sortiment": sorti,
            "sortiment_id": sorti_id,
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