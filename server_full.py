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
from PIL import Image, ImageOps, ImageFilter, ImageDraw
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

# ----------------------------
# Auto-Freistellen (rembg) wie MyAuktion-Admin
# ----------------------------
IMAGE_TARGET_SIZE = (750, 750)

REMBG_AVAILABLE = False
REMBG_SESSION = None
try:
    from rembg import new_session, remove
    REMBG_SESSION = new_session("u2net")
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

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



# ----------------------------
# Auto-Freistellen (rembg) Helper
# ----------------------------
def _make_cutout_jpeg_from_raw(raw_jpg_bytes: bytes) -> bytes:
    """Freistellen + Bodenschatten + 750x750 Weiß (wie MyAuktion-Admin). Returns JPEG bytes."""
    # RAW öffnen
    with Image.open(io.BytesIO(raw_jpg_bytes)) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")

        # rembg erwartet am besten PNG bytes
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)

        cut = remove(buf.read(), session=REMBG_SESSION)  # type: ignore
        fg = Image.open(io.BytesIO(cut)).convert("RGBA")

    # Alpha etwas säubern
    alpha = fg.split()[-1]
    alpha = alpha.point(lambda v: 0 if v < 20 else v)
    fg.putalpha(alpha)

    bbox = alpha.getbbox()
    if not bbox:
        # Fallback: einfach auf Weiß zentrieren
        base = Image.new("RGB", IMAGE_TARGET_SIZE, (255, 255, 255))
        im_rgb = im.convert("RGB")
        im_rgb = ImageOps.contain(im_rgb, IMAGE_TARGET_SIZE)
        x = (base.size[0] - im_rgb.size[0]) // 2
        y = (base.size[1] - im_rgb.size[1]) // 2
        base.paste(im_rgb, (x, y))
        out = io.BytesIO()
        base.save(out, "JPEG", quality=90)
        return out.getvalue()

    x0, y0, x1, y1 = bbox
    fg_cropped = fg.crop((x0, y0, x1, y1))
    w, h = fg_cropped.size

    # Bodenschatten (Ellipse + Blur)
    shadow_mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(shadow_mask)

    shadow_w = int(w * 0.7)
    shadow_h = max(6, int(h * 0.15))
    cx = w // 2
    sx0 = max(0, cx - shadow_w // 2)
    sx1 = min(w, cx + shadow_w // 2)
    sy0 = max(0, h - shadow_h - 1)
    sy1 = h

    draw.ellipse([sx0, sy0, sx1, sy1], fill=255)
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(12))
    shadow_mask = shadow_mask.point(lambda a: int(a * (90 / 255.0)))

    shadow_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow_rgba.paste((0, 0, 0, 255), mask=shadow_mask)

    base_rgba = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    base_rgba.paste(shadow_rgba, (0, 0), shadow_rgba)
    base_rgba.paste(fg_cropped, (0, 0), fg_cropped)

    result = base_rgba.convert("RGB")

    # auf 750x750 contain + zentriert auf Weiß
    result = ImageOps.contain(result, IMAGE_TARGET_SIZE)
    final = Image.new("RGB", IMAGE_TARGET_SIZE, (255, 255, 255))
    ox = (IMAGE_TARGET_SIZE[0] - result.size[0]) // 2
    oy = (IMAGE_TARGET_SIZE[1] - result.size[1]) // 2
    final.paste(result, (ox, oy))

    out = io.BytesIO()
    final.save(out, "JPEG", quality=90)
    return out.getvalue()


def _process_image_to_processed(raw_path: Path) -> None:
    """Erstellt uploads/processed/<same filename>.jpg (best-effort)."""
    if not REMBG_AVAILABLE or not REMBG_SESSION:
        return
    try:
        raw_bytes = raw_path.read_bytes()
        out_bytes = _make_cutout_jpeg_from_raw(raw_bytes)
        out_path = PROCESSED_DIR / raw_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(out_bytes)
    except Exception:
        # darf Upload nie blockieren
        return


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
    sortiment = str(meta.get("sortiment") or "")
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
        # ensure current CSV exists
        _ensure_export_csv_exists()
        csv_bytes = EXPORT_CSV.read_bytes()
        _gdrive_upload_bytes(token, folder_id, daily_name, csv_bytes, "text/csv")

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
            "Sortiment": str(d.get("sortiment", "") or ""),
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
# ----------------------------
SORTIMENTE_JSON = EXPORT_DIR / "sortimente.json"

def _default_sortimente() -> list[str]:
    # Beispiele (kannst du im Admin ändern)
    return ["(22) Tiernahrung 45"]

def _load_sortimente() -> list[str]:
    if SORTIMENTE_JSON.exists():
        try:
            data = json.loads(SORTIMENTE_JSON.read_text("utf-8"))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
    return _default_sortimente()

def _save_sortimente(items: list[str]) -> None:
    cleaned = []
    for x in items:
        s = str(x).strip()
        if not s:
            continue
        if s not in cleaned:
            cleaned.append(s)
    SORTIMENTE_JSON.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), "utf-8")


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

        pics = _list_image_paths(nr, PROCESSED_DIR)
        if not pics:
            pics = _list_image_paths(nr, RAW_DIR)
        img_url = ""
        if pics:
            rel = pics[-1].relative_to(BASE_DIR)
            img_url = "/static/" + rel.as_posix()

        items.append({
            "artikelnr": nr,
            "titel": mj.get("titel", "") or "",
            "beschreibung": mj.get("beschreibung", "") or "",
            "sortiment": (mj.get("sortiment", "") or "").strip(),
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
def export_images_zip(request: Request, from_nr: str | None = None, to_nr: str | None = None):
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

    # Freistellen im Hintergrund (wie MyAuktion-Admin)
    if background_tasks:
        background_tasks.add_task(_process_image_to_processed, out)

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
    # PROCESSED bevorzugen, sonst RAW
    pics = _list_image_paths(artikelnr, PROCESSED_DIR)
    if not pics:
        pics = _list_image_paths(artikelnr, RAW_DIR)

    for f in pics:
        rel = f.relative_to(BASE_DIR)
        files.append("/static/" + rel.as_posix())

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

    pics = _list_image_paths(artikelnr, PROCESSED_DIR)
    if not pics:
        pics = _list_image_paths(artikelnr, RAW_DIR)
    img_url = ""
    if pics:
        rel = pics[-1].relative_to(BASE_DIR)
        img_url = "/static/" + rel.as_posix()

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
    for k in ["titel", "beschreibung", "lagerort", "einlieferer", "mitarbeiter", "kategorie", "sortiment"]:
        if k in data and data[k] is not None:
            mj[k] = str(data[k])

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
    if name not in items:
        items.append(name)
        _save_sortimente(items)
    return {"ok": True, "items": _load_sortimente()}


@app.post("/api/admin/sortimente/rename")
def admin_sortimente_rename(request: Request, data: Dict[str, Any]):
    guard = _admin_guard(request)
    if guard:
        return guard
    old = str(data.get("old", "") or "").strip()
    new = str(data.get("new", "") or "").strip()
    if not old or not new:
        return JSONResponse({"ok": False, "error": "old/new fehlt"}, status_code=400)
    items = _load_sortimente()
    items = [new if x == old else x for x in items]
    _save_sortimente(items)
    return {"ok": True, "items": _load_sortimente()}


@app.post("/api/admin/sortimente/delete")
def admin_sortimente_delete(request: Request, data: Dict[str, Any]):
    guard = _admin_guard(request)
    if guard:
        return guard
    name = str(data.get("name", "") or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "name fehlt"}, status_code=400)
    items = [x for x in _load_sortimente() if x != name]
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
        sorti = (mj.get("sortiment", "") or "").strip()

        if sortiment and sorti.lower() != sortiment.strip().lower():
            continue
        if only_failed and (mj.get("ki_source") != "failed"):
            continue

        pics = _list_image_paths(nr)
        img_url = ""
        if pics:
            rel = pics[-1].relative_to(BASE_DIR)
            img_url = "/static/" + rel.as_posix()

        items.append({
            "artikelnr": nr,
            "titel": mj.get("titel", "") or "",
            "beschreibung": mj.get("beschreibung", "") or "",
            "kategorie": cat,
            "sortiment": sorti,
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

# ----------------------------
# Debug: rembg availability (deploy check)
# ----------------------------
@app.get("/api/debug/rembg")
def debug_rembg():
    """
    Quick check if background removal is available on the server.
    Useful on Render to verify rembg/onnxruntime + Python version.
    """
    processed_count = 0
    try:
        processed_count = sum(1 for _ in PROCESSED_DIR.glob("*.jpg"))
    except Exception:
        processed_count = 0

    return {
        "rembg_available": bool(REMBG_AVAILABLE),
        "has_session": bool(REMBG_SESSION is not None),
        "python_version": __import__("sys").version,
        "processed_dir": str(PROCESSED_DIR),
        "processed_jpg_count": processed_count,
    }

