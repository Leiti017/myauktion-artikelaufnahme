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
# - Admin-Only Admin-Flags (Token geschützt): /api/admin/articles
#
# Start:
#   python -m uvicorn server_full:app --host 0.0.0.0 --port 5050
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import base64



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
import io, json, time, csv, math, os, datetime, zipfile, hashlib, threading, random
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
# Image optimization (optional): rembg cutout -> 750x750 white bg + soft shadow
# Runs async in background after upload so operators don't have to wait.
# ----------------------------
IMG_TARGET = (750, 750)

REMBG_AVAILABLE = False
REMBG_SESSION = None
try:
    # rembg is optional (but recommended). If not installed, we just keep raw images.
    from rembg import new_session, remove  # type: ignore
    REMBG_SESSION = new_session("u2net")
    REMBG_AVAILABLE = True
except Exception as _e:
    REMBG_AVAILABLE = False
    REMBG_SESSION = None


def _processed_path_for_raw(raw_path: Path) -> Path:
    return PROCESSED_DIR / raw_path.name


def _best_image_path(raw_path: Path) -> Path:
    """Return processed image if it exists, otherwise the raw image."""
    pp = _processed_path_for_raw(raw_path)
    return pp if pp.exists() else raw_path


def _url_for_path(p: Path) -> str:
    rel = p.relative_to(BASE_DIR)
    return "/static/" + str(rel).replace("\\", "/")


def _openai_edit_garment_3d_low(cutout_rgba: Image.Image, api_key: str) -> Image.Image:
    """
    Bekleidung 3D (LOW) via OpenAI Images API.
    Input: freigestelltes PNG (RGBA). Output: RGB.
    Bei Fehler wirft Exception -> Caller kann auf Standard-Freistellung fallbacken.
    """
    # Wir schicken ein PNG + Prompt, erwarten b64_json zurück.
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Cutout als PNG Bytes
    buf = io.BytesIO()
    cutout_rgba.save(buf, format="PNG")
    buf.seek(0)

    prompt = (
        "Create a realistic studio product photo of this garment: "
        "clean 'invisible mannequin' 3D look (filled shape), natural fabric volume, "
        "white background, subtle realistic ground shadow. Keep original garment details."
    )

    files = {
        "image": ("garment.png", buf, "image/png"),
    }
    data = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": "1024x1024",
        "quality": "low",
        "background": "white",
        "response_format": "b64_json",
    }

    r = requests.post(url, headers=headers, files=files, data=data, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:400]}")

    payload = r.json()
    b64 = payload["data"][0]["b64_json"]
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _optimize_one_image(raw_path: Path, mode: str = "standard", sortiment_name: str = "") -> None:
    """
    mode:
      - "none"        -> keine Bearbeitung
      - "standard"    -> rembg Cutout + Weiß + real. Schatten (Default)
      - "clothing3d"  -> Bekleidung 3D (OpenAI) + danach Standard-Layout
    """
    try:
        mode = (mode or "standard").strip().lower()
        if mode in ("0", "false", "off", "disable", "disabled"):
            mode = "none"
        if mode in ("1", "rembg"):
            mode = "standard"

        if mode == "none":
            return

        out_path = _processed_path_for_raw(raw_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Wenn rembg nicht da ist, machen wir nur Resize auf Weiß (kein Cutout)
        if not REMBG_AVAILABLE:
            with Image.open(raw_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im = ImageOps.contain(im, IMG_TARGET)
                final = Image.new("RGB", IMG_TARGET, (255, 255, 255))
                ox = (IMG_TARGET[0] - im.size[0]) // 2
                oy = (IMG_TARGET[1] - im.size[1]) // 2
                final.paste(im, (ox, oy))
                final.save(out_path, "JPEG", quality=90)
            return

        # 1) rembg Cutout
        with Image.open(raw_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGBA")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            buf.seek(0)
            cut = remove(buf.read(), session=REMBG_SESSION)
            fg = Image.open(io.BytesIO(cut)).convert("RGBA")

        # Kanten minimal bereinigen
        alpha = fg.split()[-1].point(lambda v: 0 if v < 15 else v)
        fg.putalpha(alpha)

        # 2) Optional: Bekleidung 3D
        if mode == "clothing3d":
            api_key = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
            is_clothing = "bekleid" in (sortiment_name or "").lower()
            if api_key and is_clothing:
                try:
                    out_rgb = _openai_edit_garment_3d_low(fg, api_key=api_key)
                    # zurück nach RGBA, damit Layout-Teil unten gleich bleibt
                    fg = out_rgb.convert("RGBA")
                except Exception as e:
                    print("[IMG] clothing3d failed -> fallback standard:", raw_path.name, e)
            else:
                # Wenn kein Key oder nicht Bekleidung -> Standard
                pass

        # 3) Standard Layout: Weiß + real. Bodenschatten
        tw, th = IMG_TARGET
        canvas = Image.new("RGBA", (tw, th), (255, 255, 255, 255))

        # Größe so skalieren, dass Luft bleibt
        fg2 = ImageOps.contain(fg, (int(tw * 0.86), int(th * 0.86)))

        # Schatten: Ellipse unter dem Produkt, weichgezeichnet
        shadow = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        ell_w = int(fg2.size[0] * 0.78)
        ell_h = max(18, int(fg2.size[1] * 0.12))
        ell_x0 = (tw - ell_w) // 2
        # leicht weiter unten, damit er "am Boden" wirkt
        ell_y0 = min(th - ell_h - 10, int((th + fg2.size[1]) / 2) - ell_h // 3)
        draw.ellipse(
            (ell_x0, ell_y0, ell_x0 + ell_w, ell_y0 + ell_h),
            fill=(0, 0, 0, 85),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(6, tw // 80)))
        canvas.alpha_composite(shadow)

        x = (tw - fg2.size[0]) // 2
        y = (th - fg2.size[1]) // 2
        canvas.alpha_composite(fg2, (x, y))

        final = canvas.convert("RGB")
        final.save(out_path, "JPEG", quality=90)
    except Exception as e:
        print("[IMG] optimize error:", raw_path.name, e)

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

def _csv_row_from_meta(artikelnr: str, meta: Dict[str, Any]) -> list[str]:
    """Create one CSV row aligned with CSV_FIELDS."""
    art = str(artikelnr)
    bezeichnung = str(meta.get("titel") or "")
    beschr = str(meta.get("beschreibung") or "")
    menge = int(meta.get("menge") or 1)
    preis = _format_rufpreis(meta.get("rufpreis", 0))
    ladenpreis = _format_ladenpreis(meta.get("retail_price", 0))
    lagerort = str(meta.get("lagerort") or "")
    lagerstand = int(meta.get("lagerstand") or 1)
    uebernehmen = int(meta.get("uebernehmen") or 1)
    sortiment = str(meta.get("sortiment_id") or meta.get("sortiment") or "")
    einl_id = str(meta.get("einlieferer_id") or meta.get("einlieferer") or "")
    angel = str(meta.get("angeliefert") or "")
    betr = str(meta.get("betriebsmittel") or "")
    mitarb = str(meta.get("mitarbeiter") or "")

    return [
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
    sortiment = str(meta.get("sortiment_id") or meta.get("sortiment") or "")
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
# CSV -> JSON Helpers (for restore)
# ----------------------------
def _to_float_price(val) -> float:
    try:
        s = str(val or "").strip()
        if not s:
            return 0.0
        s = s.replace("€", "").replace(" ", "").replace(",", ".")
        return float(s)
    except Exception:
        return 0.0

def _to_int(val, default: int = 0) -> int:
    try:
        s = str(val or "").strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default

def _find_cover_image(artikelnr: str) -> str:
    a = str(artikelnr or "").strip()
    if not a:
        return ""
    p0 = RAW_DIR / f"{a}.jpg"
    if p0.exists():
        return p0.name
    for i in range(1, 50):
        pi = RAW_DIR / f"{a}_{i}.jpg"
        if pi.exists():
            return pi.name
    return ""
# ----------------------------
# Rebuild JSON articles from export CSV (when Drive has only CSV+JPG)
# ----------------------------
def _normalize_key(s: str) -> str:
    return "".join(ch for ch in str(s or "").strip().lower() if ch.isalnum())

def _csv_pick_delimiter(sample: str) -> str:
    # Try sniffer first, then fallbacks
    try:
        import csv as _csv
        return _csv.Sniffer().sniff(sample, delimiters=";,\t,|").delimiter
    except Exception:
        # heuristics: prefer ';' if present
        for d in [";", ",", "\t", "|"]:
            if d in sample:
                return d
        return ";"

def _row_get(row: dict, *names: str) -> str:
    # row keys already normalized
    for n in names:
        k=_normalize_key(n)
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    return ""

def _rebuild_meta_from_export_csv_if_missing() -> dict:
    """
    Create RAW_DIR/<artikelnr>.json for rows in EXPORT_CSV if they don't exist.
    Works even if the CSV is exported from Google Sheets.
    """
    if not EXPORT_CSV.exists():
        return {"ok": False, "error": "EXPORT_CSV missing", "path": str(EXPORT_CSV)}

    created = 0
    skipped = 0
    errors = 0

    try:
        raw = EXPORT_CSV.read_text("utf-8-sig", errors="ignore")
    except Exception:
        raw = EXPORT_CSV.read_text("utf-8", errors="ignore")

    # detect delimiter using first lines
    head = "\n".join(raw.splitlines()[:10])
    delim = _csv_pick_delimiter(head)

    import csv as _csv
    reader = _csv.reader(raw.splitlines(), delimiter=delim)
    rows = list(reader)
    if not rows:
        return {"ok": False, "error": "empty csv"}

    headers = [ _normalize_key(h) for h in rows[0] ]
    # build dict rows
    for vals in rows[1:]:
        if not any(str(v).strip() for v in vals):
            continue
        d = {}
        for i,h in enumerate(headers):
            if not h:
                continue
            d[h] = vals[i] if i < len(vals) else ""

        art = _row_get(d, "ArtikelNr", "Artikelnr", "Artikelnummer", "Artikel Nummer", "Artikel")
        art = str(art).strip()
        if not art:
            continue

        p = RAW_DIR / f"{art}.json"
        if p.exists():
            skipped += 1
            continue

        try:
            mj = _default_meta(art)

            mj["titel"] = _row_get(d, "Bezeichnung", "Titel", "Name")
            mj["beschreibung"] = _row_get(d, "Beschreibung", "Text")
            mj["lagerort"] = _row_get(d, "Lagerort", "Ort", "Lager Ort")
            mj["mitarbeiter"] = _row_get(d, "Mitarbeiter", "User", "Account")

            mj["rufpreis"] = _to_float_price(_row_get(d, "Preis", "Rufpreis"))
            mj["retail_price"] = _to_float_price(_row_get(d, "Ladenpreis", "Listenpreis", "Retail", "UVP"))

            mj["menge"] = _to_int(_row_get(d, "Menge"), 1)
            mj["lagerstand"] = _to_int(_row_get(d, "Lagerstand"), mj["menge"])
            mj["uebernehmen"] = _to_int(_row_get(d, "Uebernehmen", "Übernehmen"), 1)

            sort_raw = _row_get(d, "Sortiment", "Kategorie")
            if sort_raw:
                mj["sortiment"] = sort_raw
                mj["sortiment_name"] = sort_raw
                try:
                    mj["sortiment_id"] = int(str(sort_raw).strip())
                except Exception:
                    pass

            mj["einlieferer_id"] = _row_get(d, "Einlieferer-ID", "Einlieferer", "EinliefererID")
            mj["angeliefert"] = _row_get(d, "Angeliefert")
            mj["betriebsmittel"] = _row_get(d, "Betriebsmittel")

            # timestamps: try AufnahmeDatum or Datum
            dat = _row_get(d, "AufnahmeDatum", "Aufnahmedatum", "Datum")
            if dat:
                # accept DD.MM.YYYY
                try:
                    import datetime as _dt
                    if "." in dat:
                        dt=_dt.datetime.strptime(dat.strip(), "%d.%m.%Y")
                    else:
                        dt=_dt.datetime.strptime(dat.strip(), "%Y-%m-%d")
                    ts=int(dt.replace(tzinfo=_dt.timezone.utc).timestamp())
                    mj["created_at"]=ts
                    mj["updated_at"]=ts
                except Exception:
                    pass

            mj["cover"] = _find_cover_image(art)

            _save_meta_json(art, mj)
            created += 1
        except Exception:
            errors += 1

    try:
        _ensure_export_csv_exists()
    except Exception:
        pass

    return {"ok": True, "created": created, "skipped": skipped, "errors": errors, "delimiter": delim}
# ----------------------------
# Google Drive Backup (Render Free, ohne Disk)
# ----------------------------
GDRIVE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GDRIVE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GDRIVE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN", "").strip()
GDRIVE_FOLDER_NAME = (os.getenv("DRIVE_FOLDER_NAME", "MyAuktion-Backups") or "MyAuktion-Backups").strip()
GDRIVE_RETENTION_DAYS = int(os.getenv("DRIVE_RETENTION_DAYS", "30") or "30")
GDRIVE_FOLDER_ID = (os.getenv("DRIVE_FOLDER_ID", "") or "").strip()

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
    # Prefer explicit folder id if provided (lets you target an existing folder reliably)
    if _gdrive_cache["folder_id"]:
        return _gdrive_cache["folder_id"]
    if GDRIVE_FOLDER_ID:
        _gdrive_cache["folder_id"] = GDRIVE_FOLDER_ID
        return GDRIVE_FOLDER_ID
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

        # Upload/Update Meta JSON for this article (so Restore funktioniert)
        meta_path = RAW_DIR / f"{artikelnr}.json"
        if meta_path.exists():
            try:
                _gdrive_upload_bytes(token, folder_id, meta_path.name, meta_path.read_bytes(), "application/json")
            except Exception:
                pass

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
def _gdrive_list_files(token: str, folder_id: str) -> list[dict]:
    """List files in a Drive folder."""
    files: list[dict] = []
    page_token: str | None = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,size)",
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            headers=_gdrive_headers(token),
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        j = r.json()
        files.extend(j.get("files", []) or [])
        page_token = j.get("nextPageToken")
        if not page_token:
            break
    return files

def _gdrive_download_file(token: str, file_id: str, mime_type: str | None = None) -> bytes:
    """Download a file. If it's a Google Workspace file (e.g. Sheets), export as CSV."""
    if mime_type and mime_type.startswith("application/vnd.google-apps."):
        # export to CSV (works for spreadsheets)
        r = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
            headers=_gdrive_headers(token),
            params={"mimeType": "text/csv"},
            timeout=120,
        )
        r.raise_for_status()
        return r.content

    r = requests.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}",
        headers=_gdrive_headers(token),
        params={"alt": "media"},
        timeout=120,
    )
    r.raise_for_status()
    return r.content

def _gdrive_restore_all() -> dict:
    """Restore RAW_DIR + export CSV from Google Drive (best effort).
    Supports folders containing only CSV + JPG (rebuild JSON from CSV afterwards).
    """
    if not _gdrive_enabled():
        return {"ok": False, "error": "gdrive disabled"}
    try:
        token = _gdrive_access_token()
        folder_id = _gdrive_get_folder_id(token)
        files = _gdrive_list_files(token, folder_id)

        restored_json = 0
        restored_jpg = 0
        restored_csv = 0

        # 1) Restore JSON + JPG (if present)
        for f in files:
            name = str(f.get("name") or "")
            fid = str(f.get("id") or "")
            if not name or not fid:
                continue

            if name.lower().endswith(".json"):
                out = RAW_DIR / name
                if not out.exists():
                    out.write_bytes(_gdrive_download_file(token, fid, str(f.get('mimeType') or '')))
                    restored_json += 1
                continue

            if name.lower().endswith(".jpg"):
                out = RAW_DIR / name
                if not out.exists():
                    out.write_bytes(_gdrive_download_file(token, fid, str(f.get('mimeType') or '')))
                    restored_jpg += 1
                continue

        # 2) Restore an export CSV if local missing: pick newest among artikel_export*.csv
        if not EXPORT_CSV.exists():
            csv_candidates = []
            for f in files:
                name = str(f.get("name") or "")
                if not name:
                    continue
                low = name.lower()
                if low.startswith("artikel_export") and low.endswith(".csv"):
                    csv_candidates.append(f)
            if csv_candidates:
                def _mtime_key(x):
                    # modifiedTime is RFC3339 - lexicographic sort works
                    return str(x.get("modifiedTime") or "")
                csv_candidates.sort(key=_mtime_key, reverse=True)
                best = csv_candidates[0]
                EXPORT_CSV.write_bytes(_gdrive_download_file(token, str(best.get("id")), str(best.get("mimeType") or "")))
                restored_csv = 1

        # ensure local export exists and consistent
        _ensure_export_csv_exists()

        # NEW: if only CSV+JPG exist, rebuild missing JSON articles from CSV
        try:
            res_rebuild = _rebuild_meta_from_export_csv_if_missing()
        except Exception as e:
            res_rebuild = {"ok": False, "error": str(e)}

        return {
            "ok": True,
            "folder_id": folder_id,
            "restored_json": restored_json,
            "restored_jpg": restored_jpg,
            "restored_csv": restored_csv,
            "rebuild_from_csv": res_rebuild,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _maybe_restore_from_gdrive_on_start():
    if os.getenv("GDRIVE_RESTORE_ON_START", "").strip() not in {"1","true","yes","on"}:
        return
    def _run():
        try:
            _gdrive_restore_all()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

_maybe_restore_from_gdrive_on_start()


@app.get("/api/gdrive_debug")
def gdrive_debug():
    """Debug: list files visible in the configured Drive folder."""
    if not _gdrive_enabled():
        return {"ok": False, "error": "gdrive disabled (missing env vars)"}
    try:
        token = _gdrive_access_token()
        folder_id = _gdrive_get_folder_id(token)
        files = _gdrive_list_files(token, folder_id)
        names = [str(f.get("name") or "") for f in files][:30]
        return {"ok": True, "folder_id": folder_id, "file_count": len(files), "sample_names": names}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/gdrive_restore")
def gdrive_restore():
    """Manual restore trigger."""
    return _gdrive_restore_all()


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



def _best_image_url(p: Path) -> str:
    """Return URL for processed version if exists, else raw."""
    try:
        cand = PROCESSED_DIR / p.name
        use = cand if cand.exists() else p
        rel = use.relative_to(BASE_DIR)
        return "/static/" + str(rel).replace("\\", "/")
    except Exception:
        try:
            rel = p.relative_to(BASE_DIR)
            return "/static/" + str(rel).replace("\\", "/")
        except Exception:
            return ""

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
        "einlieferer_id": "",
        "mitarbeiter": "",
        "lagerstand": 1,
        "uebernehmen": 1,
        "angeliefert": "0",
        "betriebsmittel": "",
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
            "Sortiment": str(d.get("sortiment_id") or d.get("sortiment") or ""),
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



# ----------------------------
# Bild-Optimierung (Freistellen + weißer Studio-Hintergrund + Shadow)
# ----------------------------
def _run_image_background(raw_path: Path) -> None:
    try:
        from rembg import remove
        from PIL import ImageFilter, ImageDraw
    except Exception as e:
        print("[BG-IMG] rembg/pillow missing:", e)
        return

    try:
        im = Image.open(raw_path)
        im = ImageOps.exif_transpose(im).convert("RGBA")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        cut = remove(buf.getvalue())
        fg = Image.open(io.BytesIO(cut)).convert("RGBA")

        a = fg.split()[-1].point(lambda v: 0 if v < 15 else v)
        fg.putalpha(a)

        target_w, target_h = 750, 750
        fg2 = ImageOps.contain(fg, (int(target_w * 0.86), int(target_h * 0.86)))

        base = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))

        shadow = Image.new("L", (target_w, target_h), 0)
        draw = ImageDraw.Draw(shadow)

        w, h = fg2.size
        shadow_w = int(w * 0.72)
        shadow_h = max(8, int(h * 0.16))
        cx = target_w // 2
        cy = int(target_h * 0.80)

        sx0 = cx - shadow_w // 2
        sx1 = cx + shadow_w // 2
        sy0 = cy - shadow_h // 2
        sy1 = cy + shadow_h // 2

        draw.ellipse([sx0, sy0, sx1, sy1], fill=200)
        shadow = shadow.filter(ImageFilter.GaussianBlur(14))
        shadow_rgba = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        shadow_rgba.paste((0, 0, 0, 110), mask=shadow)

        base.alpha_composite(shadow_rgba)

        ox = (target_w - w) // 2
        oy = int(target_h * 0.80) - h
        base.alpha_composite(fg2, (ox, oy))

        out = base.convert("RGB")
        out_path = PROCESSED_DIR / raw_path.name
        out.save(out_path, "JPEG", quality=90)
    except Exception as e:
        print("[BG-IMG] failed:", e)

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

def _generate_sortiment_id(existing: set[int]) -> int:
    """Generate a short numeric Sortiment-ID (4-6 digits) that is unique."""
    # prefer 4 digits for readability; fall back to 5/6 if crowded
    for digits in (4, 5, 6):
        lo, hi = 10 ** (digits - 1), (10 ** digits) - 1
        for _ in range(5000):
            cand = random.randint(lo, hi)
            if cand not in existing:
                return cand
    # last resort
    cand = int(time.time()) % 1000000
    while cand in existing:
        cand = (cand + 1) % 1000000
    return cand

def _normalize_sortimente(raw: Any) -> list[dict]:
    """Accept legacy formats and normalize to: [{id:int, name:str}, ...]."""
    items: list[dict] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()

    if isinstance(raw, list):
        # Legacy: ["(22) Drogerie", "..."]
        if raw and all(isinstance(x, str) for x in raw):
            for name in raw:
                n = str(name).strip()
                if not n or n.lower() in seen_names:
                    continue
                sid = _generate_sortiment_id(seen_ids)
                items.append({"id": sid, "name": n})
                seen_ids.add(sid)
                seen_names.add(n.lower())
            return items

        # New: [{"id":2349,"name":"..."}, ...]
        if raw and all(isinstance(x, dict) for x in raw):
            for x in raw:
                try:
                    sid = int(x.get("id"))
                except Exception:
                    sid = None
                n = str(x.get("name") or "").strip()
                if not n:
                    continue
                if sid is None or sid in seen_ids:
                    sid = _generate_sortiment_id(seen_ids)
                if n.lower() in seen_names:
                    continue
                items.append({"id": sid, "name": n})
                seen_ids.add(sid)
                seen_names.add(n.lower())
            return items

    # fallback
    for name in _default_sortimente():
        n = str(name).strip()
        if not n or n.lower() in seen_names:
            continue
        sid = _generate_sortiment_id(seen_ids)
        items.append({"id": sid, "name": n})
        seen_ids.add(sid)
        seen_names.add(n.lower())
    return items

def _load_sortimente() -> list[dict]:
    """Return list of sortiments: [{id:int, name:str}, ...]."""
    if SORTIMENTE_JSON.exists():
        try:
            data = json.loads(SORTIMENTE_JSON.read_text("utf-8"))
            items = _normalize_sortimente(data)
            # auto-migrate legacy file to new format once
            if isinstance(data, list) and data and isinstance(data[0], str):
                _save_sortimente(items)
            return items
        except Exception:
            pass
    items = _normalize_sortimente(_default_sortimente())
    try:
        _save_sortimente(items)
    except Exception:
        pass
    return items

def _save_sortimente(items: list[dict]) -> None:
    cleaned: list[dict] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for x in (items or []):
        if not isinstance(x, dict):
            continue
        n = str(x.get("name") or "").strip()
        if not n:
            continue
        try:
            sid = int(x.get("id"))
        except Exception:
            sid = None
        if sid is None or sid in seen_ids:
            sid = _generate_sortiment_id(seen_ids)
        if n.lower() in seen_names:
            continue
        cleaned.append({"id": sid, "name": n})
        seen_ids.add(sid)
        seen_names.add(n.lower())
    SORTIMENTE_JSON.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), "utf-8")

def _sortiment_name_by_id(sortiment_id: Any) -> str:
    try:
        sid = int(sortiment_id)
    except Exception:
        return ""
    for x in _load_sortimente():
        if int(x.get("id")) == sid:
            return str(x.get("name") or "")
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

        pics = _list_image_paths(nr, directory=RAW_DIR)
        img_url = _url_for_path(_best_image_path(pics[-1])) if pics else ""

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
    # Gesamtartikel ist jetzt im Admin integriert
    return RedirectResponse(url="/admin", status_code=302)


@app.api_route("/api/health", methods=["GET","HEAD"])
def health():
    return {"ok": True}


@app.get("/api/sortimente")
def sortimente_public():
    return {"ok": True, "items": _load_sortimente()}

@app.get("/api/recent_articles")
def recent_articles(mitarbeiter: str = "", limit: int = 12):
    """Letzte aufgenommene Artikel – optional pro Mitarbeiter."""
    limit = max(1, min(int(limit or 12), 50))
    who = (mitarbeiter or "").strip().lower()
    items = []
    for p in RAW_DIR.glob("*.json"):
        try:
            mj = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        m = str(mj.get("mitarbeiter") or "").strip().lower()
        if who and m != who:
            continue
        nr = str(mj.get("artikelnr", p.stem))
        items.append({
            "artikelnr": nr,
            "titel": str(mj.get("titel") or ""),
            "updated_at": int(mj.get("updated_at", 0) or 0),
            "created_at": int(mj.get("created_at", 0) or 0),
            "sortiment_id": str(mj.get("sortiment_id") or ""),
            "sortiment_name": str(mj.get("sortiment_name") or mj.get("sortiment") or ""),
        })
    items.sort(key=lambda x: (x.get("updated_at", 0), x.get("created_at", 0)), reverse=True)
    return {"ok": True, "items": items[:limit]}



@app.get("/api/articles")
def articles_api(date_from: str | None = None, date_to: str | None = None, sortiment_id: str | None = None, sortiment_name: str | None = None, limit: int | None = None):
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


    # Sortiment Filter (entweder per ID oder per Name)
    if sortiment_name and not sortiment_id:
        # Name -> ID suchen
        sname = str(sortiment_name).strip().lower()
        for s in _load_sortimente():
            if str(s.get("name") or "").strip().lower() == sname:
                sortiment_id = str(s.get("id"))
                break

    if sortiment_id:
        sid = str(sortiment_id).strip()
        items = [x for x in items if str(x.get("sortiment_id") or "").strip() == sid]

    # sortiment_name in Response ergänzen (für UI)
    for x in items:
        if not x.get("sortiment_name"):
            x["sortiment_name"] = _sortiment_name_by_id(x.get("sortiment_id"))

    if limit:
        try:
            lim = max(1, int(limit))
            items = items[:lim]
        except Exception:
            pass

    return {"ok": True, "items": items}


@app.get("/api/export.csv")
def export_csv(from_nr: str | None = None, to_nr: str | None = None, sortiment_id: str | None = None, sortiment_name: str | None = None):
    """
    Export CSV (UTF-8 mit BOM für Excel). Optionaler Bereich:
      /api/export.csv?from_nr=123458&to_nr=123480
    """
    _rebuild_csv_export()

    # Kein Filter -> Datei direkt ausliefern
    if not from_nr and not to_nr and not sortiment_id and not sortiment_name:
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
    
    # Sortiment Filter: Name -> ID (falls ID nicht direkt übergeben wurde)
    if sortiment_name and not sortiment_id:
        sname = str(sortiment_name).strip().lower()
        for s in _load_sortimente():
            if str(s.get("name") or "").strip().lower() == sname:
                sortiment_id = str(s.get("id"))
                break

    rows: list[dict] = []
    for jf in sorted(RAW_DIR.glob("*.json")):
        art = jf.stem
        if not _in_range(art):
            continue
        meta = _load_meta_json(art)
        if sortiment_id:
            if str(meta.get("sortiment_id") or "").strip() != str(sortiment_id).strip():
                continue

        # WICHTIG: Immer per Feldname mappen (nie per Reihenfolge), sonst verschieben sich Spalten.
        rows.append({
            "ArtikelNr": art,
            "Bezeichnung": str(meta.get("titel", "") or ""),
            "Beschreibung": str(meta.get("beschreibung", "") or ""),
            "Menge": int(meta.get("menge", 1) or 1),
            "Preis": _format_rufpreis(meta.get("rufpreis", 0.0) or 0.0),
            "Ladenpreis": _format_ladenpreis(meta.get("retail_price", 0) or 0),
            "Lagerort": str(meta.get("lagerort", "") or ""),
            "Lagerstand": int(meta.get("lagerstand", 1) or 1),
            "Uebernehmen": int(meta.get("uebernehmen", 1) or 1),
            "Sortiment": str(meta.get("sortiment_id") or meta.get("sortiment") or ""),
            "Einlieferer-ID": str(meta.get("einlieferer_id", meta.get("einlieferer", "")) or ""),
            "Angeliefert": str(meta.get("angeliefert", "") or ""),
            "Betriebsmittel": str(meta.get("betriebsmittel", "") or ""),
            "Mitarbeiter": str(meta.get("mitarbeiter", "") or ""),
        })

    bio = io.StringIO()
    w = csv.DictWriter(bio, fieldnames=CSV_FIELDS, delimiter=";", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in CSV_FIELDS})

    csv_text = bio.getvalue()
    # UTF-8 BOM for Excel
    csv_bytes = ('\ufeff' + csv_text).encode('utf-8')
    return Response(content=csv_bytes, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=artikel_export.csv'})



# ----------------------------
# Bilder ZIP Export (Admin, Von/Bis ArtikelNr)
# ----------------------------

def _best_image_bytes_for_zip(raw_path: Path) -> tuple[str, bytes] | None:
    """Prefer processed image if present, else raw. Returns (filename, bytes)."""
    try:
        stem = raw_path.stem
        cand = PROCESSED_DIR / raw_path.name
        if cand.exists():
            return cand.name, cand.read_bytes()
        for pf in PROCESSED_DIR.glob(stem + ".*"):
            if pf.exists():
                return pf.name, pf.read_bytes()
        return raw_path.name, raw_path.read_bytes()
    except Exception:
        return None

def _images_for_art(artikelnr: str):
    """Return list of image Paths for article.

    Prefer processed images when available, but keep the raw naming scheme/order.
    """
    art = str(artikelnr).strip()
    raw = _list_image_paths(art, directory=RAW_DIR)
    return [_best_image_path(p) for p in raw]

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
def export_images_zip(request: Request, from_nr: str | None = None, to_nr: str | None = None, sortiment_id: str | None = None, sortiment_name: str | None = None):
    """ZIP mit allen Bildern (RAW) für Artikel (optional Von/Bis ArtikelNr).
    Admin-Token geschützt (X-Admin-Token oder ?token=...).
    """
    guard = _admin_guard(request)
    if guard:
        return guard

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # Artikel über JSONs bestimmen (damit nur echte Positionen drin sind)
        
            # Sortiment Filter: Name -> ID (falls ID nicht direkt übergeben wurde)
            if sortiment_name and not sortiment_id:
                sname = str(sortiment_name).strip().lower()
                for s in _load_sortimente():
                    if str(s.get("name") or "").strip().lower() == sname:
                        sortiment_id = str(s.get("id"))
                        break

            for jf in sorted(RAW_DIR.glob("*.json")):
                art = jf.stem
                if (from_nr or to_nr) and not _in_range_art(art, from_nr, to_nr):
                    continue
                if sortiment_id:
                    meta = _load_meta_json(art)
                    if str(meta.get("sortiment_id") or "").strip() != str(sortiment_id).strip():
                        continue
                for p in _images_for_art(art):
                    try:
                        # in zip nur Dateiname, weil schon eindeutig (ART...jpg)
                        tmp = _best_image_bytes_for_zip(p)
                        if tmp:
                            z.writestr(tmp[0], tmp[1])
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
    img_mode: str = Form(""),
    sortiment_name: str = Form(""),
    bg_remove: str = Form("0"),
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
        # Bild-Optimierung läuft IMMER im Hintergrund (Standard), außer man schaltet es explizit aus.
        mode = (img_mode or "").strip().lower()

        # Backward-Compat (alte Clients): bg_remove=1 -> Standard
        if not mode and str(bg_remove or "").strip() in ("1", "true", "on", "yes"):
            mode = "standard"

        # Default: Standard aktiv
        if not mode:
            mode = "standard"

        # Normalisieren
        if mode in ("0", "none", "off", "false", "disabled", "disable"):
            mode = "none"
        if mode in ("1", "rembg", "auto", "default", "on", "true", "yes"):
            mode = "standard"

        if mode not in ("none", "standard", "clothing3d"):
            mode = "standard"

        if mode != "none":
            background_tasks.add_task(_optimize_one_image, out, mode, sortiment_name or "")
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
        files.append(_url_for_path(_best_image_path(f)))

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
    img_url = _url_for_path(_best_image_path(pics[-1])) if pics else ""

    return {
        "ok": True,
        "artikelnr": artikelnr,
        "titel": mj.get("titel", "") or "",
        "beschreibung": mj.get("beschreibung", "") or "",
        "kategorie": mj.get("kategorie", "") or "",
        "retail_price": mj.get("retail_price", 0.0) or 0.0,
        "rufpreis": mj.get("rufpreis", 0.0) or 0.0,
        "lagerort": mj.get("lagerort", "") or "",
        # add missing fields so edit-mode can fill everything correctly
        "menge": int(mj.get("menge", 1) or 1),
        "lagerstand": int(mj.get("lagerstand", 1) or 1),
        "uebernehmen": int(mj.get("uebernehmen", 1) or 1),
        "angeliefert": mj.get("angeliefert", "") or "",
        "betriebsmittel": mj.get("betriebsmittel", "") or "",
        "sortiment": mj.get("sortiment", "") or "",
        "sortiment_id": mj.get("sortiment_id", "") or "",
        "sortiment_name": mj.get("sortiment_name", "") or "",
        "einlieferer": mj.get("einlieferer", "") or "",
        "einlieferer_id": mj.get("einlieferer_id", "") or "",
        "mitarbeiter": mj.get("mitarbeiter", "") or "",
        "reviewed": bool(mj.get("reviewed", False)),
        "ki_source": mj.get("ki_source", "") or "",
        "ki_last_error": mj.get("ki_last_error", "") or "",
        "last_image": mj.get("last_image", "") or "",
        "cover": mj.get("cover", "") or "",
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

    # Sortiment: Frontend sendet jetzt primär die ID (value), Name wird serverseitig ergänzt.
    sort_id = data.get("sortiment_id", None)
    if sort_id is None:
        # backward compat: "sortiment" kann entweder Name (alt) oder ID (neu) sein
        sort_id = data.get("sortiment", None)

    sort_name = ""
    sort_id_int: Optional[int] = None
    if sort_id is not None and str(sort_id).strip() != "":
        try:
            sort_id_int = int(str(sort_id).strip())
            sort_name = _sortiment_name_by_id(sort_id_int)
        except Exception:
            # treat as legacy name
            sort_name = str(sort_id).strip()
            sort_id_int = None

    # allow explicit name (optional)
    if data.get("sortiment_name") is not None:
        sort_name = str(data.get("sortiment_name") or "").strip() or sort_name

    if sort_id_int is not None:
        mj["sortiment_id"] = sort_id_int
    if sort_name:
        mj["sortiment_name"] = sort_name
        mj["sortiment"] = sort_name  # legacy field kept for older tools/UI
    elif "sortiment" in data and data["sortiment"] is not None:
        # legacy name-only
        mj["sortiment"] = str(data["sortiment"]).strip()

    # Settings-Felder (für CSV Export)
    if "menge" in data: mj["menge"] = int(data.get("menge") or 1)
    if "lagerstand" in data: mj["lagerstand"] = int(data.get("lagerstand") or 1)
    if "uebernehmen" in data: mj["uebernehmen"] = int(data.get("uebernehmen") or 1)
    if "einlieferer_id" in data: mj["einlieferer_id"] = str(data.get("einlieferer_id") or "")
    if "angeliefert" in data: mj["angeliefert"] = ("0" if str(data.get("angeliefert") or "").strip()=="" else str(data.get("angeliefert")).strip())
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
    raw_id = str(data.get("id", "") or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "name fehlt"}, status_code=400)

    items = _load_sortimente()

    # already exists (case-insensitive)
    if any(str(x.get("name","")).strip().lower() == name.lower() for x in items):
        return {"ok": True, "items": _load_sortimente()}

    existing_ids = {int(x.get("id")) for x in items if str(x.get("id","")).strip().isdigit()}

    # optional custom ID
    sid: int
    if raw_id:
        if not raw_id.isdigit():
            return JSONResponse({"ok": False, "error": "id muss eine Zahl sein"}, status_code=400)
        sid = int(raw_id)
        if sid <= 0:
            return JSONResponse({"ok": False, "error": "id muss > 0 sein"}, status_code=400)
        if sid in existing_ids:
            return JSONResponse({"ok": False, "error": "id bereits vergeben"}, status_code=409)
    else:
        sid = _generate_sortiment_id(existing_ids)

    items.append({"id": sid, "name": name})
    _save_sortimente(items)
    return {"ok": True, "item": {"id": sid, "name": name}, "items": _load_sortimente()}


@app.post("/api/admin/sortimente/rename")
def admin_sortimente_rename(request: Request, data: Dict[str, Any]):
    guard = _admin_guard(request)
    if guard:
        return guard

    # preferred: by id
    sortiment_id = data.get("id", None)
    new_name = str(data.get("new", "") or data.get("name", "") or "").strip()

    # legacy: old/new by name
    old_name = str(data.get("old", "") or "").strip()

    if not new_name:
        return JSONResponse({"ok": False, "error": "new/name fehlt"}, status_code=400)

    items = _load_sortimente()
    changed = False

    if sortiment_id is not None and str(sortiment_id).strip() != "":
        try:
            sid = int(sortiment_id)
        except Exception:
            return JSONResponse({"ok": False, "error": "id ungültig"}, status_code=400)
        for x in items:
            if int(x.get("id")) == sid:
                x["name"] = new_name
                changed = True
                break
    elif old_name:
        for x in items:
            if str(x.get("name","")).strip().lower() == old_name.lower():
                x["name"] = new_name
                changed = True
                break
    else:
        return JSONResponse({"ok": False, "error": "id oder old fehlt"}, status_code=400)

    if changed:
        _save_sortimente(items)
    return {"ok": True, "items": _load_sortimente()}



@app.post("/api/admin/sortimente/delete")
def admin_sortimente_delete(request: Request, data: Dict[str, Any]):
    guard = _admin_guard(request)
    if guard:
        return guard

    sortiment_id = data.get("id", None)
    name = str(data.get("name", "") or "").strip()  # legacy

    items = _load_sortimente()
    if sortiment_id is not None and str(sortiment_id).strip() != "":
        try:
            sid = int(sortiment_id)
        except Exception:
            return JSONResponse({"ok": False, "error": "id ungültig"}, status_code=400)
        items = [x for x in items if int(x.get("id")) != sid]
    elif name:
        items = [x for x in items if str(x.get("name","")).strip().lower() != name.lower()]
    else:
        return JSONResponse({"ok": False, "error": "id oder name fehlt"}, status_code=400)

    _save_sortimente(items)
    return {"ok": True, "items": _load_sortimente()}


@app.get("/api/admin/articles")
def admin_articles(request: Request, sortiment: str = "", sortiment_id: str = "", only_failed: int = 0):
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
        sorti_name = (mj.get("sortiment_name") or mj.get("sortiment") or "").strip()
        sorti_id = str(mj.get("sortiment_id") or "").strip()

        # Filter: bevorzugt nach ID, fallback nach Name (legacy)
        if sortiment_id:
            if sorti_id != str(sortiment_id).strip():
                continue
        elif sortiment:
            if sorti_name.lower() != sortiment.strip().lower():
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
            "sortiment": sorti_name,
            "sortiment_id": sorti_id,
            "sortiment_name": sorti_name,
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