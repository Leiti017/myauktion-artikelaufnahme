# [DATEI: ki_engine_openai.py]
# MyAuktion – Vision-KI via OpenAI (Multi-Image Fusion)
# Bezeichnung: MARKE PRODUKT TYP
# Beschreibung: Details (ml/MHD/Größe/Zustand/Zubehör), wenn nicht sichtbar: nicht erfinden.

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def _b64_data_url(image_path: Union[str, Path]) -> str:
    p = Path(image_path)
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    mime = "image/jpeg"
    if p.suffix.lower() == ".png":
        mime = "image/png"
    return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Optional[dict]:
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _to_float(v) -> float:
    try:
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace(",", ".")
        s = re.sub(r"[^0-9.]+", "", s)
        return float(s or 0.0)
    except Exception:
        return 0.0


def generate_meta(image_path: str, art_id: str, existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return generate_meta_multi([image_path], art_id, existing=existing)


def generate_meta_multi(image_paths: List[str], art_id: str, existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start = time.time()
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY fehlt")

    existing = existing or {}
    existing_title = (existing.get("titel") or existing.get("title") or "").strip()
    existing_desc = (existing.get("beschreibung") or existing.get("description") or "").strip()
    existing_cat = (existing.get("kategorie") or existing.get("category") or "").strip()

    want_ml = not re.search(r"\b\d{1,4}\s?ml\b", existing_title + " " + existing_desc, re.I)
    want_mhd = not re.search(r"(MHD|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{1,2}[./]\d{2,4}\b)", existing_title + " " + existing_desc, re.I)
    want_size = not re.search(r"\b\d{1,3}\s?cm\b", existing_title + " " + existing_desc, re.I)

    system = (
        "Du bist ein extrem präziser Assistent für die Artikelaufnahme (Auktion). "
        "Antworte NUR mit gültigem JSON, ohne zusätzlichen Text."
    )

    user = f"""
Analysiere ALLE Fotos (gehören zum selben Artikel). Nutze Foto 2/3 für Details.

Bezeichnung MUSS sein: "MARKE PRODUKT TYP"
- MARKE = Brand/Hersteller (wenn sichtbar)
- PRODUKT = Produktbezeichnung (kurz)
- TYP = Typenbezeichnung/Variante/Modell (z.B. Duft/Sorte/Nummer/Serie/Modell)
- KEINE Menge/MHD/Zustand in der Bezeichnung.

Beschreibung:
- schreibe Details (ml, MHD, Maße cm, Farbe/Größe, Zubehör, Zustand).
- Wenn MHD bei Lebensmittel NICHT sichtbar: schreibe "MHD: nicht ersichtlich".
- Erfinde keine Mengen/MHD.

Vorhandene Infos (falls vorhanden):
existing_title: {existing_title}
existing_desc: {existing_desc}
existing_category: {existing_cat}

Fehlende Fokus-Punkte:
ml fehlt? {str(want_ml).lower()}
MHD fehlt? {str(want_mhd).lower()}
Maße (cm) fehlt? {str(want_size).lower()}

Gib exakt dieses JSON-Schema zurück:
{{
  "brand": "",
  "product": "",
  "type": "",
  "title": "",
  "description": "",
  "category": "",
  "volume_ml": null,
  "mhd": "",
  "size": "",
  "retail_price_eur": 0
}}
"""

    content = [{"type": "text", "text": user.strip()}]
    for p in image_paths[:4]:
        content.append({"type": "image_url", "image_url": {"url": _b64_data_url(p)}})

    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": 600,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    txt = data["choices"][0]["message"]["content"]

    obj = _extract_json(txt) or {}

    brand = (obj.get("brand") or "").strip()
    product = (obj.get("product") or "").strip()
    typ = (obj.get("type") or "").strip()

    title = (obj.get("title") or "").strip()
    if not title:
        title = " ".join([p for p in [brand, product, typ] if p]).strip()
    title = re.sub(r"\s+", " ", title).strip() or f"Artikel {art_id}"

    desc = (obj.get("description") or "").strip()
    cat = (obj.get("category") or "").strip()
    price = _to_float(obj.get("retail_price_eur"))

    mhd = (obj.get("mhd") or "").strip()
    if re.search(r"lebensmittel|food|nahrung", cat, re.I) and not mhd:
        mhd = "nicht ersichtlich"

    vol = obj.get("volume_ml", None)
    if isinstance(vol, str):
        vol_f = _to_float(vol)
        vol = int(vol_f) if vol_f else None
    if isinstance(vol, (int, float)) and vol <= 0:
        vol = None

    size = (obj.get("size") or "").strip()

    extras = []
    if vol and not re.search(r"\b\d{1,4}\s?ml\b", desc, re.I):
        extras.append(f"Menge: {int(vol)} ml")
    if mhd and not re.search(r"\bMHD\b", desc, re.I):
        extras.append(f"MHD: {mhd}")
    if size and not re.search(r"\bcm\b", desc, re.I):
        extras.append(f"Maße: {size}")

    if extras:
        bullet = "\n".join(f"- {e}" for e in extras)
        desc = (desc + "\n" + bullet).strip() if desc else bullet

    dur = int((time.time() - start) * 1000)
    print(f"[KI] Artikel {art_id}: KI-Zeit {dur} ms, images={len(image_paths)}")

    return {
        "title": title,
        "description": desc,
        "category": cat,
        "retail_price": float(price),
    }
