# [DATEI: ki_engine_openai.py]
# MyAuktion – Vision-KI via OpenAI (Titel, Beschreibung & Preis)

import base64
import time
import requests
from pathlib import Path
import re
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"   # schnell + günstig + Vision perfekt geeignet


def encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _clean_line(s: str) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    return " ".join(s.split()).strip(' "').strip(".")


def _trim_len(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len - 3].rstrip() + "..."


def _parse_name_text_price(raw: str, art_id: str):
    default_name = f"Artikel {art_id}"
    default_desc = "Artikel wie abgebildet, Details bitte den Fotos entnehmen."
    default_price = 0.0

    name, desc, price = None, None, None

    m_name = re.search(r"NAME\s*=\s*(.+?)(?=\n|TEXT|PRICE_EUR|$)", raw, re.DOTALL)
    m_desc = re.search(r"TEXT\s*=\s*(.+?)(?=\n|PRICE_EUR|$)", raw, re.DOTALL)
    m_price = re.search(r"PRICE_EUR\s*=\s*([0-9]+(?:[.,][0-9]+)?)", raw)

    if m_name:
        name = m_name.group(1).strip()

    if m_desc:
        desc = m_desc.group(1).strip()

    if m_price:
        p = m_price.group(1).replace(",", ".")
        try:
            price = float(p)
        except:
            price = None

    if not name:
        name = default_name
    if not desc:
        desc = default_desc
    if price is None:
        price = default_price

    name = _clean_line(name)
    desc = _clean_line(desc)

    name = _trim_len(name, 80)
    desc = _trim_len(desc, 230)

    return name, desc, price


def generate_meta(image_path: str, art_id: str) -> dict:
    if not OPENAI_API_KEY:
        print("[KI] Kein OPENAI_API_KEY gesetzt!")
        return {
            "title": f"Artikel {art_id}",
            "description": "Artikel wie abgebildet, Details bitte den Fotos entnehmen.",
            "retail_price": 0.0,
        }

    b64 = encode_image(image_path)

    prompt = (
        "Du schreibst Produkttexte für MyAuktion.com.\n"
        "Analysiere das Produktbild und gib GENAU DREI Zeilen aus:\n\n"
        "NAME=<sehr kurzer deutscher Produktname>\n"
        "TEXT=<kurze neutrale Produktbeschreibung, 1 Satz>\n"
        "PRICE_EUR=<typischer NEUPREIS in Euro, lieber etwas höher angesetzt>\n\n"
        "Keine Erklärungen, keine Zusatzzeilen."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Bitte nur NAME=..., TEXT=..., PRICE_EUR=..."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                }
            ],
        },
    ],
    "temperature": 0.2,
    "max_tokens": 300,
}

    start = time.time()
    try:
        r = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=12)
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("[KI] Fehler:", e)
        return {
            "title": f"Artikel {art_id}",
            "description": "Artikel wie abgebildet, Details bitte den Fotos entnehmen.",
            "retail_price": 0.0,
        }

    dur = int((time.time() - start) * 1000)
    print(f"[KI] Artikel {art_id}: KI-Zeit {dur} ms")

    name, desc, price = _parse_name_text_price(result, art_id)

    return {
        "title": name,
        "description": desc,
        "retail_price": float(price),
    }
