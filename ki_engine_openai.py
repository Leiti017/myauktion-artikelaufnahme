# ki_engine_openai.py – OpenAI Vision Engine (FINAL, schnell + Kontext fürs 2. Foto)
# - KEIN Fallback: bei Fehler -> None
# - NAME: "Marke Produktbezeichnung Typenbezeichnung"
# - TEXT: alle Details (Menge/ml/g/Stück, Maße z.B. 23 cm, MHD wenn sichtbar, Set, Verpackung, etc.)
# - Kontext (Alt-Daten) wird genutzt, um beim 2. Foto fehlende Details zu ergänzen
# - Speed: detail="low", kleinere Bilder, weniger max_tokens

from __future__ import annotations

import base64
import os
import re
import time
from typing import Any, Dict, Optional

import requests

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Debug (nur Länge / vorhanden)
print("[KI-DEBUG] OPENAI_API_KEY vorhanden:", bool(OPENAI_API_KEY))
print("[KI-DEBUG] OPENAI_API_KEY Länge:", len(OPENAI_API_KEY) if OPENAI_API_KEY else 0)

CATEGORIES = [
    "Parfum",
    "Kosmetik",
    "Werkzeug",
    "Elektronik",
    "Haushalt",
    "Kleidung",
    "Schmuck/Uhren",
    "Spielzeug",
    "Bücher/Medien",
    "Sonstiges",
]

SYSTEM_PROMPT = f"""
Du bist ein extrem präziser Produkt-Assistent für ein Auktionssystem.
Du analysierst Produktfotos. Es kann mehrere Fotos desselben Artikels geben.

========================
BEZEICHNUNG (NAME)
========================
Die Bezeichnung MUSS IMMER genau so aufgebaut sein:

Marke Produktbezeichnung Typenbezeichnung

Regeln für NAME:
- Kurz und sachlich
- Grundformat: Marke Produktbezeichnung Typenbezeichnung
- Mengenangaben sind grundsätzlich NICHT im NAME, AUSSER bei:
  - GETRÄNKEN (z. B. Secco, Wein, Sekt, Spirituosen)
  - TIERNahrung (z. B. Hunde- oder Katzenfutter)
- Regeln für erlaubte Mengenangaben:
  - Erlaubt sind NUR ml / l (Getränke) bzw. g / kg (Tiernahrung)
  - **NUR wenn eindeutig sichtbar auf dem Foto**
  - **KEIN SCHÄTZEN, KEINE STANDARDMENGEN**
  - Wenn die Menge nicht klar lesbar ist: **WEGLASSEN**
  - WENN AUF EINEM KARTON / UMKARTON eindeutig sichtbar:
    → Stückzahl × Einzelmenge ist ERLAUBT
    → Format IMMER: 12x200 ml oder 12x200 g (kein Leerzeichen um das x)
  - Die Mengenangabe MUSS am ENDE stehen
  - Beispiele:
    - Einzelartikel Getränk: "Villa Sandi Prosecco DOC 200 ml"
    - Karton Getränk: "Villa Sandi Prosecco DOC 12x200 ml"
    - Einzelartikel Tiernahrung: "Royal Canin Puppy 200 g"
    - Karton Tiernahrung: "Royal Canin Puppy 12x200 g"
  - Keine weiteren Set-/Bundle-Texte im NAME
- KEIN MHD im NAME
- KEINE Zustandsangaben im NAME
- KEINE Verpackungshinweise im NAME
- Wenn eine Komponente nicht erkennbar ist, lasse sie weg.
- Reihenfolge IMMER einhalten.
  - Erlaubt sind NUR ml oder l und NUR wenn eindeutig sichtbar
  - Die Mengenangabe MUSS am ENDE stehen (z. B. "Villa Sandi Prosecco DOC 750 ml")
  - Keine mehrfachen Mengenangaben, keine Set/Packungsangaben im NAME
- KEINE Set-Hinweise im NAME
- KEIN MHD im NAME
- KEINE Zustandsangaben im NAME
- KEINE Verpackungshinweise im NAME
- Wenn eine Komponente nicht erkennbar ist, lasse sie weg.
- Reihenfolge IMMER einhalten (bei Getränken: Menge ganz am Ende).

========================
BESCHREIBUNG (TEXT)
========================
In die Beschreibung kommt ALLES, was nicht zur Bezeichnung gehört:

- Mengenangaben (ml/g/kg/Stück/Set/Packung) NUR wenn sichtbar
- Größenangaben (cm/mm) NUR wenn sichtbar oder mit Maßband/Lineal im Bild
  (wenn keine Referenz sichtbar: NICHT raten)
- MHD (Mindestens haltbar bis) NUR wenn klar sichtbar, Format MM/JJJJ oder TT.MM.JJJJ
- Verpackung / Zubehör / Lieferumfang (nur wenn sichtbar)
- neutrale Zustandsbeschreibung (nur wenn sichtbar)
- Ergänzungen aus weiteren Fotos (fehlende Infos ergänzen, nichts erfinden)

TEXT: 1–3 kurze, sachliche Sätze.

WICHTIG:
- Wenn etwas nicht klar sichtbar/lesbar ist: **NICHT raten**.
- Keine Standardannahmen (z. B. 750 ml, 1 l, 100 ml, 200 g, 1 kg).
- Stückzahlen (z. B. 6x, 12x) NUR verwenden, wenn sie klar auf dem Foto lesbar sind (typisch: Karton).

========================
KATEGORIE
========================
GENAU eine aus dieser Liste:
{", ".join(CATEGORIES)}

========================
PREIS
========================
PRICE_EUR = realistischer tagesaktueller Neupreis in EUR.
- eher leicht höher schätzen als zu niedrig
- Zahl mit Punkt als Dezimaltrenner (z. B. 129.99)

========================
ANTWORTFORMAT (PFLICHT)
========================
Antworte IMMER und NUR in diesem Format:

NAME=...
TEXT=...
CATEGORY=...
PRICE_EUR=...

Keine zusätzlichen Zeilen. Keine Erklärungen. Keine Fantasie.
""".strip()


def _encode_image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_response(text: str) -> Optional[Dict[str, Any]]:
    name = re.search(r"NAME\s*=\s*(.*)", text)
    desc = re.search(r"TEXT\s*=\s*(.*)", text)
    cat = re.search(r"CATEGORY\s*=\s*(.*)", text)
    price = re.search(r"PRICE_EUR\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)

    if not name:
        return None

    title = (name.group(1).strip() if name else "").strip()
    description = (desc.group(1).strip() if desc else "").strip()
    category = (cat.group(1).strip() if cat else "").strip()

    if category:
        allowed_lower = {c.lower(): c for c in CATEGORIES}
        if category.lower() in allowed_lower:
            category = allowed_lower[category.lower()]
        else:
            category = "Sonstiges"
    else:
        category = ""

    retail = 0.0
    if price:
        try:
            retail = float(price.group(1))
        except Exception:
            retail = 0.0

    return {
        "title": title,
        "description": description,
        "category": category,
        "retail_price": retail,
    }


def generate_meta(image_path: str, art_id: str, context: Dict[str, Any] | None = None) -> Optional[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        print("[KI] Fehler: OPENAI_API_KEY fehlt")
        return None

    context = context or {}
    ctx_title = (context.get("titel") or context.get("title") or "").strip()
    ctx_desc = (context.get("beschreibung") or context.get("description") or "").strip()
    ctx_cat = (context.get("kategorie") or context.get("category") or "").strip()

    ctx_text = ""
    if ctx_title or ctx_desc or ctx_cat:
        ctx_text = f"""Bisherige Daten (kann unvollständig sein):
NAME_ALT: {ctx_title}
TEXT_ALT: {ctx_desc}
CATEGORY_ALT: {ctx_cat}

Aufgabe:
- Verwende das neue Foto, um fehlende Details zu ERGÄNZEN oder zu KORRIGIEREN.
- NAME bleibt strikt: Marke Produktbezeichnung Typenbezeichnung.
- Details (Menge, Maße z.B. 23 cm, MHD, Set, Zubehör) gehören in TEXT – nur wenn sichtbar.
- Wenn etwas nicht sichtbar ist: NICHT raten.
"""

    b64 = _encode_image_to_b64(image_path)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Bitte streng im Format antworten (NAME/TEXT/CATEGORY/PRICE_EUR).\n\n" + ctx_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 220,
    }

    try:
        r = requests.post(OPENAI_URL, json=payload, headers=headers, timeout=25)

        if r.status_code != 200:
            print("[KI-DEBUG] HTTP:", r.status_code)
            print("[KI-DEBUG] BODY:", r.text)

        r.raise_for_status()

        content = r.json()["choices"][0]["message"]["content"]
        parsed = _parse_response(content)
        if not parsed:
            print("[KI] Fehler: Antwort konnte nicht geparst werden")
            return None

        return parsed

    except Exception as e:
        print("[KI] Fehler:", e)
        return None


# build:20251215_084727
