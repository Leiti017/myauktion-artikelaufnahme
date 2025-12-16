
# --- FIXED SERVER_FULL WITH VARIANTE A PERSISTENZ ---

import os
import json
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
EXPORT_DIR = os.path.join(BASE_DIR, "export")
EXPORT_JSON = os.path.join(EXPORT_DIR, "artikel_data.json")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

ARTIKEL = []

def _rebuild_data_json():
    try:
        with open(EXPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(ARTIKEL, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Fehler beim Rebuild von artikel_data.json:", e)


def load_existing_data():
    if os.path.exists(EXPORT_JSON):
        try:
            with open(EXPORT_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    ARTIKEL.extend(data)
        except Exception as e:
            print("Fehler beim Laden der Persistenz:", e)


load_existing_data()
_rebuild_data_json()


@app.get("/health")
def health():
    return {"status": "ok", "artikel": len(ARTIKEL)}
