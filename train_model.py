#!/usr/bin/env python3
"""
Train a flood-risk classifier.

Priority of data sources:
1) Local SQLite DB (streamlit_app/data/disasterguard.db)  <-- preferred
2) CSV at streamlit_app/data/sample_flood_data.csv
3) Tiny built-in demo dataset

Saves:
- streamlit_app/models/flood_model.pkl
- streamlit_app/models/metadata.json
"""

from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------- Paths ----------------
REPO_ROOT = os.path.dirname(__file__)
APP_DIR   = os.path.join(REPO_ROOT, "streamlit_app")

DATA_DIR = os.environ.get("DG_DATA_DIR", os.path.join(APP_DIR, "data"))
MODEL_DIR = os.environ.get("DG_MODEL_DIR", os.path.join(APP_DIR, "models"))

DB_PATH  = os.environ.get("DG_DB_PATH", os.path.join(DATA_DIR, "disasterguard.db"))
CSV_PATH = os.environ.get("DG_CSV_PATH", os.path.join(DATA_DIR, "sample_flood_data.csv"))

MODEL_PATH = os.path.join(MODEL_DIR, "flood_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["rainfall_mm", "river_level_m", "elevation_m", "urban_area"]
VALID_CLASSES = {"low": "Low", "medium": "Medium", "med": "Medium", "high": "High"}

# ---------------- Utils ----------------
def _normalize_label(x: str) -> str:
    if x is None:
        return None
    s = str(x).strip().lower()
    return VALID_CLASSES.get(s, s.capitalize())

def _clip_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce").clip(0, 500)
    df["river_level_m"] = pd.to_numeric(df["river_level_m"], errors="coerce").clip(0, 15)
    df["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce").clip(0, 3000)
    df["urban_area"] = pd.to_numeric(df["urban_area"], errors="coerce").clip(0, 1).round().astype(int)
    return df

# ---------------- Loaders ----------------
def load_from_db(db_path: str) -> pd.DataFrame | None:
    if not os.path.exists(db_path):
        return None
    with sqlite3.connect(db_path) as conn:
        q = """
        SELECT
            wr.rain_1h                     AS rainfall_mm,
            p.score_json                   AS score_json,
            loc.elev_m                     AS elevation_m,
            loc.urban                      AS urban_area,
            p.risk                         AS flood_risk
        FROM weather_readings wr
        JOIN predictions p ON p.reading_id = wr.id
        JOIN locations   loc ON loc.id = wr.location_id
        WHERE p.risk IS NOT NULL
        """
        df = pd.read_sql(q, conn)

    if df.empty:
        return None

    def parse_river(js):
        try:
            obj = json.loads(js or "{}")
            v = obj.get("river")
            return float(v) if v is not None else None
        except Exception:
            return None

    df["river_level_m"] = df["score_json"].map(parse_river)
    df.drop(columns=["score_json"], inplace=True, errors="ignore")

    df["flood_risk"] = df["flood_risk"].map(_normalize_label)
    for c in ["rainfall_mm", "elevation_m", "urban_area"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["rainfall_mm", "elevation_m", "urban_area", "flood_risk"]).copy()

    if "river_level_m" not in df.columns:
        df["river_level_m"] = None
    if df["river_level_m"].isna().mean() > 0.5:
        df["river_level_m"] = df["river_level_m"].fillna(3.0)
    else:
        df["river_level_m"] = df["river_level_m"].fillna(df["river_level_m"].median())

    df = df.drop_duplicates()

    if len(df) < 20:
        return None

    return _clip_clean(df)

def load_from_csv(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    required = FEATURES + ["flood_risk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["flood_risk"] = df["flood_risk"].map(_normalize_label)
    df = df.dropna(subset=FEATURES + ["flood_risk"]).copy()
    return _clip_clean(df)

def load_demo() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "rainfall_mm": [20, 50, 100, 150, 200],
            "river_level_m": [0.5, 1.5, 3.0, 5.0, 7.0],
            "elevation_m": [100, 50, 40, 20, 10],
            "urban_area": [0, 1, 1, 1, 0],
            "flood_risk": ["Low", "Low", "Medium", "High", "High"],
        }
    )
    return _clip_clean(df)

# ---------------- Trainer ----------------
def _save_model_and_meta(clf: RandomForestClassifier, df: pd.DataFrame, source: str, report_dict: dict | None):
    joblib.dump(clf, MODEL_PATH)
    meta = {
        "feature_order": FEATURES,
        "trained_on": source,
        "rows": int(len(df)),
        "model_type": "RandomForestClassifier",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if report_dict:
        summary = {
            "accuracy": report_dict.get("accuracy"),
            "macro_avg": report_dict.get("macro avg"),
            "weighted_avg": report_dict.get("weighted avg"),
            "support": {k: v.get("support") for k, v in report_dict.items() if k in ("Low", "Medium", "High")},
        }
        meta["metrics"] = summary
    try:
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            meta["feature_importances"] = {f: float(w) for f, w in zip(FEATURES, importances)}
    except Exception:
        pass
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Model saved -> {MODEL_PATH}")
    print(f"📝 Metadata -> {META_PATH}")

def train_and_save(df: pd.DataFrame, source: str):
    X = df[FEATURES].copy()
    y = df["flood_risk"].astype(str)

    print("📊 Source:", source)
    print("📊 Shape:", X.shape)
    print("📊 Class distribution:\n", y.value_counts())

    report_dict = None
    if (len(df) >= 60) and (y.nunique() >= 2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        print("🧪 Holdout report:\n", classification_report(y_test, y_pred))
    else:
        clf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1)
        clf.fit(X, y)

    _save_model_and_meta(clf, df, source, report_dict)

def main():
    df = load_from_db(DB_PATH)
    if df is not None:
        train_and_save(df, source=f"sqlite:{os.path.relpath(DB_PATH, REPO_ROOT)}")
        return
    df = load_from_csv(CSV_PATH)
    if df is not None:
        print(f"📄 Loading dataset from {CSV_PATH}")
        train_and_save(df, source=f"csv:{os.path.relpath(CSV_PATH, REPO_ROOT)}")
        return
    print("ℹ️ No DB/CSV found. Using a tiny built-in demo dataset.")
    df = load_demo()
    train_and_save(df, source="demo")

if __name__ == "__main__":
    main()