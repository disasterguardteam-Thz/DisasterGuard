#!/usr/bin/env python3
"""
Train a flood-risk classifier.

Priority of data sources:
1) Local SQLite DB (data/disasterguard.db)  <-- preferred in Phase 2A
2) CSV at data/sample_flood_data.csv
3) Tiny built-in demo dataset

Saves:
- models/flood_model.pkl
- models/metadata.json
"""

import os
import json
import sqlite3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------- Paths ----------------
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
DB_PATH = os.environ.get("DG_DB_PATH", os.path.join(DATA_DIR, "disasterguard.db"))
CSV_PATH = os.path.join(DATA_DIR, "sample_flood_data.csv")

MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "flood_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["rainfall_mm", "river_level_m", "elevation_m", "urban_area"]


# ---------------- Loaders ----------------
def load_from_db(db_path: str) -> pd.DataFrame | None:
    """
    Build a training frame from the local SQLite DB:

    - rainfall_mm  <- weather_readings.rain_1h
    - river_level_m <- predictions.score_json['river'] (if present; else NaN)
    - elevation_m  <- locations.elev_m
    - urban_area   <- locations.urban
    - flood_risk   <- predictions.risk

    Returns a DataFrame or None if not enough data.
    """
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        # weather readings + predictions + locations
        q = """
        SELECT
            wr.rain_1h                               AS rainfall_mm,
            p.score_json                              AS score_json,
            loc.elev_m                                AS elevation_m,
            loc.urban                                 AS urban_area,
            p.risk                                    AS flood_risk
        FROM weather_readings wr
        JOIN predictions p ON p.reading_id = wr.id
        JOIN locations loc ON loc.id = wr.location_id
        WHERE p.risk IS NOT NULL
        """
        df = pd.read_sql(q, conn)

    if df.empty:
        return None

    # Extract river from score_json (if available)
    def parse_river(js):
        try:
            obj = json.loads(js or "{}")
            v = obj.get("river")
            return float(v) if v is not None else None
        except Exception:
            return None

    df["river_level_m"] = df["score_json"].map(parse_river)
    df.drop(columns=["score_json"], inplace=True)

    # Coerce numeric
    for c in ["rainfall_mm", "river_level_m", "elevation_m", "urban_area"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic cleaning
    df["urban_area"] = df["urban_area"].clip(0, 1).round().astype("Int64")
    df = df.dropna(subset=["rainfall_mm", "elevation_m", "urban_area", "flood_risk"]).copy()

    # If river missing for many rows, fill with median or a sensible default
    if df["river_level_m"].isna().mean() > 0.5:
        df["river_level_m"] = df["river_level_m"].fillna(3.0)
    else:
        df["river_level_m"] = df["river_level_m"].fillna(df["river_level_m"].median())

    if len(df) < 20:
        # Not enough records to be useful
        return None

    return df


def load_from_csv(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    required = FEATURES + ["flood_risk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Coerce numeric + clean
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + ["flood_risk"]).copy()
    df["urban_area"] = df["urban_area"].clip(0, 1).astype(int)

    return df


def load_demo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rainfall_mm": [20, 50, 100, 150, 200],
            "river_level_m": [0.5, 1.5, 3.0, 5.0, 7.0],
            "elevation_m": [100, 50, 40, 20, 10],
            "urban_area": [0, 1, 1, 1, 0],
            "flood_risk": ["Low", "Low", "Medium", "High", "High"],
        }
    )


# ---------------- Trainer ----------------
def train_and_save(df: pd.DataFrame, source: str):
    # Light clipping to keep ranges sane
    df["rainfall_mm"] = df["rainfall_mm"].clip(0, 500)
    df["river_level_m"] = df["river_level_m"].clip(0, 15)
    df["elevation_m"] = df["elevation_m"].clip(0, 2000)
    df["urban_area"] = df["urban_area"].clip(0, 1).astype(int)

    X = df[FEATURES].copy()
    y = df["flood_risk"].astype(str)

    print("📊 Source:", source)
    print("📊 Shape:", X.shape)
    print("📊 Class distribution:\n", y.value_counts())

    # If we have enough samples, do a quick holdout to sanity-check
    if len(df) >= 60 and y.nunique() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("🧪 Holdout report:\n", classification_report(y_test, y_pred))
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=42)
        clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    meta = {
        "feature_order": FEATURES,
        "trained_on": source,
        "rows": int(len(df)),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Model saved -> {MODEL_PATH}")
    print(f"📝 Metadata -> {META_PATH}")


def main():
    # 1) Try DB
    df = load_from_db(DB_PATH)
    if df is not None:
        train_and_save(df, source=f"sqlite:{os.path.relpath(DB_PATH, HERE)}")
        return

    # 2) Try CSV
    df = load_from_csv(CSV_PATH)
    if df is not None:
        print(f"📄 Loading dataset from {CSV_PATH}")
        train_and_save(df, source=f"csv:{os.path.relpath(CSV_PATH, HERE)}")
        return

    # 3) Demo
    print("ℹ️ No DB/CSV found. Using a tiny built-in demo dataset.")
    df = load_demo()
    train_and_save(df, source="demo")


if __name__ == "__main__":
    main()