# db.py
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


# ========= Paths (robust to current working dir) =========
HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "data" / "disasterguard.db"      # streamlit_app/data/disasterguard.db
SCHEMA_PATH = HERE / "schema.sql"

# Optional override via env var (e.g., export DG_DB_PATH=/tmp/dg.db)
_env_db = os.environ.get("DG_DB_PATH")
DB_PATH = Path(_env_db).resolve() if _env_db else DEFAULT_DB
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ========= Connection helper =========
@contextmanager
def get_conn():
    """
    SQLite connection with FK enabled. Commits & closes automatically.
    check_same_thread=False allows use inside Streamlit callbacks.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


# ========= Schema / init =========
def init_db():
    """(Re)create tables from schema.sql if needed."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = f.read()
    with get_conn() as c:
        c.executescript(schema)


# ========= Core writes =========
def upsert_location(name: str, lat: float, lon: float, elev_m: float, urban: int) -> int:
    """
    Returns an existing location id that matches (name, lat, lon) or inserts a new one.
    """
    with get_conn() as c:
        cur = c.execute(
            "SELECT id FROM locations WHERE name=? AND ABS(lat-?)<1e-6 AND ABS(lon-?)<1e-6",
            (name, lat, lon),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur = c.execute(
            "INSERT INTO locations(name, lat, lon, elev_m, urban) VALUES (?,?,?,?,?)",
            (name, lat, lon, elev_m, urban),
        )
        return cur.lastrowid


def insert_reading(
    location_id: int,
    ts_utc: str,
    rain_1h: Optional[float],
    temp_c: Optional[float],
    humidity: Optional[float],
    wind_ms: Optional[float],
    source: str = "openweather",
) -> int:
    with get_conn() as c:
        cur = c.execute(
            """
            INSERT INTO weather_readings(location_id, ts_utc, rain_1h, temp_c, humidity, wind_ms, source)
            VALUES (?,?,?,?,?,?,?)
            """,
            (location_id, ts_utc, rain_1h, temp_c, humidity, wind_ms, source),
        )
        return cur.lastrowid


def insert_prediction(
    reading_id: int,
    risk: str,
    model: str,
    score_dict: Optional[dict] = None,
) -> int:
    with get_conn() as c:
        cur = c.execute(
            """
            INSERT INTO predictions(reading_id, risk, model, score_json)
            VALUES (?,?,?,?)
            """,
            (reading_id, risk, model, json.dumps(score_dict or {})),
        )
        return cur.lastrowid


def insert_alert(
    location_id: int,
    ts_utc: str,
    risk: str,
    channel: str = "telegram",
    delivered: int = 0,
) -> int:
    with get_conn() as c:
        cur = c.execute(
            """
            INSERT INTO alerts(location_id, ts_utc, risk, channel, delivered)
            VALUES (?,?,?,?,?)
            """,
            (location_id, ts_utc, risk, channel, delivered),
        )
        return cur.lastrowid


def mark_alert_delivered(alert_id: int):
    with get_conn() as c:
        c.execute("UPDATE alerts SET delivered=1 WHERE id=?", (alert_id,))


# ========= Basic fetch (compatibility) =========
def fetch_recent_readings(location_id: int, days: int = 30):
    """Time-ordered readings (joined with risk) for the last N days."""
    with get_conn() as c:
        cur = c.execute(
            f"""
            SELECT wr.ts_utc, wr.rain_1h, wr.temp_c, wr.humidity, wr.wind_ms, p.risk
            FROM weather_readings wr
            LEFT JOIN predictions p ON p.reading_id = wr.id
            WHERE wr.location_id=?
              AND datetime(wr.ts_utc) >= datetime('now','-{days} days')
            ORDER BY wr.ts_utc ASC
            """,
            (location_id,),
        )
        rows = cur.fetchall()
    import pandas as pd
    return pd.DataFrame(rows, columns=["ts_utc","rain_1h","temp_c","humidity","wind_ms","risk"])


# ========= Analytics helpers (for Trends/Export) =========
def fetch_readings_df(location_id: int, days: int = 30):
    """Return tidy DF of readings + risk for the last N days."""
    with get_conn() as c:
        cur = c.execute(
            f"""
            SELECT wr.ts_utc,
                   wr.rain_1h, wr.temp_c, wr.humidity, wr.wind_ms,
                   COALESCE(p.risk, 'Unknown') AS risk
            FROM weather_readings wr
            LEFT JOIN predictions p ON p.reading_id = wr.id
            WHERE wr.location_id = ?
              AND datetime(wr.ts_utc) >= datetime('now','-{days} days')
            ORDER BY wr.ts_utc ASC
            """,
            (location_id,),
        )
        rows = cur.fetchall()
    import pandas as pd
    df = pd.DataFrame(rows, columns=["ts_utc","rain_1h","temp_c","humidity","wind_ms","risk"])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        # For daily grouping/plots
        df["date"] = df["ts_utc"].dt.tz_convert("UTC").dt.date.astype(str)
    return df


def fetch_daily_aggregates(location_id: int, days: int = 30):
    """Daily sums/means: rain sum, temp mean, humidity mean, wind mean."""
    df = fetch_readings_df(location_id, days)
    import pandas as pd
    if df.empty:
        return pd.DataFrame(columns=["date","rain_mm","temp_c","humidity","wind_ms"])
    g = df.groupby("date", as_index=False).agg(
        rain_mm=("rain_1h","sum"),
        temp_c=("temp_c","mean"),
        humidity=("humidity","mean"),
        wind_ms=("wind_ms","mean"),
    )
    return g


def fetch_risk_counts(location_id: int, days: int = 30):
    """Count of risk levels by day (for stacked bar chart)."""
    df = fetch_readings_df(location_id, days)
    import pandas as pd
    if df.empty:
        return pd.DataFrame(columns=["date","risk","count"])
    out = df.groupby(["date","risk"], as_index=False).size().rename(columns={"size":"count"})
    return out


def recent_snapshots(location_id: int, limit: int = 200):
    """Recent raw rows (joined with risk) for table/export."""
    with get_conn() as c:
        cur = c.execute(
            """
            SELECT wr.ts_utc, wr.rain_1h, wr.temp_c, wr.humidity, wr.wind_ms,
                   COALESCE(p.risk,'Unknown') AS risk
            FROM weather_readings wr
            LEFT JOIN predictions p ON p.reading_id = wr.id
            WHERE wr.location_id = ?
            ORDER BY wr.ts_utc DESC
            LIMIT ?
            """,
            (location_id, limit),
        )
        rows = cur.fetchall()
    import pandas as pd
    return pd.DataFrame(rows, columns=["ts_utc","rain_1h","temp_c","humidity","wind_ms","risk"])