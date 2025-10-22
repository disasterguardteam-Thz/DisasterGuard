# db.py (upgraded)
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# ========= Paths (robust to current working dir) =========
HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "data" / "disasterguard.db"
SCHEMA_PATH = HERE / "schema.sql"

_env_db = os.environ.get("DG_DB_PATH")
DB_PATH = Path(_env_db).resolve() if _env_db else DEFAULT_DB
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ========= Connection helper =========
def _apply_pragmas(conn: sqlite3.Connection):
    """
    Pragmas chosen to improve durability & perf for a small local app.
    WAL allows concurrent readers; synchronous=NORMAL is a good balance.
    """
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -20000;")  # ~20MB page cache


@contextmanager
def get_conn():
    """
    SQLite connection with FK enabled. Uses WAL + sane pragmas.
    - Commits on success, rolls back on error, always closes.
    - check_same_thread=False allows use inside Streamlit callbacks.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        _apply_pragmas(conn)
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


# ========= Schema / init =========
def _ensure_indexes(conn: sqlite3.Connection):
    """
    Create helpful indexes if they don't exist (schema-compatible).
    Safe to call multiple times. Duplicates in schema.sql are OK—IF NOT EXISTS.
    """
    conn.executescript(
        """
        -- Locations
        CREATE INDEX IF NOT EXISTS idx_locations_name       ON locations(name);
        CREATE INDEX IF NOT EXISTS idx_locations_lat_lon    ON locations(lat, lon);

        -- Weather readings
        CREATE UNIQUE INDEX IF NOT EXISTS uq_reading_loc_ts
            ON weather_readings(location_id, ts_utc);
        CREATE INDEX IF NOT EXISTS idx_weather_location_ts
            ON weather_readings(location_id, ts_utc);
        CREATE INDEX IF NOT EXISTS idx_weather_ts
            ON weather_readings(ts_utc);

        -- Predictions
        CREATE INDEX IF NOT EXISTS idx_predictions_reading
            ON predictions(reading_id);
        CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions(created_at);
        CREATE INDEX IF NOT EXISTS idx_predictions_risk_created
            ON predictions(risk, created_at);

        -- Alerts
        CREATE INDEX IF NOT EXISTS idx_alerts_location_ts
            ON alerts(location_id, ts_utc);
        CREATE INDEX IF NOT EXISTS idx_alerts_delivered_time
            ON alerts(delivered, ts_utc);
        """
    )


def init_db():
    """(Re)create tables from schema.sql if needed and ensure indexes."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema.sql not found at: {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = f.read()
    with get_conn() as c:
        c.executescript(schema)
        _ensure_indexes(c)


# ========= Core writes =========
def upsert_location(name: str, lat: float, lon: float, elev_m: float, urban: int) -> int:
    """
    Returns an existing location id that matches (name, lat, lon) or inserts a new one.
    If it exists, also refresh elev_m/urban to keep attributes current.
    """
    with get_conn() as c:
        cur = c.execute(
            """
            SELECT id, elev_m, urban
            FROM locations
            WHERE name = ?
              AND ABS(lat - ?) < 1e-6
              AND ABS(lon - ?) < 1e-6
            """,
            (name, lat, lon),
        )
        row = cur.fetchone()
        if row:
            loc_id, prev_elev, prev_urban = row
            if (prev_elev != elev_m) or (prev_urban != urban):
                c.execute(
                    "UPDATE locations SET elev_m = ?, urban = ? WHERE id = ?",
                    (elev_m, int(bool(urban)), loc_id),
                )
            return loc_id

        cur = c.execute(
            "INSERT INTO locations(name, country, lat, lon, elev_m, urban) VALUES (?,?,?,?,?,?)",
            (name, "Unknown", lat, lon, elev_m, int(bool(urban))),
        )
        return cur.lastrowid


def _json_dump_safe(obj: dict | None) -> str:
    """JSON dump that tolerates numpy types and other non-serializables."""
    def _default(o):
        try:
            import numpy as np  # optional
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
        except Exception:
            pass
        # last resort
        try:
            return float(o)  # e.g., pandas NA/num
        except Exception:
            try:
                return int(o)
            except Exception:
                return str(o)
    return json.dumps(obj or {}, default=_default)


def insert_reading(
    location_id: int,
    ts_utc: str,
    rain_1h: Optional[float],
    temp_c: Optional[float],
    humidity: Optional[float],
    wind_ms: Optional[float],
    source: str = "openweather",
) -> Optional[int]:
    """
    Idempotent insert: thanks to uq_reading_loc_ts (location_id, ts_utc),
    a repeat insert returns the existing row id instead of failing.
    """
    with get_conn() as c:
        try:
            cur = c.execute(
                """
                INSERT INTO weather_readings(location_id, ts_utc, rain_1h, temp_c, humidity, wind_ms, source)
                VALUES (?,?,?,?,?,?,?)
                """,
                (location_id, ts_utc, rain_1h, temp_c, humidity, wind_ms, source),
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            # Duplicate snapshot -> fetch existing id
            cur = c.execute(
                "SELECT id FROM weather_readings WHERE location_id=? AND ts_utc=?",
                (location_id, ts_utc),
            )
            row = cur.fetchone()
            return row[0] if row else None


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
            (reading_id, risk, model, _json_dump_safe(score_dict)),
        )
        return cur.lastrowid


def insert_alert(
    location_id: int,
    ts_utc: str,
    risk: str,
    channel: str = "telegram",
    delivered: int = 0,
    message: str | None = None,
) -> int:
    with get_conn() as c:
        cur = c.execute(
            """
            INSERT INTO alerts(location_id, ts_utc, risk, channel, delivered, message)
            VALUES (?,?,?,?,?,?)
            """,
            (location_id, ts_utc, risk, channel, int(bool(delivered)), message),
        )
        return cur.lastrowid


def mark_alert_delivered(alert_id: int):
    with get_conn() as c:
        c.execute("UPDATE alerts SET delivered = 1 WHERE id = ?", (alert_id,))


# ========= Optional helpers =========
def get_location_id(name: str, lat: float, lon: float) -> Optional[int]:
    """Return a location id if an exact (name,lat,lon) exists, else None."""
    with get_conn() as c:
        cur = c.execute(
            """
            SELECT id
            FROM locations
            WHERE name = ?
              AND ABS(lat - ?) < 1e-6
              AND ABS(lon - ?) < 1e-6
            """,
            (name, lat, lon),
        )
        row = cur.fetchone()
        return row[0] if row else None


# ========= Basic fetch (compatibility) =========
def fetch_recent_readings(location_id: int, days: int = 30):
    """
    Time-ordered readings (joined with risk) for the last N days.
    NOTE: days is int-clamped to avoid accidental huge windows.
    """
    days = max(1, int(days))
    with get_conn() as c:
        cur = c.execute(
            """
            SELECT wr.ts_utc, wr.rain_1h, wr.temp_c, wr.humidity, wr.wind_ms, p.risk
            FROM weather_readings wr
            LEFT JOIN predictions p ON p.reading_id = wr.id
            WHERE wr.location_id = ?
              AND datetime(wr.ts_utc) >= datetime('now', ?)
            ORDER BY wr.ts_utc ASC
            """,
            (location_id, f"-{days} days"),
        )
        rows = cur.fetchall()
    import pandas as pd
    return pd.DataFrame(rows, columns=["ts_utc","rain_1h","temp_c","humidity","wind_ms","risk"])


# ========= Analytics helpers =========
def fetch_readings_df(location_id: int, days: int = 30):
    """Return tidy DF of readings + risk for the last N days."""
    days = max(1, int(days))
    with get_conn() as c:
        cur = c.execute(
            """
            SELECT wr.ts_utc,
                   wr.rain_1h, wr.temp_c, wr.humidity, wr.wind_ms,
                   COALESCE(p.risk, 'Unknown') AS risk
            FROM weather_readings wr
            LEFT JOIN predictions p ON p.reading_id = wr.id
            WHERE wr.location_id = ?
              AND datetime(wr.ts_utc) >= datetime('now', ?)
            ORDER BY wr.ts_utc ASC
            """,
            (location_id, f"-{days} days"),
        )
        rows = cur.fetchall()

    import pandas as pd
    df = pd.DataFrame(rows, columns=["ts_utc","rain_1h","temp_c","humidity","wind_ms","risk"])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df["date"] = df["ts_utc"].dt.date.astype(str)
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
    limit = max(1, int(limit))
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


# ========= Extra: absolute window fetch (useful for compare) =========
def fetch_readings_window(location_id: int, start_iso: str, end_iso: str):
    """
    Fetch readings in an absolute [start, end) ISO-8601 window.
    start_iso/end_iso should be in UTC ('YYYY-MM-DDTHH:MM:SSZ' or a format SQLite accepts).
    """
    with get_conn() as c:
        cur = c.execute(
            """
            SELECT wr.ts_utc,
                   wr.rain_1h, wr.temp_c, wr.humidity, wr.wind_ms,
                   COALESCE(p.risk, 'Unknown') AS risk
            FROM weather_readings wr
            LEFT JOIN predictions p ON p.reading_id = wr.id
            WHERE wr.location_id = ?
              AND datetime(wr.ts_utc) >= datetime(?)
              AND datetime(wr.ts_utc) <  datetime(?)
            ORDER BY wr.ts_utc ASC
            """,
            (location_id, start_iso, end_iso),
        )
        rows = cur.fetchall()

    import pandas as pd
    df = pd.DataFrame(rows, columns=["ts_utc","rain_1h","temp_c","humidity","wind_ms","risk"])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df["date"] = df["ts_utc"].dt.date.astype(str)
    return df