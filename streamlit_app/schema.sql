PRAGMA foreign_keys = ON;

-- ============================================================
-- 🌍 Table: locations
-- ============================================================
CREATE TABLE IF NOT EXISTS locations (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  name      TEXT NOT NULL,
  country   TEXT DEFAULT 'Unknown',
  lat       REAL NOT NULL,
  lon       REAL NOT NULL,
  elev_m    REAL NOT NULL,
  urban     INTEGER NOT NULL DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now'))
);

-- Helpful lookups
CREATE INDEX IF NOT EXISTS idx_locations_name       ON locations(name);
CREATE INDEX IF NOT EXISTS idx_locations_lat_lon    ON locations(lat, lon);

-- ============================================================
-- ☁️ Table: weather_readings
-- ============================================================
CREATE TABLE IF NOT EXISTS weather_readings (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  location_id INTEGER NOT NULL,
  ts_utc      TEXT NOT NULL,
  rain_1h     REAL,
  rain_24h    REAL,
  temp_c      REAL,
  humidity    REAL,
  wind_ms     REAL,
  source      TEXT DEFAULT 'openweather',
  FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
);

-- Prevent duplicate snapshots for the same location+timestamp
CREATE UNIQUE INDEX IF NOT EXISTS uq_reading_loc_ts
  ON weather_readings(location_id, ts_utc);

-- Time & location access paths
CREATE INDEX IF NOT EXISTS idx_weather_location_ts
  ON weather_readings(location_id, ts_utc);
CREATE INDEX IF NOT EXISTS idx_weather_ts
  ON weather_readings(ts_utc);

-- ============================================================
-- 🤖 Table: predictions
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  reading_id    INTEGER NOT NULL,
  risk          TEXT NOT NULL,
  model         TEXT NOT NULL,
  model_version TEXT DEFAULT 'v1.0',
  score_json    TEXT,
  created_at    TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (reading_id) REFERENCES weather_readings(id) ON DELETE CASCADE
);

-- Common query paths
CREATE INDEX IF NOT EXISTS idx_predictions_reading
  ON predictions(reading_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
  ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_created
  ON predictions(risk, created_at);

-- ============================================================
-- 🚨 Table: alerts
-- ============================================================
CREATE TABLE IF NOT EXISTS alerts (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  location_id INTEGER NOT NULL,
  ts_utc      TEXT NOT NULL,
  risk        TEXT NOT NULL,
  channel     TEXT NOT NULL,
  delivered   INTEGER NOT NULL DEFAULT 0,
  message     TEXT,
  created_at  TEXT DEFAULT (datetime('now')),
  FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alerts_location_ts
  ON alerts(location_id, ts_utc);
CREATE INDEX IF NOT EXISTS idx_alerts_delivered_time
  ON alerts(delivered, ts_utc);