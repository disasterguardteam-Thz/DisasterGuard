PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS locations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  lat REAL NOT NULL,
  lon REAL NOT NULL,
  elev_m REAL NOT NULL,
  urban INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS weather_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  location_id INTEGER NOT NULL,
  ts_utc TEXT NOT NULL,
  rain_1h REAL,
  temp_c REAL,
  humidity REAL,
  wind_ms REAL,
  source TEXT DEFAULT 'openweather',
  FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reading_id INTEGER NOT NULL,
  risk TEXT NOT NULL,
  model TEXT NOT NULL,
  score_json TEXT,
  FOREIGN KEY (reading_id) REFERENCES weather_readings(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  location_id INTEGER NOT NULL,
  ts_utc TEXT NOT NULL,
  risk TEXT NOT NULL,
  channel TEXT NOT NULL,
  delivered INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE CASCADE
);