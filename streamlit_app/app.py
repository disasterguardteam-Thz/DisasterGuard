#!/usr/bin/env python3
# DisasterGuard • Analyst Studio (modular UI)
# Routes: Profile | Global View | Monitoring | Statistics

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import requests
import streamlit as st

import ui  # ui is a sibling package (folder) of app.py
from ui import styles as ui_styles
from ui.pages.statistics_dashboard import run_statistics_page
from db import (
    init_db,
    upsert_location,
    insert_reading,
    insert_prediction,
    insert_alert,
    mark_alert_delivered,
    fetch_daily_aggregates,
    fetch_risk_counts,
    recent_snapshots,
)

# -----------------------------------------------------------------------------
# Page config & base styles
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DisasterGuard • Analyst Studio", layout="wide")
ui.inject_base_css()
ui_styles.inject_base_css()
ui.spacer(6)

# -----------------------------------------------------------------------------
# Paths that work both locally and on Render
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(__file__)  # -> streamlit_app/
MODEL_DIR = os.environ.get("DG_MODEL_DIR", os.path.join(APP_DIR, "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "flood_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Secrets / Env
# -----------------------------------------------------------------------------
def _secret_or_env(key: str, env_fallback: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(env_fallback or key)

OWM_KEY  = _secret_or_env("OWM_KEY")
TG_TOKEN = _secret_or_env("TELEGRAM_BOT_TOKEN")
TG_CHAT  = _secret_or_env("TELEGRAM_CHAT_ID")

# -----------------------------------------------------------------------------
# Load model / metadata (optional)
# -----------------------------------------------------------------------------
MODEL: Optional[object] = None
CLASSES = ["Low", "Medium", "High"]
FEATURE_ORDER = ["rainfall_mm", "river_level_m", "elevation_m", "urban_area"]

try:
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh) or {}
            if isinstance(meta, dict) and "feature_order" in meta:
                FEATURE_ORDER = list(meta["feature_order"])
except Exception:
    pass

try:
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        if hasattr(MODEL, "classes_"):
            CLASSES = [str(c) for c in MODEL.classes_]
except Exception:
    MODEL = None

# -----------------------------------------------------------------------------
# DB init
# -----------------------------------------------------------------------------
init_db()

# -----------------------------------------------------------------------------
# Location helpers
# -----------------------------------------------------------------------------
COUNTRY_CENTROIDS: Dict[str, Tuple[float, float, int]] = {
    "Singapore":       (1.3521, 103.8198, 11),
    "United States":   (39.8283, -98.5795, 4),
    "United Kingdom":  (55.3781,  -3.4360, 5),
    "Malaysia":        (4.2105, 102.9758, 5),
    "Thailand":        (15.8700, 100.9925, 5),
    "Vietnam":         (14.0583, 108.2772, 5),
    "Indonesia":       (-0.7893, 113.9213, 4),
    "Philippines":     (12.8797, 121.7740, 5),
    "India":           (20.5937,  78.9629, 4),
    "Australia":       (-25.2744, 133.7751, 4),
    "Japan":           (36.2048, 138.2529, 5),
    "South Korea":     (35.9078, 127.7669, 6),
    "Other":           (0.0, 0.0, 2),
}

REGION_COORDS: Dict[str, Dict[str, Tuple[float, float, int]]] = {
    "Singapore": {
        "— Country wide —": (1.3521, 103.8198, 11),
        "Downtown Core": (1.2831, 103.8510, 13),
        "Orchard":       (1.3049, 103.8318, 13),
        "Toa Payoh":     (1.3343, 103.8506, 12),
        "Bedok":         (1.3236, 103.9300, 12),
        "Tampines":      (1.3526, 103.9442, 12),
        "Yishun":        (1.4304, 103.8353, 12),
        "Woodlands":     (1.4360, 103.7863, 12),
        "Jurong East":   (1.3325, 103.7432, 12),
        "Sentosa":       (1.2494, 103.8303, 14),
    },
    "United States": {
        "— Country wide —": (39.8283, -98.5795, 4),
        "New York":      (40.7128, -74.0060, 10),
        "Los Angeles":   (34.0522, -118.2437, 10),
        "Chicago":       (41.8781, -87.6298, 10),
        "Houston":       (29.7604, -95.3698, 10),
        "Miami":         (25.7617, -80.1918, 11),
        "San Francisco": (37.7749, -122.4194, 11),
    },
    "United Kingdom": {
        "— Country wide —": (55.3781,  -3.4360, 5),
        "London":     (51.5074, -0.1278, 11),
        "Manchester": (53.4808, -2.2426, 11),
        "Bristol":    (51.4545, -2.5879, 11),
    },
}

def resolve_target_center(country: str, region: str) -> Tuple[float, float, int]:
    c_lat, c_lon, c_zoom = COUNTRY_CENTROIDS.get(country, COUNTRY_CENTROIDS["Other"])
    if not region or region.strip() in ("— Select —", "— Country wide —"):
        return c_lat, c_lon, c_zoom
    r = REGION_COORDS.get(country, {}).get(region)
    return r if r else (c_lat, c_lon, c_zoom)

# -----------------------------------------------------------------------------
# Data helpers (weather, simple river surrogate)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_current_weather(lat: float, lon: float) -> Dict[str, float | None]:
    rain_1h = 0.0
    temp = humidity = wind = None
    try:
        if OWM_KEY:
            r = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric"},
                timeout=12,
            )
            if r.ok:
                j = r.json()
                rb = j.get("rain") or {}
                if rb.get("1h") is not None:
                    rain_1h = float(rb["1h"])
                elif rb.get("3h") is not None:
                    rain_1h = float(rb["3h"]) / 3.0
                main = j.get("main") or {}
                windj = j.get("wind") or {}
                temp = main.get("temp")
                humidity = main.get("humidity")
                wind = windj.get("speed")
    except Exception:
        pass

    if (rain_1h is None or abs(rain_1h) < 1e-9):
        try:
            om = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "precipitation",
                    "forecast_days": 1,
                    "timezone": "auto",
                },
                timeout=10,
            ).json()
            precs = (om.get("hourly") or {}).get("precipitation") or []
            if precs:
                rain_1h = float(precs[-1] or 0.0)
        except Exception:
            pass

    return {"rain_1h": float(rain_1h or 0.0), "temp": temp, "humidity": humidity, "wind": wind}

def _clip_nonneg(x: float) -> float:
    return x if x is not None and x > 0 else 0.0

@st.cache_data(ttl=900, show_spinner=False)
def openmeteo_hourly(lat: float, lon: float, hours: int = 24) -> list[float]:
    try:
        j = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation",
                "past_hours": hours,
                "forecast_days": 1,
                "timezone": "UTC",
            },
            timeout=12,
        ).json()
        prec = (j.get("hourly") or {}).get("precipitation") or []
        return [float(x or 0) for x in prec][-hours:] if prec else []
    except Exception:
        return []

def fused_rain_now_and_24h(lat: float, lon: float) -> Tuple[float, float, Dict[str, float | None]]:
    now = fetch_current_weather(lat, lon)
    r1 = float(now["rain_1h"] or 0.0)
    last24 = openmeteo_hourly(lat, lon, hours=24)
    r24 = float(sum(last24)) if last24 else r1
    if r1 <= 1e-9 and last24:
        r1 = float(last24[-1])
    return r1, r24, now

def surrogate_river_level(rain_1h: float, rain_24h: float, elev_m: float) -> float:
    est = 0.6 + 0.05 * max(0.0, rain_24h) + 0.35 * max(0.0, rain_1h) - (elev_m / 300.0)
    return max(0.0, min(8.0, float(est)))

def model_predict_with_confidence(rain_mm: float, river_m: float, elev_m: float, urban: int):
    row = {
        "rainfall_mm": float(rain_mm),
        "river_level_m": float(river_m),
        "elevation_m": float(elev_m),
        "urban_area": int(urban),
    }
    df_X = pd.DataFrame([row])
    try:
        df_X = df_X[FEATURE_ORDER]
    except Exception:
        pass

    if MODEL is not None:
        try:
            if hasattr(MODEL, "predict_proba"):
                probs = MODEL.predict_proba(df_X)[0]
                labels = [str(c) for c in getattr(MODEL, "classes_", CLASSES)]
                pm = {lab: float(p) for lab, p in zip(labels, probs)}
                for k in ["Low", "Medium", "High"]:
                    pm.setdefault(k, 0.0)
                s = sum(pm.values()) or 1.0
                pm = {k: v / s for k, v in pm.items()}
                label = max(pm, key=lambda k: pm[k])
                return label, pm, "model"
            else:
                label = str(MODEL.predict(df_X)[0])
                return label, {label: 1.0}, "model"
        except Exception:
            pass

    # Fallback heuristic
    score = (rain_mm / 100.0) * 0.5 + (river_m / 8.0) * 0.35 + (1.0 - (elev_m / 120.0)) * 0.1 + (urban * 0.05)
    if score >= 0.6:
        label, conf = "High", min(0.5 + (score - 0.6), 0.95)
        pm = {"High": conf, "Medium": (1 - conf) * 0.6, "Low": (1 - conf) * 0.4}
    elif score >= 0.35:
        label, conf = "Medium", 0.55 + (0.6 - abs(score - 0.475)) * 0.4
        pm = {"Medium": conf, "High": (1 - conf) * 0.5, "Low": (1 - conf) * 0.5}
    else:
        label, conf = "Low", min(0.5 + (0.35 - score), 0.9)
        pm = {"Low": conf, "Medium": (1 - conf) * 0.6, "High": (1 - conf) * 0.4}
    s = sum(pm.values()) or 1.0
    return label, {k: v / s for k, v in pm.items()}, "rules"

def telegram_send(text: str) -> bool:
    if not (TG_TOKEN and TG_CHAT):
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(
            url, json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=12
        )
        return bool(resp.ok)
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Sidebar (simple routes)
# -----------------------------------------------------------------------------
menu = ui.sidebar_menu()
route = (menu.get("route") if isinstance(menu, dict) else (menu if isinstance(menu, str) else "Profile"))

# Ensure some defaults for cross-page state
st.session_state.setdefault("country", "Singapore")
st.session_state.setdefault("region", "— Country wide —")
st.session_state.setdefault("window", "30 days")
st.session_state.setdefault("simulate", False)
st.session_state.setdefault("manual_river", False)

# A safe default center (SG)
lat = float(st.session_state.get("lat_input", 1.3521))
lon = float(st.session_state.get("lon_input", 103.8198))
zoom_default = int(st.session_state.get("zoom_input", 11))

COUNTRY_OPTIONS = list(COUNTRY_CENTROIDS.keys())

# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------
if route in ("Profile", "Home"):
    ui.section_header("DisasterGuard • Analyst Studio", right="<span class='dg-badge'>Environment: Prod</span>")
    st.caption("Professional flood risk monitoring & analytics.")
    ui.render_rainviewer_embed(title="Global Weather Map Live", subtitle="Worldwide radar, animation enabled.",
                               center=None, zoom=2, height=620)

elif route in ("Global View", "Global Overview", "Global"):
    ui.section_header("🌐 Global Overview", right="<span class='dg-badge'>Radar: RainViewer</span>")
    ui.render_rainviewer_embed(title="", subtitle="Global RainViewer (worldwide radar).",
                               center=None, zoom=2, height=600)

elif route == "Monitoring":
    ui.section_header("📡 Monitoring — Local Ops Console", right="<span class='dg-badge'>Live</span>")

    go_stats_col, _ = st.columns([1, 6])
    with go_stats_col:
        if st.button("↗️ View full analysis in Statistics", key="btn_to_stats"):
            st.session_state["route"] = "Statistics"
            st.rerun()

    # controls
    settings = ui.monitoring_controls(
        country_options=COUNTRY_OPTIONS,
        default_country=st.session_state.get("country", "Singapore"),
        default_region=st.session_state.get("region", "— Country wide —"),
        default_window=st.session_state.get("window", "30 days"),
        default_simulate=st.session_state.get("simulate", False),
        default_manual_river=st.session_state.get("manual_river", False),
    )
    st.session_state.update(
        country=settings["country"],
        region=settings["region"],
        window=settings["window"],
        simulate=settings["simulate"],
        manual_river=settings["manual_river"],
    )

    lat, lon, zoom_default = resolve_target_center(settings["country"], settings["region"])
    st.session_state.update(lat_input=lat, lon_input=lon, zoom_input=zoom_default)

    elev_m = 20.0
    urban = 1
    rain_1h, rain_24h, now = fused_rain_now_and_24h(lat, lon)
    river_m = surrogate_river_level(rain_1h, rain_24h, elev_m)
    river_mode = "manual" if settings["manual_river"] else "estimated"
    if settings["manual_river"]:
        river_m = st.slider("Manual river level (m)", 0.0, 8.0, float(river_m), 0.1, key="manual_river_slider")
    if settings["simulate"]:
        import random
        rain_1h = max(0.0, rain_1h + random.uniform(-2.0, 8.0))
        rain_24h = max(0.0, rain_24h + random.uniform(-5.0, 15.0))
        river_m = max(0.0, min(8.0, river_m + random.uniform(-0.2, 0.6)))

    risk, prob_map, src = model_predict_with_confidence(rain_1h, river_m, elev_m, urban)

    ui.render_divider()
    ui.render_rainviewer_embed(title="Local Radar — Focus View",
                               subtitle=f"Centered near {settings['region'] or settings['country']}.",
                               center=(lat, lon), zoom=zoom_default, height=300)
    ui.render_divider()

    ui.render_metric_tiles(rain_1h=rain_1h, rain_24h=rain_24h, river_m=river_m, elev_m=elev_m, prob_map=prob_map)
    cA, cB = st.columns(2)
    with cA:
        ui.section_header("Risk Probability")
        ui.render_probability_breakdown(prob_map)
    with cB:
        ui.render_now_card(now=now, rain_1h=rain_1h, rain_24h=rain_24h, river_m=river_m, river_mode=river_mode)

    ui.render_divider()

    days = 7 if st.session_state["window"] == "7 days" else (30 if st.session_state["window"] == "30 days" else 90)

    loc_id = upsert_location(name=f"{st.session_state['country']} • Current",
                             lat=lat, lon=lon, elev_m=elev_m, urban=urban)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rid = insert_reading(loc_id, ts, rain_1h, now.get("temp"), now.get("humidity"), now.get("wind"))
    insert_prediction(rid, risk, "rf_v1" if src == "model" else "rules",
                      {"rain": rain_1h, "rain24": rain_24h, "river": river_m, "river_source": river_mode,
                       "elev": elev_m, "urban": urban, "probs": prob_map})

    if settings["telegram_clicked"] or risk == "High":
        sent = False
        if TG_TOKEN and TG_CHAT:
            msg = (
                f"🚨 <b>{'Manual' if settings['telegram_clicked'] else 'Auto'} Flood Update</b>\n"
                f"📍 {st.session_state['country']} — {st.session_state['region']}\n"
                f"Risk: <b>{risk}</b> (High {int(prob_map.get('High',0)*100)}%)\n"
                f"🌧 Rain(1h): {rain_1h:.1f} mm • 24h: {rain_24h:.1f} mm\n"
                f"🌊 River: {river_m:.2f} m ({river_mode})\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            sent = telegram_send(msg)
        insert_alert(loc_id, ts, risk, "telegram", delivered=1 if sent else 0)
        st.success("Telegram alert sent." if sent else "Telegram send failed or not configured.")

    daily_m = fetch_daily_aggregates(loc_id, days=days)
    risks_m = fetch_risk_counts(loc_id, days=days)
    ui.render_daily_rainfall(daily_m)
    ui.render_temp_humidity(daily_m)
    ui.render_risk_histogram(risks_m)
    ui.render_divider()
    ui.render_export_table(recent_snapshots(loc_id, limit=200))

elif route in ("Statistics", "Analytics"):
    # Delegate to statistics page (using current state)
    days = 7 if st.session_state.get("window") == "7 days" else (30 if st.session_state.get("window") == "30 days" else 90)
    lat, lon, _ = resolve_target_center(st.session_state.get("country", "Singapore"),
                                        st.session_state.get("region", "— Country wide —"))
    loc_id = upsert_location(name=f"{st.session_state.get('country','Singapore')} • Current",
                             lat=lat, lon=lon, elev_m=20.0, urban=1)
    run_statistics_page(loc_id=loc_id,
                        country=st.session_state.get("country", "Singapore"),
                        region=st.session_state.get("region", "— Country wide —"),
                        days=days)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.caption(
    f"🛰 {st.session_state.get('country','Singapore')} • Lat {lat:.4f}, Lon {lon:.4f} • Radar: RainViewer • UI: modular • "
    f"Build: {os.getenv('RENDER_GIT_COMMIT', 'local')[:7]}"
)