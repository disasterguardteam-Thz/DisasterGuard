#!/usr/bin/env python3
# ============================================================
# DisasterGuard AI – v4.2 (Hybrid UI, 2D map, real demo mode)
# - Real Live/Demo switch (demo simulates sensors visibly)
# - Search city ABOVE “Use my location”; pulsing device marker
# - Auto elevation via Open-Elevation (fallback 20 m)
# - SG river live (if key), others surrogate (US/UK next step)
# - Weather radar overlay with opacity slider
# - Auto Telegram alerts (High) + manual send button
# - Keeps: DB logging, Trends, Backfill, Retrain, Export
# ============================================================

import os, io, json, random, requests, joblib, pandas as pd, streamlit as st
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from streamlit_js_eval import get_geolocation  # “Use my location” button

from db import (
    init_db, upsert_location, insert_reading, insert_prediction,
    insert_alert, mark_alert_delivered,
    fetch_daily_aggregates, fetch_risk_counts, recent_snapshots
)

# ------------------------------------------------------------
# 🌐  App configuration
# ------------------------------------------------------------
st.set_page_config(page_title="DisasterGuard AI • Dashboard", layout="wide")

# ------------------------------------------------------------
# 🎨  Global styles
# ------------------------------------------------------------
st.markdown("""
<style>
:root { --low:#2e7d32; --med:#f9a825; --high:#d32f2f; --ink:#0f172a; --muted:#64748b; }
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-weight: 700; }
.prob-card { background: rgba(15,23,42,.04); padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(2,6,23,.06); box-shadow: 0 4px 14px rgba(2,6,23,.05); }
.prob-title { font-weight: 600; color: var(--ink); font-size: 0.95rem; margin-bottom: 8px; }
.strip { height: 12px; border-radius: 999px; overflow: hidden; display: flex; background: rgba(2,6,23,.08); }
.strip > div { height: 12px; }
.strip .low  { background: var(--low); }
.strip .med  { background: var(--med); }
.strip .high { background: var(--high); }
.strip-wrap { position: relative; }
.strip-label { position: absolute; top: -22px; transform: translateX(-50%); font-size: 0.78rem; color: var(--muted); white-space: nowrap; }
.legend { display:flex; gap:10px; margin-top: 8px; font-size:.85rem; color: var(--muted);}
.legend .dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; transform: translateY(1px); }
.chip { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border-radius:999px; font-weight:600; font-size:.90rem; border:1px solid rgba(2,6,23,.08); background: rgba(2,6,23,.04); }
.chip.low  { color: var(--low);  }
.chip.med  { color: var(--med);  }
.chip.high { color: var(--high); }
.small { color: var(--muted); font-size:.85rem; }

.inline-legend {display:flex;gap:14px;align-items:center;justify-content:flex-end;font-size:.9rem;margin-top:6px;opacity:.9}
.inline-legend .sw{width:12px;height:12px;border-radius:3px;display:inline-block;margin-right:6px}

.badge {display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border-radius:999px;font-weight:600;font-size:.85rem;border:1px solid rgba(2,6,23,.08);background:rgba(2,6,23,.04);color:#6941C6}
.pulse-dot{width:10px;height:10px;border-radius:50%;background:#1976d2;box-shadow:0 0 0 rgba(25,118,210,.7);animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(25,118,210,.7)}70%{box-shadow:0 0 0 12px rgba(25,118,210,0)}100%{box-shadow:0 0 0 0 rgba(25,118,210,0)}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🔐  Secrets helper
# ------------------------------------------------------------
def _secret_or_env(key: str, env_fallback: str | None = None) -> str | None:
    try: return st.secrets[key]
    except Exception: return os.environ.get(env_fallback or key)

OWM_KEY   = _secret_or_env("OWM_KEY")
TG_TOKEN  = _secret_or_env("TELEGRAM_BOT_TOKEN")
TG_CHAT   = _secret_or_env("TELEGRAM_CHAT_ID")
SG_APIKEY = (_secret_or_env("DATA_GOV_SG_API_KEY") or _secret_or_env("SG_API_KEY") or _secret_or_env("PUB_API_KEY"))

MODEL_PATH = "models/flood_model.pkl"
META_PATH  = "models/metadata.json"

# ------------------------------------------------------------
# 🤖  Load model + metadata
# ------------------------------------------------------------
MODEL = None
CLASSES = ["Low","Medium","High"]
FEATURE_ORDER = ["rainfall_mm","river_level_m","elevation_m","urban_area"]

try:
    if os.path.exists(META_PATH):
        with open(META_PATH,"r",encoding="utf-8") as f:
            meta = json.load(f)
            if isinstance(meta,dict) and "feature_order" in meta:
                FEATURE_ORDER = list(meta["feature_order"])
except Exception: pass

try:
    MODEL = joblib.load(MODEL_PATH)
    if hasattr(MODEL,"classes_"): CLASSES = [str(c) for c in MODEL.classes_]
except Exception: MODEL = None

init_db()

# ============================================================
# 🧭  LOCATION & SOURCE CONTROLS
# ============================================================
st.sidebar.title("DisasterGuard AI")

# Source mode (REAL vs DEMO) — now actually changes behavior:
data_mode = st.sidebar.radio("Data source", ["Live", "Demo"], index=0, horizontal=True)
st.sidebar.caption("Live = real APIs. Demo = simulated sensors for demos without keys.")

# Search city ABOVE “Use my location”
def geocode_place(place:str):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q":place,"format":"json","limit":1},
                         headers={"User-Agent":"DisasterGuardAI"})
        j = r.json()
        if j: return float(j[0]["lat"]), float(j[0]["lon"]), j[0].get("display_name","")
    except Exception: pass
    return None, None, None

place = st.sidebar.text_input("🔎 Search city or area")
lat, lon = 1.3521, 103.8198  # default (SG)
place_label = "Singapore"

if place:
    glat, glon, label = geocode_place(place)
    if glat and glon:
        lat, lon = glat, glon
        place_label = label or place
        st.sidebar.success(f"📍 {place_label} ({lat:.4f}, {lon:.4f})")
    else:
        st.sidebar.warning("No matching location found.")

# “Use my location” (device GPS)
if st.sidebar.button("📡 Use my current location"):
    pos = get_geolocation()
    if pos and isinstance(pos, dict) and "coords" in pos:
        lat = float(pos["coords"]["latitude"])
        lon = float(pos["coords"]["longitude"])
        place_label = "Device location"
        st.sidebar.success(f"Using device GPS: {lat:.4f}, {lon:.4f}")
    else:
        st.sidebar.warning("Could not read device GPS.")

# Radar overlay
show_radar = st.sidebar.toggle("🌧 Weather radar overlay", value=False)
radar_opacity = st.sidebar.slider("Radar opacity", 0.0, 1.0, 0.65, 0.05, disabled=not show_radar)

# Telegram alerts + Demo toggle lives here
send_alerts = st.sidebar.toggle("Enable Telegram Alerts (High risk)", value=True)
simulate = (data_mode == "Demo")  # <- demo mode drives simulation automatically

# Manual river slider only when no live API
prefer_manual_when_no_api = st.sidebar.toggle("Use manual river slider when no live API", value=False)

# Elevation is auto; we show it read-only (with fallback)
@st.cache_data(ttl=1800)
def fetch_elevation(lat: float, lon: float):
    try:
        r = requests.get("https://api.open-elevation.com/api/v1/lookup",
                         params={"locations": f"{lat},{lon}"}, timeout=10)
        j = r.json()
        results = j.get("results") or []
        if results and "elevation" in results[0]:
            return float(results[0]["elevation"])
    except Exception:
        pass
    return 20.0  # fallback
elev_m = fetch_elevation(lat, lon)
st.sidebar.markdown(f"**Elevation (auto):** {elev_m:.0f} m")

# Urban flag: keep simple
urban  = 1 if st.sidebar.selectbox("Urban Area?", ["Yes","No"], index=0) == "Yes" else 0

# Trends window selector (moved out of original sidebar order a bit cleaner)
trend_period = st.sidebar.selectbox("Trends window", ["7 days","30 days","90 days"], index=1)

# Persist shown coordinates
st.session_state["lat_input"] = lat
st.session_state["lon_input"] = lon

# Ensure location row exists
loc_id = upsert_location(name="Current Location", lat=lat, lon=lon, elev_m=elev_m, urban=urban)

# ============================================================
# ☁️  WEATHER & RIVER DATA
# ============================================================
@st.cache_data(ttl=600)
def fetch_current_weather(lat: float, lon: float):
    """Return rain_1h (mm), temp (°C), humidity (%), wind (m/s)."""
    rain_1h = 0.0
    temp = humidity = wind = None

    # Live: OpenWeather + fallback to Open-Meteo
    try:
        if OWM_KEY:
            r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                             params={"lat": lat, "lon": lon, "appid": OWM_KEY, "units": "metric"},
                             timeout=12)
            j = r.json() if r.ok else {}
            rain_block = (j.get("rain") or {})
            if rain_block.get("1h") is not None: rain_1h = float(rain_block["1h"])
            elif rain_block.get("3h") is not None: rain_1h = float(rain_block["3h"]) / 3.0
            main = j.get("main") or {}; windj = j.get("wind") or {}
            temp, humidity, wind = main.get("temp"), main.get("humidity"), windj.get("speed")
    except Exception:
        pass

    # Fallback (or if no OWM key)
    if (rain_1h is None) or (abs(rain_1h) < 1e-9):
        try:
            om = requests.get("https://api.open-meteo.com/v1/forecast",
                              params={"latitude":lat,"longitude":lon,"hourly":"precipitation","forecast_days":1,"timezone":"auto"},
                              timeout=10).json()
            precs = (om.get("hourly") or {}).get("precipitation") or []
            if precs: rain_1h = float(precs[-1] or 0.0)
        except Exception:
            pass

    return {"rain_1h": float(rain_1h or 0.0), "temp": temp, "humidity": humidity, "wind": wind}

@st.cache_data(ttl=900)
def openmeteo_hourly(lat: float, lon: float, hours: int = 24):
    """Return last `hours` hourly precipitation (mm) from Open-Meteo."""
    try:
        j = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon, "hourly": "precipitation",
                                 "past_hours": hours, "forecast_days": 1, "timezone": "UTC"},
                         timeout=12).json()
        prec = (j.get("hourly") or {}).get("precipitation") or []
        return [float(x or 0) for x in prec][-hours:] if prec else []
    except Exception:
        return []

def fused_rain_now_and_24h(lat: float, lon: float):
    now = fetch_current_weather(lat, lon)
    rain_1h = float(now["rain_1h"] or 0.0)
    last24 = openmeteo_hourly(lat, lon, hours=24)
    rain_24h = float(sum(last24)) if last24 else float(rain_1h)
    if rain_1h <= 1e-9 and last24:
        rain_1h = float(last24[-1])
    return rain_1h, rain_24h, now

@st.cache_data(ttl=300)
def sg_nearest_river_level(lat: float, lon: float, api_key: str | None):
    """Singapore PUB water-level feed. Returns (river_level_m, station_label)."""
    try:
        url = "https://api.data.gov.sg/v1/environment/water-level"
        headers = {"api-key": api_key} if api_key else {}
        r = requests.get(url, headers=headers, timeout=10)
        if not r.ok: return (None, None)
        j = r.json()
        items = j.get("items") or []
        readings = items[0].get("readings") or [] if items else []
        if not readings: return (None, None)
        best = max(readings, key=lambda rr: float(rr.get("value") or 0.0))
        return float(best.get("value") or 0.0), f"PUB {best.get('station_id')}"
    except Exception:
        return (None, None)

def surrogate_river_level(rain_1h: float, rain_24h: float, elev_m: float):
    """Simple bounded proxy in meters (0..8) using rain + elevation."""
    est = 0.6 + 0.05 * max(0.0, rain_24h) + 0.35 * max(0.0, rain_1h) - (elev_m / 300.0)
    return max(0.0, min(8.0, float(est)))

def telegram_send(text: str):
    if not (TG_TOKEN and TG_CHAT): return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=12)
        return resp.ok
    except Exception:
        return False

# ============================================================
# 🤖  MODEL INFERENCE
# ============================================================
def model_predict_with_confidence(rain_mm: float, river_m: float, elev_m: float, urban: int):
    row = {"rainfall_mm": float(rain_mm), "river_level_m": float(river_m), "elevation_m": float(elev_m), "urban_area": int(urban)}
    df_X = pd.DataFrame([row])
    try: df_X = df_X[FEATURE_ORDER]
    except Exception: pass

    if MODEL is not None:
        try:
            if hasattr(MODEL, "predict_proba"):
                probs = MODEL.predict_proba(df_X)[0]
                labels = [str(c) for c in getattr(MODEL, "classes_", CLASSES)]
                prob_map = {lab: float(p) for lab, p in zip(labels, probs)}
                for k in ["Low", "Medium", "High"]: prob_map.setdefault(k, 0.0)
                s = sum(prob_map.values()) or 1.0
                prob_map = {k: v/s for k, v in prob_map.items()}
                label = max(prob_map, key=prob_map.get)
                return label, prob_map, "model"
            else:
                label = str(MODEL.predict(df_X)[0]); return label, {label: 1.0}, "model"
        except Exception:
            pass

    # Rule-based fallback
    score = (rain_mm / 100.0) * 0.5 + (river_m / 8.0) * 0.35 + (1.0 - (elev_m / 120.0)) * 0.1 + (urban * 0.05)
    if score >= 0.6:
        return "High", {"High": 0.9, "Medium": 0.07, "Low": 0.03}, "rules"
    elif score >= 0.35:
        return "Medium", {"Medium": 0.8, "High": 0.1, "Low": 0.1}, "rules"
    return "Low", {"Low": 0.85, "Medium": 0.1, "High": 0.05}, "rules"

# ============================================================
# 📡  READ SENSORS (Live/Demo), RIVER ROUTING, RISK
# ============================================================
rain_1h, rain_24h, now = fused_rain_now_and_24h(lat, lon)

# Demo mode: simulate sensors (visible)
if simulate:
    # gentle drift + bursts
    rain_1h = max(0.0, rain_1h + random.uniform(-2.0, 10.0))
    rain_24h = max(0.0, rain_24h + random.uniform(-5.0, 30.0))
    now["temp"] = (now["temp"] or 26.0) + random.uniform(-1.0, 1.5)
    now["humidity"] = (now["humidity"] or 80.0) + random.uniform(-5, 7)
    now["wind"] = (now["wind"] or 2.0) + random.uniform(-0.5, 1.2)

# River: SG live if key + in bbox, else surrogate (US/UK live in next step)
SG_BBOX = (0.8 <= lat <= 1.6) and (103.4 <= lon <= 104.2)
river_source = "surrogate"; river_station = None
if SG_BBOX and SG_APIKEY:
    lvl, station = sg_nearest_river_level(lat, lon, SG_APIKEY)
    if (lvl is not None) and (lvl > 0):
        river_m = float(lvl); river_source = "live"; river_station = station
    else:
        river_m = surrogate_river_level(rain_1h, rain_24h, elev_m)
else:
    river_m = surrogate_river_level(rain_1h, rain_24h, elev_m)

# Manual slider if preferred and not live
if (river_source != "live") and prefer_manual_when_no_api:
    river_m = st.sidebar.slider("Manual river level (m)", 0.0, 8.0, float(river_m), 0.1)

# Predict
risk, prob_map, src = model_predict_with_confidence(rain_1h, river_m, elev_m, urban)
risk_chip_cls = "chip " + ("high" if risk=="High" else "med" if risk=="Medium" else "low")

# ============================================================
# 📊  OVERVIEW METRICS (original look)
# ============================================================
st.markdown("## Overview")
col1, col2, col3 = st.columns([1.25,1,1.2])

with col1:
    st.metric("Rain (1h, mm)", f"{rain_1h:.1f}")
    st.metric("Rain (24h, mm)", f"{rain_24h:.1f}")
    st.metric("Temperature (°C)", None if now["temp"] is None else f"{now['temp']:.1f}")
    st.metric("Humidity (%)", None if now["humidity"] is None else f"{now['humidity']:.0f}")
    st.metric("Wind (m/s)", None if now["wind"] is None else f"{now['wind']:.1f}")

with col2:
    st.markdown(f"**Risk:** <span class='{risk_chip_cls}'>{risk}</span> <span class='small'>(from {src})</span>", unsafe_allow_html=True)

    # Probability strip (unchanged)
    def render_prob_strip(prob_map: dict):
        order = [("Low","low"), ("Medium","med"), ("High","high")]
        vals = [float(prob_map.get(lbl, 0.0)) for lbl,_ in order]
        s = sum(vals) or 1.0
        vals = [v/s for v in vals]
        w_low,w_med,w_high = [max(0.0, min(100.0, v*100.0)) for v in vals]
        pos_low, pos_med, pos_high = w_low/2, w_low+w_med/2, w_low+w_med+w_high/2
        html = f"""
        <div class="prob-card">
          <div class="prob-title">Confidence</div>
          <div class="strip-wrap">
            <div class="strip" role="meter" aria-valuemin="0" aria-valuemax="1" aria-valuenow="{max(vals)}">
              <div class="low"  style="width:{w_low:.2f}%"></div>
              <div class="med"  style="width:{w_med:.2f}%"></div>
              <div class="high" style="width:{w_high:.2f}%"></div>
            </div>
            <div class="strip-label" style="left:{pos_low:.2f}%">Low {vals[0]*100:.1f}%</div>
            <div class="strip-label" style="left:{pos_med:.2f}%">Med {vals[1]*100:.1f}%</div>
            <div class="strip-label" style="left:{pos_high:.2f}%">High {vals[2]*100:.1f}%</div>
          </div>
          <div class="legend">
            <span><span class="dot" style="background:var(--low)"></span>Low</span>
            <span><span class="dot" style="background:var(--med)"></span>Medium</span>
            <span><span class="dot" style="background:var(--high)"></span>High</span>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    render_prob_strip(prob_map)

    # Source chips
    river_help = ("Live PUB" if river_source=="live" else "Estimated from rainfall+elevation" if river_source=="surrogate" else "Manual")
    chip_cls = "chip " + ("high" if river_source=="live" else "med" if river_source=="surrogate" else "")
    st.metric("River Level (m)", f"{river_m:.2f}", help=river_help)
    st.markdown(f'<span class="{chip_cls}">{river_help}</span>', unsafe_allow_html=True)
    if simulate:
        st.markdown('<span class="badge"><span class="pulse-dot"></span> Demo sensors active</span>', unsafe_allow_html=True)

with col3:
    st.markdown("### Flood-Risk Map")

# ============================================================
# 🗺  INTERACTIVE MAP (2D pydeck)
# ============================================================
CARTO_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
CARTO_DARK  = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
RAINVIEWER  = "https://tilecache.rainviewer.com/v2/radar/{z}/{x}/{y}/256/3/1_1.png"

user_theme = st.get_option("theme.base")
map_style = CARTO_DARK if user_theme == "dark" else CARTO_LIGHT

# Colors
def hex_to_rgba(h, a): h=h.lstrip("#"); return [int(h[0:2],16), int(h[2:4],16), int(h[4:6],16), a]
color_hex = "#2e7d32" if risk=="Low" else "#f9a825" if risk=="Medium" else "#d32f2f"
halo_outer, halo_inner = hex_to_rgba(color_hex,40), hex_to_rgba(color_hex,120)
dot_color=[25,118,210,240]  # device marker
p_high = float(prob_map.get("High", 0.0))
radius_m = int(400 + 700 * p_high)

layers = []

# Optional radar overlay
if show_radar:
    layers.append(pdk.Layer(
        "TileLayer",
        data=RAINVIEWER,
        min_zoom=0,
        max_zoom=19,
        opacity=float(radar_opacity),
        pickable=False,
    ))

# Risk halo + center
layers.extend([
    pdk.Layer(
        "CircleLayer",
        data=[{"lat": lat, "lon": lon, "radius": radius_m + 250, "color": halo_outer}],
        get_position=["lon", "lat"], get_radius="radius", radius_units="meters",
        get_fill_color="color", stroked=False, pickable=False,
    ),
    pdk.Layer(
        "CircleLayer",
        data=[{"lat": lat, "lon": lon, "radius": radius_m, "color": halo_inner}],
        get_position=["lon", "lat"], get_radius="radius", radius_units="meters",
        get_fill_color="color", get_line_color="color", stroked=True,
        line_width_min_pixels=1, pickable=True,
    ),
    # Pulsing device marker (represented as a solid dot; pulse is CSS in legend/badge)
    pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": lat, "lon": lon}],
        get_position=["lon", "lat"], get_radius=60, radius_units="meters",
        get_fill_color=dot_color, stroked=False, pickable=True,
    ),
])

# View
def zoom_for_radius(rad_m):
    if rad_m > 1200: return 11.5
    if rad_m > 900:  return 12.0
    if rad_m > 700:  return 12.5
    if rad_m > 500:  return 13.0
    return 13.5

tooltip_html = (
    f"<b>Flood Risk:</b> <span style='color:{color_hex}'>{risk}</span><br>"
    f"📍 {place_label}<br>"
    f"🌧 Rain (1h): <b>{rain_1h:.1f} mm</b> (24h: <b>{rain_24h:.1f} mm</b>)<br>"
    f"🌊 River: <b>{river_m:.2f} m</b> ({'Live' if river_source=='live' else 'Estimated' if river_source=='surrogate' else 'Manual'})<br>"
    f"⛰ Elev: <b>{elev_m:.0f} m</b> &nbsp;|&nbsp; 🏙 Urban: <b>{'Yes' if urban else 'No'}</b><br>"
    f"Confidence — Low: {prob_map.get('Low',0):.0%}, Med: {prob_map.get('Medium',0):.0%}, High: {prob_map.get('High',0):.0%}"
)

deck = pdk.Deck(
    map_style=map_style,
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom_for_radius(radius_m), pitch=0, bearing=0),
    layers=layers,
    tooltip={"html": tooltip_html, "style": {"backgroundColor": "rgba(18,18,18,0.85)", "color": "white"}},
    height=600,
)
st.pydeck_chart(deck, use_container_width=True)

# Inline legend below map
st.markdown(
    f"""
    <div class="inline-legend">
      <span><span class="sw" style="background:#2e7d32"></span>Low</span>
      <span><span class="sw" style="background:#f9a825"></span>Medium</span>
      <span><span class="sw" style="background:#d32f2f"></span>High</span>
      <span style="margin-left:auto;opacity:.6;">Base: {'Dark' if user_theme=='dark' else 'Light'}{' + Radar' if show_radar else ''}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 💾  LOG TO DB + ALERTS
# ============================================================
ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
reading_id = insert_reading(loc_id, ts, rain_1h, now["temp"], now["humidity"], now["wind"])
insert_prediction(
    reading_id, risk, ("rf_v1" if src == "model" else "rules"),
    {
        "rain": rain_1h, "rain24": rain_24h, "river": river_m,
        "river_source": "live" if (SG_BBOX and river_source=="live") else river_source,
        "elev": elev_m, "urban": urban, "probs": prob_map
    },
)

# Auto alert on High
if send_alerts and risk == "High":
    source_tag = f"({'Live' if river_source=='live' else 'Estimated' if river_source=='surrogate' else 'Manual'})"
    alert_id = insert_alert(loc_id, ts, risk, "telegram", delivered=0)
    sent = telegram_send(
        f"🚨 <b>High Flood Risk</b>\n"
        f"📍 {place_label}\n"
        f"🧭 Lat {lat:.4f}, Lon {lon:.4f}\n"
        f"🌧 Rain(1h): {rain_1h:.1f} mm (24h: {rain_24h:.1f} mm)\n"
        f"🌊 River: {river_m:.2f} m {source_tag}\n"
        f"⛰ Elev: {elev_m:.0f} m | 🏙 Urban: {'Yes' if urban else 'No'}\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    if sent:
        mark_alert_delivered(alert_id)
        st.info("Telegram alert sent.")
    else:
        st.warning("Telegram alert failed. Check token/chat_id.")

# Manual alert button (send under any conditions)
if st.button("📣 Send Telegram alert now"):
    if TG_TOKEN and TG_CHAT:
        _ = telegram_send(
            f"📣 <b>Manual Flood Risk Alert</b>\n"
            f"📍 {place_label}\n"
            f"🧭 Lat {lat:.4f}, Lon {lon:.4f}\n"
            f"🌧 Rain(1h): {rain_1h:.1f} mm (24h: {rain_24h:.1f} mm)\n"
            f"🌊 River: {river_m:.2f} m ({river_source})\n"
            f"⛰ Elev: {elev_m:.0f} m | 🏙 Urban: {'Yes' if urban else 'No'}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        st.success("Manual alert sent (check Telegram).")
    else:
        st.warning("Telegram token/chat not set.")

# ----------------- SNAPSHOT BUTTON -----------------
if st.button("📸 Capture snapshot now"):
    ts2 = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    r_id = insert_reading(loc_id, ts2, rain_1h, now["temp"], now["humidity"], now["wind"])
    insert_prediction(
        r_id, risk, ("rf_v1" if src == "model" else "rules"),
        {
            "rain": rain_1h, "rain24": rain_24h, "river": river_m,
            "river_source": "live" if (SG_BBOX and river_source=="live") else river_source,
            "elev": elev_m, "urban": urban, "probs": prob_map
        }
    )
    st.success("Snapshot saved!")

# ---------- Backfill UI ----------
with st.expander("🗃 Backfill history (Open-Meteo)"):
    days_to_backfill = st.slider("Days to backfill", 3, 30, 7, 1)
    if st.button("Backfill now"):
        try:
            j = requests.get("https://api.open-meteo.com/v1/forecast",
                             params={"latitude":lat,"longitude":lon,
                                     "daily":"precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean",
                                     "past_days":days_to_backfill,"forecast_days":0,"timezone":"UTC"},
                             timeout=15).json()
            daily = j.get("daily") or {}; dates = daily.get("time") or []
            precs = daily.get("precipitation_sum") or []; temps = daily.get("temperature_2m_mean") or []; hums = daily.get("relative_humidity_2m_mean") or []
            loc_id_local = upsert_location("Current Location", lat, lon, elev_m, urban)
            for i, d in enumerate(dates):
                ts_utc = f"{d}T12:00:00Z"
                rain_day = float(precs[i] or 0.0) if i < len(precs) else 0.0
                temp_day = float(temps[i] or 0.0) if i < len(temps) else None
                hum_day  = float(hums[i]  or 0.0) if i < len(hums)  else None
                rid = insert_reading(loc_id_local, ts_utc, rain_day, temp_day, hum_day, None)
                label, _, src2 = model_predict_with_confidence(rain_day, 3.0, elev_m, urban)
                insert_prediction(rid, label, ("rf_v1" if src2 == "model" else "rules"),
                                  {"rain": rain_day, "elev": elev_m, "urban": urban, "backfill": True})
            st.success(f"Backfilled {len(dates)} day(s) into the local DB.")
        except Exception as e:
            st.error(f"Backfill failed: {e}")

# ---------- Train / Retrain panel ----------
with st.expander("🧠 Retrain AI Model (Advanced)"):
    st.caption("Rebuilds the flood-risk classifier using local snapshots (DB), else CSV/demo.")
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta_prev = json.load(f)
            st.json(meta_prev)
        except Exception:
            pass
    if st.button("🔁 Retrain now"):
        import importlib, contextlib
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tm = importlib.import_module("train_model")
                importlib.reload(tm)
                tm.main()
            logs = buf.getvalue()
            st.success("Training finished. Reloading model…")
            new_model = joblib.load(MODEL_PATH)
            new_classes = [str(c) for c in getattr(new_model, "classes_", CLASSES)]
            globals()["MODEL"] = new_model
            globals()["CLASSES"] = new_classes
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    meta_now = json.load(f)
                if "feature_order" in meta_now:
                    globals()["FEATURE_ORDER"] = list(meta_now["feature_order"])
            except Exception:
                pass
            st.info("Model reloaded successfully.")
            st.text_area("Training logs / metrics", logs, height=240)
        except Exception as e:
            st.error(f"Training failed: {e}")

st.divider()

# ----------------- TRENDS -----------------
st.markdown("## Trends")
days = 7 if trend_period == "7 days" else 30 if trend_period == "30 days" else 90
daily = fetch_daily_aggregates(loc_id, days=days)
risk_counts = fetch_risk_counts(loc_id, days=days)

colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("Daily Rainfall (mm)")
    if daily.empty:
        st.info("No snapshots yet. Capture a few or keep the app open.")
    else:
        fig_rain = px.line(daily, x="date", y="rain_mm", markers=True)
        fig_rain.update_layout(height=280, margin=dict(l=10, r=10, b=10, t=10))
        st.plotly_chart(fig_rain, use_container_width=True)

with colB:
    st.subheader("Temperature & Humidity (mean)")
    if daily.empty:
        st.info("No local data yet.")
    else:
        fig_th = go.Figure()
        fig_th.add_trace(go.Scatter(x=daily["date"], y=daily["temp_c"], mode="lines+markers", name="Temp (°C)"))
        fig_th.add_trace(go.Scatter(x=daily["date"], y=daily["humidity"], mode="lines+markers", name="Humidity (%)", yaxis="y2"))
        fig_th.update_layout(
            height=280, margin=dict(l=10, r=10, b=10, t=10),
            yaxis=dict(title="Temp (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right")
        )
        st.plotly_chart(fig_th, use_container_width=True)

st.subheader("Risk Level Frequency")
if risk_counts.empty:
    st.info("No risk records yet.")
else:
    fig_risk = px.bar(
        risk_counts, x="date", y="count", color="risk",
        barmode="stack", category_orders={"risk": ["Low","Medium","High","Unknown"]}
    )
    fig_risk.update_layout(height=320, margin=dict(l=10, r=10, b=10, t=10))
    st.plotly_chart(fig_risk, use_container_width=True)

# ----------------- RECENT SNAPSHOTS + EXPORT -----------------
st.subheader("Recent Snapshots")
snap = recent_snapshots(loc_id, limit=200)
st.dataframe(snap, use_container_width=True, height=260)
csv = snap.to_csv(index=False).encode("utf-8") if not snap.empty else b""
st.download_button(
    label="📤 Download CSV",
    data=csv,
    file_name=f"disasterguard_snapshots_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
    disabled=snap.empty
)

# ----------------- DAILY SUMMARY -----------------
if not daily.empty:
    latest = daily.iloc[-1]
    st.success(
        f"📅 <b>Latest Daily Summary</b><br>"
        f"🌧 Rain: {latest['rain_mm']:.1f} mm | 🌡 Temp: {latest['temp_c']:.1f} °C | 💧 Humidity: {latest['humidity']:.0f}%",
        icon="📊"
    )

# ----------------- ABOUT -----------------
with st.expander("ℹ️ About & Credits"):
    st.markdown("""
**DisasterGuard AI** — 2D map with radar, geolocation, auto elevation, and live/demo modes.  
• Weather: OpenWeather (current) + Open-Meteo (hourly/daily)  
• River: PUB via Data.gov.sg (SG) or surrogate elsewhere  
• Map: deck.gl • Charts: Plotly • DB: SQLite • Alerts: Telegram  
• Geocoding: OpenStreetMap Nominatim • Elevation: Open-Elevation
    """)

# ----------------- AUTO REFRESH (defensive) -----------------
st.caption("⏳ Auto-refresh every ~2 min (rerun triggers on control change).")
try:
    st.query_params.update({"_": str(int(datetime.now().timestamp()))})
except Exception:
    try:
        st.experimental_set_query_params(_=datetime.now().timestamp())
    except Exception:
        pass