# Glassy Monitoring page wrapper – UI only (reuses app/db helpers at runtime)
from __future__ import annotations

import importlib
from datetime import datetime, timezone

import streamlit as st

from ..styles import section_header, render_divider, spacer, PALETTE
from .. import charts
from .. import components

def _days_from_window(label: str) -> int:
    if label == "7 days":
        return 7
    if label == "90 days":
        return 90
    return 30

def render():
    # Import runtime helpers from the already-loaded main app
    app = importlib.import_module("app")

    from db import (
        upsert_location,
        insert_reading,
        insert_prediction,
        insert_alert,
        mark_alert_delivered,
        fetch_daily_aggregates,
        fetch_risk_counts,
        recent_snapshots,
    )

    section_header("📡 Monitoring — Local Ops Console", right="<span class='dg-badge'>Live</span>")
    spacer(6)

    go_stats, _ = st.columns([1, 6])
    with go_stats:
        if st.button("↗️ View full analysis in Statistics", key="btn_to_stats_glassy"):
            st.session_state["ui_force_route"] = "Statistics"
            st.rerun()

    spacer(6)
    render_divider()

    country_opts = list(app.COUNTRY_CENTROIDS.keys())
    settings = components.monitoring_controls(
        country_options=country_opts,
        default_country=st.session_state.get("country", "Singapore"),
        default_region=st.session_state.get("region", "— Country wide —"),
        default_window=st.session_state.get("window", "30 days"),
        default_simulate=st.session_state.get("simulate", False),
        default_manual_river=st.session_state.get("manual_river", False),
    )

    st.session_state["country"] = settings["country"]
    st.session_state["region"] = settings["region"]
    st.session_state["window"] = settings["window"]
    st.session_state["simulate"] = settings["simulate"]
    st.session_state["manual_river"] = settings["manual_river"]

    lat, lon, zoom = app.resolve_target_center(settings["country"], settings["region"])
    st.session_state["lat_input"] = float(lat)
    st.session_state["lon_input"] = float(lon)
    st.session_state["zoom_input"] = int(zoom)

    elev_m = 20.0
    urban = 1

    rain_1h, rain_24h, now = app.fused_rain_now_and_24h(lat, lon)

    river_mode = "manual" if settings["manual_river"] else "estimated"
    river_m = app.surrogate_river_level(rain_1h, rain_24h, elev_m)
    if settings["manual_river"]:
        river_m = st.slider(
            "Manual river level (m)", 0.0, 8.0, float(river_m), 0.1,
            key="mon_manual_river_slider",
            help="Override the estimated river level for what-if analysis.",
        )

    if settings["simulate"]:
        import random
        rain_1h = max(0.0, rain_1h + random.uniform(-2.0, 8.0))
        rain_24h = max(0.0, rain_24h + random.uniform(-5.0, 15.0))
        river_m = max(0.0, min(8.0, river_m + random.uniform(-0.2, 0.6)))

    risk, prob_map, src = app.model_predict_with_confidence(rain_1h, river_m, elev_m, urban)

    render_divider()

    charts.render_rainviewer_embed(
        title="Local Radar — Focus View",
        subtitle=f"{settings['country']} • {settings['region']}",
        center=(lat, lon),
        zoom=zoom,
        height=320,
    )

    spacer(8)

    charts.render_metric_tiles(
        rain_1h=rain_1h, rain_24h=rain_24h, river_m=river_m, elev_m=elev_m, prob_map=prob_map
    )

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        section_header("Risk Probability")
        charts.render_probability_breakdown(prob_map)
    with c2:
        charts.render_now_card(now=now, rain_1h=rain_1h, rain_24h=rain_24h, river_m=river_m, river_mode=river_mode)

    render_divider()

    days = _days_from_window(settings["window"])

    loc_id = upsert_location(
        name=f"{settings['country']} • {settings['region']}",
        lat=lat, lon=lon, elev_m=elev_m, urban=urban,
    )

    daily = fetch_daily_aggregates(loc_id, days=days)
    risk_counts = fetch_risk_counts(loc_id, days=days)

    charts.render_daily_rainfall(daily)
    charts.render_temp_humidity(daily)
    charts.render_risk_histogram(risk_counts)

    render_divider()

    snap = recent_snapshots(loc_id, limit=200)
    charts.render_export_table(snap)

    if settings["telegram_clicked"]:
        if not (app.TG_TOKEN and app.TG_CHAT):
            st.warning("Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as environment variables.")
        else:
            sent = app.telegram_send(
                f"🛰 <b>Manual Flood Update</b>\n"
                f"📍 {settings['country']} — {settings['region']} (lat {lat:.4f}, lon {lon:.4f})\n"
                f"Risk: <b>{int((prob_map.get('High',0.0) or 0)*100)}% High</b> ({risk}/{src})\n"
                f"🌧 Rain(1h): {rain_1h:.1f} mm • 24h: {rain_24h:.1f} mm\n"
                f"🌊 River: {river_m:.2f} m ({river_mode.title()})\n"
                f"⛰ Elev: {elev_m:.0f} m • 🏙 Urban: {'Yes' if urban else 'No'}\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            if sent:
                st.success("Manual alert sent.")
            else:
                st.warning("Manual alert failed (check token/chat id).")

    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reading_id = insert_reading(
        location_id=loc_id,
        ts_utc=ts_utc,
        rain_1h=rain_1h,
        temp_c=now.get("temp") if isinstance(now, dict) else None,
        humidity=now.get("humidity") if isinstance(now, dict) else None,
        wind_ms=now.get("wind") if isinstance(now, dict) else None,
    )
    insert_prediction(
        reading_id=reading_id,
        risk=risk,
        model="rf_v1" if src == "model" else "rules",
        score_dict={
            "rain": rain_1h, "rain24": rain_24h, "river": river_m,
            "river_source": river_mode, "elev": elev_m, "urban": urban, "probs": prob_map,
        },
    )

    if risk == "High":
        alert_id = insert_alert(loc_id, ts_utc, risk, channel="telegram", delivered=0)
        ok = app.telegram_send(
            f"🚨 <b>High Flood Risk</b>\n"
            f"📍 {settings['country']} — {settings['region']} (lat {lat:.4f}, lon {lon:.4f})\n"
            f"🌧 Rain(1h): {rain_1h:.1f} mm • 24h: {rain_24h:.1f} mm\n"
            f"🌊 River: {river_m:.2f} m ({river_mode.title()})\n"
            f"⛰ Elev: {elev_m:.0f} m • 🏙 Urban: {'Yes' if urban else 'No'}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        if ok:
            mark_alert_delivered(alert_id)
            st.info("Telegram alert sent.")
        else:
            st.warning("Telegram alert failed (check token/chat id).")

    spacer(8)
    st.caption(
        f"🛰 {settings['country']} • {settings['region']} • "
        f"Lat {lat:.4f}, Lon {lon:.4f} • Radar: RainViewer • UI: glassy"
    )