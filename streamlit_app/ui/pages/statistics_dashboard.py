# streamlit_app/ui/pages/statistics_dashboard.py
from __future__ import annotations

from datetime import datetime
import pandas as pd
import streamlit as st

from ..styles import PALETTE, section_header, spacer
from ..charts import (
    render_daily_rainfall,
    render_temp_humidity,
    render_risk_histogram,
    render_exceedance_curve,
    render_calendar_heatmap,
    render_correlation_matrix,
    render_top_events_table,
    render_risk_share,
    render_trend_anomaly,
)
from db import fetch_daily_aggregates, fetch_risk_counts

MODEL_VERSION = "v1.0"


@st.cache_data(ttl=300, show_spinner=False)
def aggregate_data(loc_id: int, days: int):
    daily = fetch_daily_aggregates(loc_id, days=days)
    risk_counts = fetch_risk_counts(loc_id, days=days)
    return daily, risk_counts


def _kpi(label: str, value_html: str, color: str):
    st.markdown(
        f"<div class='dg-card' style='padding:14px'>"
        f"<div style='color:{PALETTE['muted']};font-size:.85rem'>{label}</div>"
        f"<div style='font-weight:800;font-size:1.25rem;color:{color}'>{value_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def run_statistics_page(loc_id: int, country: str, region: str, days: int):
    # Header
    section_header(
        "Risk Analytics & Forecasting",
        right=f"<span class='dg-badge'>Model: {MODEL_VERSION}</span>",
    )

    # Fetch
    daily, risk_counts = aggregate_data(loc_id, days)

    # ---- KPIs (robust to missing columns) ----
    total_days = int(len(daily) if isinstance(daily, pd.DataFrame) else 0)
    avg_rain = (
        float(daily["rain_mm"].mean())
        if isinstance(daily, pd.DataFrame)
        and not daily.empty
        and "rain_mm" in daily.columns
        else 0.0
    )
    high_days = (
        int(
            risk_counts.loc[risk_counts["risk"] == "High", "count"].sum()
        )
        if isinstance(risk_counts, pd.DataFrame) and not risk_counts.empty
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Days analysed", f"{total_days}", PALETTE.get("cyan", PALETTE["accent"]))
    with c2:
        _kpi("Avg rainfall", f"{avg_rain:.1f} mm", PALETTE["purple"])
    with c3:
        _kpi("High-risk events", f"{high_days}", PALETTE["danger"])
    with c4:
        _kpi("Window", f"{days} days", PALETTE["ok"])

    spacer(8)

    # ---- Trend + composition ----
    col1, col2 = st.columns([1.45, 1.0])
    with col1:
        section_header("Rainfall Trend & Anomaly")
        render_trend_anomaly(daily, window=7)
    with col2:
        section_header("Risk Composition")
        render_risk_share(risk_counts)

    spacer(8)

    # ---- Heatmap + correlations ----
    col3, col4 = st.columns(2)
    with col3:
        section_header("Monthly Rainfall Heatmap")
        render_calendar_heatmap(daily)
    with col4:
        section_header("Metric Correlations")
        render_correlation_matrix(daily)

    spacer(8)

    # ---- Top events + temp/humidity ----
    col5, col6 = st.columns(2)
    with col5:
        section_header("Top Rain Events")
        render_top_events_table(daily, n=15)
    with col6:
        section_header("Temperature & Humidity")
        render_temp_humidity(daily)

    spacer(8)

    # ---- Stacked risk by day ----
    section_header("Risk Level Frequency (Stacked)")
    render_risk_histogram(risk_counts)

    spacer(8)

    # ---- Exceedance / Return period curve ----
    section_header("Exceedance Probability (Return Period)")
    render_exceedance_curve(daily)

    st.caption(f"Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")