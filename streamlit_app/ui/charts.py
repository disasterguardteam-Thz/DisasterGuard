# charts.py — UI-only visual components (glassy theme)
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .styles import PALETTE, PLOT_TEMPLATE, PLOT_BG

# ---------- Safe color fallbacks (prevents KeyError on missing keys) ----------
OK = PALETTE.get("ok", "#22c55e")
WARN = PALETTE.get("warn", "#f59e0b")
DANGER = PALETTE.get("danger", "#ef4444")
MUTED = PALETTE.get("muted", "#94a3b8")
BLUE = PALETTE.get("blue", "#60a5fa")
PURPLE = PALETTE.get("purple", "#8b5cf6")
ACCENT = PALETTE.get("accent", "#a78bfa")
CYAN = PALETTE.get("cyan", "#06b6d4")  # <- key that caused the deploy crash


# ---------- Mini util ----------
def _safe_has_cols(df: pd.DataFrame | None, cols) -> bool:
    return (df is not None) and (not df.empty) and all(c in df.columns for c in cols)


# ---------- RainViewer embed ----------
def render_rainviewer_embed(
    *,
    title: str,
    subtitle: str = "",
    center: Optional[Tuple[float, float]] = None,
    zoom: int = 9,
    height: int = 620,
    dark: bool = False,
):
    if title:
        st.markdown(
            "<div class='dg-section'><h3>{}</h3><div></div></div>".format(title),
            unsafe_allow_html=True,
        )
    if subtitle:
        st.caption(subtitle)

    theme = 1 if dark else 0
    if center is None:
        url = (
            "https://www.rainviewer.com/map.html?"
            f"loc=0.00000,0.00000,2&layer=radar&overlay=0&op=100&lm=1&sm=1&sn=1&hu=0&bp={theme}"
        )
    else:
        lat, lon = center
        url = (
            "https://www.rainviewer.com/map.html?"
            f"loc={lat:.5f},{lon:.5f},{zoom}&layer=radar&overlay=0&op=100&lm=1&sm=1&sn=1&hu=0&bp={theme}"
        )
    components.iframe(url, height=height, scrolling=False)


# ---------- KPI tiles ----------
def _tile(label: str, value: str, *, value_color: Optional[str] = None):
    st.markdown("<div class='dg-tile'>", unsafe_allow_html=True)
    st.markdown(f"<div class='dg-t-label'>{label}</div>", unsafe_allow_html=True)
    if value_color:
        st.markdown(
            f"<div class='dg-t-value' style='color:{value_color}'>{value}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div class='dg-t-value'>{value}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_tiles(
    *,
    rain_1h: float,
    rain_24h: float,
    river_m: float,
    elev_m: float,
    prob_map: Dict[str, float],
):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        _tile("Rain (1h)", f"{rain_1h:.1f} mm")
    with c2:
        _tile("Rain (24h)", f"{rain_24h:.1f} mm")
    with c3:
        _tile("River", f"{river_m:.2f} m")
    with c4:
        _tile("Elevation", f"{elev_m:.0f} m")
    with c5:
        high = float(prob_map.get("High", 0.0)) * 100.0
        color = DANGER if high >= 60 else (WARN if high >= 30 else OK)
        _tile("Confidence (High)", f"{high:.0f}%", value_color=color)


# ---------- Probability bar ----------
def render_probability_breakdown(prob_map: Dict[str, float]):
    order = ["Low", "Medium", "High"]
    vals = [float(prob_map.get(k, 0.0)) for k in order]
    df = pd.DataFrame({"Risk": order, "Probability": vals})

    fig = px.bar(
        df,
        x="Probability",
        y="Risk",
        orientation="h",  # <-- must be 'h', not 'horizontal'
        text=[f"{v * 100:.0f}%" for v in vals],
        color="Risk",
        color_discrete_map={
            "Low": OK,
            "Medium": WARN,
            "High": DANGER,
        },
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=8, t=10, b=8),
        showlegend=False,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        xaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Snapshot card ----------
def render_now_card(
    now: Dict[str, float | None],
    rain_1h: float,
    rain_24h: float,
    river_m: float,
    river_mode: str,
):
    st.markdown("<div class='dg-card'>", unsafe_allow_html=True)
    st.markdown(
        "<h4 style='margin:0 0 6px 0;'>Latest Snapshot</h4>", unsafe_allow_html=True
    )
    rows = [
        ("Rain (1h)", f"{rain_1h:.1f} mm"),
        ("Rain (24h)", f"{rain_24h:.1f} mm"),
        (
            "River level",
            f"{river_m:.2f} m  •  " + ("Manual" if river_mode == "manual" else "Estimated"),
        ),
        ("Temperature", f"{now.get('temp')} °C" if now and now.get("temp") is not None else "—"),
        ("Humidity", f"{int(now.get('humidity'))} %" if now and now.get("humidity") is not None else "—"),
        ("Wind", f"{now.get('wind')} m/s" if now and now.get("wind") else "—"),
    ]
    for k, v in rows:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;margin:6px 0;color:#e2e8f0'>"
            f"<span style='color:{MUTED}'>{k}</span><span>{v}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Daily rainfall ----------
def render_daily_rainfall(daily: pd.DataFrame | None):
    st.markdown(
        "<div class='dg-section'><h3>Daily Rainfall</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    if not _safe_has_cols(daily, ["date", "rain_mm"]):
        st.info("No snapshots yet.")
        return
    fig = px.line(
        daily,
        x="date",
        y="rain_mm",
        markers=True,
        template=PLOT_TEMPLATE,
        color_discrete_sequence=[CYAN],
    )
    fig.update_traces(hovertemplate="%{x}<br>Rain: %{y:.1f} mm")
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Temp / Humidity dual-axis ----------
def render_temp_humidity(daily: pd.DataFrame | None):
    st.markdown(
        "<div class='dg-section'><h3>Temperature & Humidity (mean)</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    if not _safe_has_cols(daily, ["date", "temp_c", "humidity"]):
        st.info("No local data yet.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["temp_c"],
            mode="lines+markers",
            name="Temp (°C)",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["humidity"],
            mode="lines+markers",
            name="Humidity (%)",
            yaxis="y2",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=300,
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        yaxis=dict(title="Temp (°C)"),
        yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Stacked risk histogram ----------
def render_risk_histogram(risk_counts: pd.DataFrame | None):
    st.markdown(
        "<div class='dg-section'><h3>Risk Level Frequency</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    if not _safe_has_cols(risk_counts, ["date", "risk", "count"]):
        st.info("No risk records yet.")
        return
    fig = px.bar(
        risk_counts,
        x="date",
        y="count",
        color="risk",
        barmode="stack",
        category_orders={"risk": ["Low", "Medium", "High", "Unknown"]},
        color_discrete_map={
            "Low": OK,
            "Medium": WARN,
            "High": DANGER,
            "Unknown": BLUE,
        },
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Export table ----------
def render_export_table(snap: pd.DataFrame | None):
    st.markdown(
        "<div class='dg-section'><h3>Export</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    snap = snap if snap is not None else pd.DataFrame()
    st.dataframe(snap, use_container_width=True, height=280)
    if not snap.empty:
        csv = snap.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📤 Download CSV",
            data=csv,
            file_name="disasterguard_snapshots.csv",
            mime="text/csv",
        )


# ---------- Trend + anomaly ----------
def render_trend_anomaly(daily: pd.DataFrame | None, window: int = 7):
    st.markdown(
        f"<div class='dg-section'><h3>Rainfall Trend & Anomaly</h3>"
        f"<div class='dg-badge'>{window}-day mean</div></div>",
        unsafe_allow_html=True,
    )
    if not _safe_has_cols(daily, ["date", "rain_mm"]):
        st.info("No daily rainfall yet.")
        return

    df = daily.sort_values("date").copy()
    df["ma"] = df["rain_mm"].rolling(window, min_periods=max(1, window // 2)).mean()
    df["anomaly"] = df["rain_mm"] - df["ma"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["anomaly"],
            name="Anomaly (mm)",
            opacity=0.38,
            marker_color=PURPLE,
            hovertemplate="%{x}<br>Anomaly: %{y:.1f} mm",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rain_mm"],
            mode="lines+markers",
            name="Daily rain (mm)",
            line=dict(width=2, color=CYAN),
            hovertemplate="%{x}<br>Rain: %{y:.1f} mm",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ma"],
            mode="lines",
            name=f"{window}-day mean",
            line=dict(width=3, dash="dash", color=ACCENT),
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        barmode="relative",
        legend=dict(orientation="h"),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Exceedance probability (Return period) ----------
def render_exceedance_curve(daily: pd.DataFrame | None):
    st.markdown(
        "<div class='dg-section'><h3>Exceedance Probability (Rain Events)</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    if not _safe_has_cols(daily, ["date", "rain_mm"]):
        st.info("No daily rainfall yet.")
        return

    df = daily[["date", "rain_mm"]].dropna().copy()
    if df.empty:
        st.info("No rain values to plot.")
        return

    df = df.sort_values("rain_mm", ascending=False).reset_index(drop=True)
    n = len(df)
    df["rank"] = df.index + 1
    df["exceed_prob"] = df["rank"] / (n + 1.0)
    df["return_period_days"] = 1.0 / df["exceed_prob"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["return_period_days"],
            y=df["rain_mm"],
            mode="lines+markers",
            name="Event curve",
            line=dict(width=2, color=BLUE),
            hovertemplate="Return period: %{x:.1f} d<br>Rain: %{y:.1f} mm",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        xaxis_title="Return period (days, lower = more frequent)",
        yaxis_title="Rain (mm)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Correlation heatmap ----------
def render_correlation_matrix(daily: pd.DataFrame | None):
    st.markdown(
        "<div class='dg-section'><h3>Metric Correlations</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    if daily is None or daily.empty:
        st.info("No daily data yet.")
        return

    cols = [c for c in ["rain_mm", "temp_c", "humidity"] if c in daily.columns]
    if len(cols) < 2:
        st.info("Need at least two metrics (rain, temp, humidity) to correlate.")
        return

    df = daily[cols].copy()
    corr = df.corr(numeric_only=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar=dict(title="ρ"),
            hovertemplate="%{y} vs %{x}<br>ρ: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=300,
        margin=dict(l=10, r=10, b=10, t=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Calendar heatmap ----------
def render_calendar_heatmap(daily: pd.DataFrame | None):
    st.markdown(
        "<div class='dg-section'><h3>Calendar Heatmap (Rain mm)</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    if not _safe_has_cols(daily, ["date", "rain_mm"]):
        st.info("No daily rainfall yet.")
        return

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["weekday"] = df["date"].dt.weekday

    pivot = df.pivot_table(
        index="weekday", columns="week", values="rain_mm", aggfunc="sum"
    ).sort_index()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(int(c)) for c in pivot.columns],
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(pivot.index)],
            coloraxis="coloraxis",
            hovertemplate="Week %{x} • %{y}<br>Rain: %{z:.1f} mm<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=300,
        margin=dict(l=10, r=10, b=10, t=10),
        coloraxis=dict(colorscale="Blues"),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Top N rain events ----------
def render_top_events_table(daily: pd.DataFrame | None, n: int = 10):
    st.markdown(
        f"<div class='dg-section'><h3>Top {n} Rain Days</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    if not _safe_has_cols(daily, ["date", "rain_mm"]):
        st.info("No daily rainfall yet.")
        return

    df = daily.sort_values("rain_mm", ascending=False).head(n)[
        ["date", "rain_mm"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(df, use_container_width=True, height=300)


# ---------- Risk share ----------
def render_risk_share(risk_counts: pd.DataFrame | None):
    st.markdown(
        "<div class='dg-section'><h3>Risk Composition</h3><div></div></div>",
        unsafe_allow_html=True,
    )
    if (risk_counts is None) or risk_counts.empty:
        st.info("No risk records yet.")
        return

    pivot = risk_counts.groupby("risk", as_index=False)["count"].sum()
    total = float(pivot["count"].sum() or 1.0)
    pivot["share"] = pivot["count"] / total

    fig = px.bar(
        pivot,
        x="risk",
        y="share",
        text=pivot["share"].map(lambda v: f"{v * 100:.0f}%"),
        category_orders={"risk": ["Low", "Medium", "High", "Unknown"]},
        color="risk",
        color_discrete_map={
            "Low": OK,
            "Medium": WARN,
            "High": DANGER,
            "Unknown": BLUE,
        },
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        showlegend=False,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig, use_container_width=True)