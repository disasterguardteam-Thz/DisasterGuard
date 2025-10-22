# streamlit_app/ui/pages/dashboard_analytics.py
from __future__ import annotations
import streamlit as st
from ..styles import section_header, spacer
from .statistics_dashboard import run_statistics_page
from db import upsert_location  # only to ensure a Location row exists

def _get_state(key: str, default):
    return st.session_state.get(key, default)

def _days_from_window(label: str) -> int:
    if label == "7 days": return 7
    if label == "90 days": return 90
    return 30

def render():
    """Thin wrapper that reads app state and delegates to the statistics page."""
    country = _get_state("country", "Singapore")
    region  = _get_state("region", "— Country wide —")
    window  = _get_state("window", "30 days")
    days    = _days_from_window(window)

    lat = float(_get_state("lat_input", 1.3521))
    lon = float(_get_state("lon_input", 103.8198))

    loc_name = f"{country} • {region}"
    loc_id = upsert_location(name=loc_name, lat=lat, lon=lon, elev_m=20.0, urban=1)  # <- fix: lat=lat

    st.markdown(
        "<h2 style='margin:0 0 8px 0;font-weight:900;"
        "background:linear-gradient(90deg,#60A5FA,#22D3EE);"
        "-webkit-background-clip:text;color:transparent'>Statistics</h2>",
        unsafe_allow_html=True,
    )
    section_header(f"Analytics for {country} — {region}", right=f"<span class='dg-badge'>Window: {days}d</span>")
    spacer(8)

    run_statistics_page(loc_id=loc_id, country=country, region=region, days=days)