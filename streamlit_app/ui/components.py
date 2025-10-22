# streamlit_app/ui/components.py
from __future__ import annotations
from typing import Dict
import streamlit as st
from .styles import PALETTE, section_header

__all__ = ["monitoring_controls"]


def monitoring_controls(
    *,
    country_options,
    default_country: str,
    default_region: str,
    default_window: str,
    default_simulate: bool,
    default_manual_river: bool,
) -> Dict:
    """
    Pure UI block for Monitoring controls.
    No DB/model logic here; only stateful widgets + display.
    """
    section_header("DG • Analyst Studio", right="")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
    with c1:
        country = st.selectbox(
            "Country",
            country_options,
            index=country_options.index(default_country) if default_country in country_options else 0,
            key="ctl_country",
        )

    curated = {
        "Singapore": ["— Country wide —","Downtown Core","Orchard","Toa Payoh","Bedok","Tampines","Yishun","Woodlands","Jurong East","Sentosa"],
        "United States": ["— Country wide —","New York","Los Angeles","Chicago","Houston","Miami","San Francisco"],
        "United Kingdom": ["— Country wide —","London","Manchester","Bristol"],
        "Other": ["— Country wide —"],
    }
    region_options = curated.get(country, curated["Other"])

    with c2:
        region = st.selectbox(
            "Region / City",
            region_options,
            index=(region_options.index(default_region) if default_region in region_options else 0),
            key="ctl_region",
        )

    with c3:
        window = st.radio(
            "Window",
            options=["7 days", "30 days", "90 days"],
            index=["7 days", "30 days", "90 days"].index(default_window) if default_window in ["7 days","30 days","90 days"] else 1,
            horizontal=True,
            key="ctl_window",
        )

    with c4:
        telegram_clicked = st.button("📲 Telegram", key="ctl_telegram")

    t1, t2, _ = st.columns([1.0, 1.0, 2.0])
    with t1:
        simulate = st.toggle("Simulate", value=default_simulate, key="ctl_simulate")
    with t2:
        manual_river = st.toggle("Manual river", value=default_manual_river, key="ctl_manual_river")

    # status chips
    st.markdown(
        f"""
        <div>
          <span class="dg-chip">Window: {window}</span>
          <span class="dg-chip">Sim: {"On" if simulate else "Off"}</span>
          <span class="dg-chip">River: {"Manual" if manual_river else "Auto"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return {
        "country": country,
        "region": region,
        "window": window,
        "simulate": simulate,
        "manual_river": manual_river,
        "telegram_clicked": telegram_clicked,
    }