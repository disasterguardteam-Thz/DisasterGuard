# streamlit_app/ui/pages/home.py
from __future__ import annotations
import streamlit as st
from ..styles import section_header
from ..components import render_rainviewer_embed

def render():
    section_header("DisasterGuard • Analyst Studio", right="<span class='dg-badge'>Environment: Prod</span>")
    st.caption("Professional flood risk monitoring & analytics.")
    render_rainviewer_embed(
        title="Global Weather Map Live",
        subtitle="Worldwide radar, animation enabled.",
        center=None, zoom=2, height=640
    )