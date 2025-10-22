# sidebar.py — Glassy sidebar (visual only)
from __future__ import annotations
import streamlit as st

# Single source of truth for nav labels
ROUTES = ["Profile", "Global View", "Monitoring", "Statistics"]


def _route_state_key() -> str:
    # centralize the key name in case we ever rename it
    return "route"


def _current_route(default: str = "Profile") -> str:
    """
    Resolve the current route from session_state.
    - respects a one-shot redirect hint `ui_force_route` (then clears it)
    - falls back to `default` if nothing set
    """
    # One-shot redirect (e.g., when clicking "View full analysis in Statistics")
    force = st.session_state.pop("ui_force_route", None)
    if isinstance(force, str) and force in ROUTES:
        st.session_state[_route_state_key()] = force

    cur = st.session_state.get(_route_state_key(), default)
    if cur not in ROUTES:
        cur = default
        st.session_state[_route_state_key()] = cur
    return cur


def sidebar_menu() -> dict:
    """
    Render the fixed glassy sidebar and return {"route": <label>}.
    *Pure UI/UX*: no data logic; safe to call anytime.
    """
    # --- Title / Branding (kept minimal & glassy) ---
    st.sidebar.markdown(
        '<div class="dg-sb-title">DISASTER GUARD<br/>DASHBOARD</div>',
        unsafe_allow_html=True,
    )

    # --- Active route ---
    current = _current_route(default="Profile")

    # --- Nav items ---
    for label in ROUTES:
        active = (label == current)
        st.sidebar.markdown(
            f'<div class="dg-sb-item {"active" if active else ""}">', unsafe_allow_html=True
        )
        if st.sidebar.button(
            label,
            key=f"sb_{label.replace(' ', '_').lower()}",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state[_route_state_key()] = label
            st.rerun()
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # --- Subtle footer (brand/legal/etc.) ---
    st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='color:#93A3B3;font-size:.78rem;opacity:.85'>© DisasterGuard</div>",
        unsafe_allow_html=True,
    )

    return {"route": st.session_state.get(_route_state_key(), "Profile")}