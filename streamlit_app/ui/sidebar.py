# streamlit_app/ui/sidebar.py — SaaS Navigation Panel (Dark / Glassy)
from __future__ import annotations
import streamlit as st

# Router keys that your app understands (do not change)
ROUTES = ["Profile", "Global View", "Monitoring", "Statistics"]

def _route_state_key() -> str:
    return "route"

def _current_route(default: str = "Profile") -> str:
    # one-shot redirect support
    force = st.session_state.pop("ui_force_route", None)
    if isinstance(force, str) and force in ROUTES:
        st.session_state[_route_state_key()] = force

    cur = st.session_state.get(_route_state_key(), default)
    if cur not in ROUTES:
        cur = default
        st.session_state[_route_state_key()] = cur
    return cur

def _nav_items():
    """
    Display label -> router key mapping.
    We **display** '⚙️ Settings' but route to 'Profile' to avoid touching app.py.
    """
    return [
        {"label": "🌐 Global Overview", "route": "Global View"},
        {"label": "📍 Monitoring",     "route": "Monitoring"},
        {"label": "📊 Analytics",      "route": "Statistics"},
        {"label": "⚙️ Settings",       "route": "Profile"},   # maps to existing 'Profile'
    ]

def sidebar_menu() -> dict:
    # ── Brand / Logo (glassy gradient text logo) ─────────────────────────────
    st.sidebar.markdown(
        """
        <div class="dg-sb-title" style="margin-bottom:10px;">
          <span style="
            font-weight:900; font-size:1.18rem; line-height:1.1;
            background: linear-gradient(90deg, #22D3EE, #93C5FD);
            -webkit-background-clip: text; background-clip: text; color: transparent;
            letter-spacing:.02em;">
            DisasterGuard&nbsp;AI
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active route
    current = _current_route(default="Profile")

    # ── Main Navigation Buttons (with icons) ─────────────────────────────────
    for item in _nav_items():
        is_active = (item["route"] == current)
        st.sidebar.markdown(
            f'<div class="dg-sb-item {"active" if is_active else ""}">', unsafe_allow_html=True
        )
        if st.sidebar.button(
            item["label"],
            key=f"sb_{item['route'].replace(' ', '_').lower()}",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state[_route_state_key()] = item["route"]
            st.rerun()
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # ── Footer (version / copyright) ─────────────────────────────────────────
    st.sidebar.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='color:#93A3B3;font-size:.78rem;opacity:.85'>"
        "Version 1.0 • © 2025 DisasterGuard Team"
        "</div>",
        unsafe_allow_html=True,
    )

    return {"route": st.session_state.get(_route_state_key(), "Profile")}