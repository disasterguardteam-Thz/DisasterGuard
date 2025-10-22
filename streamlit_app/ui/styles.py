# styles.py — Glassy UI/UX theme (presentation-only)
from __future__ import annotations
import streamlit as st

# ---- Design Tokens -----------------------------------------------------------
PALETTE = {
    # Core
    "bg": "#0A0F1C",
    "text": "#EAF0F7",
    "muted": "#9AA4B2",
    # Surfaces
    "panel": "rgba(13, 18, 32, 0.65)",        # glassy base
    "panel_border": "rgba(148,163,184,0.18)",
    "chip_bg": "rgba(148,163,184,0.08)",
    "chip_border": "rgba(148,163,184,0.22)",
    "badge_bg": "rgba(16, 23, 44, 0.78)",
    "badge_border": "rgba(148,163,184,0.18)",
    "divider": "rgba(148,163,184,0.28)",
    # Accents
    "accent": "#22D3EE",   # cyan
    "accent2": "#93C5FD",  # light blue
    "cyan": "#22D3EE",     # keep explicit key (many charts reference "cyan")
    "blue": "#60A5FA",
    "purple": "#A78BFA",
    # Status
    "ok": "#22C55E",
    "warn": "#F59E0B",
    "danger": "#EF4444",
}

PLOT_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)"  # transparent

# ---- Global Glassy CSS -------------------------------------------------------
def inject_base_css():
    """Global, presentation-only CSS. Safe to re-run; no app logic."""
    st.markdown(
        f"""
<style>
:root {{
  --dg-bg: {PALETTE["bg"]};
  --dg-text: {PALETTE["text"]};
  --dg-muted: {PALETTE["muted"]};
  --dg-panel: {PALETTE["panel"]};
  --dg-panel-border: {PALETTE["panel_border"]};
  --dg-chip-bg: {PALETTE["chip_bg"]};
  --dg-chip-border: {PALETTE["chip_border"]};
  --dg-badge-bg: {PALETTE["badge_bg"]};
  --dg-badge-border: {PALETTE["badge_border"]};
  --dg-divider: {PALETTE["divider"]};
  --dg-accent: {PALETTE["accent"]};
  --dg-accent2: {PALETTE["accent2"]};
  --dg-ok: {PALETTE["ok"]};
  --dg-warn: {PALETTE["warn"]};
  --dg-danger: {PALETTE["danger"]};
}}

html, body, [class*="css"] {{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
  color: var(--dg-text);
  background: radial-gradient(140% 120% at 10% 0%, #0a1021 0%, #0b1220 45%, #0a0f1c 100%) fixed;
}}
.block-container {{ padding-top: .6rem; padding-bottom: 2rem; }}

 /* ---------- Frosted cards everywhere ---------- */
.dg-glass {{
  background: var(--dg-panel);
  border: 1px solid var(--dg-panel-border);
  border-radius: 16px;
  backdrop-filter: saturate(130%) blur(14px);
  -webkit-backdrop-filter: saturate(130%) blur(14px);
  box-shadow:
     0 0 0 1px rgba(255,255,255,0.03) inset,
     0 10px 30px rgba(2,6,23,0.55);
}}

 /* ---------- FIXED SIDEBAR (no collapse) ---------- */
section[data-testid="stSidebar"] {{
  width: 280px !important;
  min-width: 280px !important;
  max-width: 280px !important;
  border-right: 1px solid rgba(148,163,184,.12);
  background:
    linear-gradient(180deg, rgba(147,197,253,0.08), rgba(34,211,238,0.04)) padding-box,
    radial-gradient(120% 120% at -10% -20%, rgba(34,211,238,0.25), rgba(147,197,253,0.08) 35%, rgba(11,19,34,0.85) 80%) border-box,
    #0B1322;
}}
section[data-testid="stSidebar"] .block-container {{
  padding: 18px 14px 28px 14px;
}}
/* Hide any collapse buttons Streamlit may render */
button[kind="header"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[title="Collapse sidebar"],
button[aria-label="Toggle sidebar"] {{ display: none !important; }}

 /* ---------- Sidebar Title ---------- */
.dg-sb-title {{
  text-transform: uppercase;
  font-weight: 900;
  letter-spacing: .08em;
  line-height: 1.12;
  color: #eef4ff;
  font-size: 1.08rem;
  margin: 6px 6px 16px 6px;
}}

 /* ---------- Sidebar Menu Buttons ---------- */
.dg-sb-item {{ margin: 8px 6px; position: relative; }}
.dg-sb-item .stButton>button {{
  width: 100%;
  text-align: left;
  padding: 11px 14px;
  border-radius: 12px;
  border: 1px solid rgba(148,163,184,.16);
  color: #e6edf6;
  font-weight: 650;
  transition: transform .12s ease, border-color .18s ease, box-shadow .18s ease;
  box-shadow:
     0 1px 0 0 rgba(255,255,255,.03) inset,
     0 10px 24px rgba(2,6,23,.42);
  background:
    linear-gradient(180deg, rgba(17,26,42,.85), rgba(11,18,32,.92)) padding-box,
    linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,0)) border-box;
  backdrop-filter: blur(10px);
}}
.dg-sb-item .stButton>button:hover {{
  transform: translateY(-1px);
  border-color: rgba(125,211,252,.40);
}}
.dg-sb-item.active .stButton>button {{
  border-color: rgba(34,211,238,.55);
  box-shadow:
     0 0 0 1px rgba(34,211,238,.10) inset,
     0 14px 36px rgba(2,6,23,.55);
}}
.dg-sb-item.active::before {{
  content: "";
  position: absolute; left: -6px; top: 8px; bottom: 8px; width: 4px;
  border-radius: 999px;
  background: linear-gradient(180deg, var(--dg-accent), var(--dg-accent2));
}}

 /* ---------- Sections / Headers / Badges ---------- */
.dg-section {{
  display:flex; align-items:center; justify-content:space-between;
  margin: 6px 0 10px 0;
}}
.dg-section h3 {{
  margin:0; font-size:1.06rem; font-weight:800;
  background: linear-gradient(90deg, #93c5fd, #22d3ee);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  letter-spacing:.2px;
}}
.dg-divider {{
  height:1px; margin: 14px 0;
  background: linear-gradient(90deg, transparent, var(--dg-divider), transparent);
}}
.dg-badge {{
  display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:10px;
  background: var(--dg-badge-bg); border: 1px solid var(--dg-badge-border);
  font-weight:600; color:#e2e8f0;
  backdrop-filter: blur(10px);
}}
.dg-chip {{
  display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px;
  background: var(--dg-chip-bg); border: 1px solid var(--dg-chip-border);
  font-weight:600; color:#e2e8f0;
  backdrop-filter: blur(8px);
}}

 /* ---------- Cards / Tiles ---------- */
.dg-card {{ composes: dg-glass; }}
.dg-tile {{ composes: dg-glass; }}
.dg-t-label {{ color: var(--dg-muted); font-size:.86rem; margin-bottom:6px; }}
.dg-t-value {{ font-size:1.35rem; font-weight:700; color:#e2e8f0; }}

 /* ---------- Tables ---------- */
[data-testid="stDataFrame"] .styled-table, [data-testid="stDataFrame"] table {{
  border-radius: 12px !important;
  overflow: hidden !important;
}}
</style>
        """,
        unsafe_allow_html=True,
    )

# ---- Tiny helpers used across the UI ----------------------------------------
def spacer(px: int = 10):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

def render_divider():
    st.markdown('<div class="dg-divider"></div>', unsafe_allow_html=True)

def section_header(title: str, right: str = ""):
    st.markdown(
        f"""
        <div class="dg-section">
          <h3>{title}</h3>
          <div>{right}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )