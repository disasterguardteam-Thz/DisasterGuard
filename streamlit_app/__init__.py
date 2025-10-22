"""
Expose UI helpers so the app can `import ui` safely.
Pure UI/UX – no business logic here.
"""
from .styles import inject_base_css, spacer, render_divider, section_header, PALETTE
from .sidebar import sidebar_menu
from .components import monitoring_controls
from .charts import (
    render_rainviewer_embed,
    render_metric_tiles,
    render_probability_breakdown,
    render_now_card,
    render_daily_rainfall,
    render_temp_humidity,
    render_risk_histogram,
    render_export_table,
    render_trend_anomaly,
    render_exceedance_curve,     # ← ensure this is imported
    render_correlation_matrix,
    render_calendar_heatmap,
    render_top_events_table,
    render_risk_share,
)

__all__ = [
    "inject_base_css", "spacer", "render_divider", "section_header", "PALETTE",
    "sidebar_menu",
    "monitoring_controls",
    "render_rainviewer_embed", "render_metric_tiles", "render_probability_breakdown",
    "render_now_card", "render_daily_rainfall", "render_temp_humidity",
    "render_risk_histogram", "render_export_table", "render_trend_anomaly",
    "render_exceedance_curve", "render_correlation_matrix", "render_calendar_heatmap",
    "render_top_events_table", "render_risk_share",
]