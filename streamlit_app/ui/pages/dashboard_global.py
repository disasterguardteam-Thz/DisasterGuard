from __future__ import annotations
from ..styles import section_header
from ..charts import render_rainviewer_embed  # fixed import

def render():
    section_header("🌐 Global Overview", right="<span class='dg-badge'>Radar: RainViewer</span>")
    render_rainviewer_embed(
        title="",
        subtitle="Global RainViewer (worldwide radar).",
        center=None,
        zoom=2,
        height=620,
    )