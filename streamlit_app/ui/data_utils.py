# streamlit_app/ui/data_utils.py
from __future__ import annotations
import pandas as pd

def safe_has_cols(df: pd.DataFrame, cols) -> bool:
    return (df is not None) and (not df.empty) and all(c in df.columns for c in cols)