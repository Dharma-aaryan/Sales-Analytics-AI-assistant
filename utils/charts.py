# utils/charts.py
from __future__ import annotations
import re
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import streamlit as st

from .aliases import COLUMN_ALIASES
from .tools import tool_query  # to compute period_revenue_usd when needed

# --- NL axis aliases (in addition to COLUMN_ALIASES) ---
EXTRA_ALIASES = {
    "segments": "segment", "segment": "segment",
    "industry": "industry", "vertical": "industry",
    "region": "region", "geo": "region",
    "tier": "tier", "channel": "channel",
    "customer": "company_name", "customers": "company_name",
    "account": "company_name", "accounts": "company_name",

    # churn wording
    "churn%": "churn_probability_percent",
    "churn probability": "churn_probability_percent",
    "probability of churn": "churn_probability_percent",

    # revenue wording (virtual)
    "revenue": "period_revenue_usd",
    "total revenue": "period_revenue_usd",
    "sales": "period_revenue_usd",
}

AXES_PATTERNS = [
    r"^\s*(?P<x>.+?)\s+against\s+(?P<y>.+?)\s*$",
    r"^\s*(?P<x>.+?)\s+vs\.?\s+(?P<y>.+?)\s*$",
    r"^\s*(?P<x>.+?)\s+versus\s+(?P<y>.+?)\s*$",
    r"^\s*(?P<x>.+?)\s+x\s+(?P<y>.+?)\s*$",
]

# ---------- helpers ----------

def _canonical(term: str, schema: set) -> Optional[str]:
    t = (term or "").strip().lower()
    if t == "churn":
        return "churn_probability_percent"
    if t in EXTRA_ALIASES:
        cand = EXTRA_ALIASES[t]
        if cand == "period_revenue_usd":
            return cand
        if cand in schema:
            return cand
    mapped = COLUMN_ALIASES.get(t)
    if mapped:
        if mapped == "period_revenue_usd":
            return mapped
        if mapped in schema:
            return mapped
    if term in schema:
        return term
    return None

def parse_axes_command(text: str) -> Optional[Tuple[str, str]]:
    txt = (text or "").strip().lower()
    for pat in AXES_PATTERNS:
        m = re.match(pat, txt)
        if m:
            return m.group("x").strip(), m.group("y").strip()
    return None

def _pick_default_category(schema: set) -> Optional[str]:
    for c in ("industry", "segment", "region", "tier", "channel", "company_name"):
        if c in schema:
            return c
    return None

def _bin_numeric(s: pd.Series, bins: int = 10) -> pd.Series:
    try:
        nunq = s.nunique(dropna=True)
        if nunq <= 1:
            return s.astype(str)
        q = pd.qcut(s, q=min(bins, nunq), duplicates="drop")
        return q.astype(str)
    except Exception:
        try:
            return pd.cut(s, bins=bins).astype(str)
        except Exception:
            return s.astype(str)

# ---------- special comparison intents ----------

_SPECIAL_TOTAL_VS_TOP = re.compile(
    r"\btotal\s+revenue\b.*\b(against|vs\.?|versus)\b.*\b(highest|top)\s+revenue\s+company\b",
    re.IGNORECASE,
)

def _special_total_vs_top(df_full: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    if "company_name" not in df_full.columns:
        return None

    if "period_revenue_usd" not in df_full.columns:
        q = {
            "select": ["company_name", "period_revenue_usd"],
            "filters": [],
            "time_window": {"start": "2000-01-01", "end": "2100-01-01"},
            "group_by": ["company_name"],
            "aggregations": {"period_revenue_usd": "sum"},
            "order_by": [{"col": "period_revenue_usd", "desc": True}],
            "computed": True,
            "limit": 100000
        }
        grp = tool_query(df_full, q)
    else:
        grp = (
            df_full.groupby("company_name", as_index=False)["period_revenue_usd"]
            .sum().sort_values("period_revenue_usd", ascending=False)
        )

    if grp is None or grp.empty:
        return None

    top_row = grp.iloc[0]
    top_name = str(top_row["company_name"])
    top_rev = float(top_row["period_revenue_usd"])
    total_rev = float(grp["period_revenue_usd"].sum())

    comp = pd.DataFrame({
        "label": ["All companies", f"Top company: {top_name}"],
        "period_revenue_usd": [total_rev, top_rev],
    })
    comp = comp.loc[:, ~comp.columns.duplicated()]
    return comp, {"x": "label", "y": "period_revenue_usd", "title": "Total revenue vs highest revenue company"}

def _match_special(text: str, df_full: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    if _SPECIAL_TOTAL_VS_TOP.search(text or ""):
        return _special_total_vs_top(df_full)
    return None

# ---------- public API ----------

def prepare_chart_frame(
    df_full: pd.DataFrame,
    last_table: Optional[pd.DataFrame],
    schema: set,
    user_text: str
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    # 1) Special comparisons first
    special = _match_special(user_text, df_full)
    if special:
        return special

    # 2) Axes-style command
    parsed = parse_axes_command(user_text)
    if not parsed:
        return None

    x_raw, y_raw = parsed
    x = _canonical(x_raw, schema)
    y = _canonical(y_raw, schema)

    if not x or not y:
        x = _canonical(y_raw, schema) if not x else x
        y = _canonical(x_raw, schema) if not y else y
    if not x or not y:
        return None

    if x == y:
        cat = _pick_default_category(schema)
        if cat:
            x = cat
        else:
            return None

    if last_table is not None and not last_table.empty:
        cols = set(last_table.columns)
        if x in cols and y in cols:
            plot_df = last_table[[x, y]].dropna()
            plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
            return plot_df, {"x": x, "y": y, "title": f"{y} vs {x}"}

    if y == "period_revenue_usd":
        q = {
            "select": [x, "period_revenue_usd"],
            "filters": [],
            "time_window": {"start": "2000-01-01", "end": "2100-01-01"},
            "group_by": [x],
            "aggregations": {"period_revenue_usd": "sum"},
            "order_by": [{"col": "period_revenue_usd", "desc": True}],
            "computed": True,
            "limit": 1000
        }
        plot_df = tool_query(df_full, q)
        if plot_df is not None and not plot_df.empty:
            plot_df = plot_df[[x, "period_revenue_usd"]]
            plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
            return plot_df, {"x": x, "y": "period_revenue_usd", "title": f"Revenue vs {x}"}

    if x in df_full.columns and y in df_full.columns:
        plot_df = df_full[[x, y]].dropna()
        plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
        return plot_df, {"x": x, "y": y, "title": f"{y} vs {x}"}

    return None

def build_bar_agg(df: pd.DataFrame, x: str, y: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """
    Compute the aggregated frame used by the bar chart.
    Returns (agg_df, y_key).
    """
    if df is None or df.empty or x not in df.columns:
        return pd.DataFrame(), ""

    cols = [x] + ([y] if y and y in df.columns else [])
    seen, uniq = set(), []
    for c in cols:
        if c not in seen:
            uniq.append(c); seen.add(c)
    plot = df[uniq].copy()
    if len(plot) > 2000:
        plot = plot.sample(2000, random_state=42)

    x_is_num = pd.api.types.is_numeric_dtype(plot[x])
    y_is_num = (y in plot.columns and pd.api.types.is_numeric_dtype(plot[y])) if y else False

    if x_is_num:
        plot["__xbin__"] = _bin_numeric(plot[x], bins=10)
        x_key = "__xbin__"
    else:
        x_key = x

    if y_is_num:
        agg = plot.groupby(x_key, dropna=False, as_index=False)[y].sum()
        y_key = y
    else:
        agg = plot.groupby(x_key, dropna=False, as_index=False).size().rename(columns={"size": "__count__"})
        y_key = "__count__"

    agg = agg.rename(columns={x_key: x})
    agg = agg.loc[:, ~agg.columns.duplicated()]
    return agg, y_key

def render_bar_only(df: pd.DataFrame, x: str, y: Optional[str], title: str = ""):
    agg, y_key = build_bar_agg(df, x, y)
    if agg.empty or not y_key:
        st.info("Nothing to chart.")
        return
    if title:
        st.caption(f"**{title}**")
    try:
        st.bar_chart(agg.set_index(x)[y_key], use_container_width=True)
    except Exception:
        st.info("Could not render bar chart.")
