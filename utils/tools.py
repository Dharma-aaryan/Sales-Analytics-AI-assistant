from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, Any, List, Set
from .aliases import resolve_col

DATE_COLS: Set[str] = {"signup_date", "contract_start", "contract_end"}

def _is_date_col(col: str) -> bool:
    return col in DATE_COLS or col.endswith("_date")

def _to_pydate(x):
    if pd.isna(x): return None
    if isinstance(x, date): return x
    if isinstance(x, datetime): return x.date()
    try:
        return pd.to_datetime(x, errors="coerce").date()
    except Exception:
        return None

def _coerce_scalar_for_series(val, ser: pd.Series, col_name: str):
    if _is_date_col(col_name):
        return _to_pydate(val)
    try:
        import pandas.api.types as pdt
        if pdt.is_numeric_dtype(ser.dtype):
            return pd.to_numeric(val, errors="coerce")
    except Exception:
        pass
    return val

def _overlap_months(cs, ce, ws, we) -> int:
    if cs is None or ce is None or ws is None or we is None:
        return 0
    a = max(cs, ws); b = min(ce, we)
    if b < a:
        return 0
    return int(np.ceil(((b - a).days + 1) / 30.4))

def _apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]] | None, schema: Set[str]) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for f in filters:
        raw_col = f.get("col")
        op = (f.get("op") or "==").strip()
        val = f.get("value")
        col = resolve_col(raw_col, schema)
        if not col or col not in out.columns:
            continue
        ser = out[col]

        if op == "between":
            lo, hi = None, None
            if isinstance(val, dict):
                lo, hi = val.get("start"), val.get("end")
            elif isinstance(val, (list, tuple)) and len(val) == 2:
                lo, hi = val
            lo = _coerce_scalar_for_series(lo, ser, col)
            hi = _coerce_scalar_for_series(hi, ser, col)
            if lo is not None:
                out = out[out[col] >= lo]
            if hi is not None:
                out = out[out[col] <= hi]
            continue

        if op == "in":
            vals = list(val) if isinstance(val, (list, tuple, set)) else [val]
            vals = [_coerce_scalar_for_series(v, ser, col) for v in vals]
            out = out[out[col].isin(vals)]
            continue

        val = _coerce_scalar_for_series(val, ser, col)
        if op == "==":
            out = out[out[col] == val]
        elif op == "!=":
            out = out[out[col] != val]
        elif op == ">":
            out = out[out[col] > val]
        elif op == "<":
            out = out[out[col] < val]
        elif op == ">=":
            out = out[out[col] >= val]
        elif op == "<=":
            out = out[out[col] <= val]
        elif op == "contains":
            out = out[out[col].astype(str).str.contains(str(val), case=False, na=False)]
    return out

def tool_query(df: pd.DataFrame, args: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute a structured query safely against df.
    """
    schema = set(df.columns)
    out = df.copy()

    # Normalize date columns
    for c in (DATE_COLS & schema):
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.date

    # Compute period revenue if needed/mentioned
    wants_period_rev = (
        "period_revenue_usd" in (args.get("select") or []) or
        ("aggregations" in args and "period_revenue_usd" in (args.get("aggregations") or {})) or
        any((ob.get("col") == "period_revenue_usd") for ob in (args.get("order_by") or []))
    )
    win = args.get("time_window") or {}
    ws = _to_pydate(win.get("start")) if win else None
    we = _to_pydate(win.get("end")) if win else None
    if (args.get("computed") or wants_period_rev) and {"contract_start","contract_end","mrr_usd"} <= schema:
        if not (ws and we):
            ws, we = _to_pydate("2000-01-01"), _to_pydate("2100-01-01")
        out["overlap_months"] = out.apply(
            lambda r: _overlap_months(r["contract_start"], r["contract_end"], ws, we),
            axis=1
        )
        out["period_revenue_usd"] = (out["mrr_usd"] * out["overlap_months"]).round(2)
        schema = set(out.columns)

    # Apply filters
    out = _apply_filters(out, args.get("filters"), schema)

    # Build projection
    raw_select = args.get("select") or []
    select: List[str] = []
    for c in raw_select:
        rc = resolve_col(c, schema)
        if rc and rc not in select:
            select.append(rc)
    if not select:
        select = list(out.columns)

    # Aggregations (explicit)
    aggs_in = args.get("aggregations") or {}
    agg_map: Dict[str, str] = {}
    for k, how in aggs_in.items():
        rc = resolve_col(k, schema) or k
        if rc in out.columns and how in ("sum", "mean", "count", "nunique"):
            agg_map[rc] = how

    # Group-by
    group_by: List[str] = []
    for c in (args.get("group_by") or []):
        rc = resolve_col(c, schema)
        if rc and rc not in group_by:
            group_by.append(rc)

    if group_by:
        # Ensure filtered/order-by numeric columns are aggregated & retained
        default_numeric = [
            c for c in select
            if c not in group_by and c in out.columns and pd.api.types.is_numeric_dtype(out[c])
        ]
        for c in default_numeric:
            agg_map.setdefault(c, "sum")

        if "period_revenue_usd" in out.columns:
            agg_map.setdefault("period_revenue_usd", "sum")
        if "mrr_usd" in out.columns:
            agg_map.setdefault("mrr_usd", "sum")

        referenced_cols = set()
        for f in (args.get("filters") or []):
            col = resolve_col(f.get("col"), set(out.columns))
            if col: referenced_cols.add(col)
        for ob in (args.get("order_by") or []):
            col = resolve_col(ob.get("col"), set(out.columns))
            if col: referenced_cols.add(col)

        for col in referenced_cols:
            if col in group_by:
                continue
            if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
                if col == "churn_probability_percent":
                    agg_map.setdefault(col, "max")
                elif col in ("period_revenue_usd", "mrr_usd"):
                    agg_map.setdefault(col, "sum")
                else:
                    agg_map.setdefault(col, "sum")

        if not agg_map:
            for c in ("period_revenue_usd", "mrr_usd"):
                if c in out.columns:
                    agg_map[c] = "sum"
                    break

        res = out.groupby(group_by, as_index=False).agg(agg_map)

        if "churn" in agg_map and agg_map["churn"] == "mean" and "churn" in res.columns:
            res = res.rename(columns={"churn": "churn_rate"})
    else:
        keep = [c for c in select if c in out.columns]
        res = out[keep]

    # Order & limit
    order_by = []
    for ob in (args.get("order_by") or []):
        rc = resolve_col(ob.get("col"), set(res.columns))
        if rc:
            order_by.append({"col": rc, "desc": bool(ob.get("desc"))})
    for ob in order_by[::-1]:
        col = ob["col"]
        if col in res.columns:
            res = res.sort_values(col, ascending=not ob["desc"])
    if args.get("limit"):
        res = res.head(int(args["limit"]))

    return res.reset_index(drop=True)
