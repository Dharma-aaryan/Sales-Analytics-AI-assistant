# utils/aliases.py
from typing import Dict, Any, List, Tuple

COLUMN_ALIASES = {
    "customer_name": "company_name",
    "client_name": "company_name",
    "account_name": "company_name",
    "company": "company_name",
    "customer": "company_name",
    "client": "company_name",
    "name": "company_name",

    "revenue": "period_revenue_usd",
    "total_revenue": "period_revenue_usd",
    "revenue_usd": "period_revenue_usd",
    "period_revenue": "period_revenue_usd",
    "mrr": "mrr_usd",
    "arr": "mrr_usd",

    "start_date": "contract_start",
    "end_date": "contract_end",
    "contract_start_date": "contract_start",
    "contract_end_date": "contract_end",

    # keep bare 'churn' mapped to binary column by default;
    # we'll smart-switch to churn_probability_percent inside _fix_filters when threshold > 1
    "is_churn": "churn",
    "churn_flag": "churn",

    "churn_probability": "churn_probability_percent",
    "churn_prob": "churn_probability_percent",
    "churn_risk": "churn_probability_percent",
    "probability_of_churn": "churn_probability_percent",

    "services_used": "services_used_count",
    "service_count": "services_used_count",
    "num_services": "services_used_count",
    "number_of_services": "services_used_count",

    "feedback_text": "feedback",
    "customer_feedback": "feedback",
    "notes": "feedback",
}

# columns we treat as revenue-like
REVENUE_LIKE = {
    "period_revenue_usd", "revenue", "total_revenue",
    "revenue_usd", "period_revenue", "mrr_usd"
}

# columns allowed to pass even if not present in schema (computed later)
VIRTUAL_COLS = {"period_revenue_usd"}

def _as_col_name(x: Any) -> str | None:
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("col", "name", "field", "column"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        expr = x.get("expr")
        if isinstance(expr, str):
            if "period_revenue_usd" in expr: return "period_revenue_usd"
            if "mrr_usd" in expr: return "mrr_usd"
    return None

def resolve_col(col_like: Any, schema: set) -> str | None:
    """
    Map a user/LLM-provided column-like thing to a real column name.
    Allow virtual columns (e.g., 'period_revenue_usd') to pass even if not in schema yet.
    """
    name = _as_col_name(col_like)
    if not name:
        return None
    # exact
    if name in schema or name in VIRTUAL_COLS:
        return name
    # alias
    mapped = COLUMN_ALIASES.get(name.lower())
    if mapped:
        if mapped in schema or mapped in VIRTUAL_COLS:
            return mapped
    return None

def _fix_list(cols: List[Any] | None, schema: set) -> List[str]:
    out: List[str] = []
    for c in cols or []:
        rc = resolve_col(c, schema)
        if rc and rc not in out:
            out.append(rc)
    return out

def _parse_percentish(val: Any) -> Tuple[float | None, bool]:
    """
    Returns (numeric_value, is_percent_string)
    e.g., "70%" -> (70.0, True), "70" -> (70.0, False), 0.7 -> (0.7, False)
    """
    if isinstance(val, (int, float)):
        return float(val), False
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("%"):
            try:
                return float(s[:-1]), True
            except Exception:
                return None, True
        try:
            return float(s), False
        except Exception:
            return None, False
    return None, False

def _fix_filters(filters: List[Dict[str, Any]] | None, schema: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in (filters or []):
        raw_col = f.get("col")
        op = (f.get("op") or "==").strip()
        val = f.get("value")

        # initial mapping (may be churn or churn_probability_percent depending on threshold)
        name = _as_col_name(raw_col) or str(raw_col or "")
        mapped = COLUMN_ALIASES.get(name.lower(), name)

        # Smart churn disambiguation:
        # If user wrote "churn > 70%" but the LLM/plan used "churn",
        # interpret as churn_probability_percent because binary churn (0/1) can't be > 1.
        if str(mapped).lower() == "churn":
            num, is_pct = _parse_percentish(val)
            if num is not None:
                # thresholds > 1 imply they meant probability, not 0/1 flag
                if num > 1:
                    mapped = "churn_probability_percent"
                    # if value looks like 0-1 fraction with "%", leave as-is (user intent was percent)
                # if num in [0,1] and is_pct is False, leave as binary churn
            # if unparsable, leave mapped as churn

        # resolve (allow virtual revenue col)
        col = resolve_col(mapped, schema) or (mapped if mapped in VIRTUAL_COLS else None)
        if not col:
            continue

        # push cleaned filter through
        out.append({"col": col, "op": op, "value": val})
    return out

def _fix_order_by(order_by: List[Any] | None, schema: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ob in (order_by or []):
        if isinstance(ob, dict):
            col = resolve_col(ob.get("col") or ob, schema)
            if col:
                out.append({"col": col, "desc": bool(ob.get("desc"))})
        else:
            col = resolve_col(ob, schema)
            if col:
                out.append({"col": col, "desc": True})
    return out

def _cols_from_filters(filters: List[Dict[str, Any]] | None) -> List[str]:
    return [f["col"] for f in (filters or []) if isinstance(f, dict) and f.get("col")]

def _mentions_revenue(select, aggs, order_by, filt_cols):
    sel_rev = any((s in REVENUE_LIKE) for s in (select or []))
    agg_rev = any(((k in REVENUE_LIKE) for k in (aggs or {}).keys()))
    ord_rev = any(((ob.get("col") in REVENUE_LIKE) for ob in (order_by or [])))
    fil_rev = any(((c in REVENUE_LIKE) for c in (filt_cols or [])))
    return sel_rev or agg_rev or ord_rev or fil_rev

def _ensure_selected(args: Dict[str, Any], cols: List[str]) -> None:
    sel = args.get("select") or []
    for c in cols:
        if c not in sel:
            sel.append(c)
    args["select"] = sel

def _ensure_revenue_defaults(args: Dict[str, Any], schema: set) -> None:
    _ensure_selected(args, ["period_revenue_usd"])
    args["computed"] = True if args.get("computed") is None else bool(args.get("computed"))
    tw = args.get("time_window") or {}
    if not (tw.get("start") and tw.get("end")):
        args["time_window"] = {"start": "2000-01-01", "end": "2100-01-01"}
    if "company_name" in schema and "company_name" in (args.get("select") or []):
        g = args.get("group_by") or []
        if "company_name" not in g:
            g.append("company_name")
        args["group_by"] = g
    aggs = args.get("aggregations") or {}
    aggs.setdefault("period_revenue_usd", "sum")
    args["aggregations"] = aggs
    if not args.get("order_by"):
        args["order_by"] = [{"col": "period_revenue_usd", "desc": True}]

def sanitize_plan(plan: Dict[str, Any], schema: set) -> Dict[str, Any]:
    if not plan or "steps" not in plan:
        return plan

    for step in plan["steps"]:
        if step.get("tool") != "query":
            continue
        args = step.setdefault("args", {})

        args["select"] = _fix_list(args.get("select"), schema)
        args["group_by"] = _fix_list(args.get("group_by"), schema)
        args["filters"] = _fix_filters(args.get("filters"), schema)
        args["order_by"] = _fix_order_by(args.get("order_by"), schema)

        # fix aggregations
        aggs_in = args.get("aggregations") or {}
        fixed_aggs: Dict[str, str] = {}
        if isinstance(aggs_in, dict):
            for k, v in aggs_in.items():
                col = resolve_col(k, schema) or _as_col_name(k)
                if col and (col in schema or col in VIRTUAL_COLS) and v in ("sum","mean","count","nunique"):
                    fixed_aggs[col] = v
        args["aggregations"] = fixed_aggs

        # ensure filtered cols appear in output
        filt_cols = _cols_from_filters(args["filters"])
        _ensure_selected(args, filt_cols)

        # default select
        if not args["select"]:
            if "company_name" in schema:
                args["select"] = ["company_name"]

        # revenue triggers defaults
        if _mentions_revenue(args.get("select"), args.get("aggregations"), args.get("order_by"), filt_cols):
            _ensure_revenue_defaults(args, schema)

        # if revenue only in filters, still group by company
        if any(c in ("period_revenue_usd","mrr_usd","revenue","period_revenue") for c in filt_cols):
            g = args.get("group_by") or []
            if "company_name" not in g and "company_name" in schema:
                g.append("company_name")
            args["group_by"] = g

        # if user filtered on churn_probability but didn't select it, include it
        if "churn_probability_percent" in filt_cols:
            _ensure_selected(args, ["churn_probability_percent"])

    return plan
