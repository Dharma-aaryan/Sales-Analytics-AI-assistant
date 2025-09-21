from __future__ import annotations
import pandas as pd
from typing import Dict, Any

def build_schema_context(df: pd.DataFrame) -> str:
    parts = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        parts.append(f"- {c}: {dtype}")
    return "Columns:\n" + "\n".join(parts)

def slice_to_text(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "No rows."
    head = df.head(max_rows)
    return head.to_csv(index=False)

def attach_context_to_plan(plan: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    plan = plan or {}
    steps = plan.get("steps") or []
    has_narrate = any(s.get("tool") == "narrate" for s in steps)
    if not has_narrate:
        return plan
    context = build_schema_context(df)
    plan["_context"] = context
    return plan
