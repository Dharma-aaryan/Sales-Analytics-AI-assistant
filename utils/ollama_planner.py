from __future__ import annotations
import json
import requests
import os
from typing import Dict, Any, List

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3:latest")

SYSTEM = """
You are an analytics planner for a tabular sales/churn dataset. Convert the user's request into a JSON plan of tool calls.

ALWAYS return ONLY valid JSON like:
{ "steps": [ { "tool": "query" | "chart" | "narrate", "args": { ... } }, ... ] }

Tools:

- query.args = {
  "select": [string],
  "filters": [ { "col": "string", "op": "==|!=|>|<|>=|<=|in|between|contains", "value": any } ],
  "time_window": { "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" } | null,
  "group_by": [string],
  "order_by": [ { "col": "string", "desc": true|false } ],
  "limit": int | null,
  "computed": true|false,
  "aggregations": { "col": "sum|mean|count|nunique" }
}

- chart.args = {
  "kind": "auto|bar|line",
  "x": "string|null",
  "y": "string|null",
  "title": "string|null"
}

- narrate.args = { "focus": "string" }

Rules & heuristics:
1) If the user mentions "revenue" and no time window, use time_window={start:"2000-01-01", end:"2100-01-01"} and computed=true.
2) For “top N by revenue”: aggregations={"period_revenue_usd":"sum"}, group_by=["company_name"], order_by desc, limit N.
3) For churn rate by a dimension: aggregations={"churn":"mean"}, group_by on that dimension, order desc on "churn".
4) ALWAYS include a chart step after the query. Prefer "kind":"auto" with x/y left null unless obvious.
5) ALWAYS include a narrate step after chart.

Schema hints:
company_name, industry, segment, region, tier, channel,
churn (0/1), churn_probability_percent (1–100),
mrr_usd, period_revenue_usd (computed),
signup_date, contract_start, contract_end,
feature_adoption_rate, utilization_rate_90d,
support_tickets_90d, failed_payments_180d, discount_pct
"""

def _chat_ollama(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")

def plan_with_ollama(user_text: str, rag_context: str = "") -> Dict[str, Any] | None:
    messages = [
        {"role": "system", "content": SYSTEM.strip()},
        {"role": "user", "content": user_text.strip()}
    ]
    try:
        out = _chat_ollama(messages)
        start = out.find("{"); end = out.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(out[start:end+1])
    except Exception:
        return None
