import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import json

from utils.ollama_planner import plan_with_ollama
from utils.tools import tool_query
from utils.aliases import sanitize_plan
from utils.ui_tools import tool_narrate_streamlit
from utils.charts import prepare_chart_frame, render_bar_only, build_bar_agg  # <= build_bar_agg added

# --- Path to enriched dataset
DATA_PATH = Path("data/cyber_subscriptions_churn_enriched.csv")

st.set_page_config(page_title="Agentic Analytics Chat", layout="centered")

# --- Load dataset (cached)
@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(f"Missing {DATA_PATH}. Put your CSV at {DATA_PATH}.")
    st.stop()

df = load_df(DATA_PATH)
schema = set(df.columns)

# --- Session state
if "messages" not in st.session_state:
    # messages: [{role, content}]
    # content can be str | DataFrame | {"chart_spec": {...}}
    st.session_state.messages = []

# --- Replay history (now re-renders charts)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if isinstance(content, dict) and "chart_spec" in content:
            spec = content["chart_spec"]
            agg_df = pd.DataFrame(spec["data"])
            x = spec["x"]; y_key = spec["y_key"]; title = spec.get("title", "")
            if title:
                st.caption(f"**{title}**")
            try:
                st.bar_chart(agg_df.set_index(x)[y_key], use_container_width=True)
            except Exception:
                st.dataframe(agg_df, use_container_width=True, hide_index=True)
        elif isinstance(content, pd.DataFrame):
            st.dataframe(content, use_container_width=True, hide_index=True)
        else:
            st.markdown(content)

# --- Chat input at bottom
if user_text := st.chat_input("Ask me about sales, churn, or revenue analytics…"):
    # Show user bubble
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # --- Try explicit "X against Y" chart command first (no LLM) → BAR CHART
    last_assistant_df = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant" and isinstance(m["content"], pd.DataFrame):
            last_assistant_df = m["content"]
            break

    prepared = prepare_chart_frame(df_full=df, last_table=last_assistant_df, schema=schema, user_text=user_text)
    if prepared:
        plot_df, chart_args = prepared
        plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()]
        x = chart_args["x"]; y = chart_args.get("y"); title = chart_args.get("title", "")

        # Render chart
        with st.chat_message("assistant"):
            render_bar_only(plot_df, x=x, y=y, title=title)

        # Persist a replayable chart spec
        agg_df, y_key = build_bar_agg(plot_df, x=x, y=y)
        spec = {"chart_spec": {"x": x, "y_key": y_key, "title": title, "data": agg_df.to_dict("records")}}
        st.session_state.messages.append({"role": "assistant", "content": spec})
        st.rerun()

    # --- Otherwise: Ask Ollama for plan
    try:
        plan = plan_with_ollama(user_text, rag_context="")
    except Exception as e:
        plan = None
        st.warning(f"Ollama planning failed: {e}")

    # --- Fallback plan if Ollama didn’t produce a valid query
    if not plan or not any(s.get("tool") == "query" for s in plan.get("steps", [])):
        plan = {
            "steps": [
                {
                    "tool": "query",
                    "args": {
                        "select": ["company_name", "period_revenue_usd", "churn_probability_percent"],
                        "filters": [],
                        "time_window": {"start": "2000-01-01", "end": "2100-01-01"},
                        "group_by": ["company_name"],
                        "aggregations": {"period_revenue_usd": "sum"},
                        "order_by": [{"col": "period_revenue_usd", "desc": True}],
                        "limit": 5,
                        "computed": True,
                    },
                },
                {"tool": "chart", "args": {"x": "company_name", "y": "period_revenue_usd", "title": "Top 5 by revenue"}},
                {"tool": "narrate", "args": {"focus": "default retention insights"}}
            ]
        }

    # --- Sanitize plan
    plan = sanitize_plan(plan, schema=schema)

    # --- Assistant bubble for planned steps
    with st.chat_message("assistant"):
        last_table = None

        for step in plan.get("steps", []):
            tool = step.get("tool")
            args: Dict[str, Any] = step.get("args", {}) or {}

            if tool == "query":
                last_table = tool_query(df, args)

                filtered_cols = [f.get("col") for f in (args.get("filters") or []) if isinstance(f, dict)]
                needs_relax = (
                    (last_table is None or last_table.empty)
                    and any(c in {"period_revenue_usd", "churn_probability_percent"} for c in filtered_cols)
                )

                if needs_relax:
                    rev_probe_args = {
                        "select": ["company_name", "period_revenue_usd"],
                        "filters": [],
                        "time_window": {"start": "2000-01-01", "end": "2100-01-01"},
                        "group_by": ["company_name"],
                        "aggregations": {"period_revenue_usd": "sum"},
                        "order_by": [{"col": "period_revenue_usd", "desc": True}],
                        "computed": True,
                    }
                    rev_tbl = tool_query(df, rev_probe_args)
                    rev_cut = float(rev_tbl["period_revenue_usd"].quantile(0.80)) if (
                        rev_tbl is not None and not rev_tbl.empty and "period_revenue_usd" in rev_tbl.columns
                    ) else 0.0
                    churn_cut = float(df["churn_probability_percent"].quantile(0.90)) if (
                        "churn_probability_percent" in df.columns
                    ) else 0.0

                    relaxed_filters = []
                    for f in (args.get("filters") or []):
                        if not isinstance(f, dict): continue
                        c = f.get("col")
                        if c == "period_revenue_usd":
                            relaxed_filters.append({"col": c, "op": ">", "value": rev_cut})
                        elif c == "churn_probability_percent":
                            relaxed_filters.append({"col": c, "op": ">=", "value": churn_cut})
                        else:
                            relaxed_filters.append(f)

                    relaxed_args = dict(args)
                    relaxed_args["filters"] = relaxed_filters
                    relaxed_args.setdefault("group_by", ["company_name"])
                    relaxed_args.setdefault("aggregations", {}).setdefault("period_revenue_usd", "sum")
                    relaxed_args["computed"] = True
                    relaxed_args.setdefault("time_window", {"start": "2000-01-01", "end": "2100-01-01"})

                    last_table = tool_query(df, relaxed_args)
                    if last_table is not None and not last_table.empty:
                        st.info(f"No exact matches. Thresholds relaxed to: revenue ≥ {rev_cut:,.0f}, churn risk ≥ {churn_cut:.0f}%")

                if last_table is not None and not last_table.empty:
                    front = [c for c in ["company_name", "period_revenue_usd", "churn_probability_percent"] if c in last_table.columns]
                    cols = front + [c for c in last_table.columns if c not in front]
                    st.dataframe(last_table[cols], use_container_width=True, hide_index=True)
                    st.session_state.messages.append({"role": "assistant", "content": last_table})
                else:
                    st.info("No results found for this query.")

            elif tool == "chart" and last_table is not None and not last_table.empty:
                x = args.get("x") or ("company_name" if "company_name" in last_table.columns else last_table.columns[0])
                y = args.get("y")
                title = args.get("title", "")
                render_bar_only(last_table, x=x, y=y, title=title)

                # store replayable chart spec
                agg_df, y_key = build_bar_agg(last_table, x=x, y=y)
                spec = {"chart_spec": {"x": x, "y_key": y_key, "title": title, "data": agg_df.to_dict("records")}}
                st.session_state.messages.append({"role": "assistant", "content": spec})

            elif tool == "narrate":
                tool_narrate_streamlit(last_table if last_table is not None else pd.DataFrame(), args)

        with st.expander("🔍 Executed Plan", expanded=False):
            st.code(json.dumps(plan, indent=2))

    st.rerun()
