from __future__ import annotations
import streamlit as st
import pandas as pd

def tool_narrate_streamlit(df: pd.DataFrame, args: dict | None = None):
    """Deterministic narration: 2–3 bullets + 2–3 actions."""
    args = args or {}
    if df is None or df.empty:
        st.info("No rows matched. Try relaxing filters.")
        return

    bullets = []

    if "period_revenue_usd" in df.columns and pd.api.types.is_numeric_dtype(df["period_revenue_usd"]):
        total_rev = float(df["period_revenue_usd"].sum())
        bullets.append(f"Total revenue in result: **${total_rev:,.0f}**.")
        top = df.sort_values("period_revenue_usd", ascending=False).head(1)
        if not top.empty and "company_name" in df.columns:
            bullets.append(f"Top account by revenue: **{top.iloc[0]['company_name']}** "
                           f"(${top.iloc[0]['period_revenue_usd']:,.0f}).")

    if "churn_probability_percent" in df.columns and pd.api.types.is_numeric_dtype(df["churn_probability_percent"]):
        avg_cp = float(df["churn_probability_percent"].mean())
        max_cp = float(df["churn_probability_percent"].max())
        bullets.append(f"Avg churn risk: **{avg_cp:.1f}%** (max **{max_cp:.0f}%**).")

    actions = []
    if "feature_adoption_rate" in df.columns and df["feature_adoption_rate"].mean() < 0.4:
        actions.append("Deploy **adoption playbooks** (guided onboarding, QBR training).")
    if "support_tickets_90d" in df.columns and df["support_tickets_90d"].mean() > 5:
        actions.append("Prioritize **proactive support** for accounts with high ticket volume.")
    if "discount_pct" in df.columns and df["discount_pct"].mean() > 0:
        actions.append("Use **targeted retention incentives** instead of blanket discounts.")
    if not actions:
        actions = [
            "Schedule an **executive check-in** with top-risk accounts.",
            "Share a **value recap** highlighting outcomes achieved.",
            "Offer a **pilot of an underused module** to increase stickiness."
        ]

    st.markdown("**Insights**")
    st.markdown("\n".join([f"- {b}" for b in bullets]) or "- No obvious insights.")
    st.markdown("**Suggested actions**")
    st.markdown("\n".join([f"- {a}" for a in actions]))
