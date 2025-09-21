# scripts/smoke_test.py
import pandas as pd
from pathlib import Path
import sys

# --- ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "cyber_subscriptions_churn_enriched.csv"

REQUIRED = [
    "company_name","mrr_usd","contract_start","contract_end","churn",
    "churn_probability_percent","services_used_count","feedback"
]
DATE_COLS = {"signup_date","contract_start","contract_end"}
NUM_COLS  = {
    "mrr_usd","seats_committed","seats_active_90d","feature_adoption_rate",
    "utilization_rate_90d","support_tickets_90d","failed_payments_180d",
    "discount_pct","churn_probability_percent","services_used_count","churn"
}
RANGES = {
    "feature_adoption_rate": (0,1),
    "utilization_rate_90d": (0,1),
    "discount_pct": (0,100),
    "churn": (0,1),
    "churn_probability_percent": (1,100),
    "services_used_count": (1,10),
}

def schema_sanity(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in DATE_COLS & set(df.columns):
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    for c in NUM_COLS & set(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print("Null % (top 10):")
    print((df.isna().mean()*100).round(2).sort_values(ascending=False).head(10))
    issues = []
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns: continue
        s = df[col]
        if lo is not None and (s < lo).any(): issues.append(f"{col} < {lo}")
        if hi is not None and (s > hi).any(): issues.append(f"{col} > {hi}")
    print("Range checks:", "OK" if not issues else issues)
    if {"contract_start","contract_end"} <= set(df.columns):
        bad = pd.to_datetime(df["contract_start"]) > pd.to_datetime(df["contract_end"])
        print("contract_start > contract_end rows:", int(bad.sum()))
    print(f"Schema OK. Rows: {len(df)}, Cols: {len(df.columns)}")
    return df

from utils.tools import tool_query

if __name__ == "__main__":
    # 1) Load + sanity
    df = pd.read_csv(DATA_PATH)
    df = schema_sanity(df)

    # 2) Probe revenue distribution (wide window)
    probe = {
        "select": ["company_name", "period_revenue_usd", "churn_probability_percent"],
        "filters": [],
        "time_window": {"start": "2000-01-01", "end": "2100-01-01"},
        "group_by": ["company_name"],
        "aggregations": {"period_revenue_usd": "sum"},
        "order_by": [{"col": "period_revenue_usd", "desc": True}],
        "limit": 20,
        "computed": True,
    }
    top_rev = tool_query(df, probe)
    print("\nTop 20 by period_revenue_usd (wide window):")
    print(top_rev.to_string(index=False))
    if not top_rev.empty:
        qtiles = top_rev["period_revenue_usd"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
        print("\nRevenue quantiles (wide window):")
        print(qtiles)

    # 2b) Churn probability distribution
    print("\nChurn probability quantiles:")
    print(df["churn_probability_percent"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nTop 10 by churn_probability_percent:")
    print(
        df.sort_values("churn_probability_percent", ascending=False)
          .loc[:, ["company_name", "churn_probability_percent", "mrr_usd"]]
          .head(10)
          .to_string(index=False)
    )

    # 3) Percentile-aware intersection: high revenue AND high churn risk
    churn_cut = float(df["churn_probability_percent"].quantile(0.90))  # 90th pct
    rev_plan = {
        "select": ["company_name", "period_revenue_usd"],
        "filters": [],
        "time_window": {"start": "2000-01-01", "end": "2100-01-01"},
        "group_by": ["company_name"],
        "aggregations": {"period_revenue_usd": "sum"},
        "order_by": [{"col": "period_revenue_usd", "desc": True}],
        "computed": True
    }
    rev_tbl = tool_query(df, rev_plan)
    rev_cut = float(rev_tbl["period_revenue_usd"].quantile(0.80))      # 80th pct

    churn_tbl = df.groupby("company_name", as_index=False)["churn_probability_percent"].max()

    joined = (
        rev_tbl.merge(churn_tbl, on="company_name", how="left")
               .query("period_revenue_usd > @rev_cut and churn_probability_percent >= @churn_cut")
               .sort_values(["period_revenue_usd", "churn_probability_percent"], ascending=[False, False])
               .head(20)
    )

    print(f"\nCutoffs → revenue >= {rev_cut:,.0f}, churn_prob >= {churn_cut:.0f}")
    print("High revenue ∩ High churn risk (top 20):")
    print(joined.to_string(index=False))
