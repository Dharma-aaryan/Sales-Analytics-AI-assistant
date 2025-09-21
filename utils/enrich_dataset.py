from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np

REQUIRED = ["company_name","mrr_usd","contract_start","contract_end"]
TARGETS = ["churn_probability_percent", "services_used_count", "feedback"]

def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("signup_date","contract_start","contract_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _ensure_ranges(df: pd.DataFrame) -> pd.DataFrame:
    if "churn_probability_percent" in df.columns:
        df["churn_probability_percent"] = df["churn_probability_percent"].clip(1, 100)
    if "services_used_count" in df.columns:
        df["services_used_count"] = df["services_used_count"].clip(1, 10).astype(int)
    return df

FEEDBACK_TEMPLATES = [
    "Great coverage, but onboarding took longer than expected.",
    "Seeing value, but need better alert tuning.",
    "Performance is solid; support response could be faster.",
    "Considering expansion if we can consolidate tools.",
    "Struggling with adoption in 1-2 teams; need training."
]

def enrich(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df = _coerce_dates(df)

    if "churn_probability_percent" not in df.columns:
        base = rng.normal(20, 10, size=len(df))
        if "support_tickets_90d" in df.columns:
            base += df["support_tickets_90d"].fillna(0) * 1.5
        if "failed_payments_180d" in df.columns:
            base += df["failed_payments_180d"].fillna(0) * 2.0
        if "feature_adoption_rate" in df.columns:
            base += (1 - df["feature_adoption_rate"].fillna(0)) * 30
        df["churn_probability_percent"] = np.clip(base, 1, 100).round(0).astype(int)

    if "services_used_count" not in df.columns:
        df["services_used_count"] = rng.integers(1, 11, size=len(df))

    if "feedback" not in df.columns:
        picks = rng.integers(0, len(FEEDBACK_TEMPLATES), size=len(df))
        df["feedback"] = [FEEDBACK_TEMPLATES[i] for i in picks]

    df = _ensure_ranges(df)
    return df

def main(in_path: str, out_path: str | None = None):
    src = Path(in_path)
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")
    df = pd.read_csv(src)
    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    out = enrich(df)
    if out_path is None:
        out_path = str(src.with_name(src.stem + "_enriched.csv"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved enriched dataset → {out_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Input CSV")
    p.add_argument("--out", dest="out_path", default=None, help="Output CSV path")
    args = p.parse_args()
    main(args.in_path, args.out_path)
