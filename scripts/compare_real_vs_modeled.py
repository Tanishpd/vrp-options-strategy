"""
Compare the MODEL-based v2 backtest (Black-Scholes at real DVOL IV) against
the REAL-data backtest (actual Deribit trade prices, actual settlements) on
the entries both pipelines attempted, to answer: did the model over- or
under-state the credit actually available in the market?

Run this AFTER both scripts/run_v2.py and scripts/real_backtest.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CV = {"BTC": 0.001, "ETH": 0.01}


def main():
    modeled = pd.read_csv(ROOT / "results" / "v2_trades.csv", parse_dates=["entry"])
    real = pd.read_csv(ROOT / "results" / "real_trades.csv", parse_dates=["entry"])

    for df, name in ((modeled, "modeled"), (real, "real")):
        df["credit_per_unit"] = df["credit"] / (df["lots"] * df["sym"].map(CV))
        df["pnl_per_credit"] = df["pnl"] / df["credit"]

    m = modeled.set_index(["sym", "entry"])[["credit_per_unit", "credit", "fees", "pnl", "pnl_per_credit"]]
    r = real.set_index(["sym", "entry"])[["credit_per_unit", "credit", "fees", "pnl", "pnl_per_credit"]]
    joined = m.join(r, how="inner", lsuffix="_model", rsuffix="_real")

    print("=" * 90)
    print("COVERAGE")
    print("=" * 90)
    print(f"  modeled entries:  {len(modeled)}")
    print(f"  real entries:     {len(real)}  (some modeled entries had no real market -> skipped)")
    print(f"  matched on (sym, entry date): {len(joined)}  "
          f"({len(joined)/len(modeled):.1%} of modeled entries)")
    if joined.empty:
        print("\nNo overlapping (sym, entry) pairs — nothing to compare yet.")
        print("(This is expected if either backtest hasn't been (re)run against the same date range.)")
        return

    print("\n" + "=" * 90)
    print("CREDIT PER UNIT OF UNDERLYING — did the model over/understate what the market paid?")
    print("=" * 90)
    ratio = joined["credit_per_unit_real"] / joined["credit_per_unit_model"]
    print(f"  n matched          {len(joined)}")
    print(f"  mean  real/model   {ratio.mean():.2f}x   (1.00 = model was accurate)")
    print(f"  median real/model  {ratio.median():.2f}x")
    print(f"  modeled credit/unit  mean ${joined['credit_per_unit_model'].mean():.2f}")
    print(f"  real credit/unit     mean ${joined['credit_per_unit_real'].mean():.2f}")

    print("\n" + "=" * 90)
    print("BY SYMBOL")
    print("=" * 90)
    for sym, g in joined.groupby(level="sym"):
        rr = g["credit_per_unit_real"] / g["credit_per_unit_model"]
        print(f"  {sym}  n={len(g):>3}  real/model credit ratio: mean {rr.mean():.2f}x  "
              f"median {rr.median():.2f}x")

    print("\n" + "=" * 90)
    print("P&L PER $1 OF CREDIT COLLECTED (normalizes away the credit-size difference)")
    print("=" * 90)
    print(f"  modeled: mean {joined['pnl_per_credit_model'].mean():+.2f}   "
          f"median {joined['pnl_per_credit_model'].median():+.2f}")
    print(f"  real:    mean {joined['pnl_per_credit_real'].mean():+.2f}   "
          f"median {joined['pnl_per_credit_real'].median():+.2f}")

    print("\n" + "=" * 90)
    print("DISTRIBUTION OF THE REAL/MODEL RATIO (is the model biased, or just noisy?)")
    print("=" * 90)
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:>2}  {np.percentile(ratio, q):.2f}x")

    out = ROOT / "results" / "real_vs_modeled_comparison.csv"
    joined.reset_index().to_csv(out, index=False)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    sys.exit(main())
