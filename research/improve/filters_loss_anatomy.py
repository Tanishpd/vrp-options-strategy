"""Avenue: entry filter refinement — Part 1: loss anatomy.

Loads research/results/final_trades.csv (84 baseline trades) and joins each
trade's ENTRY date against research/data/BTC.csv market state:
  - trailing 5d / 20d log return
  - trailing 30d realized vol (rv_past) and iv / rv_past ratio
  - day of week, IV level, IV rank
For losers, also determines WHICH short strike was breached at settlement
(put side = crash week, call side = rally week).

Read-only. Writes nothing outside stdout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

tr = pd.read_csv(ROOT / "results" / "final_trades.csv", parse_dates=["entry", "expiry"])
px = pd.read_csv(ROOT / "data" / "BTC.csv", index_col=0, parse_dates=True)

px["logret"] = np.log(px["close"]).diff()
px["ret5"] = px["close"].pct_change(5)
px["ret20"] = px["close"].pct_change(20)
px["rv_past"] = px["logret"].rolling(30).std() * np.sqrt(365)
px["ivrv"] = px["iv"] / px["rv_past"]
pxn = px.copy()
pxn.index = pxn.index.tz_localize(None).normalize()

state = pxn[["close", "ret5", "ret20", "rv_past", "ivrv", "iv"]]
tr = tr.join(state, on="entry", rsuffix="_mkt")
tr["dow"] = tr["entry"].dt.day_name()
tr["settle_spot"] = tr["expiry"].map(pxn["close"])
tr["win"] = tr["pnl"] > 0

# which side breached (settle > 0 means intrinsic was paid)
def breach(r):
    if r["settle"] <= 0.005:
        return "none"
    s = r["settle_spot"]
    if pd.notna(r["sC"]) and s > r["sC"]:
        return "CALL"
    if pd.notna(r["sP"]) and s < r["sP"]:
        return "PUT"
    return "none"

tr["breach"] = tr.apply(breach, axis=1)

lose, winr = tr[~tr["win"]], tr[tr["win"]]
print(f"n={len(tr)}  winners={len(winr)}  losers={len(lose)}")

print("\n--- BREACH SIDE (losers) ---")
print(lose["breach"].value_counts().to_string())
print("losers with settle>0 but pnl>0 exist?", ((tr.settle > 0) & tr.win).sum())
print("\nbreach side among ALL settle>0 trades:")
print(tr[tr.settle > 0.005].groupby("breach")["win"].agg(["count", "mean"]).to_string())

print("\n--- CONTINUOUS STATE: losers vs winners (mean / median) ---")
for c in ["ret5", "ret20", "rv_past", "ivrv", "iv_mkt", "iv_rank", "dte"]:
    print(f"{c:>8}: L mean {lose[c].mean():+.3f} med {lose[c].median():+.3f} | "
          f"W mean {winr[c].mean():+.3f} med {winr[c].median():+.3f}")

print("\n--- ret5 buckets ---")
b = pd.cut(tr["ret5"], [-1, -0.10, -0.05, 0, 0.05, 0.10, 1])
print(tr.groupby(b, observed=True).agg(n=("win", "size"), win=("win", "mean"),
      pnl=("pnl", "sum"), breachP=("breach", lambda s: (s == "PUT").sum()),
      breachC=("breach", lambda s: (s == "CALL").sum())).to_string())

print("\n--- ret20 buckets ---")
b = pd.cut(tr["ret20"], [-1, -0.15, -0.05, 0.05, 0.15, 1])
print(tr.groupby(b, observed=True).agg(n=("win", "size"), win=("win", "mean"),
      pnl=("pnl", "sum"), breachP=("breach", lambda s: (s == "PUT").sum()),
      breachC=("breach", lambda s: (s == "CALL").sum())).to_string())

print("\n--- iv/rv_past buckets ---")
b = pd.cut(tr["ivrv"], [0, 0.8, 1.0, 1.2, 1.5, 10])
print(tr.groupby(b, observed=True).agg(n=("win", "size"), win=("win", "mean"),
      pnl=("pnl", "sum")).to_string())

print("\n--- day of week ---")
print(tr.groupby("dow").agg(n=("win", "size"), win=("win", "mean"),
      pnl=("pnl", "sum")).to_string())

print("\n--- iv_rank buckets ---")
b = pd.cut(tr["iv_rank"], [0.6, 0.7, 0.8, 0.9, 1.0])
print(tr.groupby(b, observed=True).agg(n=("win", "size"), win=("win", "mean"),
      pnl=("pnl", "sum")).to_string())

print("\n--- IV level buckets ---")
b = pd.cut(tr["iv"], [0, 0.5, 0.7, 0.9, 2])
print(tr.groupby(b, observed=True).agg(n=("win", "size"), win=("win", "mean"),
      pnl=("pnl", "sum")).to_string())

print("\n--- all 21 losers, sorted by pnl ---")
cols = ["entry", "expiry", "pnl", "breach", "ret5", "ret20", "ivrv", "iv_rank", "dow"]
print(lose.sort_values("pnl")[cols].to_string(index=False,
      formatters={"ret5": "{:+.3f}".format, "ret20": "{:+.3f}".format,
                  "ivrv": "{:.2f}".format}))

print("\n--- losers by year ---")
print(lose.groupby(lose.entry.dt.year)["pnl"].agg(["count", "sum"]).to_string())
