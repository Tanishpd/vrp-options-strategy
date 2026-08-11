"""Tenor avenue (b,c,d): REAL monthly expiries (last Friday of month, 12:00 UTC),
min_dte=7, max_dte=35 — the honest headline test. Plus:
  - neighbor cells (min_dte 5/7/10, ladder of 2 monthlies) for plateau check
  - fee share vs baseline's 36%
  - bootstrap 95% CI on mean P&L/trade
  - mixed book: monthly core + weekly overlay only when iv_rank >= 0.80
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tenor_lib import (BASELINE, MixedBacktest, MonthlyBacktest, bootstrap_ci,
                       fee_share, metrics_line)
from engine import Backtest, Config

WINDOWS = [("full", None), ("2024+", "2024-01-01"), ("2026", "2026-01-01")]


def run_all(tag, make_bt):
    out = {}
    for wtag, start in WINDOWS:
        bt = make_bt().run(start, None)
        out[wtag] = (bt, metrics_line(f"{tag} [{wtag}]", bt))
    bt_full = out["full"][0]
    f, p, s = fee_share(bt_full)
    pnls = [t.pnl for t in bt_full.closed]
    mean, lo, hi = bootstrap_ci(pnls)
    print(f"{'':<34} fees=${f:.2f}  netP&L=${p:.2f}  fee_share={s:.1%}  "
          f"mean P&L/trade=${mean:+.2f}  boot95CI=[{lo:+.2f},{hi:+.2f}]"
          f"{'  (CI excludes 0)' if lo > 0 else '  (CI INCLUDES 0)'}")
    return out


print("=" * 112)
print("REAL MONTHLY EXPIRIES (last Friday of month) vs baseline weekly")
print("=" * 112)

run_all("baseline real-weekly", lambda: Backtest(Config(**BASELINE)))
print("-" * 112)

def mk(min_dte=7, max_dte=35, conc=1, **extra):
    kw = dict(BASELINE)
    kw.update(min_dte=min_dte, max_dte=max_dte, max_concurrent_per_symbol=conc, **extra)
    return lambda: MonthlyBacktest(Config(**kw))

# headline
head = run_all("monthly min_dte=7 max_dte=35", mk())
print("-" * 112)
# neighbors (plateau check)
run_all("monthly min_dte=5", mk(min_dte=5))
print("-" * 112)
run_all("monthly min_dte=10", mk(min_dte=10))
print("-" * 112)
run_all("monthly max_dte=42 (never skip)", mk(max_dte=42))
print("-" * 112)
run_all("monthly ladder x2", mk(conc=2))
print("-" * 112)

# mixed book: monthly core + weekly only at iv_rank >= 0.80
def mk_mixed(rank=0.80):
    kw = dict(BASELINE)
    kw.update(min_dte=7, max_dte=35)
    return lambda: MixedBacktest(Config(**kw), weekly_min_rank=rank)

run_all("mixed monthly + weekly@rank>=0.8", mk_mixed(0.80))
print("-" * 112)
run_all("mixed monthly + weekly@rank>=0.7", mk_mixed(0.70))
print("-" * 112)

# detail on the headline monthly: tenor distribution and per-book stats
bt = head["full"][0]
dtes = [ (t.expiry - t.entry).total_seconds()/86400 for t in bt.closed ]
print(f"headline monthly: tenor distribution  min={min(dtes):.1f}  median={np.median(dtes):.1f}  "
      f"max={max(dtes):.1f} days;  trades/yr={len(bt.closed)/4.9:.1f}")
med_margin = np.median([t.margin for t in bt.closed])
print(f"headline monthly: median margin=${med_margin:.0f}, worst trade=${min(t.pnl for t in bt.closed):.2f}")
