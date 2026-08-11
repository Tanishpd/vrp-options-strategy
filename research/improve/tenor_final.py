"""Tenor avenue — FINAL reproducer.

Headline variant: REAL monthly expiries (last Friday of month, 12:00 UTC
settlement, Deribit/Delta convention), min_dte=7, max_dte=35, all other
parameters identical to the validated weekly baseline.

Verdict: NEGATIVE. Reproduces the exact headline numbers:
  full   : n=31  win 77.4%  PF 2.32  CAGR +4.52%  maxDD -4.34%  Sharpe 0.80
  2024+  : n=22  win 72.7%  PF 1.91  CAGR +4.50%  maxDD -4.34%  Sharpe 0.67
  2026   : n=5   win 80.0%  PF 2.72  CAGR +6.86%  maxDD -3.37%  Sharpe 1.03
  fee share 20.2% of net P&L (vs baseline 36.1%), boot95CI on mean P&L/trade
  [+0.51, +10.52] — excludes zero, barely.

Full grid: research/improve/tenor_synthetic_scan.py (synthetic dte 7/10/14/21/30)
and research/improve/tenor_monthly.py (real monthly neighbors + mixed book).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tenor_lib import BASELINE, MonthlyBacktest, bootstrap_ci, fee_share, metrics_line
from engine import Backtest, Config

KW = dict(BASELINE)
KW.update(min_dte=7, max_dte=35)          # real monthly calendar bounds

print("baseline (validated weekly), for reference:")
for wtag, start in [("full", None), ("2024+", "2024-01-01"), ("2026", "2026-01-01")]:
    metrics_line(f"  baseline [{wtag}]", Backtest(Config(**BASELINE)).run(start, None))

print("\nheadline: real monthly expiries (last Friday of month):")
full = None
for wtag, start in [("full", None), ("2024+", "2024-01-01"), ("2026", "2026-01-01")]:
    bt = MonthlyBacktest(Config(**KW)).run(start, None)
    metrics_line(f"  monthly  [{wtag}]", bt)
    if wtag == "full":
        full = bt

f, p, s = fee_share(full)
mean, lo, hi = bootstrap_ci([t.pnl for t in full.closed])
print(f"\n  fees=${f:.2f}  net P&L=${p:.2f}  fee share={s:.1%}  (baseline: 36.1%)")
print(f"  mean P&L/trade=${mean:+.2f}  bootstrap 95% CI=[{lo:+.2f}, {hi:+.2f}]")
