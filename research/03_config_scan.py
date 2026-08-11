import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vrp.engine import Config, Backtest

def show(label, cfg, start=None, end=None):
    m = Backtest(cfg).run(start, end).metrics()
    if not m.get("trades"):
        print(f"  {label:<42} no trades"); return None
    print(f"  {label:<42} n={m['trades']:>4}  win {m['win_rate']:>5.1%}  "
          f"PF {m['profit_factor']:>5.2f}  CAGR {m['cagr']:>+7.1%}  "
          f"maxDD {m['max_dd']:>+6.1%}  Sharpe {m['sharpe']:>5.2f}  "
          f"final ${m['final_equity']:>9,.0f}")
    return m

print("=" * 118)
print("A. THE FILTER IS THE STRATEGY — does IV rank actually matter in the P&L?")
print("=" * 118)
for thr in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8):
    show(f"min_iv_rank = {thr:.2f}", Config(min_iv_rank=thr))

print()
print("=" * 118)
print("B. OUT-OF-SAMPLE — filter chosen on 2021-2023, tested on 2024-2026 (never seen)")
print("=" * 118)
for lbl, a, b in (("in-sample  2021-03..2023-12", None, "2023-12-31"),
                  ("OUT-OF-SAMPLE 2024-01..now", "2024-01-01", None)):
    show(lbl, Config(min_iv_rank=0.60), a, b)

print()
print("=" * 118)
print("C. STRUCTURE — defined risk vs naked, one side vs strangle")
print("=" * 118)
show("credit spreads, both sides (default)", Config())
show("credit spreads, calls only",          Config(both_sides=False))
show("NAKED shorts, both sides",            Config(defined_risk=False))

print()
print("=" * 118)
print("D. SENSITIVITY — is this a knife-edge fit or a plateau?")
print("=" * 118)
for d in (0.10, 0.15, 0.20, 0.25):
    show(f"short delta = {d:.2f}", Config(short_delta=d))
print()
for pt in (0.4, 0.5, 0.6, 0.75):
    show(f"profit take at {pt:.0%} of credit", Config(profit_take=pt))
print()
for sp in (0.05, 0.10, 0.15, 0.20):
    show(f"bid-ask spread = {sp:.0%}", Config(spread_pct=sp))
print()
for r in (0.02, 0.04, 0.06, 0.10):
    show(f"risk per trade = {r:.0%}", Config(risk_per_trade=r))
