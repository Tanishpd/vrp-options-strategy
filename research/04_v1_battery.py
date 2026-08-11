"""Try to break the result. Out-of-sample, bootstrap, cost stress, and a null test."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vrp.engine import Config, Backtest

BEST = dict(short_delta=0.25, dte=7, defined_risk=True, hold_to_expiry=True,
            profit_take=0.99, min_iv_rank=0.60)

def run(**kw):
    c = Config(**{**BEST, **kw})
    return Backtest(c).run(kw.pop("_start", None), kw.pop("_end", None))

def line(lbl, bt):
    m = bt.metrics()
    if not m.get("trades"):
        print(f"  {lbl:<44} no trades"); return None
    print(f"  {lbl:<44} n={m['trades']:>4} win {m['win_rate']:>5.1%} PF {m['profit_factor']:>5.2f} "
          f"CAGR {m['cagr']:>+7.1%} maxDD {m['max_dd']:>+6.1%} Sharpe {m['sharpe']:>5.2f}")
    return m

print("=" * 108)
print("1. OUT-OF-SAMPLE — every parameter above was chosen looking at the full sample.")
print("   Split it: does the same config work on data that had no say in choosing it?")
print("=" * 108)
for lbl, a, b in (("2021-03 .. 2023-12  (in-sample)", None, "2023-12-31"),
                  ("2024-01 .. 2026-08  (OUT-OF-SAMPLE)", "2024-01-01", None),
                  ("2025-01 .. 2026-08  (most recent)", "2025-01-01", None),
                  ("2026 only  (VRP was ~0 this year)", "2026-01-01", None)):
    c = Config(**BEST)
    line(lbl, Backtest(c).run(a, b))

print()
print("=" * 108)
print("2. BOOTSTRAP — resample trades with replacement. Is the edge distinguishable from luck?")
print("=" * 108)
bt = Backtest(Config(**BEST)).run()
pnl = np.array([t.pnl for t in bt.closed])
rng = np.random.default_rng(0)
boot = np.array([rng.choice(pnl, len(pnl), replace=True).mean() for _ in range(20000)])
print(f"  trades n={len(pnl)}   mean P&L/trade ${pnl.mean():+.3f}   sd ${pnl.std():.2f}")
print(f"  bootstrap 95% CI on mean P&L/trade: [${np.percentile(boot,2.5):+.3f}, ${np.percentile(boot,97.5):+.3f}]")
print(f"  P(mean <= 0) = {(boot<=0).mean():.1%}   {'<-- NOT significant' if (boot<=0).mean()>0.05 else '<-- significant at 5%'}")

print()
print("=" * 108)
print("3. COST STRESS — my spread curve is fitted to ONE snapshot. What if it is wrong?")
print("=" * 108)
for mult, lbl in ((1.0, "as measured"), (1.5, "50% wider"), (2.0, "double"), (3.0, "triple")):
    line(f"spread x{mult:.1f} ({lbl})", Backtest(Config(**{**BEST, "spread_k": 0.0065*mult})).run())
print()
for f, lbl in ((0.0003, "Delta taker 0.03%"), (0.0006, "double fees"), (0.0010, "0.10%")):
    line(f"fees {lbl}", Backtest(Config(**{**BEST, "fee_notional_pct": f})).run())

print()
print("=" * 108)
print("4. NULL TEST — remove the edge thesis. If the IV-rank filter is real,")
print("   trading the LOW-IV days should lose, and random entry should be ~flat.")
print("=" * 108)
import vrp.engine as E
line("IV rank >= 0.60 (the thesis)", Backtest(Config(**BEST)).run())
# invert: trade only when IV is CHEAP
orig = E.Backtest._try_open
def inverted(self, sym, row, date):
    if row["iv_rank"] >= 0.40: return
    r = row.copy(); r["iv_rank"] = 1.0
    orig(self, sym, r, date)
E.Backtest._try_open = inverted
line("IV rank < 0.40 (inverted thesis)", Backtest(Config(**BEST)).run())
E.Backtest._try_open = orig
line("no filter at all", Backtest(Config(**{**BEST, "min_iv_rank": 0.0})).run())

print()
print("=" * 108)
print("5. PARAMETER SURFACE — plateau or spike? (CAGR across the delta x profit-take grid)")
print("=" * 108)
print(f"  {'':>10}" + "".join(f"{f'pt={p:.2f}':>10}" for p in (0.50, 0.70, 0.85, 0.99)))
for d in (0.15, 0.20, 0.25, 0.30):
    cells = []
    for p in (0.50, 0.70, 0.85, 0.99):
        m = Backtest(Config(**{**BEST, "short_delta": d, "profit_take": p})).run().metrics()
        cells.append(f"{m.get('cagr', float('nan')):>+9.1%}" if m.get("trades") else f"{'-':>9}")
    print(f"  delta={d:.2f}" + "".join(f"{c:>10}" for c in cells))
