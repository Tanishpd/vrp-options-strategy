"""
AVENUE: size by edge strength (IV rank) instead of flat 4% risk_per_trade.

Baseline: flat risk_per_trade=0.04 on every entry (rank >= 0.60).
Variants tested (whole grid reported, no cherry-picking):
  a) linear ramp:  rpt = lo + (hi-lo) * clip((rank-0.60)/0.40, 0, 1)
  b) step:         rpt = lo below cut, hi at/above cut
  c) inverse ramp: rpt = hi at rank 0.60 down to lo at rank 1.00  (null control:
     if this ALSO "improves" things the avenue is noise)
  d) hump:         peak sizing at rank ~0.85, tapering into the top bucket
     (motivated by the 0.9-1.0 VRP dip; flagged as in-sample-informed)
  plus a risk-matched FLAT control at the realized mean rpt of the best variant.

Implementation: subclass Backtest, set self.cfg.risk_per_trade from the row's
iv_rank before delegating to the stock _try_open, restore afterwards.
Engine, data, and every other parameter are untouched.

Run:  "/Users/tanishpd/Desktop/crypto 2/.venv/bin/python" research/improve/sizing_by_edge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from engine import Backtest, Config  # noqa: E402

BASE = dict(symbols=("BTC",), capital=600.0, min_iv_rank=0.60, short_delta=0.25,
            wing_width_deltas=0.07, defined_risk=True, both_sides=True,
            real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
            profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
            max_margin_util=0.50, max_leverage=3.0)

PERIODS = [("full", None, None), ("oos24", "2024-01-01", None), ("y2026", "2026-01-01", None)]


class SizedBacktest(Backtest):
    """Risk per trade is a function of the entry day's IV rank."""

    def __init__(self, cfg: Config, size_fn):
        super().__init__(cfg)
        self.size_fn = size_fn
        self.rpt_used: list[float] = []          # rpt actually applied on opened trades

    def _try_open(self, sym, row, date):
        old = self.cfg.risk_per_trade
        rpt = float(self.size_fn(float(row["iv_rank"])))
        n_before = len(self.open)
        self.cfg.risk_per_trade = rpt
        try:
            super()._try_open(sym, row, date)
        finally:
            self.cfg.risk_per_trade = old
        if len(self.open) > n_before:
            self.rpt_used.append(rpt)


# ---- sizing rules --------------------------------------------------------
def flat(x):
    return lambda r: x

def linear(lo, hi):
    return lambda r: lo + (hi - lo) * min(max((r - 0.60) / 0.40, 0.0), 1.0)

def step(lo, hi, cut):
    return lambda r: lo if r < cut else hi

def inverse(hi, lo):          # hi at rank 0.60, lo at rank 1.00
    return lambda r: hi + (lo - hi) * min(max((r - 0.60) / 0.40, 0.0), 1.0)

def hump(lo, peak, tail):     # lo@0.60 -> peak@0.85 -> tail@1.00
    def f(r):
        if r <= 0.85:
            return lo + (peak - lo) * min(max((r - 0.60) / 0.25, 0.0), 1.0)
        return peak + (tail - peak) * min((r - 0.85) / 0.15, 1.0)
    return f


VARIANTS = [
    ("baseline flat 4%",        flat(0.04)),
    ("a1 linear 2%->8%",        linear(0.02, 0.08)),
    ("a2 linear 3%->6%",        linear(0.03, 0.06)),
    ("a3 linear 2%->6%",        linear(0.02, 0.06)),
    ("a4 linear 4%->8%",        linear(0.04, 0.08)),
    ("b1 step 3%/6% @0.75",     step(0.03, 0.06, 0.75)),
    ("b2 step 4%/8% @0.75",     step(0.04, 0.08, 0.75)),
    ("b3 step 3%/6% @0.80",     step(0.03, 0.06, 0.80)),
    ("c1 INVERSE 8%->2%",       inverse(0.08, 0.02)),
    ("c2 INVERSE step 6%/3%",   step(0.06, 0.03, 0.75)),
    ("d1 hump 3/8/5 pk@.85",    hump(0.03, 0.08, 0.05)),
    ("ctrl flat 5.5% (=a4 avg)", flat(0.055)),
]


def run_variant(size_fn, start, end):
    bt = SizedBacktest(Config(**BASE), size_fn).run(start, end)
    m = bt.metrics()
    m["avg_rpt"] = float(np.mean(bt.rpt_used)) if bt.rpt_used else float("nan")
    m["ranks"] = [t.entry_iv_rank for t in bt.closed]
    m["lots"] = [t.legs[0].lots for t in bt.closed]
    return m


def fmt(m):
    return (f"n={m['trades']:>3}  win={m['win_rate']:>5.1%}  PF={m['profit_factor']:>5.2f}  "
            f"CAGR={m['cagr']:>+7.2%}  maxDD={m['max_dd']:>+7.2%}  Sh={m['sharpe']:>5.2f}  "
            f"CAGR/DD={m['cagr'] / abs(m['max_dd']):>5.2f}  eq=${m['final_equity']:>8.2f}  "
            f"avg_rpt={m['avg_rpt']:.3f}")


def main():
    results = {}
    for name, fn in VARIANTS:
        results[name] = {p: run_variant(fn, s, e) for p, s, e in PERIODS}

    # risk-matched flat control at the mean rpt of the primary scaled variant (a1)
    matched = results["a1 linear 2%->8%"]["full"]["avg_rpt"]
    name = f"ctrl flat @{matched:.3f} (rpt-matched to a1)"
    results[name] = {p: run_variant(flat(matched), s, e) for p, s, e in PERIODS}

    for name, per in results.items():
        print(f"\n{name}")
        for p, _, _ in PERIODS:
            print(f"  {p:<6} {fmt(per[p])}")

    r = np.array(results["baseline flat 4%"]["full"]["ranks"])
    print("\nentry iv_rank distribution (baseline, full sample):")
    print("  " + "  ".join(f"[{a:.2f},{a + 0.1:.2f}): {((r >= a) & (r < a + 0.1)).sum()}"
                           for a in np.arange(0.6, 1.0, 0.1)) + f"  =1.00: {(r >= 0.999).sum()}")

    # granularity diagnostic: with $600 capital lots are integer-floored, so how
    # much of the intended sizing signal survives quantization?
    for v in ("baseline flat 4%", "a1 linear 2%->8%", "c1 INVERSE 8%->2%"):
        lots = np.array(results[v]["full"]["lots"], dtype=float)
        print(f"lots ({v:<22}): min={lots.min():.0f} med={np.median(lots):.0f} "
              f"max={lots.max():.0f} mean={lots.mean():.2f} sd={lots.std():.2f}")
    b = np.array(results["baseline flat 4%"]["full"]["lots"], dtype=float)
    a = np.array(results["a1 linear 2%->8%"]["full"]["lots"], dtype=float)
    if len(a) == len(b):
        print(f"a1 vs baseline: identical lots on {int((a == b).sum())}/{len(b)} trades")


if __name__ == "__main__":
    main()
