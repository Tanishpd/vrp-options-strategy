"""Avenue: capital utilization via laddering and sizing — FINAL runner + diagnostics.

BEST VARIANT: baseline Config + max_concurrent_per_symbol=2 (risk_per_trade
unchanged at 0.04). Reproduces baseline vs best on full sample, OOS 2024-01+,
and 2026-only, plus:
  - daily concurrent-position distribution and margin-utilization percentiles
  - peak stacked max_loss (worst-case tail when 2 rungs are open)
  - entry-weekday / DTE / credit composition shift (why the ladder wins)
  - year-by-year comparison
  - P&L correlation of time-overlapping rung pairs
  - plateau checks: rpt neighbors and iv-rank-filter neighbors around the best cell

Run: ./.venv/bin/python research/improve/ladder_sizing_final.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Config, Backtest

BASE = dict(symbols=("BTC",), capital=600.0, min_iv_rank=0.60, short_delta=0.25,
            wing_width_deltas=0.07, defined_risk=True, both_sides=True,
            real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
            profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
            max_margin_util=0.50, max_leverage=3.0)
BEST = {**BASE, "max_concurrent_per_symbol": 2}


class InstrumentedBacktest(Backtest):
    """Backtest.run plus daily open-count / margin / stacked-max-loss recording."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.daily = []  # (date, n_open, margin_open, equity, stacked_max_loss)

    def run(self, start=None, end=None):
        idx = sorted(set().union(*[set(d.index) for d in self.data.values()]))
        if start:
            idx = [d for d in idx if d >= pd.Timestamp(start, tz="UTC")]
        if end:
            idx = [d for d in idx if d <= pd.Timestamp(end, tz="UTC")]
        for date in idx:
            self._manage(date)
            for sym in self.cfg.symbols:
                d = self.data[sym]
                if date in d.index:
                    self._try_open(sym, d.loc[date], date)
            self.curve.append((date, self.equity))
            self.daily.append((date, len(self.open),
                               sum(t.margin for t in self.open), self.equity,
                               sum(t.max_loss for t in self.open)))
        for tr in list(self.open):
            self.open.remove(tr)
        return self


def fmt(m):
    if m.get("trades", 0) == 0:
        return "n=0"
    return (f"n={m['trades']:>3} win={m['win_rate']:.1%} PF={m['profit_factor']:.2f} "
            f"CAGR={m['cagr']:+.2%} maxDD={m['max_dd']:.2%} Sharpe={m['sharpe']:.2f} "
            f"final=${m['final_equity']:.2f}")


def run3(kw, label):
    print(f"\n--- {label} ---")
    bt = InstrumentedBacktest(Config(**kw)).run()
    print(f"  full : {fmt(bt.metrics())}")
    for tag, s in (("OOS24", "2024-01-01"), ("2026 ", "2026-01-01")):
        m = InstrumentedBacktest(Config(**kw)).run(s, None).metrics()
        print(f"  {tag}: {fmt(m)}")
    return bt


def trade_df(bt):
    return pd.DataFrame([dict(entry=t.entry, exit=t.exit_date, expiry=t.expiry,
                              dte=(t.expiry - t.entry).total_seconds() / 86400,
                              credit=t.credit, fees=t.fees, pnl=t.pnl,
                              max_loss=t.max_loss, margin=t.margin,
                              wd=t.entry.weekday()) for t in bt.closed])


if __name__ == "__main__":
    print("=" * 88)
    print("LADDERING & SIZING — final comparison")
    print("=" * 88)
    bt_base = run3(BASE, "BASELINE  mc=1 rpt=0.04")
    bt_best = run3(BEST, "BEST      mc=2 rpt=0.04  (only change: max_concurrent_per_symbol=2)")

    # ---- concurrency & margin utilization -----------------------------------
    for label, bt in (("baseline", bt_base), ("ladder mc=2", bt_best)):
        d = pd.DataFrame(bt.daily, columns=["date", "n", "margin", "eq", "stackloss"])
        dist = d["n"].value_counts(normalize=True).sort_index()
        util = d["margin"] / d["eq"]
        stack = d["stackloss"] / d["eq"]
        print(f"\n[{label}] concurrent positions (share of days): "
              + "  ".join(f"{int(k)}: {v:.1%}" for k, v in dist.items()))
        print(f"[{label}] margin util pcts  p50={util.quantile(.5):.1%} "
              f"p75={util.quantile(.75):.1%} p90={util.quantile(.9):.1%} "
              f"p95={util.quantile(.95):.1%} p99={util.quantile(.99):.1%} "
              f"max={util.max():.1%}")
        print(f"[{label}] stacked max_loss/equity  p95={stack.quantile(.95):.1%} "
              f"p99={stack.quantile(.99):.1%} max={stack.max():.1%}")

    # ---- composition shift: why the ladder wins -----------------------------
    tb, tl = trade_df(bt_base), trade_df(bt_best)
    wd_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("\nentry weekday distribution:")
    for label, t in (("baseline", tb), ("ladder", tl)):
        c = t["wd"].value_counts(normalize=True).sort_index()
        print(f"  {label:<9}" + "  ".join(f"{wd_names[int(k)]}:{v:.0%}" for k, v in c.items()))
    for label, t in (("baseline", tb), ("ladder", tl)):
        print(f"  {label:<9} avg dte={t.dte.mean():.1f}  avg credit=${t.credit.mean():.2f}  "
              f"fees/credit={t.fees.sum() / t.credit.sum():.1%}  "
              f"fees/netPnL={t.fees.sum() / t.pnl.sum():.0%}")

    # ---- year by year -------------------------------------------------------
    print("\nyear-by-year net P&L ($):")
    for label, t in (("baseline", tb), ("ladder", tl)):
        g = t.groupby(t.entry.dt.year).pnl.agg(["count", "sum"])
        print(f"  {label:<9}" + "  ".join(f"{y}: n={int(r['count'])},{r['sum']:+.0f}"
                                          for y, r in g.iterrows()))

    # ---- overlap-pair P&L correlation ---------------------------------------
    tr = bt_best.closed
    pairs = []
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            a, b = tr[i], tr[j]
            if a.entry <= b.exit_date and b.entry <= a.exit_date:  # time overlap
                pairs.append((a.pnl / a.max_loss, b.pnl / b.max_loss,
                              (min(a.exit_date, b.exit_date) - max(a.entry, b.entry)).days))
    if pairs:
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        ov = np.array([p[2] for p in pairs])
        r = float(np.corrcoef(x, y)[0, 1])
        both_neg = float(np.mean((x < 0) & (y < 0)))
        print(f"\noverlapping rung pairs: {len(pairs)}  "
              f"avg overlap={ov.mean():.1f} days  "
              f"corr(pnl/max_loss)={r:+.2f}  both-lose share={both_neg:.1%}")
    # control: consecutive NON-overlapping trades
    xs, ys = [], []
    for i in range(len(tr) - 1):
        a, b = tr[i], tr[i + 1]
        if b.entry > a.exit_date:
            xs.append(a.pnl / a.max_loss); ys.append(b.pnl / b.max_loss)
    if len(xs) > 2:
        print(f"control (consecutive non-overlapping pairs, n={len(xs)}): "
              f"corr={float(np.corrcoef(xs, ys)[0, 1]):+.2f}")

    # ---- plateau checks around the best cell --------------------------------
    print("\nplateau checks (mc=2):")
    for rpt in (0.04, 0.05, 0.06):
        for ivr in (0.55, 0.60, 0.65):
            kw = {**BEST, "risk_per_trade": rpt, "min_iv_rank": ivr}
            m = InstrumentedBacktest(Config(**kw)).run().metrics()
            mo = InstrumentedBacktest(Config(**kw)).run("2024-01-01", None).metrics()
            print(f"  rpt={rpt:.2f} ivr={ivr:.2f}: full {fmt(m)}")
            print(f"                        OOS  {fmt(mo)}")
