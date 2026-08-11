"""Avenue: capital utilization via laddering and sizing — exploration grid.

Grid: max_concurrent_per_symbol x risk_per_trade x max_dte, each evaluated on
full sample, OOS 2024-01+, and 2026-only. Also records daily concurrency,
margin utilization, and lot-cap diagnostics via an instrumented subclass.
Nothing in research/ outside improve/ is modified.
"""
import sys, math
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


class InstrumentedBacktest(Backtest):
    """Same run loop as Backtest.run, plus daily open-count / margin recording."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.daily = []  # (date, n_open, margin_open, equity)

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
                               sum(t.margin for t in self.open), self.equity))
        for tr in list(self.open):
            self.open.remove(tr)
        return self


def run_variant(kw, start=None, end=None):
    bt = InstrumentedBacktest(Config(**{**BASE, **kw})).run(start, end)
    m = bt.metrics()
    return bt, m


def fmt(m):
    if m.get("trades", 0) == 0:
        return "n=0"
    return (f"n={m['trades']:>3} win={m['win_rate']:.1%} PF={m['profit_factor']:.2f} "
            f"CAGR={m['cagr']:+.2%} maxDD={m['max_dd']:.2%} Sharpe={m['sharpe']:.2f} "
            f"final=${m['final_equity']:.0f}")


def concurrency_dist(bt):
    n = np.array([x[1] for x in bt.daily])
    return {k: float((n == k).mean()) for k in range(int(n.max()) + 1)}


def margin_util_pcts(bt):
    u = np.array([x[2] / x[3] for x in bt.daily if x[3] > 0])
    return {p: float(np.percentile(u, p)) for p in (50, 75, 90, 95, 99, 100)}


if __name__ == "__main__":
    rows = []
    for mc in (1, 2, 3):
        for mdte in (10, 14):
            for rpt in (0.04, 0.06, 0.08, 0.12):
                kw = dict(max_concurrent_per_symbol=mc, max_dte=mdte,
                          risk_per_trade=rpt)
                bt_full, mf = run_variant(kw)
                _, mo = run_variant(kw, "2024-01-01", None)
                _, m6 = run_variant(kw, "2026-01-01", None)
                cd = concurrency_dist(bt_full)
                mu = margin_util_pcts(bt_full)
                ratio = mf["cagr"] / abs(mf["max_dd"]) if mf.get("max_dd") else float("nan")
                rows.append(dict(mc=mc, max_dte=mdte, rpt=rpt,
                                 n=mf["trades"], win=mf["win_rate"],
                                 pf=mf["profit_factor"], cagr=mf["cagr"],
                                 dd=mf["max_dd"], sharpe=mf["sharpe"],
                                 ratio=ratio,
                                 oos_cagr=mo.get("cagr", float("nan")),
                                 oos_pf=mo.get("profit_factor", float("nan")),
                                 oos_dd=mo.get("max_dd", float("nan")),
                                 y26_cagr=m6.get("cagr", float("nan")),
                                 y26_pf=m6.get("profit_factor", float("nan")),
                                 pct_days_ge2=sum(v for k, v in cd.items() if k >= 2),
                                 max_open=max(cd),
                                 mu50=mu[50], mu90=mu[90], mu99=mu[99]))
                print(f"mc={mc} max_dte={mdte} rpt={rpt:.2f} | full {fmt(mf)}")
                print(f"    OOS24 {fmt(mo)}")
                print(f"    2026  {fmt(m6)}")
                print(f"    days with >=2 open: {rows[-1]['pct_days_ge2']:.1%}  "
                      f"max open: {rows[-1]['max_open']}  "
                      f"margin util p50/p90/p99: {mu[50]:.1%}/{mu[90]:.1%}/{mu[99]:.1%}")
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "ladder_sizing_grid_results.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
