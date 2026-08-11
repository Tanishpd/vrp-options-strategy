"""Attribution + plateau follow-up for the laddering avenue.

1) mc=1 vs mc=2 delta at neighboring iv-rank filters (is the improvement from
   mc=2 itself robust, or specific to ivr=0.60?)
2) Decomposition: mc=1 with min_dte=8 forces Wed/Thu entries (8.5-9.5 dte)
   WITHOUT any concurrency — how much of the mc=2 gain is just the longer,
   fee-cheaper tenor vs genuine laddering?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Config, Backtest

BASE = dict(symbols=("BTC",), capital=600.0, min_iv_rank=0.60, short_delta=0.25,
            wing_width_deltas=0.07, defined_risk=True, both_sides=True,
            real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
            profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
            max_margin_util=0.50, max_leverage=3.0)


def fmt(m):
    if m.get("trades", 0) == 0:
        return "n=0"
    return (f"n={m['trades']:>3} win={m['win_rate']:.1%} PF={m['profit_factor']:.2f} "
            f"CAGR={m['cagr']:+.2%} maxDD={m['max_dd']:.2%} Sharpe={m['sharpe']:.2f}")


def show(kw, label):
    mf = Backtest(Config(**kw)).run().metrics()
    mo = Backtest(Config(**kw)).run("2024-01-01", None).metrics()
    m6 = Backtest(Config(**kw)).run("2026-01-01", None).metrics()
    print(f"{label:<34} full {fmt(mf)}")
    print(f"{'':<34} OOS  {fmt(mo)}")
    print(f"{'':<34} 2026 {fmt(m6)}")


if __name__ == "__main__":
    print("mc=1 vs mc=2 at neighboring iv-rank filters (rpt=0.04):")
    for ivr in (0.55, 0.60, 0.65, 0.70):
        for mc in (1, 2):
            show({**BASE, "min_iv_rank": ivr, "max_concurrent_per_symbol": mc},
                 f"  ivr={ivr:.2f} mc={mc}")

    print("\ndecomposition — tenor effect without concurrency:")
    show({**BASE, "min_dte": 8}, "  mc=1 min_dte=8 (Wed/Thu only)")
    show({**BASE, "max_concurrent_per_symbol": 2}, "  mc=2 (best)")
