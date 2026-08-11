"""
One-shot validation battery for a candidate v2 config.

Discipline: this script is written BEFORE the improvement search finished and is
run ONCE on the combined winner config. If v2 fails here, we ship the baseline —
we do not iterate on v2 until it passes.

Usage:
    .venv/bin/python research/v2_validate.py '{"min_iv_rank": 0.60, ...}'
"""
import sys, json, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vrp.engine import Config, Backtest
import vrp.engine as E

BASELINE = dict(symbols=("BTC",), capital=600.0, min_iv_rank=0.60, short_delta=0.25,
                wing_width_deltas=0.07, defined_risk=True, both_sides=True,
                real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
                profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
                max_margin_util=0.50, max_leverage=3.0)
BASE_REF = {"cagr": 0.1035, "sharpe": 1.11, "maxdd": -0.0944, "pf": 2.44}


def metrics_line(label, cfg_kwargs, start=None, end=None):
    m = Backtest(Config(**cfg_kwargs)).run(start, end).metrics()
    if not m.get("trades"):
        print(f"  {label:<46} no trades")
        return None
    print(f"  {label:<46} n={m['trades']:>3} win {m['win_rate']:>5.1%} PF {m['profit_factor']:>5.2f} "
          f"CAGR {m['cagr']:>+7.1%} maxDD {m['max_dd']:>+6.1%} Sharpe {m['sharpe']:>5.2f}")
    return m


def main(v2_overrides: dict):
    v2 = {**BASELINE, **v2_overrides}
    # tuples arrive as lists from JSON
    if "symbols" in v2 and isinstance(v2["symbols"], list):
        v2["symbols"] = tuple(v2["symbols"])
    print("=" * 104)
    print("V2 CANDIDATE — ONE-SHOT VALIDATION")
    print("overrides:", json.dumps(v2_overrides, default=str))
    print("=" * 104)

    checks = {}

    print("\n1. HEADLINE vs BASELINE")
    mb = metrics_line("baseline", BASELINE)
    mv = metrics_line("v2", v2)
    if not mv:
        print("   FAIL: v2 produces no trades"); return
    checks["risk_adjusted"] = (mv["cagr"] / abs(mv["max_dd"])) >= (mb["cagr"] / abs(mb["max_dd"]))
    checks["sharpe"] = mv["sharpe"] >= mb["sharpe"] * 0.95

    print("\n2. OUT-OF-SAMPLE")
    oos = {}
    for lbl, a, b in (("in-sample ..2023-12", None, "2023-12-31"),
                      ("OOS 2024-01..", "2024-01-01", None),
                      ("2025-01..", "2025-01-01", None),
                      ("2026 only", "2026-01-01", None)):
        oos[lbl] = metrics_line(lbl, v2, a, b)
    checks["oos_positive"] = bool(oos["OOS 2024-01.."] and oos["OOS 2024-01.."]["cagr"] > 0
                                  and oos["2026 only"] and oos["2026 only"]["cagr"] > -0.02)

    print("\n3. BOOTSTRAP (20k resamples of trade P&L)")
    bt = Backtest(Config(**v2)).run()
    p = np.array([t.pnl for t in bt.closed])
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(p, len(p), True).mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"  n={len(p)}  mean ${p.mean():+.2f}/trade  95% CI [${lo:+.2f}, ${hi:+.2f}]  P(<=0)={(bs <= 0).mean():.2%}")
    checks["bootstrap"] = lo > 0

    print("\n4. COST STRESS (must stay >= baseline under the SAME stress)")
    s1 = metrics_line("v2 spread x2", {**v2, "spread_k": v2.get("spread_k", 0.0065) * 2})
    s2 = metrics_line("v2 fees x2", {**v2, "fee_notional_pct": v2.get("fee_notional_pct", 0.0003) * 2})
    b1 = metrics_line("baseline spread x2", {**BASELINE, "spread_k": 0.0130})
    b2 = metrics_line("baseline fees x2", {**BASELINE, "fee_notional_pct": 0.0006})
    checks["cost_stress"] = bool(s1 and b1 and s1["cagr"] >= b1["cagr"] - 0.01
                                 and s2 and b2 and s2["cagr"] >= b2["cagr"] - 0.01)

    print("\n5. NULL TEST (filter inverted must still destroy it)")
    orig = E.Backtest._try_open
    def inv(self, s, row, d):
        if row["iv_rank"] >= 0.40:
            return
        r = row.copy(); r["iv_rank"] = 1.0
        orig(self, s, r, d)
    E.Backtest._try_open = inv
    mn = metrics_line("v2 with INVERTED filter", v2)
    E.Backtest._try_open = orig
    checks["null"] = bool(mn is None or mn["cagr"] < mv["cagr"] - 0.05)

    print("\n6. VERDICT")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print(f"\n  {'V2 ACCEPTED' if ok else 'V2 REJECTED — SHIP THE BASELINE'}")
    return ok


if __name__ == "__main__":
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    main(overrides)
