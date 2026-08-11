"""
Avenue: execution & fee engineering.

Tests (all vs the validated baseline):
  a) maker execution      — spread_k=0.00325 + fee 0.02% (full maker),
                            spread_k=0.0049 + fee 0.025% (half maker / half taker),
                            spread_k=0.0049 + fee 0.03%  (spread-only pessimist)
  b) credit/fee gate      — skip entries whose net credit < K * round-trip fees
                            (round-trip = 2 * entry fees), K in {3, 5, 8}
  c) wing economics       — wing_width_deltas in {0.05, 0.07, 0.10, 0.12},
                            plus defined_risk=False (naked strangle)
  d) combined stack       — best of a+b+c together

Every variant reports full sample, OOS 2024-01+, and 2026-only.
Run: .venv/bin/python research/improve/execfees_grid.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Config, Backtest  # noqa: E402

BASE = dict(symbols=("BTC",), capital=600.0, min_iv_rank=0.60, short_delta=0.25,
            wing_width_deltas=0.07, defined_risk=True, both_sides=True,
            real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
            profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
            max_margin_util=0.50, max_leverage=3.0)


class GatedBacktest(Backtest):
    """Baseline entry logic + a credit/fee gate: refuse trades whose net credit
    is below k_gate x round-trip fees (round-trip modeled as 2x entry fees)."""

    def __init__(self, cfg: Config, k_gate: float):
        super().__init__(cfg)
        self.k_gate = k_gate
        self.gated = 0

    def _try_open(self, sym, row, date):
        n_before = len(self.open)
        super()._try_open(sym, row, date)
        if len(self.open) > n_before:
            tr = self.open[-1]
            if tr.credit < self.k_gate * (2.0 * tr.fees):
                # undo the open exactly: entry fees were already debited
                self.equity += tr.fees
                self.open.pop()
                self.gated += 1


def run3(make_bt):
    """Fresh backtest per window: full sample, OOS 2024+, 2026 only."""
    out = {}
    for tag, start in (("full", None), ("oos24", "2024-01-01"), ("y26", "2026-01-01")):
        bt = make_bt()
        bt.run(start, None)
        m = bt.metrics()
        m["gated"] = getattr(bt, "gated", 0)
        out[tag] = m
    return out


def fmt(m):
    if m.get("trades", 0) == 0:
        return "n=0"
    return (f"n={m['trades']:<3} win={m['win_rate']*100:5.1f}% PF={m['profit_factor']:5.2f} "
            f"CAGR={m['cagr']*100:+6.2f}% DD={m['max_dd']*100:6.2f}% "
            f"Sh={m['sharpe']:5.2f} worst={m['worst_trade']:+7.2f} "
            f"fin=${m['final_equity']:7.2f}"
            + (f" gated={m['gated']}" if m.get("gated") else ""))


def report(name, res):
    print(f"\n{name}")
    for tag in ("full", "oos24", "y26"):
        print(f"  {tag:<6} {fmt(res[tag])}")


def main():
    variants = []

    # ---- baseline ------------------------------------------------------
    variants.append(("BASELINE (taker, spread_k=.0065, fee .03%)",
                     lambda: Backtest(Config(**BASE))))

    # ---- a) maker execution -------------------------------------------
    for name, sk, feep in [
        ("a1 MAKER full   (spread_k=.00325, fee .02%)", 0.00325, 0.0002),
        ("a2 MAKER half   (spread_k=.0049,  fee .025%)", 0.0049, 0.00025),
        ("a3 MAKER spread-only pessimist (spread_k=.0049, fee .03%)", 0.0049, 0.0003),
    ]:
        kw = dict(BASE, spread_k=sk, fee_notional_pct=feep)
        variants.append((name, lambda kw=kw: Backtest(Config(**kw))))

    # ---- b) credit/fee gate (on baseline taker costs) ------------------
    # K=2 and K=4 are neighbor cells to check whether K=3 sits on a plateau
    for K in (2, 3, 4, 5, 8):
        variants.append((f"b  GATE K={K} (skip if credit < {K}x2x entry fees), taker costs",
                         lambda K=K: GatedBacktest(Config(**BASE), K)))

    # ---- c) wing economics --------------------------------------------
    # 0.15 is a neighbor cell to check whether 0.10-0.12 sits on a plateau
    for w in (0.05, 0.07, 0.10, 0.12, 0.15):
        kw = dict(BASE, wing_width_deltas=w)
        variants.append((f"c  WING width={w:.2f} (defined risk), taker costs",
                         lambda kw=kw: Backtest(Config(**kw))))
    kw = dict(BASE, defined_risk=False)
    variants.append(("c  NAKED strangle (defined_risk=False), taker costs",
                     lambda kw=kw: Backtest(Config(**kw))))

    # ---- d) combined stacks -------------------------------------------
    MAKER = dict(spread_k=0.00325, fee_notional_pct=0.0002)
    HALF = dict(spread_k=0.0049, fee_notional_pct=0.00025)
    for name, cost, wing, gate in [
        ("d  a1 maker + wing 0.10",                 MAKER, 0.10, None),
        ("d  a1 maker + wing 0.12",                 MAKER, 0.12, None),
        ("d  a1 maker + gate K=3",                  MAKER, 0.07, 3),
        ("d  a1 maker + wing 0.10 + gate K=3",      MAKER, 0.10, 3),
        ("d  a1 maker + wing 0.12 + gate K=3",      MAKER, 0.12, 3),
        ("d  a2 half-maker + wing 0.10",            HALF, 0.10, None),
        ("d  a2 half-maker + wing 0.10 + gate K=3", HALF, 0.10, 3),
        ("d  a2 half-maker + wing 0.12",            HALF, 0.12, None),
    ]:
        kw = dict(BASE, wing_width_deltas=wing, **cost)
        if gate is None:
            variants.append((name, lambda kw=kw: Backtest(Config(**kw))))
        else:
            variants.append((name, lambda kw=kw, g=gate: GatedBacktest(Config(**kw), g)))

    for name, make in variants:
        report(name, run3(make))

    # ---- fee share of net P&L (full sample) ---------------------------
    print("\nFEE SHARE OF NET P&L (full sample)")
    for name, kw in [
        ("baseline", BASE),
        ("best: a1 maker + wing 0.10",
         dict(BASE, wing_width_deltas=0.10, spread_k=0.00325, fee_notional_pct=0.0002)),
        ("pessimist: a2 half-maker + wing 0.10",
         dict(BASE, wing_width_deltas=0.10, spread_k=0.0049, fee_notional_pct=0.00025)),
    ]:
        bt = Backtest(Config(**kw)).run()
        fees = sum(t.fees for t in bt.closed)
        pnl = sum(t.pnl for t in bt.closed)
        print(f"  {name:38s} fees=${fees:7.2f}  netPnL=${pnl:8.2f}  fees/netPnL={fees/pnl:5.1%}")


if __name__ == "__main__":
    main()
