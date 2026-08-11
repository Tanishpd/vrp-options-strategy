"""Avenue: entry filter refinement — Part 2 runner.

Loss anatomy (research/improve/filters_loss_anatomy.py) showed:
  - 13 of 21 baseline losers breached the SHORT CALL (rally/bounce weeks).
  - Call breaches cluster after mild 5d dips (ret5 in (-10%,-5%]: 5 breaches,
    win 40%) and in strong 20d uptrends (ret20 > +15%: 5 breaches, win 65%).
  - iv/rv_past does NOT separate losers from winners (losers ivrv 1.24 vs
    winners 1.15) — that family is not tested.

Two filter families, six variants total (anti-mining budget):
  A: skip the CALL side (sell put-spread only) when trailing 5d return <= t,
     t in {-0.04, -0.06, -0.08}
  B: skip the CALL side when trailing 20d return >= t, t in {0.10, 0.15, 0.20}

Everything else identical to the validated baseline. Reproduce with:
  .venv/bin/python research/improve/filters_runner.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import (Backtest, Config, Leg, Trade, bs, fee, margin_req,  # noqa: E402
                    spread_for, strike_for_delta)

BASE = dict(symbols=("BTC",), capital=600.0, min_iv_rank=0.60, short_delta=0.25,
            wing_width_deltas=0.07, defined_risk=True, both_sides=True,
            real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
            profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
            max_margin_util=0.50, max_leverage=3.0)


class SideFilterBacktest(Backtest):
    """Baseline engine + a momentum-conditional call-side skip.

    mode='ret5'  : skip the call leg when trailing 5d return <= thr
    mode='ret20' : skip the call leg when trailing 20d return >= thr
    Both use only past closes up to the entry bar — no lookahead.
    """

    def __init__(self, cfg: Config, mode: str, thr: float):
        super().__init__(cfg)
        self.mode, self.thr = mode, thr
        for s in self.data:
            d = self.data[s]
            d["ret5"] = d["close"].pct_change(5)
            d["ret20"] = d["close"].pct_change(20)

    def _skip_call(self, row) -> bool:
        if self.mode == "ret5":
            v = row["ret5"]
            return bool(pd.notna(v) and v <= self.thr)
        if self.mode == "ret20":
            v = row["ret20"]
            return bool(pd.notna(v) and v >= self.thr)
        return False

    # copied from engine.Backtest._try_open; the ONLY change is the `sides`
    # selection marked below. engine.py itself is untouched.
    def _try_open(self, sym: str, row: pd.Series, date: pd.Timestamp):
        cfg = self.cfg
        if sum(t.symbol == sym for t in self.open) >= cfg.max_concurrent_per_symbol:
            return
        thr = (cfg.min_iv_rank_by_symbol or {}).get(sym, cfg.min_iv_rank)
        if row["iv_rank"] < thr:
            return

        spot, iv = float(row["close"]), float(row["iv"])
        if cfg.real_expiries:
            ahead = (cfg.expiry_weekday - date.weekday()) % 7
            expiry = (date + pd.Timedelta(days=ahead)).normalize() + pd.Timedelta(hours=cfg.expiry_hour)
            dte_actual = (expiry - date).total_seconds() / 86400
            while dte_actual < cfg.min_dte:
                expiry += pd.Timedelta(days=7)
                dte_actual = (expiry - date).total_seconds() / 86400
            if dte_actual > cfg.max_dte:
                return
        else:
            dte_actual = float(cfg.dte)
            expiry = date + pd.Timedelta(days=cfg.dte)
        if any(t.symbol == sym and t.expiry == expiry for t in self.open):
            return
        T = dte_actual / 365.0
        cv = cfg.contract_value[sym]
        step = cfg.strike_step[sym]

        # ── modification: momentum-conditional call-side skip ──
        sides = ("C", "P") if cfg.both_sides else ("C",)
        if cfg.both_sides and self._skip_call(row):
            sides = ("P",)
        # ───────────────────────────────────────────────────────
        legs, credit_unit, wing_unit = [], 0.0, 0.0
        widths = []
        for side in sides:
            Ks = strike_for_delta(spot, iv, T, side, cfg.short_delta, step)
            short_px = bs(spot, Ks, iv, T, side)
            sp_s = spread_for(cfg, spot, Ks, iv, T, side)
            credit_unit += short_px * (1 - sp_s / 2)
            legs.append(Leg(side, Ks, True, 1.0, short_px))
            if cfg.defined_risk:
                Kl = strike_for_delta(spot, iv, T, side,
                                      max(cfg.short_delta - cfg.wing_width_deltas, 0.01), step)
                long_px = bs(spot, Kl, iv, T, side)
                sp_l = spread_for(cfg, spot, Kl, iv, T, side)
                wing_unit += long_px * (1 + sp_l / 2)
                legs.append(Leg(side, Kl, False, 1.0, long_px))
                widths.append(abs(Kl - Ks))

        net_credit_unit = credit_unit - wing_unit
        if net_credit_unit <= 0:
            return

        if cfg.defined_risk:
            max_loss_unit = max(widths) - net_credit_unit
            if max_loss_unit <= 0:
                return
        else:
            max_loss_unit = spot * iv * math.sqrt(T) * 3.0

        risk_budget = self.equity * cfg.risk_per_trade
        lots = risk_budget / (max_loss_unit * cv)
        lots = min(lots, cfg.max_leverage * self.equity / (spot * cv))

        m = 0.0
        for leg in legs:
            if leg.short:
                m += margin_req(leg.entry_px, spot, leg.K, leg.side, 1.0, cv)
        if cfg.defined_risk:
            m = min(m, max(widths) * cv)
        if m <= 0:
            return
        free = self.equity * cfg.max_margin_util - sum(t.margin for t in self.open)
        if free <= 0:
            return
        lots = min(lots, free / m)
        lots = math.floor(lots)
        if lots < 1:
            return

        for leg in legs:
            leg.lots = lots

        credit = net_credit_unit * lots * cv
        f = sum(fee(leg.entry_px, spot, lots, cv, cfg) for leg in legs)
        tr = Trade(
            symbol=sym, entry=date, expiry=expiry,
            legs=legs, credit=credit,
            max_loss=(max(widths) - net_credit_unit) * lots * cv if cfg.defined_risk else float("inf"),
            margin=m * lots, entry_iv=iv, entry_iv_rank=float(row["iv_rank"]),
            entry_spot=spot, fees=f,
        )
        self.equity -= f
        self.open.append(tr)


WINDOWS = {"full": (None, None), "oos24": ("2024-01-01", None), "y2026": ("2026-01-01", None)}


def evaluate(make_bt):
    out = {}
    for label, (s, e) in WINDOWS.items():
        bt = make_bt()
        bt.run(s, e)
        m = bt.metrics()
        out[label] = m
    return out


def fmt(m):
    return (f"n={m['trades']:>3} win={m['win_rate']:.1%} PF={m['profit_factor']:.2f} "
            f"CAGR={m['cagr']:+.2%} DD={m['max_dd']:.2%} Sh={m['sharpe']:.2f} "
            f"eq=${m['final_equity']:.2f}")


def main():
    variants = [("baseline", None, None)]
    variants += [("A_ret5", "ret5", t) for t in (-0.04, -0.06, -0.08)]
    variants += [("B_ret20", "ret20", t) for t in (0.10, 0.15, 0.20)]

    for name, mode, thr in variants:
        if mode is None:
            res = evaluate(lambda: Backtest(Config(**BASE)))
            tag = name
        else:
            res = evaluate(lambda m=mode, t=thr: SideFilterBacktest(Config(**BASE), m, t))
            tag = f"{name} thr={thr:+.2f}"
        print(f"\n=== {tag} ===")
        for label in WINDOWS:
            print(f"  {label:<6} {fmt(res[label])}")


if __name__ == "__main__":
    main()
