"""
AVENUE: Add an ETH sleeve alongside the validated BTC iron-condor seller.

ETH VRP is only reliably positive at high IV rank (>= ~0.8 per research/vrp.py),
so ETH gets its own, stricter entry threshold via min_iv_rank_by_symbol while
BTC stays at 0.60. This script:

  1. Reproduces the BTC-only baseline.
  2. Runs the combined BTC+ETH portfolio at ETH thresholds 0.65..0.90
     (0.70/0.75/0.80/0.85 are the requested grid; 0.65 and 0.90 are plateau
     neighbours).
  3. Runs ETH-only sleeves at each threshold (standalone economics).
  4. Instruments entry sizing to detect margin contention (entries blocked or
     shrunk by the shared max_margin_util=0.50 budget) and records daily
     margin utilization.
  5. Reports entry-timing overlap and monthly sleeve P&L correlation.

Run:  .venv/bin/python research/improve/eth_sleeve.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import (Backtest, Config, Leg, Trade, bs, fee, margin_req,
                    spread_for, strike_for_delta)

BASE = dict(capital=600.0, min_iv_rank=0.60, short_delta=0.25,
            wing_width_deltas=0.07, defined_risk=True, both_sides=True,
            real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
            profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
            max_margin_util=0.50, max_leverage=3.0)

PERIODS = [("full", None, None),
           ("oos24", "2024-01-01", None),
           ("y2026", "2026-01-01", None)]


class InstrumentedBacktest(Backtest):
    """Exact copy of Backtest._try_open with sizing-cap logging added, plus a
    run() that records daily margin utilization. No behaviour change."""

    def __init__(self, cfg, risk_by_symbol=None):
        super().__init__(cfg)
        self.entry_log = []      # every signal day: what capped/blocked sizing
        self.util_curve = []     # (date, margin_used / equity)
        # optional per-symbol risk_per_trade override; empty dict = engine default
        self.risk_by_symbol = risk_by_symbol or {}

    def _try_open(self, sym, row, date):
        cfg = self.cfg
        if sum(t.symbol == sym for t in self.open) >= cfg.max_concurrent_per_symbol:
            return
        thr = (cfg.min_iv_rank_by_symbol or {}).get(sym, cfg.min_iv_rank)
        if row["iv_rank"] < thr:
            return

        other_open = any(t.symbol != sym for t in self.open)
        util_before = sum(t.margin for t in self.open) / self.equity

        spot, iv = float(row["close"]), float(row["iv"])
        if cfg.real_expiries:
            ahead = (cfg.expiry_weekday - date.weekday()) % 7
            expiry = (date + pd.Timedelta(days=ahead)).normalize() + pd.Timedelta(hours=cfg.expiry_hour)
            dte_actual = (expiry - date).total_seconds() / 86400
            while dte_actual < cfg.min_dte:
                expiry += pd.Timedelta(days=7)
                dte_actual = (expiry - date).total_seconds() / 86400
            if dte_actual > cfg.max_dte:
                self._log(date, sym, other_open, util_before, "no_expiry", 0, 0)
                return
        else:
            dte_actual = float(cfg.dte)
            expiry = date + pd.Timedelta(days=cfg.dte)
        if any(t.symbol == sym and t.expiry == expiry for t in self.open):
            return
        T = dte_actual / 365.0
        cv = cfg.contract_value[sym]
        step = cfg.strike_step[sym]

        sides = ("C", "P") if cfg.both_sides else ("C",)
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
            self._log(date, sym, other_open, util_before, "no_credit", 0, 0)
            return

        if cfg.defined_risk:
            max_loss_unit = max(widths) - net_credit_unit
            if max_loss_unit <= 0:
                return
        else:
            max_loss_unit = spot * iv * math.sqrt(T) * 3.0

        risk_budget = self.equity * self.risk_by_symbol.get(sym, cfg.risk_per_trade)
        lots = risk_budget / (max_loss_unit * cv)
        lots_risk = lots
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
            self._log(date, sym, other_open, util_before, "blocked_margin",
                      lots_risk, 0)
            return
        lots_pre_margin = lots
        lots = min(lots, free / m)
        margin_shrunk = lots < lots_pre_margin - 1e-9
        lots = math.floor(lots)
        if lots < 1:
            self._log(date, sym, other_open, util_before,
                      "blocked_margin_lots0" if margin_shrunk else "blocked_size_lots0",
                      lots_risk, 0)
            return

        self._log(date, sym, other_open, util_before,
                  "shrunk_by_margin" if lots < math.floor(lots_pre_margin) else "opened",
                  lots_risk, lots)

        for leg in legs:
            leg.lots = lots

        credit = net_credit_unit * lots * cv
        f = sum(fee(leg.entry_px, spot, lots, cv, cfg) for leg in legs)
        tr = Trade(symbol=sym, entry=date, expiry=expiry, legs=legs, credit=credit,
                   max_loss=(max(widths) - net_credit_unit) * lots * cv if cfg.defined_risk else float("inf"),
                   margin=m * lots, entry_iv=iv, entry_iv_rank=float(row["iv_rank"]),
                   entry_spot=spot, fees=f)
        self.equity -= f
        self.open.append(tr)

    def _log(self, date, sym, other_open, util_before, outcome, lots_risk, lots_final):
        self.entry_log.append(dict(date=date, sym=sym, other_symbol_open=other_open,
                                   util_before=util_before, outcome=outcome,
                                   lots_risk=lots_risk, lots_final=lots_final))

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
            self.util_curve.append((date, sum(t.margin for t in self.open) / self.equity))
        for tr in list(self.open):
            self.open.remove(tr)
        return self


def mline(bt):
    m = bt.metrics()
    if m.get("trades", 0) == 0:
        return "n=0 (no trades)"
    return (f"n={m['trades']:>3}  win {m['win_rate']:.1%}  PF {m['profit_factor']:.2f}  "
            f"CAGR {m['cagr']:+.2%}  maxDD {m['max_dd']:.2%}  Sharpe {m['sharpe']:.2f}  "
            f"final ${m['final_equity']:.2f}")


def run_variant(name, cfg_kwargs, risk_by_symbol=None):
    print(f"\n### {name}")
    bts = {}
    for pname, s, e in PERIODS:
        bt = InstrumentedBacktest(Config(**cfg_kwargs), risk_by_symbol).run(s, e)
        bts[pname] = bt
        print(f"  {pname:<6} {mline(bt)}")
    return bts


def dd_episode(bt, label):
    eq = pd.Series(dict(bt.curve)).sort_index()
    dd = (eq - eq.cummax()) / eq.cummax()
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    tr = pd.DataFrame([dict(sym=t.symbol, exit=t.exit_date, pnl=t.pnl)
                       for t in bt.closed])
    ep = tr[(tr["exit"] >= peak) & (tr["exit"] <= trough)]
    by_sym = ep.groupby("sym")["pnl"].agg(["sum", "count"])
    print(f"    [{label}] maxDD {dd.min():.2%}: peak {peak.date()} -> trough {trough.date()}")
    print(f"    losses in episode by sleeve:\n{by_sym.to_string()}".replace("\n", "\n      "))


def sleeve_analysis(bt, thr):
    """Post-hoc diagnostics on a combined full-sample run."""
    tr = pd.DataFrame([dict(sym=t.symbol, entry=t.entry, exit=t.exit_date,
                            pnl=t.pnl, margin=t.margin, rank=t.entry_iv_rank)
                       for t in bt.closed])
    eth, btc = tr[tr.sym == "ETH"], tr[tr.sym == "BTC"]

    # entry-timing overlap: was a BTC trade live when each ETH trade opened?
    def btc_live(d):
        return ((btc.entry <= d) & (btc.exit >= d)).any()
    eth_when_btc_live = sum(btc_live(d) for d in eth.entry)

    # signal-level overlap on tradeable days
    db, de = bt.data["BTC"], bt.data["ETH"]
    common = db.index.intersection(de.index)
    b_sig = db.loc[common, "iv_rank"] >= 0.60
    e_sig = de.loc[common, "iv_rank"] >= thr
    both = int((b_sig & e_sig).sum())
    eth_only = int((e_sig & ~b_sig).sum())
    btc_only = int((b_sig & ~e_sig).sum())

    # monthly sleeve P&L correlation (zero-filled months, exit-date attribution)
    tr["ym"] = tr["exit"].dt.to_period("M")
    piv = tr.pivot_table(index="ym", columns="sym", values="pnl",
                         aggfunc="sum").fillna(0.0)
    corr = piv["BTC"].corr(piv["ETH"]) if {"BTC", "ETH"} <= set(piv.columns) and len(piv) > 2 else float("nan")
    active = tr.pivot_table(index="ym", columns="sym", values="pnl", aggfunc="sum")
    both_active = active.dropna()
    corr_active = (both_active["BTC"].corr(both_active["ETH"])
                   if len(both_active) > 2 else float("nan"))

    # margin contention
    log = pd.DataFrame(bt.entry_log)
    eth_log = log[log.sym == "ETH"]
    blocked = eth_log[eth_log.outcome.str.startswith("blocked_margin")]
    shrunk = eth_log[eth_log.outcome == "shrunk_by_margin"]
    btc_log = log[log.sym == "BTC"]
    btc_blocked = btc_log[btc_log.outcome.str.startswith("blocked_margin")]
    util = pd.Series(dict(bt.util_curve))
    util_pos = util[util > 0]

    print(f"    ETH trades {len(eth)} (P&L ${eth.pnl.sum():+.2f}, win {(eth.pnl>0).mean():.1%}), "
          f"BTC trades {len(btc)} (P&L ${btc.pnl.sum():+.2f})")
    print(f"    ETH entries with a BTC position live: {eth_when_btc_live}/{len(eth)}")
    print(f"    signal days  both:{both}  ETH-only:{eth_only}  BTC-only:{btc_only}")
    print(f"    monthly P&L corr (zero-filled): {corr:+.2f}   "
          f"(both-active months only, n={len(both_active)}): {corr_active:+.2f}")
    print(f"    ETH sizing outcomes: {eth_log.outcome.value_counts().to_dict()}")
    print(f"    ETH entries blocked by margin: {len(blocked)}, shrunk by margin: {len(shrunk)}")
    print(f"    BTC entries blocked by margin: {len(btc_blocked)}")
    print(f"    margin util: median(all days) {util.median():.1%}, "
          f"median(days with positions) {util_pos.median():.1%}, max {util.max():.1%}")
    if len(eth):
        print(f"    ETH margin/trade median ${eth.margin.median():.0f} "
              f"vs BTC ${btc.margin.median():.0f}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    print("=" * 90)
    print("BASELINE: BTC-only")
    print("=" * 90)
    run_variant("BTC-only baseline", dict(symbols=("BTC",), **BASE))

    print("\n" + "=" * 90)
    print("COMBINED BTC+ETH  (BTC thr 0.60, ETH thr varies)")
    print("=" * 90)
    combined = {}
    for thr in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        cfg = dict(symbols=("BTC", "ETH"), min_iv_rank_by_symbol={"ETH": thr}, **BASE)
        combined[thr] = run_variant(f"BTC 0.60 + ETH {thr:.2f}", cfg)
        sleeve_analysis(combined[thr]["full"], thr)

    print("\n" + "=" * 90)
    print("ETH-ONLY sleeve economics (capital $600, same structure)")
    print("=" * 90)
    for thr in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        cfg = dict(symbols=("ETH",), **{**BASE, "min_iv_rank": thr})
        run_variant(f"ETH-only thr {thr:.2f}", cfg)

    print("\n" + "=" * 90)
    print("DRAWDOWN EPISODES (full sample)")
    print("=" * 90)
    for thr in (0.75, 0.85):
        dd_episode(combined[thr]["full"], f"combined ETH thr {thr:.2f}")

    # ── risk-normalized comparison ──────────────────────────────────────
    # The naive combined book doubles trade count, so its maxDD nearly doubles.
    # Fair risk-adjusted test: shrink sizing so the combined book's maxDD is in
    # the baseline's neighbourhood, then compare CAGR/Sharpe at equal pain.
    print("\n" + "=" * 90)
    print("RISK-NORMALIZED COMBINED (target maxDD ~ baseline -9.4%)")
    print("=" * 90)
    for thr in (0.70, 0.75, 0.80, 0.85, 0.90):
        cfg = dict(symbols=("BTC", "ETH"), min_iv_rank_by_symbol={"ETH": thr},
                   **{**BASE, "risk_per_trade": 0.025})
        run_variant(f"global risk 0.025, ETH thr {thr:.2f}", cfg)
    for thr in (0.70, 0.75, 0.80, 0.85, 0.90):
        cfg = dict(symbols=("BTC", "ETH"), min_iv_rank_by_symbol={"ETH": thr}, **BASE)
        bts = run_variant(f"BTC risk 0.04 + ETH risk 0.02, ETH thr {thr:.2f}",
                          cfg, risk_by_symbol={"ETH": 0.02})
        if thr in (0.75, 0.90):
            dd_episode(bts["full"], f"BTC 0.04/ETH 0.02 thr {thr:.2f}")
            sleeve_analysis(bts["full"], thr)
