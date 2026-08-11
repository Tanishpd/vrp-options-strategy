"""Tenor avenue — shared library.

MonthlyBacktest : real MONTHLY expiries (last Friday of the month, Deribit/Delta
                  convention, 12:00 UTC settlement). Subclasses engine.Backtest
                  and overrides only the expiry-selection logic inside _try_open;
                  every pricing/fee/margin/sizing line is copied verbatim from
                  research/engine.py (which must not be edited).

MixedBacktest   : monthly core book (iv_rank >= cfg.min_iv_rank) plus an extra
                  WEEKLY book that only trades when iv_rank >= weekly_min_rank.

Helpers: metrics_line, fee_share, bootstrap_ci.
"""
from __future__ import annotations

import calendar
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Backtest, Config, Leg, Trade, bs, fee, margin_req, spread_for, strike_for_delta


# ── monthly expiry calendar ────────────────────────────────────────────────
def last_friday(year: int, month: int, hour: int) -> pd.Timestamp:
    """Last Friday of (year, month) at `hour`:00 UTC — Deribit/Delta monthly."""
    d = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1], tz="UTC")
    d -= pd.Timedelta(days=(d.weekday() - 4) % 7)
    return d.normalize() + pd.Timedelta(hours=hour)


def monthly_expiry_for(date: pd.Timestamp, min_dte: float, max_dte: float,
                       hour: int) -> tuple[pd.Timestamp | None, float]:
    """Nearest monthly expiry at least min_dte away; None if it exceeds max_dte."""
    exp = last_friday(date.year, date.month, hour)
    dte = (exp - date).total_seconds() / 86400
    if dte < min_dte:
        y, m = (date.year + 1, 1) if date.month == 12 else (date.year, date.month + 1)
        exp = last_friday(y, m, hour)
        dte = (exp - date).total_seconds() / 86400
    if dte > max_dte:
        return None, dte
    return exp, dte


# ── generic opener: identical to engine._try_open except expiry is passed in ──
def open_with_expiry(bt: Backtest, sym: str, row: pd.Series, date: pd.Timestamp,
                     expiry: pd.Timestamp, dte_actual: float, book: str,
                     max_concurrent_this_book: int):
    """Copy of engine.Backtest._try_open with the expiry-selection block replaced
    by an explicit (expiry, dte) argument and per-book concurrency."""
    cfg = bt.cfg
    if sum(getattr(t, "book", "core") == book and t.symbol == sym for t in bt.open) \
            >= max_concurrent_this_book:
        return

    spot, iv = float(row["close"]), float(row["iv"])
    # distinct expiries only — same rule as the engine
    if any(t.symbol == sym and t.expiry == expiry for t in bt.open):
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
        return

    if cfg.defined_risk:
        max_loss_unit = max(widths) - net_credit_unit
        if max_loss_unit <= 0:
            return
    else:
        max_loss_unit = spot * iv * math.sqrt(T) * 3.0

    risk_budget = bt.equity * cfg.risk_per_trade
    lots = risk_budget / (max_loss_unit * cv)
    lots = min(lots, cfg.max_leverage * bt.equity / (spot * cv))

    m = 0.0
    for leg in legs:
        if leg.short:
            m += margin_req(leg.entry_px, spot, leg.K, leg.side, 1.0, cv)
    if cfg.defined_risk:
        m = min(m, max(widths) * cv)
    if m <= 0:
        return
    free = bt.equity * cfg.max_margin_util - sum(t.margin for t in bt.open)
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
    tr.book = book
    bt.equity -= f
    bt.open.append(tr)


# ── monthly backtest ───────────────────────────────────────────────────────
class MonthlyBacktest(Backtest):
    """Sells on real monthly expiries (last Friday, 12:00 UTC)."""

    def _try_open(self, sym: str, row: pd.Series, date: pd.Timestamp):
        cfg = self.cfg
        thr = (cfg.min_iv_rank_by_symbol or {}).get(sym, cfg.min_iv_rank)
        if row["iv_rank"] < thr:
            return
        expiry, dte = monthly_expiry_for(date, cfg.min_dte, cfg.max_dte, cfg.expiry_hour)
        if expiry is None:
            return
        open_with_expiry(self, sym, row, date, expiry, dte, "monthly",
                         cfg.max_concurrent_per_symbol)


# ── mixed book: monthly core + weekly overlay at high IV rank ──────────────
class MixedBacktest(Backtest):
    """Monthly core (iv_rank >= cfg.min_iv_rank) + weekly overlay traded only
    when iv_rank >= weekly_min_rank. One position per book. Weekly book uses
    baseline weekly calendar (Friday 12:00 UTC, weekly_min_dte..weekly_max_dte)."""

    def __init__(self, cfg: Config, weekly_min_rank: float = 0.80,
                 weekly_min_dte: float = 3, weekly_max_dte: float = 10):
        super().__init__(cfg)
        self.weekly_min_rank = weekly_min_rank
        self.weekly_min_dte = weekly_min_dte
        self.weekly_max_dte = weekly_max_dte

    def _try_open(self, sym: str, row: pd.Series, date: pd.Timestamp):
        cfg = self.cfg
        thr = (cfg.min_iv_rank_by_symbol or {}).get(sym, cfg.min_iv_rank)
        # monthly core
        if row["iv_rank"] >= thr:
            expiry, dte = monthly_expiry_for(date, cfg.min_dte, cfg.max_dte, cfg.expiry_hour)
            if expiry is not None:
                open_with_expiry(self, sym, row, date, expiry, dte, "monthly", 1)
        # weekly overlay only in the top IV regime
        if row["iv_rank"] >= self.weekly_min_rank:
            ahead = (cfg.expiry_weekday - date.weekday()) % 7
            expiry = (date + pd.Timedelta(days=ahead)).normalize() + pd.Timedelta(hours=cfg.expiry_hour)
            dte = (expiry - date).total_seconds() / 86400
            while dte < self.weekly_min_dte:
                expiry += pd.Timedelta(days=7)
                dte = (expiry - date).total_seconds() / 86400
            if dte <= self.weekly_max_dte:
                open_with_expiry(self, sym, row, date, expiry, dte, "weekly", 1)


# ── reporting helpers ──────────────────────────────────────────────────────
def fee_share(bt: Backtest) -> tuple[float, float, float]:
    """(fees, net pnl, fees/net_pnl) over closed trades."""
    fees = sum(t.fees for t in bt.closed)
    pnl = sum(t.pnl for t in bt.closed)
    return fees, pnl, (fees / pnl if pnl else float("nan"))


def bootstrap_ci(pnls, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """95% CI on mean P&L/trade by iid bootstrap."""
    rng = np.random.default_rng(seed)
    p = np.asarray(pnls, dtype=float)
    means = rng.choice(p, size=(n_boot, len(p)), replace=True).mean(axis=1)
    return float(p.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metrics_line(tag: str, bt: Backtest) -> dict:
    m = bt.metrics()
    if m.get("trades", 0) == 0:
        print(f"{tag:<34} NO TRADES")
        return m
    print(f"{tag:<34} n={m['trades']:>3}  win={m['win_rate']:.1%}  PF={m['profit_factor']:.2f}  "
          f"CAGR={m['cagr']:+.2%}  maxDD={m['max_dd']:.2%}  Sharpe={m['sharpe']:.2f}  "
          f"final=${m['final_equity']:.2f}")
    return m


BASELINE = dict(symbols=("BTC",), capital=600.0, min_iv_rank=0.60, short_delta=0.25,
                wing_width_deltas=0.07, defined_risk=True, both_sides=True,
                real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
                profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
                max_margin_util=0.50, max_leverage=3.0)
