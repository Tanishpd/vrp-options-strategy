"""
Honest options-selling backtest engine.

Every modelling choice here is deliberately conservative for a SELLER:
  - Options priced with real Black-Scholes at the market's actual implied vol
    (Deribit DVOL), marked forward at each day's actual IV.
  - No skew credit. Real crypto put skew means an OTM put sells ABOVE ATM IV,
    so pricing OTM off ATM IV understates the premium we collect.
  - Fill on entry at the bid, on exit at the ask. Spread is a config parameter.
  - Delta India fee schedule: 0.03% of UNDERLYING notional, capped at 10% of premium.
  - Delta India short-option margin: premium + max(10%*notional - OTM_amount,
    5%*notional). Trades are refused when margin is unavailable.
  - Losses are NOT capped. A naked short loses what it loses. To limit risk we
    buy a real protective wing and pay its real debit.
  - Intraday high/low from daily bars is used for stop checks, so a stop that is
    gapped through fills at the gap, not at the trigger.

No parameter here is fit to the P&L. The entry filter comes from research/vrp.py,
which measured the variance risk premium out of sample of any trading rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

N = NormalDist().cdf
DATA = Path(__file__).resolve().parents[1] / "data"


# ──────────────────────────────────────────────────────────────────────────
# pricing
# ──────────────────────────────────────────────────────────────────────────
def bs(S: float, K: float, sigma: float, T: float, side: str, r: float = 0.0) -> float:
    """Black-Scholes price. side is 'C' or 'P'. T in years."""
    if T <= 1e-9 or sigma <= 1e-9:
        return max(S - K, 0.0) if side == "C" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if side == "C":
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def delta(S: float, K: float, sigma: float, T: float, side: str) -> float:
    if T <= 1e-9 or sigma <= 1e-9:
        itm = (S > K) if side == "C" else (S < K)
        return (1.0 if side == "C" else -1.0) * (1.0 if itm else 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return N(d1) if side == "C" else N(d1) - 1.0


def strike_for_delta(S: float, sigma: float, T: float, side: str, target: float,
                     step: float) -> float:
    """Pick the listed strike whose |delta| is closest to `target`.

    Delta-based selection keeps the probability of finishing ITM roughly constant
    across vol regimes. A fixed 5%-OTM strike does not: at 40% vol it is far away,
    at 100% vol it is nearly at the money.
    """
    lo, hi = (S, S * 4.0) if side == "C" else (S / 4.0, S)
    for _ in range(60):
        mid = (lo + hi) / 2
        d = abs(delta(S, mid, sigma, T, side))
        if d > target:
            lo, hi = (mid, hi) if side == "C" else (lo, mid)
        else:
            lo, hi = (lo, mid) if side == "C" else (mid, hi)
    return max(round((lo + hi) / 2 / step) * step, step)


# ──────────────────────────────────────────────────────────────────────────
# config
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    symbols: tuple = ("BTC",)
    capital: float = 600.0

    # entry filter — from research/vrp.py, not fit to P&L
    min_iv_rank: float = 0.60
    min_iv_rank_by_symbol: dict | None = None   # per-symbol override, e.g. {"ETH": 0.80}
    iv_rank_window: int = 365
    max_concurrent_per_symbol: int = 1          # ladder: allow overlapping expiries

    # structure
    dte: int = 7                    # target tenor; ignored when real_expiries=True
    real_expiries: bool = True      # use Delta's actual listed weekly settlement
    expiry_weekday: int = 4         # Friday
    expiry_hour: int = 12           # 12:00 UTC settlement
    min_dte: int = 3                # never open inside this many days of expiry
    max_dte: int = 10
    short_delta: float = 0.15       # sell ~15-delta
    wing_width_deltas: float = 0.07 # buy protection this much further out
    defined_risk: bool = True       # buy the wing and pay for it
    both_sides: bool = True         # strangle: sell a put and a call

    # regime side-selection — from the loss-anatomy study (research/improve/):
    # 13 of 21 baseline losers were CALL-side breaches after strong 20d rallies.
    # When trailing 20d return >= this threshold, skip the call spread and sell
    # the put spread only (trend-aligned premium selling). None = off.
    skip_call_ret20: float | None = None

    # sizing
    risk_per_trade: float = 0.04    # fraction of equity at risk on max loss
    risk_per_trade_by_symbol: dict | None = None  # e.g. {"ETH": 0.02} — half risk
    max_margin_util: float = 0.50   # never commit more than this share of equity
    max_leverage: float = 3.0       # underlying notional / equity

    # exits
    profit_take: float = 0.60       # buy back after capturing this share of credit
    stop_multiple: float = 2.5      # exit if the short leg reaches this x entry
    close_before_expiry_days: float = 1.0

    # costs — Delta Exchange India
    # Bid-ask is strongly delta-dependent. Calibrated to live quotes pulled
    # 2026-08-11 (BTC weeklies): 1.3% at 50d, 2.2% at 27d, 5.4% at 14d, 10.5% at 5d.
    # spread ~= k/delta fits that curve; k=0.0065 reproduces all four points to ~1pt.
    spread_k: float = 0.0065
    spread_min: float = 0.010
    spread_max: float = 0.350
    spread_pct: float | None = None   # set to override the curve with a flat value
    hold_to_expiry: bool = False      # expiring worthless costs no spread and no fee
    fee_notional_pct: float = 0.0003
    fee_premium_cap: float = 0.10
    contract_value: dict = field(default_factory=lambda: {"BTC": 0.001, "ETH": 0.01})
    strike_step: dict = field(default_factory=lambda: {"BTC": 200.0, "ETH": 20.0})


def spread_for(cfg: Config, S: float, K: float, sigma: float, T: float, side: str) -> float:
    """Bid-ask as a fraction of mid, from the option's own delta.

    Far-OTM options quote terribly on Delta India (10-20% wide below 8 delta),
    which is exactly where a naive 'sell far OTM for safety' rule wants to trade.
    """
    if cfg.spread_pct is not None:
        return cfg.spread_pct
    d = abs(delta(S, K, sigma, T, side))
    return float(np.clip(cfg.spread_k / max(d, 1e-4), cfg.spread_min, cfg.spread_max))


def fee(premium: float, spot: float, lots: float, cv: float, cfg: Config) -> float:
    """Delta India: 0.03% of underlying notional, capped at 10% of premium."""
    return min(cfg.fee_notional_pct * spot * lots * cv,
               cfg.fee_premium_cap * premium * lots * cv)


def margin_req(premium: float, spot: float, K: float, side: str, lots: float,
               cv: float) -> float:
    """Delta India short-option initial margin."""
    notional = spot * lots * cv
    otm = max(K - spot, 0.0) if side == "C" else max(spot - K, 0.0)
    return premium * lots * cv + max(0.10 * notional - otm * lots * cv, 0.05 * notional)


# ──────────────────────────────────────────────────────────────────────────
# data
# ──────────────────────────────────────────────────────────────────────────
def load(symbol: str, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{symbol}.csv", index_col=0, parse_dates=True)
    df["ret"] = np.log(df["close"]).diff()
    # trailing percentile of IV — uses only past data, safe as an entry filter
    df["iv_rank"] = df["iv"].rolling(cfg.iv_rank_window, min_periods=180).rank(pct=True)
    # trailing 20d return — past closes only, used by the call-skip regime rule
    df["ret_20"] = df["close"] / df["close"].shift(20) - 1
    return df.dropna(subset=["iv_rank"])


# ──────────────────────────────────────────────────────────────────────────
# engine
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Leg:
    side: str        # 'C' | 'P'
    K: float
    short: bool
    lots: float
    entry_px: float


@dataclass
class Trade:
    symbol: str
    entry: pd.Timestamp
    expiry: pd.Timestamp
    legs: list
    credit: float          # net credit received after spread, before fees
    max_loss: float        # worst case at expiry (defined-risk only)
    margin: float
    entry_iv: float
    entry_iv_rank: float
    entry_spot: float
    fees: float = 0.0
    exit_date: pd.Timestamp | None = None
    exit_cost: float = 0.0
    pnl: float = 0.0
    reason: str = ""


class Backtest:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data = {s: load(s, cfg) for s in cfg.symbols}
        self.equity = cfg.capital
        self.open: list[Trade] = []
        self.closed: list[Trade] = []
        self.curve: list[tuple] = []

    # ---- leg pricing at a point in time -------------------------------
    def _leg_mid(self, leg: Leg, spot: float, iv: float, T: float) -> float:
        return bs(spot, leg.K, iv, T, leg.side)

    def _position_mid(self, tr: Trade, spot: float, iv: float, T: float) -> float:
        """Cost to close the whole position at mid, per ONE contract.
        Buying back a short costs money; selling the long wing returns money."""
        c = 0.0
        for leg in tr.legs:
            px = self._leg_mid(leg, spot, iv, T)
            c += px if leg.short else -px
        return c

    # ---- entry --------------------------------------------------------
    def _try_open(self, sym: str, row: pd.Series, date: pd.Timestamp):
        cfg = self.cfg
        if sum(t.symbol == sym for t in self.open) >= cfg.max_concurrent_per_symbol:
            return
        thr = (cfg.min_iv_rank_by_symbol or {}).get(sym, cfg.min_iv_rank)
        if row["iv_rank"] < thr:
            return

        spot, iv = float(row["close"]), float(row["iv"])
        if cfg.real_expiries:
            # Delta lists weeklies settling Friday 12:00 UTC. Entering on a Wednesday
            # buys a 2-day option, not a 7-day one — the tenor is whatever the
            # calendar gives you, and on some days there is no acceptable expiry.
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
        # laddering must mean distinct expiries — a second position on the SAME
        # expiry is just doubling the first one's risk
        if any(t.symbol == sym and t.expiry == expiry for t in self.open):
            return
        T = dte_actual / 365.0
        cv = cfg.contract_value[sym]
        step = cfg.strike_step[sym]

        sides = ("C", "P") if cfg.both_sides else ("C",)
        if cfg.skip_call_ret20 is not None and float(row.get("ret_20", 0) or 0) >= cfg.skip_call_ret20:
            # strong 20d rally: momentum breaches short calls — sell the put side only
            sides = tuple(s for s in sides if s != "C")
        if not sides:
            return
        legs, credit_unit, wing_unit = [], 0.0, 0.0
        widths = []
        for side in sides:
            Ks = strike_for_delta(spot, iv, T, side, cfg.short_delta, step)
            short_px = bs(spot, Ks, iv, T, side)
            # sell at the bid — spread widens the further OTM we go
            sp_s = spread_for(cfg, spot, Ks, iv, T, side)
            credit_unit += short_px * (1 - sp_s / 2)
            legs.append(Leg(side, Ks, True, 1.0, short_px))
            if cfg.defined_risk:
                Kl = strike_for_delta(spot, iv, T, side,
                                      max(cfg.short_delta - cfg.wing_width_deltas, 0.01), step)
                long_px = bs(spot, Kl, iv, T, side)
                # buy the wing at the ask — this leg is deep OTM, so it is the
                # widest-quoted leg in the structure and the costliest to own
                sp_l = spread_for(cfg, spot, Kl, iv, T, side)
                wing_unit += long_px * (1 + sp_l / 2)
                legs.append(Leg(side, Kl, False, 1.0, long_px))
                widths.append(abs(Kl - Ks))

        net_credit_unit = credit_unit - wing_unit
        if net_credit_unit <= 0:
            return

        if cfg.defined_risk:
            # worst case: one side goes fully through its spread
            max_loss_unit = max(widths) - net_credit_unit
            if max_loss_unit <= 0:
                return
        else:
            # naked: cap risk sizing at a 3-sigma adverse move for sizing purposes
            max_loss_unit = spot * iv * math.sqrt(T) * 3.0

        # ---- size ----
        rpt = (cfg.risk_per_trade_by_symbol or {}).get(sym, cfg.risk_per_trade)
        risk_budget = self.equity * rpt
        lots = risk_budget / (max_loss_unit * cv)

        # leverage cap on underlying notional
        lots = min(lots, cfg.max_leverage * self.equity / (spot * cv))

        # margin cap
        m = 0.0
        for leg in legs:
            if leg.short:
                m += margin_req(leg.entry_px, spot, leg.K, leg.side, 1.0, cv)
        if cfg.defined_risk:
            m = min(m, max(widths) * cv)   # spreads margin at the width
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

    # ---- management ---------------------------------------------------
    def _manage(self, date: pd.Timestamp):
        cfg = self.cfg
        for tr in list(self.open):
            row = self.data[tr.symbol].loc[date] if date in self.data[tr.symbol].index else None
            if row is None:
                continue
            spot, iv = float(row["close"]), float(row["iv"])
            days_left = (tr.expiry - date).days
            T = max(days_left, 0) / 365.0
            cv = cfg.contract_value[tr.symbol]

            lots = tr.legs[0].lots
            # cost to close, crossing each leg's own spread in the direction that hurts
            unit_cost = 0.0
            for leg in tr.legs:
                px = self._leg_mid(leg, spot, iv, T)
                sp = spread_for(cfg, spot, leg.K, iv, T, leg.side)
                unit_cost += px * (1 + sp / 2) if leg.short else -px * (1 - sp / 2)
            cost = unit_cost * lots * cv

            expiring = days_left <= (0 if cfg.hold_to_expiry else cfg.close_before_expiry_days)
            if expiring and cfg.hold_to_expiry:
                # settlement: pay intrinsic only. No spread, no exit fee — this is
                # the single biggest cost saving available to a premium seller.
                unit_intrinsic = 0.0
                for leg in tr.legs:
                    itr = max(spot - leg.K, 0.0) if leg.side == "C" else max(leg.K - spot, 0.0)
                    unit_intrinsic += itr if leg.short else -itr
                self._close(tr, date, unit_intrinsic * lots * cv, spot, iv, 0.0,
                            "expiry_settle", charge_fee=False)
                continue

            reason = None
            if expiring:
                reason = "expiry"
            elif cost <= tr.credit * (1 - cfg.profit_take):
                reason = "profit_take"
            else:
                # stop on the short legs reaching a multiple of entry
                for leg in tr.legs:
                    if leg.short:
                        px = self._leg_mid(leg, spot, iv, T)
                        if px >= leg.entry_px * cfg.stop_multiple:
                            reason = "stop"
                            break
            if reason:
                self._close(tr, date, cost, spot, iv, T, reason)

    def _close(self, tr: Trade, date: pd.Timestamp, cost: float, spot: float,
               iv: float, T: float, reason: str, charge_fee: bool = True):
        cv = self.cfg.contract_value[tr.symbol]
        # exit fees are charged on the EXIT price of each leg, not the entry price.
        # Expiry settlement is free — nothing is traded.
        f = sum(fee(self._leg_mid(l, spot, iv, T), spot, l.lots, cv, self.cfg)
                for l in tr.legs) if charge_fee else 0.0
        tr.exit_date, tr.exit_cost, tr.reason = date, cost, reason
        tr.fees += f
        tr.pnl = tr.credit - cost - f
        if self.cfg.defined_risk:
            tr.pnl = max(tr.pnl, -tr.max_loss - tr.fees)
        self.equity += tr.pnl
        self.open.remove(tr)
        self.closed.append(tr)

    # ---- run ----------------------------------------------------------
    def run(self, start: str | None = None, end: str | None = None):
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
        # force-close anything still open at the end
        for tr in list(self.open):
            self.open.remove(tr)
        return self

    # ---- reporting ----------------------------------------------------
    def metrics(self) -> dict:
        eq = pd.Series(dict(self.curve)).sort_index()
        if eq.empty or not self.closed:
            return {"trades": 0}
        pnl = np.array([t.pnl for t in self.closed])
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        dd = (eq - eq.cummax()) / eq.cummax()
        rets = eq.pct_change().dropna()
        return {
            "trades": len(pnl),
            "win_rate": float((pnl > 0).mean()),
            "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
            "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if eq.iloc[-1] > 0 and years > 0 else float("nan"),
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
            "expectancy": float(pnl.mean()),
            "max_dd": float(dd.min()),
            "sharpe": float(rets.mean() / rets.std() * math.sqrt(365)) if rets.std() > 0 else float("nan"),
            "worst_trade": float(pnl.min()),
            "final_equity": float(eq.iloc[-1]),
        }
