"""
Shared, validated v2 decision logic for LIVE trading — used identically by
the read-only simulator (scripts/paper_trade_live.py) and the real
testnet order engine (scripts/live_trade_testnet.py), so the two can never
drift into deciding differently. This module never places an order and
never touches an API key; it only reads Delta's public market data and
decides WHAT a v2 position should look like.

Every price here is real:
  - Spot: Delta India's live index/mark price (public endpoint)
  - Strike selection: the REAL listed contract whose live greeks.delta is
    closest to the target (Delta publishes delta per contract — no
    Black-Scholes needed to pick a strike, unlike the historical backtests)
  - IV-rank filter: Deribit DVOL, refreshed daily into data/BTC.csv and
    data/ETH.csv — the exact definition validated in scripts/real_backtest.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from vrp.engine import Config as EngineConfig
from vrp.engine import fee, load, margin_req  # noqa: F401  (re-exported for callers)

B = "https://api.india.delta.exchange"  # PUBLIC market data only. No key. No orders.

CV = {"BTC": 0.001, "ETH": 0.01}
MIN_IV_RANK = {"BTC": 0.60, "ETH": 0.75}
SHORT_DELTA = 0.25
WING_DELTA = 0.15                  # short_delta(0.25) - wing_width_deltas(0.10), matches v2
RISK_PER_TRADE = {"BTC": 0.04, "ETH": 0.02}
MAX_LEVERAGE = 3.0
MAX_MARGIN_UTIL = 0.50
MAX_CONCURRENT = 2
MIN_DTE, MAX_DTE = 3, 10
SKIP_CALL_RET20 = 0.20
FEE_CFG = EngineConfig()


def get_tickers(contract_types: str) -> list[dict]:
    r = requests.get(f"{B}/v2/tickers", params={"contract_types": contract_types}, timeout=20)
    r.raise_for_status()
    return r.json().get("result", [])


def get_spot(sym: str) -> float:
    r = requests.get(f"{B}/v2/tickers/{sym}USD", timeout=15)
    r.raise_for_status()
    return float(r.json()["result"]["mark_price"])


def parse_expiry(symbol: str) -> datetime:
    # e.g. C-BTC-67000-140826 -> 14 Aug 2026, weekly settlement 12:00 UTC
    return datetime.strptime(symbol.split("-")[-1], "%d%m%y").replace(hour=12, tzinfo=timezone.utc)


def live_chain(sym: str) -> pd.DataFrame:
    rows = []
    for ct, side in (("call_options", "C"), ("put_options", "P")):
        for t in get_tickers(ct):
            if t.get("underlying_asset_symbol") != sym:
                continue
            q = t.get("quotes") or {}
            g = t.get("greeks") or {}
            try:
                bb, ba = float(q.get("best_bid") or 0), float(q.get("best_ask") or 0)
                d = abs(float(g.get("delta") or 0))
            except (TypeError, ValueError):
                continue
            if bb <= 0 or ba <= 0 or d <= 0:
                continue
            try:
                expiry = parse_expiry(t["symbol"])
            except ValueError:
                continue
            rows.append(dict(symbol=t["symbol"], side=side, strike=float(t["strike_price"]),
                             expiry=expiry, best_bid=bb, best_ask=ba, delta=d,
                             oi=float(t.get("oi_contracts") or 0)))
    return pd.DataFrame(rows)


def refresh_iv_rank_data():
    """Append today's real Binance spot + real Deribit DVOL row if missing."""
    from fetch_data import dvol, spot  # scripts/fetch_data.py, added to sys.path above
    for sym, delivery in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        path = ROOT / "data" / f"{sym}.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        today = pd.Timestamp.now(tz="UTC").normalize()
        if df.index.max() >= today - pd.Timedelta(days=1):
            continue
        iv, px = dvol(sym), spot(delivery)
        merged = px.join(iv, how="inner").dropna()
        merged.to_csv(path)


def current_signal(sym: str) -> dict:
    """Real spot, real trailing-20d return, real IV rank — from data/*.csv."""
    d = load(sym, EngineConfig())
    last = d.iloc[-1]
    return dict(spot=float(last["close"]), iv_rank=float(last["iv_rank"]),
               ret_20=float(last.get("ret_20", 0) or 0), asof=d.index[-1])


def pick_expiry(chain: pd.DataFrame, now: datetime) -> datetime | None:
    """Nearest listed expiry with 3-10 real days to expiry. Delta lists DAILY
    expiries too -- this explicitly filters to the intended weekly band."""
    for exp in sorted(chain["expiry"].unique()):
        dte = (exp - now).total_seconds() / 86400
        if MIN_DTE <= dte <= MAX_DTE:
            return exp
    return None


def pick_contract(chain: pd.DataFrame, expiry: datetime, side: str, target_delta: float) -> dict | None:
    sub = chain[(chain["expiry"] == expiry) & (chain["side"] == side)]
    if sub.empty:
        return None
    sub = sub.assign(dist=(sub["delta"] - target_delta).abs()).sort_values("dist")
    return sub.iloc[0].to_dict()


def plan_entry(sym: str, open_positions: list[dict]) -> dict | None:
    """Decide what a v2 entry for `sym` should look like RIGHT NOW, from real
    live data. Returns None if the strategy says don't trade. Never places
    anything -- this is pure decision logic, shared by both live engines."""
    if sum(p["sym"] == sym for p in open_positions) >= MAX_CONCURRENT:
        return None
    sig = current_signal(sym)
    if sig["iv_rank"] < MIN_IV_RANK[sym]:
        return {"skip": "iv_rank_below_threshold", "iv_rank": round(sig["iv_rank"], 3),
                "threshold": MIN_IV_RANK[sym]}

    chain = live_chain(sym)
    if chain.empty:
        return {"skip": "empty_live_chain"}
    now = datetime.now(timezone.utc)
    expiry = pick_expiry(chain, now)
    if expiry is None:
        return {"skip": "no_expiry_in_dte_band"}
    if any(p["sym"] == sym and p["expiry"] == expiry.isoformat() for p in open_positions):
        return {"skip": "already_have_this_expiry"}

    sides = ("C", "P")
    if sig["ret_20"] >= SKIP_CALL_RET20:
        sides = ("P",)

    legs, credit_u, widths = [], 0.0, []
    for side in sides:
        short = pick_contract(chain, expiry, side, SHORT_DELTA)
        wing = pick_contract(chain, expiry, side, WING_DELTA)
        if not short or not wing or short["symbol"] == wing["symbol"]:
            return {"skip": "no_real_contract", "side": side}
        credit_u += short["best_bid"] - wing["best_ask"]
        widths.append(abs(wing["strike"] - short["strike"]))
        legs.append(dict(side=side, K=short["strike"], short=True,
                         symbol=short["symbol"], fill=short["best_bid"], delta=short["delta"]))
        legs.append(dict(side=side, K=wing["strike"], short=False,
                         symbol=wing["symbol"], fill=wing["best_ask"], delta=wing["delta"]))
    if credit_u <= 0:
        return {"skip": "non_positive_credit", "credit_unit": round(credit_u, 2)}
    max_loss_u = max(widths) - credit_u
    if max_loss_u <= 0:
        return {"skip": "credit_exceeds_width"}

    return dict(sym=sym, sides="".join(sides), expiry=expiry, legs=legs,
               credit_unit=credit_u, max_loss_unit=max_loss_u, widths=widths,
               spot=sig["spot"], iv_rank=sig["iv_rank"])


def size_plan(plan: dict, equity: float, committed_margin: float) -> tuple[int, float]:
    """Given a plan from plan_entry() and current account state, return
    (lots, margin_per_lot). lots may be 0 if sizing floors to nothing."""
    sym = plan["sym"]
    lots = (equity * RISK_PER_TRADE[sym]) / (plan["max_loss_unit"] * CV[sym])
    lots = min(lots, MAX_LEVERAGE * equity / (plan["spot"] * CV[sym]))
    m_unit = sum(margin_req(l["fill"], plan["spot"], l["K"], l["side"], 1.0, CV[sym])
                for l in plan["legs"] if l["short"])
    m_unit = min(m_unit, max(plan["widths"]) * CV[sym])
    free = equity * MAX_MARGIN_UTIL - committed_margin
    if free <= 0 or m_unit <= 0:
        return 0, m_unit
    import math
    return math.floor(min(lots, free / m_unit)), m_unit
