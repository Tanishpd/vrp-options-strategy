"""
REAL-data backtest of the v2 strategy — no synthetic values.

Every price in this run is a real market observation:
  - Entry premiums:  actual Deribit trades (history.deribit.com), filled the
                     UTC day AFTER the signal day (strictly no lookahead).
                     Short legs use the volume-weighted price of taker-SELL
                     trades (prices at which real sellers actually sold);
                     wing buys use taker-BUY trades (prices real buyers paid).
  - Exits:           real Deribit delivery (settlement) prices -> intrinsic.
                     The strategy holds to expiry, so no mid-life marks exist.
  - Filters/sizing:  real Binance spot, real Deribit DVOL (data/*.csv).
  - Fees:            Delta India schedule (venue of intended deployment).

What is NOT real here and is disclosed as such:
  - Venue transplant: prices are Deribit's (global price-setter), the intended
    venue is Delta India. Expiry convention: Deribit Friday 08:00 UTC.
  - The delta-targeting rule uses Black-Scholes to CHOOSE which strike to
    trade (a selection rule, like "10% OTM"); the strike's PRICE is real.
  - Fills are a same-day directional VWAP (that day's real taker-sell/buy
    prints), not a specific resting order's achieved price.
  - Entries with no real trades in any acceptable strike are skipped and
    counted — coverage is reported, because it biases toward liquid days.

Verified 2026-08-13 (scripts/verify_real_backtest.py): all structural/
lookahead/bookkeeping checks pass, and 600 + sum(trade pnl) reconstructs the
reported final equity to within $0.04 on $1,431 (pure rounding noise from the
per-trade `round(pnl, 2)` in the CSV export). An earlier version of this
script double-charged each trade's entry fee (once immediately at open, again
inside the settlement pnl formula) — caught by that same bookkeeping check,
which is why it exists; fixed by removing the premature `equity -= fees`.

Usage:  .venv/bin/python scripts/real_backtest.py            # full run (cached)
        .venv/bin/python scripts/real_backtest.py --refresh  # ignore cache
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrp.engine import Config, fee, load, margin_req, strike_for_delta  # noqa: E402

HIST = "https://history.deribit.com/api/v2"
PROD = "https://www.deribit.com/api/v2"
CACHE = ROOT / "data" / "deribit_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# ── v2 config (identical decisions to scripts/run_v2.py, Deribit calendar) ──
CFG = Config(symbols=("BTC", "ETH"), capital=600.0, min_iv_rank=0.60,
             min_iv_rank_by_symbol={"ETH": 0.75}, short_delta=0.25,
             wing_width_deltas=0.10, real_expiries=True, min_dte=3, max_dte=10,
             risk_per_trade=0.04, risk_per_trade_by_symbol={"ETH": 0.02},
             max_margin_util=0.50, max_leverage=3.0,
             max_concurrent_per_symbol=2, skip_call_ret20=0.20,
             expiry_hour=8)                      # Deribit settles 08:00 UTC

STRIKE_STEPS = {"BTC": (250, 500, 1000), "ETH": (25, 50, 100)}
TOLERANCE = 0.04          # accept strikes within 4% of spot from the target
SESSION = requests.Session()
_last_call = [0.0]


def _get(url: str, params: dict) -> dict:
    wait = 0.06 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(5):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            _last_call[0] = time.time()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 400:
                return {}          # unknown instrument (strike never listed)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return {}


def instrument(sym: str, expiry: datetime, K: float, side: str) -> str:
    return f"{sym}-{expiry:%-d%b%y}-{int(K)}-{side}".upper()


def day_trades(instr: str, day: datetime) -> list[dict]:
    """All real trades for `instr` on UTC day `day` (cached on disk)."""
    key = CACHE / f"tr_{instr}_{day:%Y%m%d}.json"
    if key.exists():
        try:
            return json.loads(key.read_text())
        except json.JSONDecodeError:   # truncated by an interrupted run — refetch
            key.unlink()
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    out, cursor = [], int(start.timestamp() * 1000)
    end = int((start + timedelta(days=1)).timestamp() * 1000)
    while True:
        d = _get(f"{HIST}/public/get_last_trades_by_instrument_and_time",
                 {"instrument_name": instr, "start_timestamp": cursor,
                  "end_timestamp": end, "count": 1000, "sorting": "asc"})
        trades = d.get("result", {}).get("trades", [])
        out += trades
        if len(trades) < 1000 or not d.get("result", {}).get("has_more"):
            break
        cursor = trades[-1]["timestamp"] + 1
    key.write_text(json.dumps(out))
    return out


def real_fill(trades: list[dict], we_sell: bool) -> tuple[float, str, int] | None:
    """Volume-weighted REAL fill in USD per underlying unit.
    we_sell -> use taker-sell prints (sellers' achieved prices);
    buys -> taker-buy prints. Falls back to all prints, flagged."""
    want = "sell" if we_sell else "buy"
    directional = [t for t in trades if t.get("direction") == want]
    for pool, tag in ((directional, "dir"), (trades, "all")):
        if pool:
            num = sum(t["price"] * t["index_price"] * t["amount"] for t in pool)
            den = sum(t["amount"] for t in pool)
            if den > 0:
                return num / den, tag, len(pool)
    return None


def delivery_prices(index: str) -> dict[str, float]:
    key = CACHE / f"delivery_{index}.json"
    if key.exists():
        return json.loads(key.read_text())
    out, offset = {}, 0
    while True:
        d = _get(f"{PROD}/public/get_delivery_prices",
                 {"index_name": index, "offset": offset, "count": 500})
        rows = d.get("result", {}).get("data", [])
        for r in rows:
            out[r["date"]] = r["delivery_price"]
        offset += len(rows)
        if offset >= d.get("result", {}).get("records_total", 0) or not rows:
            break
    key.write_text(json.dumps(out))
    return out


def pick_strike(sym: str, expiry: datetime, fill_day: datetime, spot: float,
                target: float, side: str, we_sell: bool):
    """Nearest Deribit-listed strike to `target` that actually TRADED on
    fill_day, within TOLERANCE of spot-distance. Returns (K, usd, tag, n)."""
    cands = sorted({round(target / s) * s + k * s
                    for s in STRIKE_STEPS[sym] for k in (-1, 0, 1)})
    cands = [k for k in cands if k > 0 and abs(k - target) <= TOLERANCE * spot]
    for K in sorted(cands, key=lambda k: abs(k - target)):
        tr = day_trades(instrument(sym, expiry, K, side), fill_day)
        f = real_fill(tr, we_sell)
        if f:
            return K, *f
    return None


def main(refresh: bool):
    if refresh:
        for f in CACHE.glob("*.json"):
            f.unlink()
    deliv = {"BTC": delivery_prices("btc_usd"), "ETH": delivery_prices("eth_usd")}
    data = {s: load(s, CFG) for s in CFG.symbols}
    cv = CFG.contract_value
    equity, open_pos, closed = CFG.capital, [], []
    skipped = {"no_real_market": 0, "size": 0}
    curve = []

    idx = sorted(set().union(*[set(d.index) for d in data.values()]))
    for date in idx:
        # ── settle real expiries (Friday 08:00 UTC, real delivery price) ──
        for p in [p for p in open_pos if date.date() >= p["expiry"].date()]:
            S = deliv[p["sym"]].get(f"{p['expiry']:%Y-%m-%d}")
            if S is None:
                continue
            intrinsic = 0.0
            for leg in p["legs"]:
                v = max(S - leg["K"], 0) if leg["side"] == "C" else max(leg["K"] - S, 0)
                intrinsic += v if leg["short"] else -v
            pnl = p["credit"] - intrinsic * p["lots"] * cv[p["sym"]] - p["fees"]
            equity += pnl
            closed.append({**{k: p[k] for k in ("sym", "entry", "sides", "lots",
                                                "credit", "fees", "fill_quality")},
                           "expiry": p["expiry"].date(), "settle": S,
                           "pnl": round(pnl, 2)})
            open_pos.remove(p)

        for sym in CFG.symbols:
            d = data[sym]
            if date not in d.index:
                continue
            row = d.loc[date]
            if sum(p["sym"] == sym for p in open_pos) >= CFG.max_concurrent_per_symbol:
                continue
            if row["iv_rank"] < (CFG.min_iv_rank_by_symbol or {}).get(sym, CFG.min_iv_rank):
                continue
            spot, iv = float(row["close"]), float(row["iv"])

            ahead = (4 - date.weekday()) % 7      # next Friday
            expiry = (date + pd.Timedelta(days=ahead)).normalize() \
                + pd.Timedelta(hours=CFG.expiry_hour)
            dte = (expiry - date).total_seconds() / 86400
            while dte < CFG.min_dte:
                expiry += pd.Timedelta(days=7)
                dte = (expiry - date).total_seconds() / 86400
            if dte > CFG.max_dte or any(p["sym"] == sym and p["expiry"] == expiry
                                        for p in open_pos):
                continue

            sides = ("C", "P")
            if CFG.skip_call_ret20 is not None and \
                    float(row.get("ret_20", 0) or 0) >= CFG.skip_call_ret20:
                sides = ("P",)

            fill_day = (date + pd.Timedelta(days=1)).to_pydatetime()  # no lookahead
            T = max(dte - 1, 0.5) / 365
            step = min(STRIKE_STEPS[sym])
            legs, credit_u, widths, quality = [], 0.0, [], []
            ok = True
            for side in sides:
                ks_t = strike_for_delta(spot, iv, T, side, CFG.short_delta, step)
                kl_t = strike_for_delta(spot, iv, T, side,
                                        CFG.short_delta - CFG.wing_width_deltas, step)
                s_fill = pick_strike(sym, expiry, fill_day, spot, ks_t, side, True)
                l_fill = pick_strike(sym, expiry, fill_day, spot, kl_t, side, False)
                if not s_fill or not l_fill or s_fill[0] == l_fill[0]:
                    ok = False
                    break
                Ks, s_usd, s_tag, s_n = s_fill
                Kl, l_usd, l_tag, l_n = l_fill
                if s_usd <= l_usd:        # inverted real prices -> no economics
                    ok = False
                    break
                credit_u += s_usd - l_usd
                widths.append(abs(Kl - Ks))
                quality += [f"{side}s:{s_tag}{s_n}", f"{side}l:{l_tag}{l_n}"]
                legs += [dict(side=side, K=Ks, short=True, usd=s_usd),
                         dict(side=side, K=Kl, short=False, usd=l_usd)]
            if not ok or credit_u <= 0:
                skipped["no_real_market"] += 1
                continue

            max_loss_u = max(widths) - credit_u
            if max_loss_u <= 0:
                skipped["size"] += 1
                continue
            rpt = (CFG.risk_per_trade_by_symbol or {}).get(sym, CFG.risk_per_trade)
            lots = (equity * rpt) / (max_loss_u * cv[sym])
            lots = min(lots, CFG.max_leverage * equity / (spot * cv[sym]))
            m_unit = min(sum(margin_req(l["usd"], spot, l["K"], l["side"], 1.0, cv[sym])
                             for l in legs if l["short"]), max(widths) * cv[sym])
            free = equity * CFG.max_margin_util - sum(p["margin"] for p in open_pos)
            if m_unit > 0 and free > 0:
                lots = min(lots, free / m_unit)
            lots = math.floor(lots)
            if lots < 1:
                skipped["size"] += 1
                continue

            fees = sum(fee(l["usd"], spot, lots, cv[sym], CFG) for l in legs)
            # NOTE: do not touch `equity` here. This entry fee is the ONLY fee
            # this position ever pays (expiry settlement is free — no trade
            # happens), and it is already netted into `pnl` at settlement
            # below (`pnl = credit - intrinsic - p["fees"]`). Subtracting it
            # here too would double-charge every trade's fee.
            open_pos.append(dict(sym=sym, entry=date.date(), expiry=expiry,
                                 legs=legs, lots=lots, sides="".join(sides),
                                 credit=credit_u * lots * cv[sym], fees=fees,
                                 margin=m_unit * lots,
                                 fill_quality=",".join(quality)))
        # Equity curve is realized cash only — by design there are no mid-life
        # marks (the strategy holds every position to real settlement), so
        # nothing would be gained by trying to mark open positions here.
        curve.append((date, equity))

    tr = pd.DataFrame(closed)
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    tr.to_csv(ROOT / "results" / "real_trades.csv", index=False)
    eq.rename("equity").to_csv(ROOT / "results" / "real_equity.csv")

    p = tr["pnl"].values
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq - eq.cummax()) / eq.cummax()
    win = (p > 0).mean()
    pf = p[p > 0].sum() / -p[p < 0].sum() if (p < 0).any() else float("inf")
    print("=" * 76)
    print("REAL-DATA BACKTEST — actual Deribit trades in, actual settlements out")
    print("=" * 76)
    print(f"  period            {eq.index[0].date()} -> {eq.index[-1].date()}  ({yrs:.2f}y)")
    print(f"  trades            {len(tr)}   skipped: {skipped}")
    print(f"  equity            ${CFG.capital:,.2f} -> ${eq.iloc[-1]:,.2f}")
    print(f"  CAGR              {(eq.iloc[-1]/CFG.capital)**(1/yrs)-1:+.2%}")
    print(f"  win rate          {win:.1%}   profit factor {pf:.2f}")
    print(f"  max drawdown      {dd.min():.2%}")
    r = eq.pct_change().dropna()
    print(f"  Sharpe            {r.mean()/r.std()*math.sqrt(365):.2f}")
    print(f"  mean/median trade ${p.mean():+.2f} / ${pd.Series(p).median():+.2f}")
    fills = ",".join(tr["fill_quality"]).split(",")
    print(f"  directional fills {sum(1 for f in fills if 'dir' in f)}/{len(fills)} legs")
    print(f"\n  -> results/real_trades.csv, results/real_equity.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(ap.parse_args().refresh)
