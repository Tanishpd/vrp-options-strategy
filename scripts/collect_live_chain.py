"""
Collect REAL option-chain snapshots from Delta Exchange India.

No synthetic values anywhere: every row is an actual quoted market state —
best bid/ask (price and size), mark, bid/ask/mark IV, open interest, volume —
for every listed BTC and ETH option, plus the spot index.

Snapshots append to data/live_chain/chain_YYYY-MM-DD.csv (one file per UTC
day). Over weeks this becomes the real dataset that replaces the modeled
premiums in the historical backtest: real entry credits, real exit costs,
real spreads, real term structure and skew.

Usage:
    .venv/bin/python scripts/collect_live_chain.py            # one snapshot
    .venv/bin/python scripts/collect_live_chain.py --loop 900 # every 15 min

Public endpoints only — no API keys, read-only, places no orders.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "live_chain"
BASE = "https://api.india.delta.exchange"
UNDERLYINGS = ("BTC", "ETH")

COLUMNS = ["ts_utc", "underlying", "symbol", "side", "strike", "expiry_utc",
           "dte_days", "spot", "best_bid", "best_ask", "bid_size", "ask_size",
           "spread_pct", "mark_price", "mark_iv", "bid_iv", "ask_iv",
           "oi_contracts", "volume_24h", "turnover_usd_24h"]


def snapshot() -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    rows = []
    for ct in ("call_options", "put_options"):
        r = requests.get(f"{BASE}/v2/tickers",
                         params={"contract_types": ct}, timeout=30)
        r.raise_for_status()
        for t in r.json().get("result", []):
            if t.get("underlying_asset_symbol") not in UNDERLYINGS:
                continue
            q = t.get("quotes") or {}

            def f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None

            try:  # symbol format: C-BTC-64000-150826
                expiry = datetime.strptime(t["symbol"].split("-")[-1], "%d%m%y") \
                    .replace(hour=12, tzinfo=timezone.utc)
            except ValueError:
                continue
            bb, ba = f(q.get("best_bid")), f(q.get("best_ask"))
            mid = (bb + ba) / 2 if bb and ba else None
            rows.append({
                "ts_utc": now.isoformat(timespec="seconds"),
                "underlying": t["underlying_asset_symbol"],
                "symbol": t["symbol"],
                "side": "C" if ct == "call_options" else "P",
                "strike": f(t.get("strike_price")),
                "expiry_utc": expiry.isoformat(timespec="seconds"),
                "dte_days": round((expiry - now).total_seconds() / 86400, 3),
                "spot": f(t.get("spot_price")),
                "best_bid": bb, "best_ask": ba,
                "bid_size": f(q.get("bid_size")), "ask_size": f(q.get("ask_size")),
                "spread_pct": round((ba - bb) / mid, 5) if mid else None,
                "mark_price": f(t.get("mark_price")),
                "mark_iv": f(q.get("mark_iv")),
                "bid_iv": f(q.get("bid_iv")), "ask_iv": f(q.get("ask_iv")),
                "oi_contracts": f(t.get("oi_contracts")),
                "volume_24h": f(t.get("volume")),
                "turnover_usd_24h": f(t.get("turnover_usd")),
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def collect_once() -> str:
    df = snapshot()
    if df.empty:
        raise RuntimeError("empty chain from Delta — refusing to write")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"chain_{datetime.now(timezone.utc):%Y-%m-%d}.csv"
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    quoted = df.dropna(subset=["best_bid", "best_ask"])
    print(f"{datetime.now(timezone.utc):%H:%M:%S}Z  {len(df)} contracts "
          f"({len(quoted)} two-sided) -> {path.name}", flush=True)
    return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--loop", type=int, metavar="SECONDS",
                    help="snapshot repeatedly at this interval (min 60)")
    args = ap.parse_args()
    if args.loop:
        interval = max(args.loop, 60)
        while True:
            try:
                collect_once()
            except Exception as e:      # keep collecting through outages
                print(f"snapshot failed: {e}", file=sys.stderr, flush=True)
            time.sleep(interval)
    else:
        collect_once()


if __name__ == "__main__":
    main()
