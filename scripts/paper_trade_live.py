"""
Live PAPER (read-only) trading engine for the validated v2 VRP strategy.

NEVER PLACES AN ORDER. No Delta API key is read, imported, or required by any
code path here or in vrp/live_signals.py — every endpoint hit is Delta's
PUBLIC market data API. This "fills" a decision from vrp.live_signals.plan_entry
at the real best_bid/best_ask observed at decision time, and settles positions
using the real spot/index price polled at/after the real expiry (Delta
publishes no dedicated settlement-price endpoint for expired contracts; this
is the one disclosed approximation in this file).

See scripts/live_trade_testnet.py for the version that places real (testnet)
orders instead of simulating a fill — the two share every decision (signal,
strike selection, sizing) via vrp/live_signals.py and differ ONLY in how a
leg gets "filled."

State persists to disk so a restart never loses an open position:
  data/live_paper/state.json       open positions
  data/live_paper/trade_log.jsonl  append-only OPEN/CLOSE/SKIP events
  data/live_paper/equity.csv       equity after each check

Usage:
  .venv/bin/python scripts/paper_trade_live.py --once        # single check
  .venv/bin/python scripts/paper_trade_live.py --loop 1800   # every 30 min
  .venv/bin/python scripts/paper_trade_live.py --status      # print summary
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrp.live_signals import CV, FEE_CFG, fee, get_spot  # noqa: E402
from vrp.live_signals import plan_entry, refresh_iv_rank_data, size_plan  # noqa: E402

SYMBOLS = ("BTC", "ETH")
STATE_DIR = ROOT / "data" / "live_paper"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "trade_log.jsonl"
EQUITY_FILE = STATE_DIR / "equity.csv"
CAPITAL = 600.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S UTC")
logging.Formatter.converter = time.gmtime
log = logging.getLogger("paper_trade_live")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"equity": CAPITAL, "open": []}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=1, default=str))


def log_event(event: str, **fields):
    rec = {"event": event, "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def record_equity(equity: float):
    row = pd.DataFrame([{"ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "equity": round(equity, 2)}])
    row.to_csv(EQUITY_FILE, mode="a", header=not EQUITY_FILE.exists(), index=False)


def try_open(state: dict, sym: str):
    plan = plan_entry(sym, state["open"])
    if plan is None:
        return
    if "skip" in plan:
        log_event("SKIP", sym=sym, **plan)
        return

    lots, m_unit = size_plan(plan, state["equity"], sum(p["margin"] for p in state["open"]))
    if lots < 1:
        log_event("SKIP", sym=sym, reason="sizes_to_zero_lots")
        return

    fees = sum(fee(l["fill"], plan["spot"], lots, CV[sym], FEE_CFG) for l in plan["legs"])

    pos = dict(sym=sym, entry_ts=datetime.now(timezone.utc).isoformat(),
              expiry=plan["expiry"].isoformat(), sides=plan["sides"], lots=lots,
              legs=plan["legs"], credit=round(plan["credit_unit"] * lots * CV[sym], 4),
              fees=round(fees, 4), margin=round(m_unit * lots, 2))
    state["open"].append(pos)
    log_event("OPEN", **pos, entry_spot=plan["spot"], entry_iv_rank=round(plan["iv_rank"], 3))
    log.info(f"OPEN {sym} {pos['sides']} lots={lots} credit=${pos['credit']:.2f} "
            f"fees=${pos['fees']:.2f} expiry={plan['expiry'].date()} "
            f"legs={[l['symbol'] for l in plan['legs']]}")


def check_settlements(state: dict):
    now = datetime.now(timezone.utc)
    still_open = []
    for p in state["open"]:
        expiry = datetime.fromisoformat(p["expiry"])
        if now < expiry:
            still_open.append(p)
            continue
        S = get_spot(p["sym"])
        settle_poll_ts = now.isoformat()
        intrinsic = 0.0
        for leg in p["legs"]:
            v = max(S - leg["K"], 0) if leg["side"] == "C" else max(leg["K"] - S, 0)
            intrinsic += v if leg["short"] else -v
        pnl = p["credit"] - intrinsic * p["lots"] * CV[p["sym"]] - p["fees"]
        state["equity"] += pnl
        log_event("CLOSE", sym=p["sym"], entry_ts=p["entry_ts"], expiry=p["expiry"],
                  settle_spot=S, settle_poll_ts=settle_poll_ts,
                  credit=p["credit"], fees=p["fees"], pnl=round(pnl, 2),
                  equity_after=round(state["equity"], 2))
        log.info(f"CLOSE {p['sym']} {p['sides']} settle_spot=${S:,.0f} "
                f"(polled {settle_poll_ts}, real expiry was {expiry.isoformat()}) "
                f"pnl=${pnl:+.2f} equity=${state['equity']:,.2f}")
    state["open"] = still_open


def run_once():
    state = load_state()
    try:
        refresh_iv_rank_data()
    except Exception as e:
        log.warning(f"IV-rank data refresh failed (using existing data): {e}")
    check_settlements(state)
    for sym in SYMBOLS:
        try:
            try_open(state, sym)
        except Exception as e:
            log.error(f"{sym}: try_open failed, skipping this cycle: {e}")
    save_state(state)
    record_equity(state["equity"])
    log.info(f"cycle done | equity=${state['equity']:,.2f} | open positions: "
            f"{[(p['sym'], p['sides'], p['lots']) for p in state['open']]}")


def status():
    state = load_state()
    print(f"equity: ${state['equity']:,.2f}")
    print(f"open positions ({len(state['open'])}):")
    for p in state["open"]:
        print(f"  {p['sym']} {p['sides']} lots={p['lots']} credit=${p['credit']:.2f} "
              f"expiry={p['expiry']}")
    if LOG_FILE.exists():
        closes = [json.loads(l) for l in open(LOG_FILE) if '"event": "CLOSE"' in l]
        if closes:
            pnls = [c["pnl"] for c in closes]
            print(f"\nclosed trades: {len(closes)}  win rate: {sum(1 for p in pnls if p>0)/len(pnls):.1%}"
                  f"  total pnl: ${sum(pnls):+.2f}")


def main():
    ap = argparse.ArgumentParser(description="Live paper trading (read-only) for the v2 VRP strategy")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, metavar="SECONDS")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        status()
        return

    if args.loop:
        interval = max(args.loop, 300)
        stop = {"flag": False}
        signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
        signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
        while not stop["flag"]:
            try:
                run_once()
            except Exception as e:
                log.error(f"cycle error, will retry next interval: {e}")
            for _ in range(interval):
                if stop["flag"]:
                    break
                time.sleep(1)
        log.info("stopped cleanly, state saved.")
    else:
        run_once()


if __name__ == "__main__":
    main()
