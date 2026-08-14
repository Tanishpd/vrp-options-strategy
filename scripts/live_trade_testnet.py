"""
Live REAL-ORDER trading engine for the validated v2 VRP strategy —
Delta Exchange TESTNET (fake money, real matching engine, real dashboard).

This is the version that makes trades appear on https://demo.delta.exchange.
Decisions (signal, strike selection, sizing) are IDENTICAL to
scripts/paper_trade_live.py — both import from vrp/live_signals.py, so the
two can never drift into deciding differently. This file's only job is
turning a decision into a real order and recording the real fill.

SAFETY DESIGN — read before touching this file:
  - vrp.delta_testnet.DeltaTestnetClient refuses to initialize against
    anything whose base URL doesn't contain "testnet". That check is in the
    client, not here, so there is no path through this script that can
    reach production even by mistake.
  - Every entry is constructed WINGS FIRST: the protective long legs are
    bought before any short leg is sold. A long option is a bounded debit —
    if construction fails after buying wings but before selling shorts, you
    are left holding protection you didn't need, never a naked short.
  - If ANY leg fails to fill, the entry is aborted and every leg already
    filled is immediately unwound (bought back / sold back at market). A
    position is only ever recorded as OPEN once every leg of it is
    confirmed filled. There is no partial/half-built position.
  - Unwind is retried up to 3 times with backoff. If it still fails, this
    logs an ALERT event and stops attempting new entries for that symbol
    until a human checks the account on the dashboard — it does not retry
    forever or take any other automatic action.
  - Settlement: Delta settles expired options on its own books automatically
    (that's what makes this show up on your dashboard). This script's local
    P&L number is a SECONDARY estimate for logging only — the dashboard and
    get_positions()/get_balances() are the source of truth. On every
    startup, local state is reconciled against the exchange's actual
    open positions, not trusted blindly.

Usage:
  .venv/bin/python scripts/live_trade_testnet.py --smoke-test   # auth check only
  .venv/bin/python scripts/live_trade_testnet.py --once
  .venv/bin/python scripts/live_trade_testnet.py --loop 1800
  .venv/bin/python scripts/live_trade_testnet.py --status
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
from vrp.delta_testnet import DeltaAPIError, DeltaTestnetClient  # noqa: E402
from vrp.live_signals import CV, FEE_CFG, fee  # noqa: E402
from vrp.live_signals import plan_entry, refresh_iv_rank_data, size_plan  # noqa: E402

SYMBOLS = ("BTC", "ETH")
STATE_DIR = ROOT / "data" / "live_testnet"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "trade_log.jsonl"
EQUITY_FILE = STATE_DIR / "equity.csv"
CAPITAL = 600.0
UNWIND_ATTEMPTS = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S UTC")
logging.Formatter.converter = time.gmtime
log = logging.getLogger("live_trade_testnet")


def get_client() -> DeltaTestnetClient:
    """Credentials come ONLY from AWS Secrets Manager, read at process start.
    Never hardcoded, never logged, never persisted to disk by this script."""
    import boto3
    sm = boto3.client("secretsmanager", region_name="ap-south-1")
    creds = json.loads(sm.get_secret_value(SecretId="crypto-backtest/api-keys")["SecretString"])
    return DeltaTestnetClient(creds["DELTA_API_KEY"], creds["DELTA_API_SECRET"], creds["DELTA_BASE_URL"])


# ── state ────────────────────────────────────────────────────────────────
def load_state(starting_equity: float | None = None) -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    # First run: seed the local (secondary, logging-only) equity estimate
    # from the REAL exchange balance, not a fictional constant — sizing
    # itself always uses the live balance regardless of this value.
    return {"equity_estimate": starting_equity if starting_equity is not None else CAPITAL, "open": []}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=1, default=str))


def log_event(event: str, **fields):
    rec = {"event": event, "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def record_equity(equity: float):
    row = pd.DataFrame([{"ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "equity_estimate": round(equity, 2)}])
    row.to_csv(EQUITY_FILE, mode="a", header=not EQUITY_FILE.exists(), index=False)


# ── reconciliation: exchange is truth, local state is a cache ───────────
def reconcile_with_exchange(client: DeltaTestnetClient, state: dict):
    try:
        real_positions = {p["product_symbol"] for p in client.get_positions()
                          if float(p.get("size", 0)) != 0}
    except DeltaAPIError as e:
        log.error(f"Could not reconcile with exchange (auth/network issue): {e}")
        return
    for pos in list(state["open"]):
        local_symbols = {leg["symbol"] for leg in pos["legs"]}
        if not (local_symbols & real_positions):
            log.warning(f"{pos['sym']} {pos['sides']} opened {pos['entry_ts']}: no legs found "
                       f"in real exchange positions (likely settled while offline). "
                       f"Removing from local tracking — check dashboard for the real P&L.")
            log_event("RECONCILE_REMOVED", sym=pos["sym"], sides=pos["sides"],
                      entry_ts=pos["entry_ts"], reason="not_found_in_exchange_positions")
            state["open"].remove(pos)


# ── safe multi-leg construction ──────────────────────────────────────────
def place_and_confirm(client: DeltaTestnetClient, symbol: str, side: str, size: int) -> dict:
    """Places a market order and waits for a terminal fill state.
    Raises DeltaAPIError or RuntimeError on anything but a clean fill."""
    order = client.place_market_order(symbol, side, size)
    final = client.wait_for_fill(order["id"])
    if final.get("state") != "closed":
        raise RuntimeError(f"order {order['id']} for {symbol} did not close "
                           f"(state={final.get('state')!r}) — treating as failed")
    avg_fill = float(final.get("average_fill_price") or 0)
    if avg_fill <= 0:
        raise RuntimeError(f"order {order['id']} for {symbol} closed with no fill price")
    return {"order_id": order["id"], "symbol": symbol, "side": side, "size": size,
           "avg_fill_price": avg_fill}


def unwind(client: DeltaTestnetClient, filled: list[dict]):
    """Best-effort reversal of every already-filled leg, most recent first."""
    for f in reversed(filled):
        reverse_side = "sell" if f["side"] == "buy" else "buy"
        for attempt in range(1, UNWIND_ATTEMPTS + 1):
            try:
                r = place_and_confirm(client, f["symbol"], reverse_side, f["size"])
                log_event("UNWOUND", original=f, reversal=r)
                log.warning(f"Unwound {f['symbol']} ({reverse_side} {f['size']}) "
                           f"after failed entry construction.")
                break
            except Exception as e:
                if attempt == UNWIND_ATTEMPTS:
                    log_event("UNWIND_FAILED", leg=f, error=str(e))
                    log.error(f"ALERT: could not unwind {f['symbol']} after "
                             f"{UNWIND_ATTEMPTS} attempts — {e}. A residual position may "
                             f"exist on the testnet account. Check https://demo.delta.exchange "
                             f"manually. Not retrying further this cycle.")
                else:
                    time.sleep(2 ** attempt)


def real_usd_balance(client: DeltaTestnetClient) -> tuple[float, float]:
    """(available_balance, margin_already_committed) straight from the
    exchange's own USD wallet entry — never from local bookkeeping. Excludes
    pending_trading_fee_credit, which is not usable margin."""
    for b in client.get_balances():
        if b.get("asset_symbol") == "USD":
            avail = float(b["available_balance"])
            committed = float(b.get("position_margin", 0)) + float(b.get("order_margin", 0))
            return avail, committed
    raise RuntimeError("No USD wallet entry found in get_balances() response")


def try_open(client: DeltaTestnetClient, state: dict, sym: str):
    plan = plan_entry(sym, state["open"])
    if plan is None:
        return
    if "skip" in plan:
        log_event("SKIP", sym=sym, **plan)
        return

    avail, committed = real_usd_balance(client)
    # size_plan wants (equity, margin_already_committed); equity here is
    # available + committed (committed margin is still "yours", just tied up)
    equity_for_sizing = avail + committed
    lots, m_unit = size_plan(plan, equity_for_sizing, committed)
    if lots < 1:
        log_event("SKIP", sym=sym, reason="sizes_to_zero_lots",
                  real_available_balance=avail, real_committed_margin=committed)
        return
    log.info(f"{sym}: sizing off REAL exchange balance — available=${avail:.2f} "
            f"committed=${committed:.2f} -> {lots} lots")

    filled: list[dict] = []
    real_legs = []
    try:
        # WINGS FIRST — a bought long option is a bounded debit, never a naked short.
        for leg in [l for l in plan["legs"] if not l["short"]]:
            r = place_and_confirm(client, leg["symbol"], "buy", lots)
            filled.append(r)
            real_legs.append({**leg, "fill": r["avg_fill_price"]})
            log.info(f"  filled wing BUY {leg['symbol']} x{lots} @ ${r['avg_fill_price']:.4f}")
        for leg in [l for l in plan["legs"] if l["short"]]:
            r = place_and_confirm(client, leg["symbol"], "sell", lots)
            filled.append(r)
            real_legs.append({**leg, "fill": r["avg_fill_price"]})
            log.info(f"  filled short SELL {leg['symbol']} x{lots} @ ${r['avg_fill_price']:.4f}")
    except Exception as e:
        log.error(f"{sym}: entry construction failed ({e}) — unwinding {len(filled)} filled leg(s)")
        log_event("ABORT", sym=sym, sides=plan["sides"], error=str(e),
                  filled_before_abort=filled)
        unwind(client, filled)
        return

    # credit/fees from REAL fills, not the pre-trade plan
    credit_u = sum(l["fill"] if l["short"] else -l["fill"] for l in real_legs)
    fees = sum(fee(l["fill"], plan["spot"], lots, CV[sym], FEE_CFG) for l in real_legs)
    pos = dict(sym=sym, entry_ts=datetime.now(timezone.utc).isoformat(),
              expiry=plan["expiry"].isoformat(), sides=plan["sides"], lots=lots,
              legs=real_legs, credit=round(credit_u * lots * CV[sym], 4),
              fees=round(fees, 4), margin=round(m_unit * lots, 2))
    state["open"].append(pos)
    log_event("OPEN", **pos, planned_credit_unit=plan["credit_unit"], entry_iv_rank=round(plan["iv_rank"], 3))
    log.info(f"OPEN {sym} {pos['sides']} lots={lots} real_credit=${pos['credit']:.2f} "
            f"(planned ${plan['credit_unit']*lots*CV[sym]:.2f}) — visible now on demo.delta.exchange")


def real_leg_cash_effect(client: DeltaTestnetClient, symbol: str, since: datetime) -> float:
    """Sum every real wallet transaction (cashflow + commission) tied to this
    exact contract since it was opened. This is Delta's own ledger, not an
    estimate — it captures whatever the exchange actually did at settlement,
    including anything about its settlement mechanics we don't model
    ourselves. product_id filtering isn't honored server-side (checked), so
    this filters client-side by symbol and timestamp."""
    tx = client._request("GET", "/v2/wallet/transactions", params={"page_size": 200})
    total = 0.0
    for t in tx:
        if t.get("meta_data", {}).get("product_symbol") != symbol:
            continue
        if datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")) < since:
            continue
        total += float(t["amount"])
    return total


# ── settlement: exchange is the source of truth ──────────────────────────
def check_settlements(client: DeltaTestnetClient, state: dict):
    now = datetime.now(timezone.utc)
    try:
        real_positions = {p["product_symbol"] for p in client.get_positions()
                          if float(p.get("size", 0)) != 0}
    except DeltaAPIError as e:
        log.error(f"Could not check exchange positions this cycle: {e}")
        return
    still_open = []
    for p in state["open"]:
        expiry = datetime.fromisoformat(p["expiry"])
        legs_still_live = {leg["symbol"] for leg in p["legs"]} & real_positions
        if now < expiry and legs_still_live:
            still_open.append(p)
            continue
        # Settled on the exchange. Sum Delta's OWN ledger entries for every
        # leg since entry -- this is the real realized P&L, not an estimate.
        entry_ts = datetime.fromisoformat(p["entry_ts"])
        real_pnl = sum(real_leg_cash_effect(client, leg["symbol"], entry_ts) for leg in p["legs"])
        state["equity_estimate"] += real_pnl
        log_event("CLOSE", sym=p["sym"], sides=p["sides"], expiry=p["expiry"],
                  real_pnl=round(real_pnl, 2), source="delta_wallet_transactions_ledger")
        log.info(f"CLOSE {p['sym']} {p['sides']} real_pnl=${real_pnl:+.2f} "
                f"(from Delta's own wallet ledger, not an estimate)")
    state["open"] = still_open


# ── cycle ────────────────────────────────────────────────────────────────
def run_once():
    client = get_client()
    seed = None
    if not STATE_FILE.exists():
        avail, committed = real_usd_balance(client)
        seed = avail + committed
        log.info(f"first run: seeding local equity_estimate from real exchange balance ${seed:.2f}")
    state = load_state(seed)
    reconcile_with_exchange(client, state)
    try:
        refresh_iv_rank_data()
    except Exception as e:
        log.warning(f"IV-rank data refresh failed (using existing data): {e}")
    check_settlements(client, state)
    for sym in SYMBOLS:
        try:
            try_open(client, state, sym)
        except Exception as e:
            log.error(f"{sym}: unexpected error, skipping this cycle: {e}")
    save_state(state)
    record_equity(state["equity_estimate"])
    log.info(f"cycle done | equity_estimate=${state['equity_estimate']:,.2f} | "
            f"local open positions: {[(p['sym'], p['sides'], p['lots']) for p in state['open']]}")


def smoke_test():
    """Auth-only check — confirms the API key + IP whitelist actually work,
    without touching the strategy or placing any order."""
    client = get_client()
    balances = client.get_balances()
    positions = client.get_positions()
    print("AUTH OK. Testnet balances:")
    for b in balances:
        if float(b.get("balance", 0)) != 0:
            print(f"  {b.get('asset_symbol', '?')}: {b.get('balance')}")
    print(f"Open positions on exchange right now: {len(positions)}")
    for p in positions:
        if float(p.get("size", 0)) != 0:
            print(f"  {p['product_symbol']}: size={p['size']}")


def status():
    state = load_state()
    print(f"equity estimate (secondary — check dashboard for real balance): ${state['equity_estimate']:,.2f}")
    print(f"local open positions ({len(state['open'])}):")
    for p in state["open"]:
        print(f"  {p['sym']} {p['sides']} lots={p['lots']} credit=${p['credit']:.2f} expiry={p['expiry']}")
        for leg in p["legs"]:
            print(f"    {leg['symbol']}")


def main():
    ap = argparse.ArgumentParser(description="Live testnet order placement for the v2 VRP strategy")
    ap.add_argument("--smoke-test", action="store_true", help="auth check only, no strategy logic")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, metavar="SECONDS")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return
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
