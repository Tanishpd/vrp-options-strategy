"""
Telegram command listener for the VRP live trading engine — the other
direction from vrp/telegram_notify.py's push notifications. Message the bot
and get an immediate reply with real, freshly-queried data.

Runs as its own process, separate from vrp-live-testnet.service (which only
wakes every 30 minutes to check for trades). This listens continuously via
Telegram long-polling so a command gets an answer in seconds, not whenever
the next trading cycle happens to run.

READ-ONLY. Every command here only calls GET endpoints (real balance, real
positions, the local trade log) or reads files. Nothing in this file can
place, cancel, or modify an order — same safety boundary as
scripts/paper_trade_live.py, just for queries instead of simulated fills.

Only responds to messages from the configured TELEGRAM_CHAT_ID (from the
tradingbot/telegram secret) — a stray message from any other chat is
ignored, never answered, so real balance/position data is never leaked to
an unexpected chat even if the bot's username were ever discovered.

Commands:
  /help        list commands
  /status      local equity estimate + local open positions
  /balance     REAL Delta balance (available + committed margin)
  /positions   REAL Delta open positions, straight from the exchange
  /trades      last 10 closed trades (real P&L, from the wallet ledger)
  /summary     same content as the automatic daily summary, on demand

Usage:
  .venv/bin/python scripts/telegram_bot_listener.py
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vrp.telegram_notify as tg  # noqa: E402
from scripts.live_trade_testnet import (  # noqa: E402
    LOG_FILE, STATE_FILE, get_client, real_usd_balance, send_daily_summary,
)

OFFSET_FILE = ROOT / "data" / "telegram_state" / "listener_offset.json"
OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S UTC")
logging.Formatter.converter = time.gmtime
log = logging.getLogger("telegram_bot_listener")


def load_offset() -> int:
    if OFFSET_FILE.exists():
        try:
            return json.loads(OFFSET_FILE.read_text())["offset"]
        except (json.JSONDecodeError, KeyError):
            pass
    return 0


def save_offset(offset: int):
    OFFSET_FILE.write_text(json.dumps({"offset": offset}))


def cmd_help(_client) -> str:
    return ("*Commands*\n"
           "/status — local equity estimate + open positions\n"
           "/balance — real Delta balance right now\n"
           "/positions — real Delta open positions right now\n"
           "/trades — last 10 closed trades (real P&L)\n"
           "/summary — full daily-summary content, on demand")


def cmd_status(_client) -> str:
    if not STATE_FILE.exists():
        return "No local state yet — the trading engine hasn't run a cycle."
    state = json.loads(STATE_FILE.read_text())
    lines = [f"Equity estimate: ${state['equity_estimate']:.2f}",
            f"Local open positions: {len(state['open'])}"]
    for p in state["open"]:
        lines.append(f"  • {p['sym']} {p['sides']} x{p['lots']} lots, expiry {p['expiry'][:10]}")
    return "\n".join(lines)


def cmd_balance(client) -> str:
    avail, committed = real_usd_balance(client)
    return f"Real Delta balance:\nAvailable: ${avail:.2f}\nCommitted to margin: ${committed:.2f}\nTotal: ${avail + committed:.2f}"


def cmd_positions(client) -> str:
    positions = [p for p in client.get_positions() if float(p.get("size", 0)) != 0]
    if not positions:
        return "No open positions on the exchange right now."
    lines = ["*Real open positions:*"]
    for p in positions:
        lines.append(f"  • {p['product_symbol']}: size={p['size']}")
    return "\n".join(lines)


def cmd_trades(_client) -> str:
    if not LOG_FILE.exists():
        return "No trade log yet."
    closes = []
    for line in open(LOG_FILE):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "CLOSE":
            closes.append(rec)
    if not closes:
        return "No closed trades yet."
    lines = ["*Last 10 closed trades:*"]
    for c in closes[-10:]:
        emoji = "\U0001f7e2" if c.get("real_pnl", 0) >= 0 else "\U0001f534"
        lines.append(f"{emoji} {c['sym']} {c['sides']} ${c.get('real_pnl', 0):+.2f} "
                     f"(expiry {c['expiry'][:10]})")
    return "\n".join(lines)


def cmd_summary(client) -> str:
    if not STATE_FILE.exists():
        return "No local state yet."
    state = json.loads(STATE_FILE.read_text())
    send_daily_summary(client, state)
    return None  # send_daily_summary already sent its own message


COMMANDS = {
    "/help": cmd_help, "/start": cmd_help,
    "/status": cmd_status,
    "/balance": cmd_balance,
    "/positions": cmd_positions,
    "/trades": cmd_trades,
    "/summary": cmd_summary,
}


def handle_message(client, text: str) -> str | None:
    cmd = text.strip().split()[0].lower() if text.strip() else ""
    handler = COMMANDS.get(cmd)
    if handler is None:
        return f"Unknown command {cmd!r}. Send /help for the list."
    try:
        return handler(client)
    except Exception as e:
        log.error(f"command {cmd} failed: {e}")
        return f"Error handling {cmd}: {e}"


def run():
    allowed_chat_id = tg.configured_chat_id()
    if not allowed_chat_id:
        log.error("No TELEGRAM_CHAT_ID configured — cannot verify who to respond to. Exiting.")
        return
    offset = load_offset()
    client = get_client()
    log.info(f"listening for commands (only from chat_id={allowed_chat_id})")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    while not stop["flag"]:
        updates = tg.get_updates(offset, timeout=25)
        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = msg.get("text", "")
            if not text:
                continue
            if chat_id != str(allowed_chat_id):
                log.warning(f"ignoring message from unrecognized chat_id={chat_id}")
                continue
            log.info(f"received: {text!r}")
            reply = handle_message(client, text)
            if reply:
                tg.send(reply)
        save_offset(offset)


if __name__ == "__main__":
    run()
