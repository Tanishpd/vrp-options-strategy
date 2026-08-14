"""
Minimal Telegram Bot API notifier for the VRP live trading engine.

Credentials come from AWS Secrets Manager (tradingbot/telegram) — the same
bot/chat used by the separate equities trading bot. That means crypto and
equities notifications land in the SAME Telegram chat, so every message
this module sends is prefixed "[VRP Crypto]" to stay distinguishable from
the equities bot's messages. Never hardcoded, never logged.

A notification failure must never crash or block the trading loop: every
function here catches its own errors, logs a warning, and returns False.
Nothing that calls send()/send_error()/send_alert() should ever wrap it in
its own try/except — that's handled here so call sites stay simple.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "telegram_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
DEDUPE_FILE = STATE_DIR / "last_error.json"
DAILY_FILE = STATE_DIR / "last_daily_summary.json"

_cached_creds: dict | None = None


def _get_creds() -> dict:
    global _cached_creds
    if _cached_creds is None:
        try:
            import boto3
            sm = boto3.client("secretsmanager", region_name="ap-south-1")
            raw = sm.get_secret_value(SecretId="tradingbot/telegram")["SecretString"]
            _cached_creds = json.loads(raw)
        except Exception:
            _cached_creds = {}
    return _cached_creds


def send(text: str) -> bool:
    """Best-effort send. Returns False (never raises) if creds are missing,
    Telegram is unreachable, or anything else goes wrong."""
    creds = _get_creds()
    token, chat_id = creds.get("TELEGRAM_BOT_TOKEN"), creds.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": f"[VRP Crypto] {text}",
                               "parse_mode": "Markdown"},
                          timeout=10)
        return r.ok
    except Exception:
        return False


def send_error(text: str, min_repeat_interval_s: int = 3600) -> bool:
    """Error alert, but suppresses an IDENTICAL repeated error within the
    interval so a persistent bug doesn't send one message every 30-minute
    cycle. A DIFFERENT error always goes through immediately."""
    last = {}
    if DEDUPE_FILE.exists():
        try:
            last = json.loads(DEDUPE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    now = time.time()
    if last.get("text") == text and now - last.get("ts", 0) < min_repeat_interval_s:
        return False
    ok = send(f"⚠️ *Error*\n{text}")
    DEDUPE_FILE.write_text(json.dumps({"text": text, "ts": now}))
    return ok


def send_alert(text: str) -> bool:
    """For things that need attention NOW (e.g. a failed unwind that may
    have left a residual position). Never deduplicated or suppressed."""
    return send(f"\U0001f6a8 *ALERT*\n{text}")


def due_for_daily_summary() -> bool:
    """True once per UTC calendar day, regardless of restarts."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
    last = {}
    if DAILY_FILE.exists():
        try:
            last = json.loads(DAILY_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return last.get("date") != today


def mark_daily_summary_sent():
    DAILY_FILE.write_text(json.dumps({"date": time.strftime("%Y-%m-%d", time.gmtime())}))


def configured_chat_id() -> str | None:
    """The one chat this bot is allowed to talk to and take commands from."""
    return _get_creds().get("TELEGRAM_CHAT_ID")


def get_updates(offset: int, timeout: int = 25) -> list[dict]:
    """Long-poll for new incoming messages since `offset`. Returns [] (never
    raises) on any error, so a listener loop can just retry next iteration."""
    creds = _get_creds()
    token = creds.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return []
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                         params={"offset": offset, "timeout": timeout}, timeout=timeout + 10)
        if not r.ok:
            return []
        return r.json().get("result", [])
    except Exception:
        return []
