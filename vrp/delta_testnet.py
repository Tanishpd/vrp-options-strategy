"""
Minimal Delta Exchange order-placement client — TESTNET ONLY.

Hard safety rail: __init__ refuses to run unless "testnet" appears in the
base URL. This client is used by exactly one script (scripts/live_trade_testnet.py)
whose entire purpose is placing real orders against Delta's testnet (fake
money, real matching engine) so they appear on your Delta demo dashboard.
It must never be pointed at production — that assertion is not a warning
comment, it's a raised exception.

Only 4 authenticated calls exist in this file: place a market order, get an
order's status, get open positions, get wallet balances. Nothing else is
implemented — no withdrawals, no API-key management, no anything not needed
by the v2 strategy.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass

import requests


class DeltaTestnetClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        if "testnet" not in base_url:
            raise RuntimeError(
                f"REFUSING to initialize: base_url '{base_url}' does not contain "
                f"'testnet'. DeltaTestnetClient only ever trades Delta's testnet — "
                f"this assertion exists specifically to prevent it from ever being "
                f"pointed at production by a config mistake."
            )
        self.key, self.secret, self.base = api_key, api_secret, base_url.rstrip("/")
        self.session = requests.Session()

    def _sign(self, method: str, path: str, query: str, body: str) -> tuple[str, str]:
        ts = str(int(time.time()))
        prehash = method + ts + path + query + body
        sig = hmac.new(self.secret.encode(), prehash.encode(), hashlib.sha256).hexdigest()
        return ts, sig

    def _request(self, method: str, path: str, params: dict | None = None,
                 body: dict | None = None) -> dict:
        params = params or {}
        query = "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())) if params else ""
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        ts, sig = self._sign(method, path, query, body_str)
        headers = {"api-key": self.key, "timestamp": ts, "signature": sig,
                  "Content-Type": "application/json", "Accept": "application/json"}
        # Build the URL with the EXACT query string that was signed. Passing
        # params= to requests instead would let it re-serialize the dict in
        # insertion order, which can differ from the sorted order signed
        # above whenever there are 2+ params -- Delta then sees a different
        # query string than the one the signature covers, and rejects it
        # with "Signature Mismatch". Caught by testing multi-param calls
        # before ever placing a real order.
        r = self.session.request(method, self.base + path + query,
                                 data=body_str if body else None, headers=headers, timeout=15)
        try:
            data = r.json()
        except ValueError:
            r.raise_for_status()
            raise
        if not data.get("success", False):
            raise DeltaAPIError(r.status_code, data)
        return data.get("result")

    # ---- read-only (used to verify auth works, and to reconcile state) ----
    def get_balances(self) -> list[dict]:
        return self._request("GET", "/v2/wallet/balances")

    def get_positions(self) -> list[dict]:
        return self._request("GET", "/v2/positions/margined")

    def get_order(self, order_id: int) -> dict:
        return self._request("GET", f"/v2/orders/{order_id}")

    # ---- the only way this client can move money (fake money) ----
    def place_market_order(self, product_symbol: str, side: str, size: int) -> dict:
        """side: 'buy' or 'sell'. size: integer number of lots/contracts.
        Returns Delta's order object (includes state, average_fill_price once filled)."""
        assert side in ("buy", "sell"), side
        assert isinstance(size, int) and size >= 1, size
        body = {"product_symbol": product_symbol, "size": size, "side": side,
               "order_type": "market_order"}
        return self._request("POST", "/v2/orders", body=body)

    def wait_for_fill(self, order_id: int, timeout_s: float = 10.0, poll_s: float = 0.5) -> dict:
        """Market orders should fill near-instantly. Poll briefly; if it hasn't
        reached a terminal state in `timeout_s`, return whatever we last saw
        so the caller can decide how to handle an unusual non-fill."""
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            last = self.get_order(order_id)
            if last.get("state") in ("closed", "cancelled", "rejected"):
                return last
            time.sleep(poll_s)
        return last or {}


@dataclass
class DeltaAPIError(Exception):
    status_code: int
    body: dict

    def __str__(self):
        return f"Delta API error (HTTP {self.status_code}): {self.body}"
