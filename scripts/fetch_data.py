"""Fetch real BTC/ETH implied vol (Deribit DVOL) + spot (Binance) for VRP research."""
import time, requests, pandas as pd, numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data"; OUT.mkdir(exist_ok=True)

def dvol(currency="BTC"):
    """Deribit DVOL index — 30-day forward-looking implied vol, daily OHLC."""
    rows, start = [], int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
    end_all = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    while start < end_all:
        end = min(start + 300 * 86400_000, end_all)
        r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                         params={"currency": currency, "start_timestamp": start,
                                 "end_timestamp": end, "resolution": "1D"}, timeout=30)
        r.raise_for_status()
        d = r.json().get("result", {}).get("data", [])
        if not d: break
        rows += d
        start = d[-1][0] + 86400_000
        time.sleep(0.25)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize()
    return df.set_index("date")[["close"]].rename(columns={"close": "iv"}) / 100.0

def spot(symbol="BTCUSDT"):
    rows, start = [], int(pd.Timestamp("2020-12-01", tz="UTC").timestamp() * 1000)
    while True:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol, "interval": "1d",
                                 "startTime": start, "limit": 1000}, timeout=30)
        r.raise_for_status()
        d = r.json()
        if not d: break
        rows += d
        if len(d) < 1000: break
        start = d[-1][0] + 86400_000
        time.sleep(0.2)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    return df.drop_duplicates("date").set_index("date")[["open", "high", "low", "close", "volume"]]

if __name__ == "__main__":
    for cur, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        iv, px = dvol(cur), spot(sym)
        df = px.join(iv, how="inner").dropna()
        df.to_csv(OUT / f"{cur}.csv")
        print(f"{cur}: {len(df)} days  {df.index.min().date()} -> {df.index.max().date()}  "
              f"IV mean {df['iv'].mean():.1%}  spot {df['close'].iloc[-1]:,.0f}")
