"""Is the BTC VRP still there? Test decay, tenor, and whether the IV-rank filter holds recently."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data"

def load(cur, h):
    df = pd.read_csv(D / f"{cur}.csv", index_col=0, parse_dates=True)
    df["ret"] = np.log(df["close"]).diff()
    df["rv_fwd"] = df["ret"].shift(-1).rolling(h).std().shift(-(h - 1)) * np.sqrt(365)
    df["vrp"] = df["iv"] - df["rv_fwd"]
    df["iv_rank"] = df["iv"].rolling(365, min_periods=180).rank(pct=True)
    return df.dropna(subset=["vrp", "iv_rank"])

print("=" * 84)
print("1. VRP BY TENOR — the strategy sells weeklies, DVOL is a 30-day index")
print("=" * 84)
print(f"{'horizon':>9}  {'BTC mean':>10} {'BTC win%':>9}  {'ETH mean':>10} {'ETH win%':>9}")
for h in (7, 14, 30, 60):
    r = []
    for cur in ("BTC", "ETH"):
        d = load(cur, h)
        r += [f"{d['vrp'].mean():>+9.2%}", f"{(d['vrp']>0).mean():>8.1%}"]
    print(f"{h:>7}d  {r[0]:>10} {r[1]:>9}  {r[2]:>10} {r[3]:>9}")

print()
print("=" * 84)
print("2. DECAY — is the edge disappearing? (BTC, 30d, split by era)")
print("=" * 84)
d = load("BTC", 30)
eras = [("2021-03/2022-12", "2021-03-24", "2022-12-31"),
        ("2023-01/2024-12", "2023-01-01", "2024-12-31"),
        ("2025-01/now",     "2025-01-01", "2026-12-31")]
for lbl, a, b in eras:
    s = d.loc[a:b]
    v = s["vrp"]; ne = len(v)/30
    t = v.mean()/(v.std()/np.sqrt(ne)) if ne > 1 else float("nan")
    hi = s[s["iv_rank"] >= 0.6]["vrp"]
    print(f"  {lbl:<16} all: mean {v.mean():>+7.2%} win {(v>0).mean():>5.1%} t={t:>5.2f} n={len(v):>4}"
          f"   |  IV-rank>=0.6: mean {hi.mean():>+7.2%} win {(hi>0).mean():>5.1%} n={len(hi):>4}")

print()
print("=" * 84)
print("3. DOES THE IV-RANK FILTER STILL WORK IN 2025-2026? (the deployability test)")
print("=" * 84)
for cur in ("BTC", "ETH"):
    d = load(cur, 30)
    recent = d.loc["2025-01-01":]
    print(f"  {cur} 2025-01 -> now  (n={len(recent)})")
    for lo, hi_ in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        s = recent[(recent["iv_rank"] >= lo) & (recent["iv_rank"] < hi_)]["vrp"]
        if len(s) < 20:
            print(f"    IV rank {lo:.1f}-{hi_:.1f}   n={len(s):<4} (too few)")
            continue
        print(f"    IV rank {lo:.1f}-{hi_:.1f}   mean {s.mean():>+7.2%}  win {(s>0).mean():>5.1%}  n={len(s)}")
    print()

print("=" * 84)
print("4. WORST CASE — what does the left tail look like for a vol seller?")
print("=" * 84)
d = load("BTC", 30)
for q in (0.01, 0.05, 0.10, 0.25):
    print(f"  {q:>5.0%} percentile VRP: {d['vrp'].quantile(q):>+7.2%}")
w = d.nsmallest(5, "vrp")[["iv", "rv_fwd", "vrp"]]
print("\n  5 worst 30-day windows (entry IV vs what actually realized):")
for i, r in w.iterrows():
    print(f"    {i.date()}  sold vol at {r['iv']:.0%}, realized {r['rv_fwd']:.0%}  -> VRP {r['vrp']:+.1%}")
