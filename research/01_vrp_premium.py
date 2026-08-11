"""Does a variance risk premium exist in BTC/ETH, and where is it concentrated?"""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data"
pd.set_option("display.width", 200)

def load(cur, horizon=30):
    df = pd.read_csv(D / f"{cur}.csv", index_col=0, parse_dates=True)
    df["ret"] = np.log(df["close"]).diff()
    # RV over the NEXT `horizon` days, annualized 365 (crypto trades 24/7)
    df["rv_fwd"] = df["ret"].shift(-1).rolling(horizon).std().shift(-(horizon - 1)) * np.sqrt(365)
    # RV over the PAST horizon — this is knowable at t, safe for filters
    df["rv_past"] = df["ret"].rolling(horizon).std() * np.sqrt(365)
    df["vrp"] = df["iv"] - df["rv_fwd"]                 # what a vol seller earns
    df["iv_rank"] = df["iv"].rolling(365, min_periods=180).rank(pct=True)   # knowable at t
    df["iv_over_rvpast"] = df["iv"] / df["rv_past"]                         # knowable at t
    return df

for cur in ("BTC", "ETH"):
    d = load(cur).dropna(subset=["vrp"])
    v = d["vrp"]
    n_eff = len(v) / 30                     # overlapping windows -> effective independent obs
    t = v.mean() / (v.std() / np.sqrt(n_eff))
    print("=" * 78)
    print(f"{cur}   n={len(d)} days ({d.index.min().date()} -> {d.index.max().date()})")
    print(f"  mean IV {d['iv'].mean():>7.1%}   mean forward RV {d['rv_fwd'].mean():>7.1%}")
    print(f"  MEAN VRP {v.mean():>+7.2%}  median {v.median():>+7.2%}  sd {v.std():.2%}")
    print(f"  share of days VRP > 0: {(v>0).mean():.1%}      t-stat (n_eff={n_eff:.0f}) = {t:.2f}")
    print()
    print("  VRP by IV RANK at entry (percentile of trailing 1y IV) — knowable in advance:")
    d["bkt"] = pd.qcut(d["iv_rank"], 5, labels=["Q1 lowest IV", "Q2", "Q3", "Q4", "Q5 highest IV"])
    g = d.groupby("bkt", observed=True)["vrp"].agg(["mean", "median", "std", "count",
                                                   lambda s: (s > 0).mean()])
    g.columns = ["mean", "median", "sd", "n", "win%"]
    for k, r in g.iterrows():
        print(f"    {str(k):<15} mean {r['mean']:>+7.2%}  median {r['median']:>+7.2%}  "
              f"win {r['win%']:>5.1%}  n={int(r['n'])}")
    print()
    print("  VRP by IV / trailing-30d RV at entry — the direct 'is vol rich' signal:")
    d["bkt2"] = pd.qcut(d["iv_over_rvpast"], 5, labels=["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 richest"])
    g2 = d.groupby("bkt2", observed=True)["vrp"].agg(["mean", "median", "count",
                                                     lambda s: (s > 0).mean()])
    g2.columns = ["mean", "median", "n", "win%"]
    for k, r in g2.iterrows():
        print(f"    {str(k):<15} mean {r['mean']:>+7.2%}  median {r['median']:>+7.2%}  "
              f"win {r['win%']:>5.1%}  n={int(r['n'])}")
    print()
    print("  VRP by calendar year (consistency check):")
    for y, s in d.groupby(d.index.year)["vrp"]:
        print(f"    {y}   mean {s.mean():>+7.2%}   win {(s>0).mean():>5.1%}   n={len(s)}")
    print()
