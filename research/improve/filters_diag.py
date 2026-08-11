"""Diagnostics for the side-filter variants: how many entries are converted to
put-only, which baseline losers are rescued, and a plateau probe at B thr=0.25."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "improve"))
from filters_runner import BASE, SideFilterBacktest, evaluate, fmt, WINDOWS  # noqa: E402
from engine import Config  # noqa: E402

base_tr = pd.read_csv(ROOT / "results" / "final_trades.csv", parse_dates=["entry"])
base_losers = set(base_tr.loc[base_tr.pnl < 0, "entry"].dt.date)

for mode, thrs in [("ret5", (-0.04, -0.06, -0.08)), ("ret20", (0.10, 0.15, 0.20, 0.25))]:
    for thr in thrs:
        bt = SideFilterBacktest(Config(**BASE), mode, thr)
        bt.run()
        conv = [t for t in bt.closed if not any(l.side == "C" for l in t.legs)]
        conv_l = [t for t in conv if t.pnl < 0]
        rescued = [t for t in conv if t.entry.date() in base_losers and t.pnl > 0]
        newly_lost = [t for t in bt.closed
                      if t.pnl < 0 and t.entry.date() not in base_losers]
        print(f"{mode} thr={thr:+.2f}: n={len(bt.closed)} converted_to_put_only="
              f"{len(conv)} (of which {len(conv_l)} still lost)  "
              f"rescued_losers={len(rescued)}  new_losers={len(newly_lost)}")
        for t in rescued:
            print(f"    rescued {t.entry.date()} pnl={t.pnl:+.2f}")
        for t in newly_lost:
            print(f"    NEW loss {t.entry.date()} pnl={t.pnl:+.2f}")

print("\n=== plateau probe: B_ret20 thr=+0.25 (full/oos/2026) ===")
res = evaluate(lambda: SideFilterBacktest(Config(**BASE), "ret20", 0.25))
for label in WINDOWS:
    print(f"  {label:<6} {fmt(res[label])}")
