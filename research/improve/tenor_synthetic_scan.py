"""Tenor avenue (a): synthetic fixed-tenor scan, real_expiries=False,
dte in {7, 10, 14, 21, 30}. dte=7 synthetic is included as the bridge to the
weekly baseline. Each cell: full sample, OOS 2024-01+, 2026-only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tenor_lib import BASELINE, fee_share, metrics_line
from engine import Backtest, Config

WINDOWS = [("full", None), ("2024+", "2024-01-01"), ("2026", "2026-01-01")]

print("=" * 110)
print("SYNTHETIC FIXED-TENOR SCAN (real_expiries=False)  — baseline weekly shown first")
print("=" * 110)

# baseline (real weekly expiries) for reference
for wtag, start in WINDOWS:
    bt = Backtest(Config(**BASELINE)).run(start, None)
    m = metrics_line(f"baseline real-weekly [{wtag}]", bt)
    if wtag == "full":
        f, p, s = fee_share(bt)
        print(f"{'':<34} fees=${f:.2f} netP&L=${p:.2f} fee_share={s:.1%}")

print("-" * 110)
for dte in (7, 10, 14, 21, 30):
    kw = dict(BASELINE)
    kw.update(real_expiries=False, dte=dte)
    for wtag, start in WINDOWS:
        bt = Backtest(Config(**kw)).run(start, None)
        m = metrics_line(f"synthetic dte={dte:>2} [{wtag}]", bt)
        if wtag == "full":
            f, p, s = fee_share(bt)
            print(f"{'':<34} fees=${f:.2f} netP&L=${p:.2f} fee_share={s:.1%}")
    print("-" * 110)
