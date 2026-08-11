"""Export every trade, the equity curve, and the full result set."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vrp.engine import Config, Backtest
from pathlib import Path
OUT=Path(__file__).resolve().parents[1] / "results"; OUT.mkdir(parents=True, exist_ok=True)
BEST=dict(short_delta=0.25,dte=7,defined_risk=True,hold_to_expiry=True,profit_take=0.99,min_iv_rank=0.60)

bt=Backtest(Config(**BEST)).run()
rows=[]
for i,t in enumerate(bt.closed,1):
    shorts=[l for l in t.legs if l.short]; longs=[l for l in t.legs if not l.short]
    rows.append(dict(
        n=i, entry=t.entry.date(), exit=t.exit_date.date(), days=(t.exit_date-t.entry).days,
        spot_in=round(t.entry_spot,0), iv=round(t.entry_iv,4), iv_rank=round(t.entry_iv_rank,3),
        lots=t.legs[0].lots,
        short_C=next((l.K for l in shorts if l.side=="C"),None),
        short_P=next((l.K for l in shorts if l.side=="P"),None),
        long_C=next((l.K for l in longs if l.side=="C"),None),
        long_P=next((l.K for l in longs if l.side=="P"),None),
        credit=round(t.credit,2), exit_cost=round(t.exit_cost,2), fees=round(t.fees,2),
        max_loss=round(t.max_loss,2), margin=round(t.margin,2),
        pnl=round(t.pnl,2), reason=t.reason))
tr=pd.DataFrame(rows)
tr.to_csv(OUT/"trades.csv",index=False)
eq=pd.Series(dict(bt.curve)).sort_index(); eq.name="equity"
eq.to_csv(OUT/"equity_curve.csv")
print(f"wrote {OUT}/trades.csv  ({len(tr)} trades)")
print(f"wrote {OUT}/equity_curve.csv  ({len(eq)} daily points)")
print()
pd.set_option("display.width",250,"display.max_rows",200,"display.max_columns",30)
print("="*150); print("ALL 99 TRADES"); print("="*150)
print(tr.to_string(index=False))
