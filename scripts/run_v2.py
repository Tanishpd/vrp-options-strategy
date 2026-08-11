"""Final v2 export: trades, equity curve, yearly breakdown, cost split."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vrp.engine import Config, Backtest
from pathlib import Path
OUT=Path(__file__).resolve().parents[1] / "results"

V2=dict(symbols=("BTC","ETH"), capital=600.0, min_iv_rank=0.60,
        min_iv_rank_by_symbol={"ETH":0.75}, short_delta=0.25,
        wing_width_deltas=0.10, defined_risk=True, both_sides=True,
        real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
        profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
        risk_per_trade_by_symbol={"ETH":0.02}, max_margin_util=0.50,
        max_leverage=3.0, max_concurrent_per_symbol=2, skip_call_ret20=0.20)

bt=Backtest(Config(**V2)).run()
rows=[]
for i,t in enumerate(bt.closed,1):
    s={l.side:l.K for l in t.legs if l.short}; g={l.side:l.K for l in t.legs if not l.short}
    rows.append(dict(n=i, sym=t.symbol, entry=t.entry.date(), expiry=t.expiry.date(),
        iv=round(t.entry_iv,3), iv_rank=round(t.entry_iv_rank,3), lots=t.legs[0].lots,
        sides="".join(sorted(s.keys())), sC=s.get("C"), sP=s.get("P"),
        credit=round(t.credit,2), settle=round(t.exit_cost,2), fees=round(t.fees,2),
        pnl=round(t.pnl,2)))
tr=pd.DataFrame(rows); tr.to_csv(OUT/"v2_trades.csv",index=False)
eq=pd.Series(dict(bt.curve)).sort_index(); eq.name="equity"; eq.to_csv(OUT/"v2_equity.csv")
p=tr.pnl.values; yrs=(eq.index[-1]-eq.index[0]).days/365.25

print(f"final ${eq.iloc[-1]:,.2f} from $600 over {yrs:.2f}y  |  {len(tr)} trades  |  files: v2_trades.csv, v2_equity.csv")
print(f"\nBY YEAR                          BY SYMBOL")
by_y=tr.groupby(pd.to_datetime(tr.entry).dt.year).agg(n=("pnl","size"),win=("pnl",lambda s:(s>0).mean()),pnl=("pnl","sum"))
by_s=tr.groupby("sym").agg(n=("pnl","size"),win=("pnl",lambda s:(s>0).mean()),pnl=("pnl","sum"))
for y,r in by_y.iterrows():
    print(f"  {y}  n={r['n']:>3}  win {r['win']:>5.1%}  ${r['pnl']:>+8.2f}")
for s,r in by_s.iterrows():
    print(f"  {s}   n={r['n']:>3}  win {r['win']:>5.1%}  ${r['pnl']:>+8.2f}")
put_only=(tr.sides=="P").sum()
print(f"\nret20 filter fired (put-spread-only entries): {put_only}/{len(tr)}")
print(f"fees ${tr.fees.sum():.2f} = {tr.fees.sum()/(p.sum()+tr.fees.sum()):.1%} of gross, {tr.fees.sum()/p.sum():.0%} of net P&L")
# maker-execution upside, reported separately (not part of the validated strategy)
mk=Backtest(Config(**{**V2,"spread_k":0.00325,"fee_notional_pct":0.0002})).run().metrics()
print(f"\nmaker-execution upside (unvalidatable from daily bars, do NOT bank on it):")
print(f"  CAGR {mk['cagr']:+.1%}  maxDD {mk['max_dd']:+.1%}  Sharpe {mk['sharpe']:.2f}")
