import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vrp.engine import Config, Backtest
from pathlib import Path
OUT=Path(__file__).resolve().parents[1] / "results"; OUT.mkdir(parents=True,exist_ok=True)

FINAL=dict(symbols=("BTC",), capital=600.0, min_iv_rank=0.60, short_delta=0.25,
           wing_width_deltas=0.07, defined_risk=True, both_sides=True,
           real_expiries=True, min_dte=3, max_dte=10, hold_to_expiry=True,
           profit_take=0.99, stop_multiple=1e9, risk_per_trade=0.04,
           max_margin_util=0.50, max_leverage=3.0)
bt=Backtest(Config(**FINAL)).run()
rows=[]
for i,t in enumerate(bt.closed,1):
    s={l.side:l.K for l in t.legs if l.short}; g={l.side:l.K for l in t.legs if not l.short}
    rows.append(dict(n=i, entry=t.entry.date(), expiry=t.expiry.date(),
        dte=round((t.expiry-t.entry).total_seconds()/86400,1), spot=round(t.entry_spot),
        iv=round(t.entry_iv,3), iv_rank=round(t.entry_iv_rank,3), lots=t.legs[0].lots,
        sC=s.get("C"), sP=s.get("P"), lC=g.get("C"), lP=g.get("P"),
        credit=round(t.credit,2), settle=round(t.exit_cost,2), fees=round(t.fees,2),
        max_loss=round(t.max_loss,2), margin=round(t.margin,2), pnl=round(t.pnl,2)))
tr=pd.DataFrame(rows); tr.to_csv(OUT/"final_trades.csv",index=False)
eq=pd.Series(dict(bt.curve)).sort_index(); eq.name="equity"; eq.to_csv(OUT/"final_equity.csv")
p=tr.pnl.values; w,l=p[p>0],p[p<0]
yrs=(eq.index[-1]-eq.index[0]).days/365.25
dd=(eq-eq.cummax())/eq.cummax(); r=eq.pct_change().dropna()

print("="*80); print("FINAL STRATEGY — RESULTS"); print("="*80)
for k,v in [("period",f"{eq.index[0].date()} -> {eq.index[-1].date()} ({yrs:.2f}y)"),
 ("trades",f"{len(p)}  ({len(p)/yrs:.1f}/yr)"),
 ("equity",f"${eq.iloc[0]:,.2f} -> ${eq.iloc[-1]:,.2f}   ({eq.iloc[-1]/eq.iloc[0]-1:+.1%} total)"),
 ("CAGR",f"{(eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1:+.2%}"),
 ("win rate",f"{(p>0).mean():.1%}   ({len(w)}W / {len(l)}L)"),
 ("profit factor",f"{w.sum()/-l.sum():.2f}"),
 ("expectancy",f"${p.mean():+.2f}/trade   median ${np.median(p):+.2f}"),
 ("avg win / loss",f"${w.mean():+.2f} / ${l.mean():+.2f}   payoff {w.mean()/-l.mean():.2f}"),
 ("best / worst",f"${p.max():+.2f} / ${p.min():+.2f}  (worst = {p.min()/600:.1%} of start equity)"),
 ("max drawdown",f"{dd.min():.2%}  on {dd.idxmin().date()}"),
 ("Sharpe / Sortino",f"{r.mean()/r.std()*np.sqrt(365):.2f} / {r.mean()/r[r<0].std()*np.sqrt(365):.2f}"),
 ("longest loss streak",f"{max((len(list(g)) for k_,g in itertools.groupby(p>0) if not k_),default=0)}"),
 ("capital deployed",f"median margin ${tr.margin.median():.0f} = {tr.margin.median()/600:.0%} of start equity"),
]: print(f"  {k:<22}{v}")

print("\n"+"="*80); print("BY YEAR"); print("="*80)
print(f"  {'yr':<7}{'n':>4}{'win%':>7}{'P&L $':>10}{'return':>9}{'maxDD':>8}")
for y,g in tr.groupby(pd.to_datetime(tr.entry).dt.year):
    e=eq[eq.index.year==y]; d2=(e-e.cummax())/e.cummax()
    print(f"  {y:<7}{len(g):>4}{(g.pnl>0).mean():>6.1%}{g.pnl.sum():>+10.2f}"
          f"{e.iloc[-1]/e.iloc[0]-1:>+9.1%}{d2.min():>+8.1%}")

print("\n"+"="*80); print("COST BREAKDOWN"); print("="*80)
print(f"  gross credit collected   ${tr.credit.sum():>9,.2f}")
print(f"  paid at settlement       ${-tr.settle.sum():>9,.2f}")
print(f"  exchange fees            ${-tr.fees.sum():>9,.2f}   ({tr.fees.sum()/tr.credit.sum():.1%} of credit)")
print(f"  {'-'*42}")
print(f"  net P&L                  ${p.sum():>9,.2f}")
print(f"  fees / net P&L           {tr.fees.sum()/p.sum():>9.0%}")

print("\n"+"="*80); print("P&L DISTRIBUTION"); print("="*80)
print("  " + "  ".join(f"p{q}=${np.percentile(p,q):+.1f}" for q in (0,10,25,50,75,90,100)))
print(f"\n  -> {OUT}/final_trades.csv   ({len(tr)} rows)")
print(f"  -> {OUT}/final_equity.csv   ({len(eq)} daily points)")
print("\n"+"="*80); print("ALL TRADES"); print("="*80)
pd.set_option("display.width",240,"display.max_rows",200)
print(tr.to_string(index=False))
