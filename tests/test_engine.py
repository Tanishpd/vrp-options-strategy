"""Sanity checks. If these fail, no backtest result from this engine means anything."""
import math, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from vrp.engine import bs, delta, strike_for_delta, fee, margin_req, Config

ok = True
def chk(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}   {detail}")

print("Black-Scholes")
S, K, sig, T = 60000, 60000, 0.75, 7/365
c, p = bs(S,K,sig,T,"C"), bs(S,K,sig,T,"P")
chk("put-call parity (r=0)", abs((c-p)-(S-K)) < 1e-6, f"C-P={c-p:.4f}, S-K={S-K}")
chk("ATM call ~ 0.4*S*sig*sqrt(T)", abs(c - 0.4*S*sig*math.sqrt(T))/c < 0.02, f"{c:.0f} vs {0.4*S*sig*math.sqrt(T):.0f}")
chk("deep OTM call ~ 0", bs(S, S*3, sig, T, "C") < 1.0)
chk("T=0 call = intrinsic", abs(bs(70000,60000,sig,0,"C") - 10000) < 1e-9)
chk("monotone in vol", bs(S,K,0.9,T,"C") > bs(S,K,0.5,T,"C"))
chk("monotone in T", bs(S,K,sig,30/365,"C") > bs(S,K,sig,7/365,"C"))

print("\nDelta / strike selection")
chk("ATM call delta ~0.5", abs(delta(S,K,sig,T,"C")-0.5) < 0.06, f"{delta(S,K,sig,T,'C'):.3f}")
chk("put delta negative", delta(S,K,sig,T,"P") < 0)
for tgt in (0.10, 0.15, 0.25):
    Kc = strike_for_delta(S, sig, T, "C", tgt, 200.0)
    Kp = strike_for_delta(S, sig, T, "P", tgt, 200.0)
    dc, dp = abs(delta(S,Kc,sig,T,"C")), abs(delta(S,Kp,sig,T,"P"))
    chk(f"{tgt:.0%}-delta call strike {Kc:>7.0f}", abs(dc-tgt) < 0.02, f"actual delta {dc:.3f}")
    chk(f"{tgt:.0%}-delta put  strike {Kp:>7.0f}", abs(dp-tgt) < 0.02, f"actual delta {dp:.3f}")
chk("higher vol -> further OTM strike",
    strike_for_delta(S,1.2,T,"C",0.15,200.) > strike_for_delta(S,0.5,T,"C",0.15,200.),
    f"{strike_for_delta(S,1.2,T,'C',0.15,200.):.0f} vs {strike_for_delta(S,0.5,T,'C',0.15,200.):.0f}")

print("\nFees (Delta India: 0.03% of underlying notional, capped 10% of premium)")
cfg = Config()
prem, lots, cv = 1000.0, 23, 0.001
f = fee(prem, S, lots, cv, cfg)
chk("uses notional not premium", abs(f - 0.0003*S*lots*cv) < 1e-9, f"${f:.4f} on ${S*lots*cv:,.0f} notional")
f2 = fee(0.5, S, lots, cv, cfg)
chk("10%-of-premium cap binds on cheap options", abs(f2 - 0.10*0.5*lots*cv) < 1e-9, f"${f2:.6f}")

print("\nMargin (premium + max(10%*notional - OTM, 5%*notional))")
m = margin_req(1000.0, S, S*1.10, "C", 23, 0.001)
notional = S*23*0.001
chk("margin >> premium", m > 1000*23*0.001, f"${m:.2f} vs premium ${1000*23*0.001:.2f}")
chk("margin >= 5% notional floor", m >= 0.05*notional, f"${m:.2f} >= ${0.05*notional:.2f}")

print("\nEconomic invariants")
chk("selling a 15d strangle collects less than ATM straddle",
    2*bs(S,K,sig,T,"C") > bs(S,strike_for_delta(S,sig,T,"C",0.15,200.),sig,T,"C")
                        + bs(S,strike_for_delta(S,sig,T,"P",0.15,200.),sig,T,"P"))
Ks, Kl = strike_for_delta(S,sig,T,"C",0.15,200.), strike_for_delta(S,sig,T,"C",0.08,200.)
chk("credit spread credit < width", bs(S,Ks,sig,T,"C")-bs(S,Kl,sig,T,"C") < abs(Kl-Ks), f"credit {bs(S,Ks,sig,T,'C')-bs(S,Kl,sig,T,'C'):.0f} width {abs(Kl-Ks):.0f}")

print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
sys.exit(0 if ok else 1)
