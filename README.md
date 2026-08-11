# VRP Options Strategy

Systematic BTC/ETH options selling on Delta Exchange India, built around the one
edge that is actually measurable in this market: the **variance risk premium**
(implied volatility trading rich to subsequently realized volatility).

The strategy sells weekly ~25-delta strangles with protective wings (an iron
condor), **only when IV rank ≥ 0.60**, holds to Friday settlement, and skips the
call side after strong 20-day rallies. Everything is priced, costed, and
margined the way Delta Exchange India actually works.

## Results (backtest, 2021-09 → 2026-08, $600 start)

| | Baseline (BTC) | v2-BTC | v2-Full (+ETH) |
|---|---|---|---|
| CAGR | +10.4% | +17.0% | **+24.6%** |
| Max drawdown | −9.4% | −4.7% | −7.0% |
| Sharpe | 1.11 | 1.84 | 2.08 |
| Profit factor | 2.44 | 3.57 | 3.31 |
| Win rate | 75.0% | 83.1% | 81.4% |
| Trades | 84 | 89 | 177 |
| OOS 2024+ CAGR | +13.3% | +23.4% | +34.8% |

Trade-level data: `results/v2_trades.csv`, `results/v2_equity.csv` (and
`final_*.csv` for the baseline).

## Layout

```
vrp/            the strategy engine (single source of truth)
  engine.py     Black-Scholes pricing, delta-based strikes, delta-dependent
                bid-ask (calibrated to live Delta quotes), Delta India fees
                (0.03% of underlying notional, capped 10% of premium), real
                short-option margin, real Friday-12:00-UTC expiries, free
                expiry settlement, no artificial loss caps
scripts/
  fetch_data.py     pull spot (Binance) + implied vol (Deribit DVOL) → data/
  run_baseline.py   reproduce the baseline result exactly
  run_v2.py         reproduce the v2 result exactly (writes results/v2_*.csv)
  validate.py       the six-check validation battery (see below)
tests/
  test_engine.py    22 sanity checks: put-call parity, delta accuracy,
                    fee/margin correctness, economic invariants
data/           BTC.csv, ETH.csv — daily spot + DVOL implied vol, 2021→
results/        trade logs and equity curves for baseline and v2
research/       the numbered studies that produced the strategy, in order:
  01_vrp_premium.py   is there a VRP at all, and where does it concentrate?
  02_decay_tenor.py   is the edge decaying? does it exist at weekly tenor?
  03_config_scan.py   structure/parameter scans on the honest engine
  04_v1_battery.py    OOS + bootstrap + null + stress for the v1 config
  05_export_v1.py     v1 trade export (pre-calendar; superseded)
  improve/            the six-avenue improvement search (agent-run studies)
  engine.py           compat shim → vrp/engine.py (keeps old studies runnable)
  data, results       symlinks → ../data, ../results (same reason)
archive/        the pre-audit codebase and its outputs. An audit confirmed
                63 defects (10 critical) and found every result in it invalid.
                Kept for reference only — nothing in it should be run.
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make fetch        # refresh data/ from Binance + Deribit (public, no keys)
make test         # engine sanity checks — run before trusting anything
make v2           # reproduce the headline result
make validate-v2  # full six-check battery on the v2 config
```

## Validation discipline

Every accepted config passed, in one pre-committed run of `scripts/validate.py`:

1. **Risk-adjusted beat** — CAGR/maxDD and Sharpe vs baseline (leverage that
   scales return and drawdown together does not count)
2. **Out-of-sample** — positive on 2024+ and 2026-only windows
3. **Bootstrap** — 95% CI on mean $/trade excludes zero (20k resamples)
4. **Cost stress** — still beats baseline with spreads ×2 and fees ×2
5. **Null test** — inverting the IV-rank filter must still destroy the P&L
   (it does: −8.2% CAGR, −57% maxDD — the filter *is* the strategy)
6. **Plateau** — neighboring parameter values behave similarly (no spikes)

The improvement search also produced two **clean negative results**, kept in
`research/improve/`: longer tenors lose more premium than they save in fees,
and sizing by IV rank is indistinguishable from noise (the inverse-sizing null
control "won" too). Flat 4% risk per trade stands.

## Known limitations — read before funding anything

- **Option prices are synthesized** from the DVOL 30-day ATM index via
  Black-Scholes. Term structure and skew are not captured; ~76% of v2 entries
  are 9.5-DTE positions priced at 30-day IV. This is the largest model risk.
- **n=177 trades**, and many configurations were examined during research.
  The one-shot discipline limits but does not eliminate selection effects.
- **Fees and spreads decide viability.** At double fees the baseline goes
  negative. Delta's actual fee schedule (0.03% of *underlying* notional) is
  modelled; confirm your account's tier before trading.
- **The call-skip rule's benefit is concentrated** in one 2024 episode. It is
  risk-reducing by construction, but its standalone edge is not statistically
  significant.
- **Next step is live paper trading** (3–6 months against real Delta quotes,
  logging real bid/ask fills) — not deployment. At $600 the strategy earns
  ~$150/yr; it starts justifying operational risk around $25k.
