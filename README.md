# VRP Options Strategy

Systematic BTC/ETH options selling on Delta Exchange India, built around the one
edge that is actually measurable in this market: the **variance risk premium**
(implied volatility trading rich to subsequently realized volatility).

The strategy sells weekly ~25-delta strangles with protective wings (an iron
condor), **only when IV rank ≥ 0.60**, holds to Friday settlement, and skips the
call side after strong 20-day rallies. Everything is priced, costed, and
margined the way Delta Exchange India actually works.

## Data policy: no synthetic values

Nothing in this repo runs on generated or simulated market data. Two classes of
results exist, and they are never mixed:

- **Real**: spot (Binance), implied vol (Deribit DVOL), spreads/fees/margin
  (Delta India schedules and live quotes), and the option-chain snapshots
  accumulating in `data/live_chain/` via `scripts/collect_live_chain.py`.
- **Model-based** (clearly labeled, never headline "live" results): the
  historical backtest below prices options with Black-Scholes at the *real
  market IV* on each day — because Delta India publishes no historical option
  chains, actual traded premiums for 2021–2026 do not exist publicly (paid
  vendors such as Tardis.dev sell Deribit history). The collector exists so
  that, going forward, real quotes replace modeled premiums entirely.

## Results — real-data backtest (actual Deribit trades in, actual settlements out)

2021-09 → 2026-08, $600 start. Every dollar below comes from a real trade that
actually happened on Deribit and a real settlement price — no Black-Scholes,
no DVOL, no synthesized premium. This is the number to trust.

| | v2-Full (BTC + ETH) |
|---|---|
| CAGR | **+19.45%** |
| Max drawdown | −6.37% |
| Sharpe | 1.72 |
| Profit factor | 2.63 |
| Win rate | 82.4% |
| Trades | 170 (of 187 signal days; 17 had no real market, 1 sized to zero lots) |

Trade-level data: `results/real_trades.csv`, `results/real_equity.csv`.
Reproduce with `make real-backtest` (first run fetches ~2,500 real Deribit
trade-history responses, cached forever after in `data/deribit_cache/` —
expect 30–60 min cold, seconds warm). Independently checked with
`make verify-real` (lookahead/bookkeeping checks + a from-scratch
re-derivation of sampled trades against the raw cached API data).

### What the real data revealed about the model it replaced

The strategy was originally developed and validated on a **model-based**
backtest — real spot and real Deribit DVOL implied vol, but *synthesized*
option premiums via Black-Scholes (Delta India publishes no historical option
chains, so real historical Delta premiums for 2021–2026 don't exist to buy).
`make compare-real` measures the gap directly, matched trade-for-trade:

| | Modeled (BS @ DVOL) | Real (Deribit trades) | Real ÷ Model |
|---|---|---|---|
| CAGR | +24.6% | +19.45% | — |
| Max drawdown | −7.0% | −6.37% | — |
| Sharpe | 2.08 | 1.72 | — |
| Win rate | 81.4% | 82.4% | — |
| Credit collected per trade | — | — | **0.72–0.75×** |

The model overstated available credit by roughly 25–30%, almost entirely
because it prices every ~7-day weekly off the 30-day DVOL index with no term
structure and no skew — the exact model risk flagged before this measurement
existed. What survived: **the edge itself.** Win rate and drawdown held up
better than the CAGR gap alone would suggest — the model's error was in the
*size* of the premium on any given day, not in *which* days were worth
selling premium on. That second thing is what the IV-rank filter is actually
selecting for, and real prices confirm it.

The old model-based backtest is kept below for the research trail, but is no
longer the number to cite.

<details>
<summary>Model-based backtest (superseded — click to expand)</summary>

| | Baseline (BTC) | v2-BTC | v2-Full (+ETH) |
|---|---|---|---|
| CAGR | +10.4% | +17.0% | +24.6% |
| Max drawdown | −9.4% | −4.7% | −7.0% |
| Sharpe | 1.11 | 1.84 | 2.08 |
| Profit factor | 2.44 | 3.57 | 3.31 |
| Win rate | 75.0% | 83.1% | 81.4% |
| Trades | 84 | 89 | 177 |
| OOS 2024+ CAGR | +13.3% | +23.4% | +34.8% |

Trade-level data: `results/v2_trades.csv`, `results/v2_equity.csv` (and
`final_*.csv` for the baseline).
</details>

## Layout

```
vrp/            the strategy engine (single source of truth)
  engine.py     Black-Scholes pricing, delta-based strikes, delta-dependent
                bid-ask (calibrated to live Delta quotes), Delta India fees
                (0.03% of underlying notional, capped 10% of premium), real
                short-option margin, real Friday-12:00-UTC expiries, free
                expiry settlement, no artificial loss caps
scripts/
  fetch_data.py             spot (Binance) + implied vol (Deribit DVOL) → data/
  collect_live_chain.py     REAL Delta India option-chain snapshots →
                            data/live_chain/ (bid/ask/sizes/IVs/OI; run on a
                            schedule — this dataset can't be backfilled later)
  run_baseline.py           reproduce the model-based baseline exactly
  run_v2.py                 reproduce the model-based v2 exactly
  validate.py               six-check validation battery for model configs
  real_backtest.py          v2 on REAL Deribit trades + real settlements —
                            the headline result (writes results/real_*.csv)
  verify_real_backtest.py   adversarial spot-check of real_backtest.py:
                            bookkeeping closure, lookahead checks, a from-
                            scratch re-derivation of sampled trades
  compare_real_vs_modeled.py  matches real vs modeled trades and reports the
                            credit gap (see "What the real data revealed")
tests/
  test_engine.py    22 sanity checks: put-call parity, delta accuracy,
                    fee/margin correctness, economic invariants
data/
  BTC.csv, ETH.csv      daily spot + DVOL implied vol, 2021→
  deribit_cache/        cached raw Deribit API responses behind the real
                        backtest (real_trades + delivery prices; ~2,500 files)
  live_chain/           accumulating REAL Delta India chain snapshots
results/        trade logs and equity curves — real_*.csv is the one to trust,
                v2_*.csv / final_*.csv are the superseded model-based runs
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
make fetch          # refresh data/ from Binance + Deribit (public, no keys)
make test           # engine sanity checks — run before trusting anything
make real-backtest  # THE headline result: real Deribit trades + settlements
make verify-real    # adversarial spot-check of that result
make compare-real   # how much did the earlier model over/understate credit?
make collect        # one real Delta India chain snapshot (run on a schedule)
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

- **Venue transplant.** The real-data backtest fills against Deribit — the
  global BTC/ETH options price-setter and the only venue with a public trade
  history — not Delta India, the intended venue. Delta tracks Deribit closely
  but the books differ; this is real market data, just not the exact market.
- **Fills are a same-day directional VWAP**, not what a specific resting limit
  order would have achieved: each leg's price is the volume-weighted average
  of that day's real taker-sell (short legs) or taker-buy (wing legs) prints,
  one day after the signal (no lookahead). A real order might do better or
  worse than the day's average.
- **Coverage is 170/187 signal days (90.9%).** 17 were skipped for having no
  real two-sided market in any acceptable strike — this biases the sample
  toward more liquid days/strikes, in a direction that's hard to sign a priori.
- **Fees and spreads decide viability.** Delta's actual fee schedule (0.03%
  of *underlying* notional) is modelled; confirm your account's tier before
  trading. `make compare-real` already shows real credit runs ~25–30% below
  what a naive model would assume — margin for error is not large.
- **The call-skip rule's benefit is concentrated** in one 2024 episode. It is
  risk-reducing by construction, but its standalone edge is not statistically
  significant.
- **Next step is live paper trading** (3–6 months against real Delta quotes,
  logging real bid/ask fills) — this backtest is historical evidence, not a
  live result. At $600 the real-data strategy earns ~$115/yr; it starts
  justifying operational risk around $25k.
