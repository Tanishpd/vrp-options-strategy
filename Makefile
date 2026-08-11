PY := .venv/bin/python

.PHONY: fetch test baseline v2 validate-baseline validate-v2

fetch:            ## refresh data/ from Binance + Deribit (public endpoints)
	$(PY) scripts/fetch_data.py

test:             ## engine sanity checks (put-call parity, fees, margin, ...)
	$(PY) tests/test_engine.py

baseline:         ## reproduce the baseline result (writes results/final_*.csv)
	$(PY) scripts/run_baseline.py

v2:               ## reproduce the v2 result (writes results/v2_*.csv)
	$(PY) scripts/run_v2.py

validate-baseline: ## six-check battery on the baseline config
	$(PY) scripts/validate.py '{}'

validate-v2:      ## six-check battery on the v2 config
	$(PY) scripts/validate.py '{"max_concurrent_per_symbol":2,"wing_width_deltas":0.10,"skip_call_ret20":0.20,"symbols":["BTC","ETH"],"min_iv_rank_by_symbol":{"ETH":0.75},"risk_per_trade_by_symbol":{"ETH":0.02}}'
