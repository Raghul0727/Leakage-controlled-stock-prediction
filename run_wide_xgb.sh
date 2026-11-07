#!/bin/bash
# tuned XGBoost + Bayesian optimisation, annual walk-forward, on the cleaned wide panels.
# Runs both panels in parallel (single-threaded each).  Several hours; leave it running.
cd "$(dirname "$0")"
python make_clean.py
python walk_forward_annual.py --source csv --data data/us2010_clean.csv --test-years 2013-2025 --trials ${TRIALS:-10} --out results_wf_xgb_us > wide_us.log 2>&1 &
python walk_forward_annual.py --source csv --data data/nifty_clean.csv --test-years 2013-2025 --trials ${TRIALS:-10} --out results_wf_xgb_nifty > wide_nifty.log 2>&1 &
wait
echo "DONE wide"
