#!/bin/bash
# Steps 1 and 3: planted-signal test on the real pipeline, then the repaired-price rerun.  About 40 minutes.
set -e
cd "$(dirname "$0")"
echo "== planted signal, phi = 0.10 =="; python xai_stock_pipeline.py --data data/planted_phi0.10.csv --out results_planted_xgb
echo "== planted signal, null (phi = 0) =="; python xai_stock_pipeline.py --data data/planted_phi0.00.csv --out results_null_xgb
echo "== repair the four bad closes (needs internet) =="
python repair_prices.py --apply 2>/dev/null | tee repair_prices_output.txt
echo "== rerun main study on repaired prices =="; python xai_stock_pipeline.py --data data/sp500_10stocks_daily_2013_2018_repaired.csv --out results_repaired
echo "DONE quick"
