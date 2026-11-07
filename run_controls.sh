#!/bin/bash
# Step 2: 100 shuffled-label SHAP controls.  About 1-2 hours.
set -e
cd "$(dirname "$0")"
python xai_stock_pipeline.py --out results_100controls --n-controls 100
echo "DONE controls"
