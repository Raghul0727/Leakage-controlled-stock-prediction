#!/usr/bin/env python3
"""
Determinism check.  Runs the FULL pipeline N times, single-threaded, fixed seeds,
and reports the mean and SD of the decision-engine Sharpe ratio and of the main test metrics.

    python repeat_runs.py                    # 5 identical runs  -> SD should be 0 if the pipeline is deterministic
    python repeat_runs.py --vary-seed        # 5 runs whose primary model seed is 0,1,2,3,4 -> seed sensitivity

Run it in a virtual environment with pinned versions (pip freeze > requirements-lock.txt).
Outputs: results_repeat/determinism_summary.json, determinism_runs.csv
"""
import argparse, json, subprocess, sys, hashlib, os
import pandas as pd, numpy as np

KEY = "Decision engine (XGB-BO), confidence + risk"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--vary-seed", action="store_true")
ap.add_argument("--base", default="results_repeat")
ap.add_argument("--extra", default="", help="extra args passed to xai_stock_pipeline.py, e.g. '--quick'")
a = ap.parse_args()

rows, hashes = [], []
for i in range(a.n):
    out = f"{a.base}/run{i + 1}"
    cmd = [sys.executable, "xai_stock_pipeline.py", "--out", out] + a.extra.split()
    if a.vary_seed:
        cmd += ["--seed-offset", str(i)]
    subprocess.run(cmd, check=True)
    bm = pd.read_csv(f"{out}/backtest_metrics.csv", index_col=0)
    ms = pd.read_csv(f"{out}/model_comparison_summary.csv")
    px = ms[(ms.model == "xgb") & (ms.variant == "bayes")].iloc[0]
    rows.append(dict(run=i + 1, sharpe=bm.loc[KEY, "sharpe"], exposure=bm.loc[KEY, "avg_exposure"],
                     xgb_bayes_acc=px.acc_mean, xgb_bayes_auc=px.auc_mean,
                     sharpe_buy_and_hold=bm.loc["Benchmark: buy & hold (equal weight)", "sharpe"]))
    hashes.append(hashlib.sha256(open(f"{out}/decision_log_with_explanations.csv", "rb").read()).hexdigest())
df = pd.DataFrame(rows)
os.makedirs(a.base, exist_ok=True)
df.to_csv(f"{a.base}/determinism_runs.csv", index=False)
summ = dict(n_runs=a.n, vary_seed=a.vary_seed,
            sharpe_mean=float(df.sharpe.mean()), sharpe_sd=float(df.sharpe.std(ddof=1)),
            sharpe_min=float(df.sharpe.min()), sharpe_max=float(df.sharpe.max()),
            exposure_mean=float(df.exposure.mean()), exposure_sd=float(df.exposure.std(ddof=1)),
            decision_logs_byte_identical=len(set(hashes)) == 1)
json.dump(summ, open(f"{a.base}/determinism_summary.json", "w"), indent=1)
print(df.to_string()); print(json.dumps(summ, indent=1))
