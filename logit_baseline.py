#!/usr/bin/env python3
"""Logistic-regression baseline on the main ten-stock split.  Same features, same pre-test/test split."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_v] = "1"
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xai_stock_pipeline as P
raw = pd.read_csv("data/sp500_10stocks_daily_2013_2018.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel = P.build_panel(raw)
pre = panel[panel.date < pd.Timestamp(P.TEST_START)]; test = panel[panel.date >= pd.Timestamp(P.TEST_START)].reset_index(drop=True)
out = {}
for C in (1.0, 0.01):                                     # C=0.01: strongly regularised
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500)).fit(pre[P.FEATURES], pre.y)
    p = m.predict_proba(test[P.FEATURES])[:, 1]; met = P.clf_metrics(test.y, p)
    bs = P.date_bootstrap(test, {"m": p}, n=1000)
    out[f"C={C}"] = dict(acc=met["acc"], auc=met["auc"], logloss=met["logloss"],
                         acc_ci=np.percentile(bs["m_acc"], [2.5, 97.5]).tolist(), auc_ci=np.percentile(bs["m_auc"], [2.5, 97.5]).tolist())
out["majority_acc"] = float(max(test.y.mean(), 1 - test.y.mean())); out["constant_logloss"] = float(P.clf_metrics(test.y, np.full(len(test), pre.y.mean()))["logloss"])
os.makedirs("results_logit", exist_ok=True); json.dump(out, open("results_logit/logit_baseline.json", "w"), indent=1); print(json.dumps(out, indent=1))
