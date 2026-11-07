#!/usr/bin/env python3
"""
Annual walk-forward on a wide panel using scikit-learn gradient-boosted trees
(HistGradientBoostingClassifier, FIXED settings, no tuning).  This is NOT the XGBoost + Bayesian-optimisation
model of the main study; it is the 'default settings' analogue, used to test whether the main conclusions
(no predictive edge; engine not distinguishable from random timing) hold on 54 stocks over 13 test years.

    python walk_forward_sklearn.py --data stocks_2010_2025.csv --test-years 2013-2025 --out results_wf_stocks
Universe rule: every ticker in the file whose history starts by --complete-by (default 2010-01-15), i.e. a complete calendar.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xai_stock_pipeline as P
import extra_analyses as rx

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", default="results_wf_stocks")
ap.add_argument("--test-years", default="2013-2025"); ap.add_argument("--n-random", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--drop", default="", help="comma-separated tickers with known unadjusted corporate actions to exclude"); ap.add_argument("--complete-by", default="2010-01-15")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
y0, y1 = (int(x) for x in a.test_years.split("-"))

def model(seed):
    return HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, l2_regularization=1.0,
                                          min_samples_leaf=200, random_state=seed)

raw = pd.read_csv(a.data, parse_dates=["date"]).sort_values(["ticker", "date"])
first = raw.groupby("ticker")["date"].min()
keep = first[first <= pd.Timestamp(a.complete_by)].index
keep = [t for t in keep if t not in set(x for x in a.drop.split(",") if x)]
raw = raw[raw.ticker.isin(keep)]
cnt_d = raw.groupby("date")["ticker"].nunique()
raw = raw[raw["date"].isin(cnt_d[cnt_d == raw["ticker"].nunique()].index)]      # common calendar: dates on which every stock traded
json.dump(dict(rule="all tickers in the file with history starting by 2010-01-15", n=len(keep), tickers=sorted(keep),
               dropped_late_start=sorted(set(first.index) - set(keep))), open(f"{a.out}/universe.json", "w"), indent=1)
panel = P.build_panel(raw)
print("stocks", len(keep), "panel", panel.shape, flush=True)

rows, tests = [], []
for Y in range(y0, y1 + 1):
    pre = panel[panel.date < pd.Timestamp(Y, 1, 1)].reset_index(drop=True)
    test = panel[(panel.date >= pd.Timestamp(Y, 1, 1)) & (panel.date <= pd.Timestamp(Y, 12, 31))].reset_index(drop=True)
    cnt = test.groupby("date")["ticker"].transform("nunique")          # keep only dates where every stock has an observation
    test = test[cnt == test.ticker.nunique()].reset_index(drop=True)
    m = model(a.seed).fit(pre[P.FEATURES], pre.y)
    p = m.predict_proba(test[P.FEATURES])[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300)).fit(pre[P.FEATURES], pre.y)   # logistic-regression baseline
    pl = lr.predict_proba(test[P.FEATURES])[:, 1]; mlr = P.clf_metrics(test.y, pl)
    met = P.clf_metrics(test.y, p)
    const_ll = log_loss(test.y, np.full(len(test), pre.y.mean())); maj = max(test.y.mean(), 1 - test.y.mean())
    # thresholds from the SAME model's out-of-fold probabilities (4 half-year folds before Y)
    oof = []
    for yy in (Y - 2, Y - 1):
        for m0, m1, d1 in ((1, 6, 30), (7, 12, 31)):
            s, e = pd.Timestamp(yy, m0, 1), pd.Timestamp(yy, m1, d1)
            tr, va = pre[pre.date < s], pre[(pre.date >= s) & (pre.date <= e)]
            oof.append(pd.Series(model(a.seed).fit(tr[P.FEATURES], tr.y).predict_proba(va[P.FEATURES])[:, 1]))
    oof = pd.concat(oof); buy, sell = float(oof.quantile(P.BUY_Q)), float(oof.quantile(P.SELL_Q))
    vcap = pre.groupby("ticker")["vol20"].quantile(P.VOL_CAP_Q).to_dict()
    test = test.assign(p=p)
    acts, pos, ret = {}, {}, {}
    for tk, g in test.groupby("ticker"):
        acts[tk], _ = P.decision_engine(g.p.values, g.vol20.values, buy, sell, vcap[tk])
        pos[tk] = P.positions_from_actions(acts[tk]); ret[tk] = g.next_ret.values
    _, o = P.portfolio(test, acts, P.COST_BPS)
    _, obh = P.portfolio(test, {tk: np.array(["BUY"] * len(g)) for tk, g in test.groupby("ticker")}, P.COST_BPS)
    rng = np.random.default_rng(rx.SEED + Y); par = {}
    for tk, q in pos.items():
        prev = np.concatenate([[0], q[:-1]])
        par[tk] = (len(q), int(q.sum()), int(((q == 1) & (prev == 0)).sum()), bool(q[-1] == 1))
    sims = np.array([rx.sharpe(rx.port_ret({tk: (rx.random_matched(T, L, k, e, rng) if L > 0 else np.zeros(T, int))
                                            for tk, (T, L, k, e) in par.items()}, ret)) for _ in range(a.n_random)])
    rows.append(dict(year=Y, n_stocks=test.ticker.nunique(), n_dates=test.date.nunique(), n_obs=len(test),
                     base_rate_up=test.y.mean(), majority_acc=maj, acc=met["acc"], auc=met["auc"], logloss=met["logloss"],
                     logloss_constant=const_ll, engine_sharpe=o["sharpe"], engine_exposure=o["avg_exposure"],
                     engine_ann_return=o["ann_return"], bh_sharpe=obh["sharpe"], bh_cum_return=obh["cum_return"],
                     bh_maxdd=obh["max_drawdown"], bh_vol=obh["ann_vol"], random_sharpe_mean=sims.mean(),
                     random_sharpe_sd=sims.std(ddof=1), engine_percentile=float((sims < o["sharpe"]).mean() * 100),
                     lr_acc=mlr["acc"], lr_auc=mlr["auc"], lr_logloss=mlr["logloss"]))
    tests.append(test); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/wf_by_year.csv", index=False)

Y_ = pd.DataFrame(rows); mv = Y_.bh_vol.median()
Y_["falling_or_volatile"] = (Y_.bh_cum_return < 0) | (Y_.bh_vol > 1.25 * mv)
Y_.to_csv(f"{a.out}/wf_by_year.csv", index=False)
Tt = pd.concat(tests); bs = P.date_bootstrap(Tt, {"m": Tt.p.values}, n=500)
pooled = dict(n_years=len(Y_), n_stocks=int(Y_.n_stocks.iloc[0]), years_auc_gt_0_5=int((Y_.auc > .5).sum()),
              years_acc_gt_majority=int((Y_.acc > Y_.majority_acc).sum()), years_logloss_lt_constant=int((Y_.logloss < Y_.logloss_constant).sum()),
              mean_auc=float(Y_.auc.mean()), sd_auc=float(Y_.auc.std(ddof=1)), mean_acc=float(Y_.acc.mean()),
              mean_engine_sharpe=float(Y_.engine_sharpe.mean()), mean_bh_sharpe=float(Y_.bh_sharpe.mean()),
              mean_engine_percentile=float(Y_.engine_percentile.mean()),
              years_engine_above_random_median=int((Y_.engine_percentile > 50).sum()),
              years_engine_sharpe_above_bh=int((Y_.engine_sharpe > Y_.bh_sharpe).sum()),
              falling_or_volatile_years=Y_.loc[Y_.falling_or_volatile, "year"].tolist(),
              pooled_auc_ci95=np.percentile(bs["m_auc"], [2.5, 97.5]).tolist(), pooled_acc_ci95=np.percentile(bs["m_acc"], [2.5, 97.5]).tolist())
json.dump(pooled, open(f"{a.out}/wf_pooled.json", "w"), indent=1)
print(Y_.round(3).to_string()); print(json.dumps(pooled, indent=1))
