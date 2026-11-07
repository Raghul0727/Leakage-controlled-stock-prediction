#!/usr/bin/env python3
"""
Planted-signal positive control.

Simulates ten stocks on the study's own calendar whose next-day return has a KNOWN weak predictable component
(AR(1) coefficient phi: r[t+1] = phi * r[t] + noise; phi = 0 is the null) and checks that the same features,
walk-forward tuning, importance check and decision engine DETECT it, and behave differently from the null.
If they do, the near-chance results on real prices are not an artefact of the pipeline.

Backends
  sklearn (default; runs anywhere): histogram gradient boosting with random-search tuning over the study's CV folds,
          logistic regression baseline, permutation importance in place of SHAP.
  Full pipeline (XGBoost + Bayesian optimisation + SHAP): write the simulated files and run the real pipeline on them:
          python planted_signal.py --write-csv data/planted_phi0.10.csv --phi 0.10 --seed 123
          python xai_stock_pipeline.py --data data/planted_phi0.10.csv --out results_planted_phi0.10
Usage
  python planted_signal.py --n-sim 10 --out results_planted
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.inspection import permutation_importance
import xai_stock_pipeline as P
import extra_analyses as rx

REAL = "data/sp500_10stocks_daily_2013_2018.csv"


def simulate(phi, seed, drift=0.0006):
    """Ten stocks on the real calendar. Daily vol of each stock copied from the real series."""
    real = pd.read_csv(REAL, parse_dates=["date"]).sort_values(["ticker", "date"])
    rng = np.random.default_rng(seed)
    frames = []
    for tk, g in real.groupby("ticker"):
        T = len(g); sig = g["close"].pct_change().std()
        e = rng.normal(0, sig, T)
        r = np.zeros(T)
        for t in range(1, T):
            r[t] = drift + phi * (r[t - 1] - drift) + np.sqrt(1 - phi ** 2) * e[t]      # keeps the unconditional vol at sig
        close = 100 * np.cumprod(1 + r)
        prev = np.concatenate([[100], close[:-1]])
        op = prev * (1 + rng.normal(0, 0.002, T))
        hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.004, T)))
        lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.004, T)))
        vol = rng.lognormal(15, 0.3, T).astype(int)
        frames.append(pd.DataFrame(dict(date=g["date"].values, ticker=tk, open=op, high=hi, low=lo, close=close, volume=vol)))
    return pd.concat(frames)


def hgb(params, seed):
    return HistGradientBoostingClassifier(random_state=seed, **params)


def sample_params(rng):
    return dict(max_depth=int(rng.integers(2, 6)), max_iter=int(rng.integers(30, 160)),
                learning_rate=float(np.exp(rng.uniform(np.log(0.005), np.log(0.2)))),
                min_samples_leaf=int(np.exp(rng.uniform(np.log(20), np.log(300)))),
                l2_regularization=float(np.exp(rng.uniform(np.log(0.01), np.log(30)))))


def cv_loss(params, pre, seed, oof=False):
    ls, o = [], []
    for tr_end, va_s, va_e in P.CV_FOLDS:
        tr = pre[pre.date < pd.Timestamp(tr_end)]
        va = pre[(pre.date >= pd.Timestamp(va_s)) & (pre.date <= pd.Timestamp(va_e))]
        p = hgb(params, seed).fit(tr[P.FEATURES], tr.y).predict_proba(va[P.FEATURES])[:, 1]
        ls.append(log_loss(va.y, p, labels=[0, 1])); o.append(pd.Series(p))
    return (np.mean(ls), pd.concat(o)) if oof else np.mean(ls)


def cv_const_loss(pre):
    ls = []
    for tr_end, va_s, va_e in P.CV_FOLDS:
        tr = pre[pre.date < pd.Timestamp(tr_end)]
        va = pre[(pre.date >= pd.Timestamp(va_s)) & (pre.date <= pd.Timestamp(va_e))]
        ls.append(log_loss(va.y, np.full(len(va), tr.y.mean()), labels=[0, 1]))
    return float(np.mean(ls))


def one_run(phi, seed, n_trials, n_random):
    panel = P.build_panel(simulate(phi, seed).sort_values(["ticker", "date"]))
    pre = panel[panel.date < pd.Timestamp(P.TEST_START)].reset_index(drop=True)
    test = panel[panel.date >= pd.Timestamp(P.TEST_START)].reset_index(drop=True)
    rng = np.random.default_rng(seed + 999)
    base = pre.y.mean(); const_ll = log_loss(test.y, np.full(len(test), base)); maj = max(test.y.mean(), 1 - test.y.mean())
    # default model (library defaults: fully grown, fast learning)
    dflt = dict(max_depth=None, max_iter=100, learning_rate=0.1, min_samples_leaf=20, l2_regularization=0.0)
    pd_ = hgb(dflt, seed).fit(pre[P.FEATURES], pre.y).predict_proba(test[P.FEATURES])[:, 1]
    # random-search tuning on the walk-forward folds
    trials = [(cv_loss(p, pre, seed), p) for p in (sample_params(rng) for _ in range(n_trials))]
    best_cv, bp = min(trials, key=lambda x: x[0])
    cv_const = cv_const_loss(pre)
    model = hgb(bp, seed).fit(pre[P.FEATURES], pre.y)
    pt = model.predict_proba(test[P.FEATURES])[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(pre[P.FEATURES], pre.y)
    pl = lr.predict_proba(test[P.FEATURES])[:, 1]
    # permutation importance (stand-in for SHAP) on the test set, and across three training windows
    def imp(m):
        r = permutation_importance(m, test[P.FEATURES], test.y, n_repeats=3, random_state=seed, scoring="neg_log_loss")
        return pd.Series(r.importances_mean, index=P.FEATURES)
    gi = imp(model); r1_rank = int(gi.rank(ascending=False)["r1"])
    wimps = []
    for end in ("2014-12-31", "2015-12-31", "2016-12-31"):
        sub = pre[pre.date <= pd.Timestamp(end)]
        wimps.append(imp(hgb(bp, seed).fit(sub[P.FEATURES], sub.y)))
    rho = pd.DataFrame(wimps).T.corr(method="spearman").values; win_stab = float(rho[np.triu_indices(3, 1)].mean())
    # decision engine on the tuned model, thresholds from its own out-of-fold probabilities
    _, oof = cv_loss(bp, pre, seed, oof=True)
    buy, sell = float(oof.quantile(P.BUY_Q)), float(oof.quantile(P.SELL_Q))
    vcap = pre.groupby("ticker")["vol20"].quantile(P.VOL_CAP_Q).to_dict()
    test = test.assign(p=pt)
    acts, pos, ret = {}, {}, {}
    for tk, g in test.groupby("ticker"):
        acts[tk], _ = P.decision_engine(g.p.values, g.vol20.values, buy, sell, vcap[tk])
        pos[tk] = P.positions_from_actions(acts[tk]); ret[tk] = g.next_ret.values
    _, o = P.portfolio(test, acts, P.COST_BPS)
    _, obh = P.portfolio(test, {tk: np.array(["BUY"] * len(g)) for tk, g in test.groupby("ticker")}, P.COST_BPS)
    par = {}
    for tk, q in pos.items():
        prev = np.concatenate([[0], q[:-1]]); par[tk] = (len(q), int(q.sum()), int(((q == 1) & (prev == 0)).sum()), bool(q[-1] == 1))
    rg = np.random.default_rng(seed)
    sims = np.array([rx.sharpe(rx.port_ret({tk: (rx.random_matched(T, L, k, e, rg) if L > 0 else np.zeros(T, int))
                                            for tk, (T, L, k, e) in par.items()}, ret)) for _ in range(n_random)])
    return dict(phi=phi, seed=seed, n_test=len(test), majority_acc=maj, const_logloss=const_ll,
                auc_default=roc_auc_score(test.y, pd_), auc_tuned=roc_auc_score(test.y, pt), auc_logit=roc_auc_score(test.y, pl),
                acc_tuned=float(((pt > .5) == test.y).mean()), logloss_default=log_loss(test.y, pd_), logloss_tuned=log_loss(test.y, pt),
                logloss_logit=log_loss(test.y, pl), cv_logloss_tuned=best_cv, cv_logloss_constant=cv_const,
                tuned_prob_sd=float(np.std(pt)), default_prob_sd=float(np.std(pd_)), tuned_learning_rate=bp["learning_rate"],
                tuned_max_iter=bp["max_iter"], r1_importance_rank=r1_rank, window_stability=win_stab,
                engine_sharpe=o["sharpe"], bh_sharpe=obh["sharpe"], engine_exposure=o["avg_exposure"],
                engine_percentile=float((sims < o["sharpe"]).mean() * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=10); ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=300); ap.add_argument("--out", default="results_planted")
    ap.add_argument("--phis", default="0,0.05,0.10"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-csv", default=None); ap.add_argument("--phi", type=float, default=0.10)
    a = ap.parse_args()
    if a.write_csv:
        os.makedirs(os.path.dirname(a.write_csv) or ".", exist_ok=True)
        simulate(a.phi, a.seed).to_csv(a.write_csv, index=False); print("wrote", a.write_csv); return
    os.makedirs(a.out, exist_ok=True); rows = []
    for phi in [float(x) for x in a.phis.split(",")]:
        for s in range(a.n_sim):
            rows.append(one_run(phi, 5000 + s, a.trials, a.n_random)); print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(f"{a.out}/planted_runs.csv", index=False)
    R = pd.DataFrame(rows)
    se = np.sqrt(0.25 / R.n_test)                                   # rough SE of an accuracy/AUC-type statistic under independence
    R["auc_detected"] = R.auc_tuned > 0.5 + 1.96 * se * 0.9
    summ = R.groupby("phi").agg(auc_tuned=("auc_tuned", "mean"), auc_tuned_sd=("auc_tuned", "std"), auc_logit=("auc_logit", "mean"),
        auc_default=("auc_default", "mean"), acc_tuned=("acc_tuned", "mean"), majority=("majority_acc", "mean"),
        ll_tuned_minus_const=("logloss_tuned", lambda s: float((s - R.loc[s.index, "const_logloss"]).mean())),
        cv_gain_vs_const=("cv_logloss_tuned", lambda s: float((R.loc[s.index, "cv_logloss_constant"] - s).mean())),
        tuned_prob_sd=("tuned_prob_sd", "mean"), default_prob_sd=("default_prob_sd", "mean"), tuned_lr=("tuned_learning_rate", "median"),
        r1_rank_median=("r1_importance_rank", "median"), window_stability=("window_stability", "mean"),
        engine_sharpe=("engine_sharpe", "mean"), bh_sharpe=("bh_sharpe", "mean"), engine_percentile=("engine_percentile", "mean"),
        share_auc_detected=("auc_detected", "mean"))
    summ.to_csv(f"{a.out}/planted_summary.csv"); print(summ.round(4).T.to_string())


if __name__ == "__main__":
    main()
