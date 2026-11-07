#!/usr/bin/env python3
"""
Explainable ML framework for next-day stock-movement prediction and decision support.

Pipeline:  Prediction -> Optimisation -> Explanation -> Decision (-> cost-aware backtest)

Usage
-----
    python xai_stock_pipeline.py                      # uses bundled CSV (reproduces the paper)
    python xai_stock_pipeline.py --source yfinance    # re-download fresh data (needs internet)
    python xai_stock_pipeline.py --quick              # tiny smoke test (few trials, 1 seed)

Everything is written to ./results (tables as CSV/JSON, figures as PNG).
Design rules that guard against the usual pitfalls:
  * every feature at day t uses data up to and including day t only
  * the label is the sign of the NEXT day's return
  * chronological split; the test set (>= 2017-01-01) is never used for tuning,
    threshold selection or model selection
  * hyper-parameters are chosen with expanding-window walk-forward CV on pre-test data
"""
import argparse, json, os, sys, time, warnings
# Determinism: force single-threaded numerics BEFORE numpy / xgboost / sklearn are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna, shap, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss, f1_score,
                             matthews_corrcoef)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------- config
TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ", "WMT", "PG", "KO", "CSCO"]
TEST_START = "2017-01-01"                       # untouched test set starts here
CV_FOLDS = [                                    # expanding-window walk-forward folds (pre-test only)
    ("2014-12-31", "2015-01-01", "2015-06-30"),
    ("2015-06-30", "2015-07-01", "2015-12-31"),
    ("2015-12-31", "2016-01-01", "2016-06-30"),
    ("2016-06-30", "2016-07-01", "2016-12-31"),
]  # (train_end, val_start, val_end)
COST_BPS = 5.0                                  # per side, applied to |change in position|
BUY_Q, SELL_Q = 0.70, 0.30                      # probability quantiles taken from OOF validation preds
VOL_CAP_Q = 0.75                                # high-risk = 20d volatility above train 75th percentile
OUT = "results"
N_JOBS = 1                                      # single-threaded: removes thread-order float differences

FEATURES = ["r1", "r2", "r3", "r4", "r5", "mom5", "mom10", "mom20",
            "c_sma5", "c_sma10", "c_sma20", "sma5_20", "macd", "macd_sig", "macd_hist",
            "rsi14", "bb_pctb", "bb_width", "vol10", "vol20", "atr14", "stoch_k",
            "hl_range", "vol_chg"]


# ----------------------------------------------------------------------------- data
def load_prices(source, path):
    if source == "yfinance":
        import yfinance as yf
        frames = []
        for t in TICKERS:
            d = yf.download(t, start="2013-02-01", end="2018-02-10", auto_adjust=False,
                            progress=False)
            d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
            d = d.reset_index().rename(columns={"Date": "date", "date": "date"})
            d["ticker"] = t
            frames.append(d[["date", "ticker", "open", "high", "low", "close", "volume"]])
        df = pd.concat(frames)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(g):
    g = g.sort_values("date").copy()
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    ret = c.pct_change()
    for k in range(1, 6):
        g[f"r{k}"] = ret.shift(k - 1)
    for k in (5, 10, 20):
        g[f"mom{k}"] = c / c.shift(k) - 1
    s5, s10, s20 = (c.rolling(w).mean() for w in (5, 10, 20))
    g["c_sma5"], g["c_sma10"], g["c_sma20"] = c / s5 - 1, c / s10 - 1, c / s20 - 1
    g["sma5_20"] = s5 / s20 - 1
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    g["macd"] = (ema12 - ema26) / c
    g["macd_sig"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_sig"]
    g["rsi14"] = _rsi(c)
    sd20 = c.rolling(20).std()
    g["bb_pctb"] = (c - s20) / (2 * sd20)
    g["bb_width"] = 4 * sd20 / s20
    g["vol10"], g["vol20"] = ret.rolling(10).std(), ret.rolling(20).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14).mean() / c
    g["stoch_k"] = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())
    g["hl_range"] = (h - l) / c
    g["vol_chg"] = np.log(v / v.rolling(20).mean())
    g["next_ret"] = c.shift(-1) / c - 1
    g["y"] = (g["next_ret"] > 0).astype(float)
    g.loc[g["next_ret"].isna(), "y"] = np.nan
    return g


def build_panel(df):
    panel = pd.concat([make_features(g) for _, g in df.groupby("ticker")])
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["y", "next_ret"])
    return panel.reset_index(drop=True)


# ----------------------------------------------------------------------------- models
def make_model(name, params, seed):
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=N_JOBS, **params)
    return xgb.XGBClassifier(random_state=seed, n_jobs=N_JOBS, tree_method="hist",
                             eval_metric="logloss", verbosity=0, **params)


DEFAULTS = {"rf": dict(n_estimators=100, min_samples_leaf=1),
            "xgb": dict(n_estimators=100, max_depth=6, learning_rate=0.3)}   # library defaults


def suggest(name, trial):
    if name == "rf":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 200),
                    max_depth=trial.suggest_int("max_depth", 3, 12),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                    max_features=trial.suggest_float("max_features", 0.15, 0.6))
    return dict(n_estimators=trial.suggest_int("n_estimators", 50, 400),
                max_depth=trial.suggest_int("max_depth", 2, 7),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_weight=trial.suggest_float("min_child_weight", 1, 60, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                gamma=trial.suggest_float("gamma", 0.0, 5.0))


def cv_logloss(name, params, pre, seed, return_oof=False, folds=None):
    losses, oof = [], []
    for tr_end, va_start, va_end in (folds or CV_FOLDS):
        # one-day purge: the last training label depends on the first validation close
        tr = pre[pre["date"] < pd.Timestamp(tr_end)]
        va = pre[(pre["date"] >= pd.Timestamp(va_start)) & (pre["date"] <= pd.Timestamp(va_end))]
        m = make_model(name, params, seed).fit(tr[FEATURES], tr["y"])
        p = np.clip(m.predict_proba(va[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
        losses.append(log_loss(va["y"], p))
        if return_oof:
            oof.append(pd.Series(p, index=va.index))
    return (np.mean(losses), pd.concat(oof)) if return_oof else np.mean(losses)


def tune(name, pre, sampler_name, n_trials, seed, folds=None):
    sampler = (optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(5, n_trials // 4))
               if sampler_name == "bayes" else optuna.samplers.RandomSampler(seed=seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda t: cv_logloss(name, suggest(name, t), pre, seed, folds=folds), n_trials=n_trials)
    return study


# ----------------------------------------------------------------------------- metrics
def clf_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, pred), auc=roc_auc_score(y, p), logloss=log_loss(y, p),
                f1=f1_score(y, pred), mcc=matthews_corrcoef(y, pred))


def date_bootstrap(test, probs, n=1000, seed=0, base=None):
    """Cluster bootstrap over trading dates (keeps cross-sectional dependence)."""
    rng = np.random.default_rng(seed)
    dates = test["date"].unique()
    idx_by_date = {d: np.where(test["date"].values == d)[0] for d in dates}
    y = test["y"].values
    res = []
    for _ in range(n):
        s = rng.choice(dates, size=len(dates), replace=True)
        ix = np.concatenate([idx_by_date[d] for d in s])
        row = {}
        for k, p in probs.items():
            row[k + "_acc"] = accuracy_score(y[ix], (p[ix] >= .5).astype(int))
            row[k + "_auc"] = roc_auc_score(y[ix], p[ix])
        res.append(row)
    return pd.DataFrame(res)


# ----------------------------------------------------------------------------- decision engine + backtest
def decision_engine(p, vol20, buy_thr, sell_thr, vol_cap):
    """Transparent rule set.  BUY: p>=buy_thr and risk not high.  SELL: p<=sell_thr.  HOLD: else."""
    action = np.where(p <= sell_thr, "SELL", np.where((p >= buy_thr) & (vol20 <= vol_cap), "BUY", "HOLD"))
    risk = np.where(vol20 > vol_cap, "HIGH", "NORMAL")
    return action, risk


def positions_from_actions(actions):
    """BUY -> long(1); SELL -> flat(0); HOLD -> keep previous position (start flat)."""
    pos, cur = [], 0
    for a in actions:
        cur = 1 if a == "BUY" else 0 if a == "SELL" else cur
        pos.append(cur)
    return np.array(pos)


def strat_returns(df_t, pos, cost_bps):
    prev = np.concatenate([[0], pos[:-1]])
    gross = pos * df_t["next_ret"].values
    cost = cost_bps / 1e4 * np.abs(pos - prev)
    return pd.Series(gross - cost, index=df_t["date"].values), np.abs(pos - prev).sum(), pos.mean()


def perf(r):
    r = pd.Series(r).dropna()
    ann_ret, ann_vol = r.mean() * 252, r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return dict(ann_return=ann_ret, ann_vol=ann_vol, sharpe=ann_ret / ann_vol if ann_vol > 0 else np.nan,
                max_drawdown=mdd, cum_return=cum.iloc[-1] - 1, win_rate=(r > 0).mean())


def portfolio(test, actions_by_key, cost_bps):
    """Equal-weight portfolio of per-ticker long/flat strategies."""
    rets, turn, expo = [], 0, []
    for tk, g in test.groupby("ticker"):
        pos = positions_from_actions(actions_by_key[tk])
        r, t, e = strat_returns(g, pos, cost_bps)
        rets.append(r.rename(tk)); turn += t; expo.append(e)
    R = pd.concat(rets, axis=1).mean(axis=1)
    out = perf(R)
    out["turnover_per_stock_per_year"] = turn / len(rets) / (test["date"].nunique() / 252)
    out["avg_exposure"] = float(np.mean(expo))
    return R, out


def block_bootstrap_sharpe(r, ref=None, n=2000, block=10, seed=0):
    rng = np.random.default_rng(seed)
    r = np.asarray(r); T = len(r)
    ref = None if ref is None else np.asarray(ref)
    out = []
    for _ in range(n):
        starts = rng.integers(0, T - block + 1, size=int(np.ceil(T / block)))
        ix = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        a = r[ix]; sh = a.mean() / a.std() * np.sqrt(252)
        if ref is not None:
            b = ref[ix]; sh = sh - b.mean() / b.std() * np.sqrt(252)
        out.append(sh)
    return np.percentile(out, [2.5, 97.5])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="csv", choices=["csv", "yfinance"])
    ap.add_argument("--data", default="data/sp500_10stocks_daily_2013_2018.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--trials-xgb", type=int, default=30)
    ap.add_argument("--trials-rf", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="model seeds are seed_offset .. seed_offset+seeds-1; the primary model uses seed_offset")
    ap.add_argument("--n-controls", type=int, default=10, help="number of shuffled-label SHAP controls")
    ap.add_argument("--n-random", type=int, default=1000, help="random exposure-matched strategies")
    a = ap.parse_args()
    global OUT
    OUT = a.out
    S0 = a.seed_offset                      # primary seed (designated in advance)
    if a.quick:
        a.seeds, a.trials_xgb, a.trials_rf, a.n_controls, a.n_random = 1, 6, 4, 2, 50
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    panel = build_panel(load_prices(a.source, a.data))
    pre = panel[panel["date"] < pd.Timestamp(TEST_START)].reset_index(drop=True)
    test = panel[panel["date"] >= pd.Timestamp(TEST_START)].reset_index(drop=True)
    info = dict(n_total=len(panel), n_pre=len(pre), n_test=len(test),
                pre_range=[str(pre.date.min().date()), str(pre.date.max().date())],
                test_range=[str(test.date.min().date()), str(test.date.max().date())],
                test_base_rate_up=float(test.y.mean()), pre_base_rate_up=float(pre.y.mean()))
    log(f"panel: {info}")

    # -------- 1) prediction + 2) optimisation: default vs random search vs Bayesian (TPE)
    rows, trial_rows, best_params, test_probs = [], [], {}, {}
    n_trials = {"rf": a.trials_rf, "xgb": a.trials_xgb}
    for name in ("rf", "xgb"):
        for seed in range(S0, S0 + a.seeds):
            # default
            m = make_model(name, DEFAULTS[name], seed).fit(pre[FEATURES], pre["y"])
            p = m.predict_proba(test[FEATURES])[:, 1]
            test_probs[(name, "default", seed)] = p
            rows.append(dict(model=name, variant="default", seed=seed,
                             cv_logloss=cv_logloss(name, DEFAULTS[name], pre, seed), **clf_metrics(test.y, p)))
            for samp in ("random", "bayes"):
                st = tune(name, pre, samp, n_trials[name], seed)
                bp = st.best_params
                best_params[(name, samp, seed)] = bp
                for t in st.trials:
                    trial_rows.append(dict(model=name, sampler=samp, seed=seed, trial=t.number, value=t.value))
                m = make_model(name, bp, seed).fit(pre[FEATURES], pre["y"])
                p = m.predict_proba(test[FEATURES])[:, 1]
                test_probs[(name, samp, seed)] = p
                rows.append(dict(model=name, variant=samp, seed=seed, cv_logloss=st.best_value,
                                 **clf_metrics(test.y, p)))
                log(f"{name:3s} {samp:6s} seed{seed}: cv_ll={st.best_value:.4f} "
                    f"test acc={rows[-1]['acc']:.4f} auc={rows[-1]['auc']:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/model_comparison_by_seed.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(f"{OUT}/bo_trials.csv", index=False)
    agg = res.groupby(["model", "variant"])[["cv_logloss", "acc", "auc", "logloss", "f1", "mcc"]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.reset_index().to_csv(f"{OUT}/model_comparison_summary.csv", index=False)
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in best_params.items()},
              open(f"{OUT}/best_hyperparameters.json", "w"), indent=1, default=float)

    # baselines for reference
    base_up = test.y.mean()
    maj_acc = max(base_up, 1 - base_up)
    always_up = dict(acc=base_up)
    # primary models (seed 0, fixed in advance; no test-based selection)
    prim = {"rf_default": test_probs[("rf", "default", S0)], "rf_bayes": test_probs[("rf", "bayes", S0)],
            "xgb_default": test_probs[("xgb", "default", S0)], "xgb_bayes": test_probs[("xgb", "bayes", S0)],
            "xgb_random": test_probs[("xgb", "random", S0)], "rf_random": test_probs[("rf", "random", S0)]}
    bs = date_bootstrap(test, prim, n=1000 if not a.quick else 50)
    ci = {}
    for k in prim:
        ci[k] = dict(acc_ci=np.percentile(bs[k + "_acc"], [2.5, 97.5]).tolist(),
                     auc_ci=np.percentile(bs[k + "_auc"], [2.5, 97.5]).tolist())
    diffs = {}
    for m in ("rf", "xgb"):
        for ref in ("default", "random"):
            d_acc = bs[f"{m}_bayes_acc"] - bs[f"{m}_{ref}_acc"]
            d_auc = bs[f"{m}_bayes_auc"] - bs[f"{m}_{ref}_auc"]
            diffs[f"{m}_bayes_minus_{ref}"] = dict(
                d_acc=float(accuracy_score(test.y, (prim[f'{m}_bayes'] >= .5)) - accuracy_score(test.y, (prim[f'{m}_{ref}'] >= .5))),
                d_acc_ci=np.percentile(d_acc, [2.5, 97.5]).tolist(),
                d_auc=float(roc_auc_score(test.y, prim[f'{m}_bayes']) - roc_auc_score(test.y, prim[f'{m}_{ref}'])),
                d_auc_ci=np.percentile(d_auc, [2.5, 97.5]).tolist())
    binom = {}
    for k, p in prim.items():
        pred = (p >= .5).astype(int)
        hits = int((pred == test.y.values).sum())
        binom[k] = float(stats.binomtest(hits, len(test), maj_acc, alternative="greater").pvalue)
    json.dump(dict(info=info, majority_acc=maj_acc, ci=ci, diffs=diffs, binom_p_vs_majority=binom),
              open(f"{OUT}/test_inference.json", "w"), indent=1, default=float)

    # -------- 3) explainability (primary tuned XGBoost, seed 0)
    log("SHAP analysis")
    bp = best_params[("xgb", "bayes", S0)]
    final = make_model("xgb", bp, S0).fit(pre[FEATURES], pre["y"])
    expl = shap.TreeExplainer(final)
    sv = expl.shap_values(test[FEATURES])
    gl = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    gl.to_csv(f"{OUT}/shap_global_importance.csv", header=["mean_abs_shap"])

    # (a) consistency across consecutive test sub-periods (frozen model)
    dates = np.sort(test["date"].unique())
    blocks = np.array_split(dates, 4)
    block_imp = {}
    for i, b in enumerate(blocks):
        mask = test["date"].isin(b).values
        block_imp[f"B{i+1} ({pd.Timestamp(b[0]).date()}..{pd.Timestamp(b[-1]).date()})"] = np.abs(sv[mask]).mean(0)
    BI = pd.DataFrame(block_imp, index=FEATURES)
    BI.to_csv(f"{OUT}/shap_importance_by_block.csv")
    rho_blocks = BI.corr(method="spearman")
    rho_blocks.to_csv(f"{OUT}/shap_block_spearman.csv")
    top5 = {c: set(BI[c].nlargest(5).index) for c in BI}
    cols = list(BI.columns)
    jacc = [len(top5[x] & top5[y]) / len(top5[x] | top5[y]) for i, x in enumerate(cols) for y in cols[i + 1:]]
    tri = rho_blocks.values[np.triu_indices(len(cols), 1)]

    # (b) consistency across different training windows (same hyper-parameters, same test data)
    win_imp = {}
    for end in ("2014-12-31", "2015-12-31", "2016-12-31"):
        sub = pre[pre["date"] <= pd.Timestamp(end)]
        mm = make_model("xgb", bp, S0).fit(sub[FEATURES], sub["y"])
        win_imp[f"train<= {end}"] = np.abs(shap.TreeExplainer(mm).shap_values(test[FEATURES])).mean(0)
    WI = pd.DataFrame(win_imp, index=FEATURES)
    WI.to_csv(f"{OUT}/shap_importance_by_training_window.csv")
    rho_win = WI.corr(method="spearman")

    # (c) does tuning change the explanation?  default vs tuned; XGB vs RF; SHAP vs permutation importance
    md = make_model("xgb", DEFAULTS["xgb"], S0).fit(pre[FEATURES], pre["y"])
    imp_def = np.abs(shap.TreeExplainer(md).shap_values(test[FEATURES])).mean(0)
    rf_b = make_model("rf", best_params[("rf", "bayes", S0)], S0).fit(pre[FEATURES], pre["y"])
    rsv = shap.TreeExplainer(rf_b).shap_values(test[FEATURES])
    rsv = rsv[..., 1] if np.ndim(rsv) == 3 else rsv
    imp_rf = np.abs(rsv).mean(0)
    pi = permutation_importance(final, test[FEATURES], test["y"], scoring="neg_log_loss", n_repeats=10,
                                random_state=0, n_jobs=1)
    sp = lambda x, y: float(stats.spearmanr(x, y)[0])
    extra = dict(spearman_default_vs_tuned_xgb=sp(imp_def, gl[FEATURES].values),
                 spearman_xgb_vs_rf_tuned=sp(imp_rf, gl[FEATURES].values),
                 spearman_shap_vs_permutation=sp(pi.importances_mean, gl[FEATURES].values))

    # (d) negative controls: SAME hyper-parameters, but TRAINING LABELS PERMUTED (no real signal).
    # Stability across test blocks of ONE frozen model is expected of any model (the explanations
    # are computed on the same features), so the informative comparison is stability across
    # retraining windows and across model seeds.  We therefore train n_controls shuffled-label
    # models (different permutation seeds) and report each statistic as a distribution.
    def imp_of(m):
        return np.abs(shap.TreeExplainer(m).shap_values(test[FEATURES])).mean(0)

    def mean_pair_rho(imps):
        M = pd.DataFrame(imps).T.corr(method="spearman").values      # .T: correlate the importance VECTORS, not the features
        return float(M[np.triu_indices(len(imps), 1)].mean())

    WINDOWS = ("2014-12-31", "2015-12-31", "2016-12-31")
    N_SEEDS_STAB = 3
    real_win = mean_pair_rho(list(win_imp.values()))
    real_seed = mean_pair_rho([imp_of(make_model("xgb", bp, S0 + k).fit(pre[FEATURES], pre["y"]))
                               for k in range(N_SEEDS_STAB)])
    ctrl_rows, ctrl_imp0 = [], None
    for c in range(a.n_controls):
        ypm = pre["y"].sample(frac=1, random_state=1000 + c).values       # permuted training labels
        w_imps = []
        for end in WINDOWS:
            msk = (pre["date"] <= pd.Timestamp(end)).values
            w_imps.append(imp_of(make_model("xgb", bp, S0).fit(pre.loc[msk, FEATURES], ypm[msk])))
        s_imps = [imp_of(make_model("xgb", bp, S0 + k).fit(pre[FEATURES], ypm)) for k in range(N_SEEDS_STAB)]
        mc = make_model("xgb", bp, S0).fit(pre[FEATURES], ypm)
        svc = shap.TreeExplainer(mc).shap_values(test[FEATURES])
        b_imps = [np.abs(svc[test["date"].isin(b).values]).mean(0) for b in blocks]
        ctrl_rows.append(dict(control=c, block_spearman=mean_pair_rho(b_imps),
                              window_spearman=mean_pair_rho(w_imps), seed_spearman=mean_pair_rho(s_imps),
                              test_auc=float(roc_auc_score(test.y, mc.predict_proba(test[FEATURES])[:, 1]))))
        if c == 0:
            ctrl_imp0 = b_imps
    CT = pd.DataFrame(ctrl_rows)
    CT.to_csv(f"{OUT}/shap_control_distribution.csv", index=False)
    dist = lambda col: dict(mean=float(CT[col].mean()), sd=float(CT[col].std(ddof=1)) if len(CT) > 1 else 0.0,
                            min=float(CT[col].min()), max=float(CT[col].max()))
    ctrl_summary = dict(n_controls=int(len(CT)),
                        real_block_spearman=float(tri.mean()),
                        real_window_spearman=real_win, real_seed_spearman=real_seed,
                        control_block_spearman=dist("block_spearman"),
                        control_window_spearman=dist("window_spearman"),
                        control_seed_spearman=dist("seed_spearman"),
                        control_test_auc=dist("test_auc"),
                        share_controls_window_ge_real=float((CT["window_spearman"] >= real_win).mean()),
                        share_controls_seed_ge_real=float((CT["seed_spearman"] >= real_seed).mean()))

    json.dump(dict(top5_global=gl.head(5).index.tolist(),
                   block_spearman_mean=float(tri.mean()), block_spearman_min=float(tri.min()),
                   block_top5_jaccard_mean=float(np.mean(jacc)),
                   window_spearman_mean=float(rho_win.values[np.triu_indices(3, 1)].mean()),
                   window_spearman_min=float(rho_win.values[np.triu_indices(3, 1)].min()),
                   control_block_spearman_mean=float(CT["block_spearman"].mean()),
                   control_test_auc=float(CT["test_auc"].iloc[0]),
                   controls=ctrl_summary, **extra), open(f"{OUT}/shap_consistency_summary.json", "w"), indent=1)

    # figures: SHAP summary + bar
    plt.figure(figsize=(7, 5.5))
    shap.summary_plot(sv, test[FEATURES], show=False, max_display=15, plot_size=None)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_summary.png", dpi=200); plt.close()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    gl.head(12)[::-1].plot.barh(ax=ax, color="#1B2A57")
    ax.set_xlabel("mean |SHAP value| (log-odds of 'up')"); ax.set_title("Global feature importance (test set)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_bar.png", dpi=200); plt.close()

    # -------- 4) decision support + backtest
    log("decision engine & backtest")
    _, oof = cv_logloss("xgb", bp, pre, S0, return_oof=True)
    buy_thr, sell_thr = float(oof.quantile(BUY_Q)), float(oof.quantile(SELL_Q))

    def oof_thr(name, params):                   # each variant gets ITS OWN out-of-fold thresholds
        _, o = cv_logloss(name, params, pre, S0, return_oof=True)
        return float(o.quantile(BUY_Q)), float(o.quantile(SELL_Q))
    thr = {"p_xgb_bayes": (buy_thr, sell_thr),
           "p_xgb_default": oof_thr("xgb", DEFAULTS["xgb"]),
           "p_rf_bayes": oof_thr("rf", best_params[("rf", "bayes", S0)])}
    vol_cap = pre.groupby("ticker")["vol20"].quantile(VOL_CAP_Q).to_dict()
    test = test.assign(p_xgb_bayes=prim["xgb_bayes"], p_xgb_default=prim["xgb_default"],
                       p_rf_bayes=prim["rf_bayes"])

    def run(pcol, buy=None, sell=None, use_risk=True, cost=COST_BPS):
        if buy is None:
            buy, sell = thr[pcol]
        acts = {}
        for tk, g in test.groupby("ticker"):
            cap = vol_cap[tk] if use_risk else np.inf
            acts[tk], _ = decision_engine(g[pcol].values, g["vol20"].values, buy, sell, cap)
        return portfolio(test, acts, cost)

    results, curves = {}, {}
    for label, kw in {
        "Decision engine (XGB-BO), confidence + risk": dict(pcol="p_xgb_bayes"),
        "Decision engine (XGB-BO), confidence only": dict(pcol="p_xgb_bayes", use_risk=False),
        "Decision engine (XGB-default), confidence + risk": dict(pcol="p_xgb_default"),
        "Decision engine (XGB-default), confidence + risk, TUNED-model thresholds (as in first submission)":
            dict(pcol="p_xgb_default", buy=buy_thr, sell=sell_thr),
        "Decision engine (RF-BO), confidence + risk": dict(pcol="p_rf_bayes"),
        "Naive rule: long if p>0.5 else flat (XGB-BO)": dict(pcol="p_xgb_bayes", buy=0.5, sell=0.4999999, use_risk=False),
    }.items():
        R, o = run(**kw); results[label] = o; curves[label] = R

    # SMA-20 trend rule (non-ML benchmark) and buy&hold
    acts = {tk: np.where(g["c_sma20"].values > 0, "BUY", "SELL") for tk, g in test.groupby("ticker")}
    R, o = portfolio(test, acts, COST_BPS); results["Benchmark: price > SMA20 (trend rule)"] = o; curves["Benchmark: price > SMA20 (trend rule)"] = R
    acts = {tk: np.array(["BUY"] * len(g)) for tk, g in test.groupby("ticker")}
    R, o = portfolio(test, acts, COST_BPS); results["Benchmark: buy & hold (equal weight)"] = o; curves["Benchmark: buy & hold (equal weight)"] = R
    BH = curves["Benchmark: buy & hold (equal weight)"]
    main_key = "Decision engine (XGB-BO), confidence + risk"
    pf = pd.DataFrame(results).T
    lo, hi = block_bootstrap_sharpe(curves[main_key].values)
    dlo, dhi = block_bootstrap_sharpe(curves[main_key].values, BH.values)
    pf["sharpe_ci_low"] = np.nan; pf["sharpe_ci_high"] = np.nan
    pf.loc[main_key, ["sharpe_ci_low", "sharpe_ci_high"]] = [lo, hi]
    pf.to_csv(f"{OUT}/backtest_metrics.csv")
    json.dump(dict(buy_threshold=buy_thr, sell_threshold=sell_thr, thresholds_by_variant=thr,
                   sharpe_diff_vs_bh_ci=[float(dlo), float(dhi)], sharpe_ci=[float(lo), float(hi)],
                   action_share={k: float(v) for k, v in pd.Series(
                       np.concatenate([decision_engine(g["p_xgb_bayes"].values, g["vol20"].values, buy_thr, sell_thr, vol_cap[tk])[0]
                                       for tk, g in test.groupby("ticker")])).value_counts(normalize=True).items()}),
              open(f"{OUT}/decision_summary.json", "w"), indent=1)

    # sensitivity: cost level and thresholds (reported, NOT used for selection)
    sens = []
    for c in (0, 5, 10, 20):
        R, o = run("p_xgb_bayes", cost=c); sens.append(dict(kind="cost_bps", value=c, **o))
    for bq, sq in ((0.6, 0.4), (0.7, 0.3), (0.8, 0.2)):
        R, o = run("p_xgb_bayes", buy=float(oof.quantile(bq)), sell=float(oof.quantile(sq)))
        sens.append(dict(kind="prob_quantiles", value=f"{bq}/{sq}", **o))
    pd.DataFrame(sens).to_csv(f"{OUT}/backtest_sensitivity.csv", index=False)

    # decision log with SHAP explanations
    names = np.array(FEATURES)
    logrows = []
    pos_cache = {}
    for tk, g in test.groupby("ticker"):
        acts, risk = decision_engine(g["p_xgb_bayes"].values, g["vol20"].values, buy_thr, sell_thr, vol_cap[tk])
        sub_sv = sv[g.index.values]
        for j in range(len(g)):
            top = np.argsort(-np.abs(sub_sv[j]))[:3]
            logrows.append(dict(date=g["date"].iloc[j].date(), ticker=tk, p_up=round(float(g["p_xgb_bayes"].iloc[j]), 4),
                                risk=risk[j], action=acts[j],
                                top_factors="; ".join(f"{names[i]} ({sub_sv[j][i]:+.3f})" for i in top)))
    dl = pd.DataFrame(logrows).sort_values(["date", "ticker"])
    dl.to_csv(f"{OUT}/decision_log_with_explanations.csv", index=False)

    # figures: BO convergence, test metrics, equity curves, SHAP consistency, example waterfall
    tr = pd.DataFrame(trial_rows)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, name in zip(axs, ("rf", "xgb")):
        for samp, col in (("random", "#9aa5c4"), ("bayes", "#1B2A57")):
            d = tr[(tr.model == name) & (tr.sampler == samp)]
            curves_ = [g.sort_values("trial")["value"].cummin().values for _, g in d.groupby("seed")]
            m = np.mean(curves_, axis=0); ax.plot(range(1, len(m) + 1), m, color=col, lw=2,
                                                  label="Random search" if samp == "random" else "Bayesian (TPE)")
        ax.set_title({"rf": "Random Forest", "xgb": "XGBoost"}[name]); ax.set_xlabel("trial")
        ax.set_ylabel("best CV log-loss so far"); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_bo_convergence.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, c in zip(curves, ["#1B2A57", "#5B6DB0", "#9aa5c4", "#7B1FA2", "#2E7D32", "#C97A0C", "#B3261E", "#444444"]):
        (1 + curves[k]).cumprod().plot(ax=ax, label=k.replace("Decision engine ", "DE "), color=c, lw=2 if "confidence + risk" in k and "XGB-BO" in k else 1.2)
    ax.set_ylabel("growth of 1 (after 5 bp/side costs)"); ax.set_title("Out-of-sample equity curves (test period)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left"); plt.tight_layout(); plt.savefig(f"{OUT}/fig_equity_curves.png", dpi=200); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, M, t in ((axs[0], rho_blocks, "Across test sub-periods"), (axs[1], rho_win, "Across training windows")):
        im = ax.imshow(M.values, vmin=0, vmax=1, cmap="Blues"); ax.set_title(t, fontsize=10)
        ax.set_xticks(range(len(M))); ax.set_yticks(range(len(M)))
        ax.set_xticklabels([f"B{i+1}" if "B" in c[:1] else c.split("<=")[1][-10:-6] for i, c in enumerate(M.columns)], fontsize=8)
        ax.set_yticklabels([f"B{i+1}" if "B" in c[:1] else c.split("<=")[1][-10:-6] for i, c in enumerate(M.columns)], fontsize=8)
        for i in range(len(M)):
            for j in range(len(M)): ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_consistency.png", dpi=200); plt.close()

    # waterfall for one illustrative BUY decision (latest BUY in test)
    buys = dl[dl.action == "BUY"]
    if len(buys):
        r = buys.iloc[-1]
        ix = test[(test.ticker == r.ticker) & (test.date == pd.Timestamp(r.date))].index[0]
        ex = shap.Explanation(values=sv[ix], base_values=expl.expected_value, data=test.loc[ix, FEATURES].values, feature_names=FEATURES)
        plt.figure(); shap.plots.waterfall(ex, max_display=8, show=False)
        plt.title(f"{r.ticker} {r.date}: P(up)={r.p_up:.3f} -> {r.action}", fontsize=9)
        plt.tight_layout(); plt.savefig(f"{OUT}/fig_waterfall_example.png", dpi=200, bbox_inches="tight"); plt.close()

    # exposure/turnover-matched random benchmark, deflated Sharpe, price checks
    import extra_analyses as rx
    rx.analyse(OUT, a.data, f"{OUT}/extra_analyses", n_sim=a.n_random)
    import importlib.metadata as md_
    def _ver(p):
        try:
            return md_.version(p)
        except Exception:
            return "n/a"
    vers = {p: _ver(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "optuna", "shap")}
    json.dump(dict(elapsed_sec=time.time() - t0, features=FEATURES, tickers=TICKERS, seed_offset=S0,
                   n_jobs=N_JOBS, python=sys.version.split()[0], versions=vers),
              open(f"{OUT}/run_meta.json", "w"))
    log("done")


if __name__ == "__main__":
    main()
