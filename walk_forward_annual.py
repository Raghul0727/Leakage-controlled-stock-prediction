#!/usr/bin/env python3
"""
Wider evidence base: >= 50 stocks, >= 10 years, annual walk-forward retraining.

  * Universe chosen by a STATED RULE, not by hand: a seeded random draw from the current S&P 500
    members (or NIFTY 50 constituents), keeping the first N tickers in the random order that have a
    complete price history over the study window.  The rule, seed and chosen tickers are saved.
    (Current members => survivorship bias; say so in the paper.)
  * For each test year Y the models are re-tuned (TPE) and re-fitted on all data before 1 January Y;
    tuning uses expanding-window half-year walk-forward folds over the two years before Y.
  * Test years include falling / volatile years (flagged from the equal-weight buy-and-hold return).
  * Each year reports accuracy, AUC, log-loss vs the constant forecast, the decision engine's Sharpe
    ratio, buy-and-hold, and the percentile of the engine in 1,000 exposure/turnover-matched random
    long/flat strategies.

Examples
    python walk_forward_annual.py --market sp500   --n-stocks 50 --universe-seed 42 --start 2008-01-01 --end 2025-12-31 --test-years 2015-2025
    python walk_forward_annual.py --market nifty50 --n-stocks 50 --universe-seed 42 --start 2008-01-01 --end 2025-12-31 --test-years 2015-2025
    python walk_forward_annual.py --source csv --data data/sp500_10stocks_daily_2013_2018.csv --test-years 2016-2017 --quick   # smoke test

NOTE: needs internet (yfinance, Wikipedia) unless --source csv.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
import xai_stock_pipeline as P
import extra_analyses as rx


# ---------------------------------------------------------------- universe + data
def members(market):
    if market == "sp500":
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted(t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())
    for t in pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50"):
        if "Symbol" in t.columns:
            return sorted(t["Symbol"].astype(str).str.strip().add(".NS").tolist())
    raise RuntimeError("could not find the NIFTY 50 table; pass --tickers-file instead")


def download(tk, start, end):
    import yfinance as yf
    d = yf.download(tk, start=start, end=end, auto_adjust=False, progress=False)
    if d is None or len(d) == 0:
        return None
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    d = d.reset_index().rename(columns={"Date": "date", "date": "date"})
    d["ticker"] = tk
    return d[["date", "ticker", "open", "high", "low", "close", "volume"]]


def build_universe(a):
    pool = [t.strip() for t in open(a.tickers_file).read().split()] if a.tickers_file else members(a.market)
    order = np.random.default_rng(a.universe_seed).permutation(len(pool))
    frames, chosen = [], []
    n_bdays = len(pd.bdate_range(a.start, a.end))
    for i in order:
        tk = pool[i]
        d = download(tk, a.start, a.end)
        if d is None or len(d) < 0.93 * n_bdays or d["date"].min() > pd.Timestamp(a.start) + pd.Timedelta(days=15):
            continue                                       # incomplete history: skip, take next in the random order
        frames.append(d); chosen.append(tk)
        if len(chosen) == a.n_stocks:
            break
    df = pd.concat(frames)
    json.dump(dict(rule="seeded random permutation of current index members; first N with complete history",
                   market=a.market, universe_seed=a.universe_seed, n_requested=a.n_stocks, tickers=sorted(chosen),
                   start=a.start, end=a.end, prices="split-adjusted, NOT dividend-adjusted (yfinance auto_adjust=False)"),
              open(f"{a.out}/universe.json", "w"), indent=1)
    return df


# ---------------------------------------------------------------- folds
def make_folds(Y):
    """Four half-year validation windows over the two years before test year Y (1-day purge, as in the paper)."""
    folds = []
    for y in (Y - 2, Y - 1):
        for (m0, m1, d1) in ((1, 6, 30), (7, 12, 31)):
            s, e = pd.Timestamp(y, m0, 1), pd.Timestamp(y, m1, d1)
            folds.append(((s - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return folds


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="sp500", choices=["sp500", "nifty50"])
    ap.add_argument("--source", default="web", choices=["web", "csv"])
    ap.add_argument("--data", default=None, help="CSV with date,ticker,open,high,low,close,volume (with --source csv)")
    ap.add_argument("--tickers-file", default=None)
    ap.add_argument("--n-stocks", type=int, default=50)
    ap.add_argument("--universe-seed", type=int, default=42)
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--test-years", default="2015-2025", help="e.g. 2015-2025")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.out = a.out or f"results_wf_{a.market}"
    if a.quick:
        a.trials, a.n_random = 6, 50
    os.makedirs(a.out, exist_ok=True)
    y0, y1 = (int(x) for x in a.test_years.split("-"))

    raw = pd.read_csv(a.data, parse_dates=["date"]) if a.source == "csv" else build_universe(a)
    raw.to_csv(f"{a.out}/prices_used.csv", index=False)
    rx.price_checks(f"{a.out}/prices_used.csv", a.out)                     # max |daily return| per stock etc.
    # common calendar so that the portfolio and the random benchmark line up
    common = raw.groupby("date")["ticker"].nunique()
    raw = raw[raw["date"].isin(common[common == raw["ticker"].nunique()].index)]
    panel = P.build_panel(raw.sort_values(["ticker", "date"]))

    rows, all_test = [], []
    for Y in range(y0, y1 + 1):
        pre = panel[panel["date"] < pd.Timestamp(Y, 1, 1)].reset_index(drop=True)
        test = panel[(panel["date"] >= pd.Timestamp(Y, 1, 1)) & (panel["date"] <= pd.Timestamp(Y, 12, 31))].reset_index(drop=True)
        cnt = test.groupby("date")["ticker"].transform("nunique")        # keep only dates on which every stock has an observation
        test = test[cnt == test["ticker"].nunique()].reset_index(drop=True)
        if len(test) == 0 or pre["date"].min() > pd.Timestamp(Y - 2, 1, 1) - pd.Timedelta(days=60):       # need training data before the first tuning fold (2 years before the test year)
            print(f"skip {Y}: not enough history / data"); continue
        folds = make_folds(Y)
        st = P.tune("xgb", pre, "bayes", a.trials, a.seed, folds=folds)
        bp = st.best_params
        m_t = P.make_model("xgb", bp, a.seed).fit(pre[P.FEATURES], pre["y"])
        m_d = P.make_model("xgb", P.DEFAULTS["xgb"], a.seed).fit(pre[P.FEATURES], pre["y"])
        p_t, p_d = m_t.predict_proba(test[P.FEATURES])[:, 1], m_d.predict_proba(test[P.FEATURES])[:, 1]
        mt, md = P.clf_metrics(test.y, p_t), P.clf_metrics(test.y, p_d)
        base = pre.y.mean()
        const_ll = log_loss(test.y, np.full(len(test), base))
        maj = max(test.y.mean(), 1 - test.y.mean())

        # decision engine: thresholds from the tuned model's OWN out-of-fold validation probabilities
        _, oof = P.cv_logloss("xgb", bp, pre, a.seed, return_oof=True, folds=folds)
        buy, sell = float(oof.quantile(P.BUY_Q)), float(oof.quantile(P.SELL_Q))
        vcap = pre.groupby("ticker")["vol20"].quantile(P.VOL_CAP_Q).to_dict()
        test = test.assign(p=p_t)
        acts, pos, ret = {}, {}, {}
        for tk, g in test.groupby("ticker"):
            acts[tk], _ = P.decision_engine(g["p"].values, g["vol20"].values, buy, sell, vcap[tk])
            pos[tk] = P.positions_from_actions(acts[tk]); ret[tk] = g["next_ret"].values
        R, o = P.portfolio(test, acts, P.COST_BPS)
        BH, obh = P.portfolio(test, {tk: np.array(["BUY"] * len(g)) for tk, g in test.groupby("ticker")}, P.COST_BPS)
        # exposure/turnover-matched random long/flat benchmark
        rng = np.random.default_rng(rx.SEED + Y)
        par = {}
        for tk, p_ in pos.items():
            prev = np.concatenate([[0], p_[:-1]])
            par[tk] = (len(p_), int(p_.sum()), int(((p_ == 1) & (prev == 0)).sum()), bool(p_[-1] == 1))
        sims = []
        for _ in range(a.n_random):
            rp = {tk: (rx.random_matched(T, L, k, e, rng) if L > 0 else np.zeros(T, int)) for tk, (T, L, k, e) in par.items()}
            sims.append(rx.sharpe(rx.port_ret(rp, ret)))
        sims = np.array(sims)
        rows.append(dict(year=Y, n_stocks=test.ticker.nunique(), n_dates=test.date.nunique(), n_obs=len(test),
                         base_rate_up=test.y.mean(), majority_acc=maj,
                         acc_tuned=mt["acc"], auc_tuned=mt["auc"], logloss_tuned=mt["logloss"], logloss_constant=const_ll,
                         acc_default=md["acc"], auc_default=md["auc"],
                         engine_sharpe=o["sharpe"], engine_exposure=o["avg_exposure"], engine_return=o["ann_return"],
                         bh_sharpe=obh["sharpe"], bh_return=obh["cum_return"], bh_maxdd=obh["max_drawdown"], bh_vol=obh["ann_vol"],
                         random_sharpe_mean=sims.mean(), random_sharpe_sd=sims.std(ddof=1),
                         engine_percentile_in_random=float((sims < o["sharpe"]).mean() * 100)))
        all_test.append(test)
        print(rows[-1], flush=True)

    Y_ = pd.DataFrame(rows)
    med_vol = Y_["bh_vol"].median()
    Y_["falling_or_volatile"] = (Y_["bh_return"] < 0) | (Y_["bh_vol"] > 1.25 * med_vol)
    Y_.to_csv(f"{a.out}/wf_by_year.csv", index=False)
    pooled = dict(n_years=int(len(Y_)), years_auc_gt_0_5=int((Y_.auc_tuned > 0.5).sum()),
                  years_acc_gt_majority=int((Y_.acc_tuned > Y_.majority_acc).sum()),
                  years_logloss_lt_constant=int((Y_.logloss_tuned < Y_.logloss_constant).sum()),
                  mean_auc=float(Y_.auc_tuned.mean()), sd_auc=float(Y_.auc_tuned.std(ddof=1)) if len(Y_) > 1 else None,
                  mean_engine_percentile=float(Y_.engine_percentile_in_random.mean()),
                  years_engine_above_random_median=int((Y_.engine_percentile_in_random > 50).sum()),
                  years_engine_sharpe_above_bh=int((Y_.engine_sharpe > Y_.bh_sharpe).sum()),
                  falling_or_volatile_years=Y_.loc[Y_.falling_or_volatile, "year"].tolist(),
                  effective_sample_note="observations on one date are correlated: effective n is closer to the number of dates than of stock-days")
    T = pd.concat(all_test)
    bs = P.date_bootstrap(T, {"m": T["p"].values}, n=1000 if not a.quick else 50)
    pooled["pooled_auc_ci95"] = np.percentile(bs["m_auc"], [2.5, 97.5]).tolist()
    pooled["pooled_acc_ci95"] = np.percentile(bs["m_acc"], [2.5, 97.5]).tolist()
    json.dump(pooled, open(f"{a.out}/wf_pooled.json", "w"), indent=1)
    print(Y_.round(3).to_string()); print(json.dumps(pooled, indent=1))


if __name__ == "__main__":
    main()
