#!/usr/bin/env python3
"""
Additional analyses that need only saved outputs (no model refit):
  (A) exposure- AND turnover-matched random long/flat benchmark (1,000 draws per run)
  (B) deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)
  (C) price-data checks: maximum absolute daily return per stock, OHLC consistency,
      split-adjustment evidence, gaps.
Usage:  python extra_analyses.py            (writes ./extra_analyses/*.csv|json)
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats

DATA = "data/sp500_10stocks_daily_2013_2018.csv"
RUNS = {"run1": "results", "run2": "results_replication_run2"}
OUT = "extra_analyses"
COST = 5e-4
N_SIM, SEED = 1000, 12345   # fixed seed: the random benchmark is itself reproducible


def load_next_ret(data_path=DATA):
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values(["ticker", "date"])
    df["next_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    return df[["date", "ticker", "next_ret"]]


def positions(log):
    pos = {}
    for tk, g in log.sort_values("date").groupby("ticker"):
        cur, p = 0, []
        for a in g["action"]:
            cur = 1 if a == "BUY" else 0 if a == "SELL" else cur
            p.append(cur)
        pos[tk] = np.array(p)
    return pos


def port_ret(pos_by_tk, ret_by_tk):
    cols = []
    for tk, p in pos_by_tk.items():
        prev = np.concatenate([[0], p[:-1]])
        cols.append(p * ret_by_tk[tk] - COST * np.abs(p - prev))
    return np.mean(cols, axis=0)


def sharpe(r):
    return r.mean() / r.std(ddof=1) * np.sqrt(252)


def random_matched(T, n_long, n_entries, ends_long, rng):
    """Random 0/1 path of length T with exactly n_long long days, n_entries long blocks,
    and the same end state, starting flat.  Same exposure and same number of position changes."""
    if n_long == 0:
        return np.zeros(T, int)
    k = n_entries
    # composition of n_long into k positive block lengths
    cuts = np.sort(rng.choice(np.arange(1, n_long), size=k - 1, replace=False)) if k > 1 else np.array([], int)
    blocks = np.diff(np.concatenate([[0], cuts, [n_long]]))
    n_flat = T - n_long
    # gaps: g0>=0 (before 1st block), g1..g_{k-1}>=1 (between), g_k>=0 (after; must be 0 if ends long, >=1 else)
    last_min = 0 if ends_long else 1
    mandatory = (k - 1) + last_min
    free = n_flat - mandatory
    if free < 0:
        raise ValueError("cannot match")
    # distribute 'free' extra flat days over g0, g1..g_{k-1}, (g_k only if not ends_long)
    slots = k + (0 if ends_long else 1)
    extra = rng.multinomial(free, np.ones(slots) / slots)
    g = np.zeros(k + 1, int)
    g[0] = extra[0]
    for i in range(1, k):
        g[i] = 1 + extra[i]
    g[k] = 0 if ends_long else 1 + extra[k]
    path = []
    for i in range(k):
        path += [0] * g[i] + [1] * blocks[i]
    path += [0] * g[k]
    return np.array(path, int)


def analyse(results_dir, data_path=DATA, out_dir=OUT, n_sim=N_SIM, tag="run"):
    """Exposure/turnover-matched random benchmark + deflated Sharpe for ONE results folder.
    Needs decision_log_with_explanations.csv, backtest_metrics.csv, backtest_sensitivity.csv there."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    nr = load_next_ret(data_path)
    log = pd.read_csv(f"{results_dir}/decision_log_with_explanations.csv", parse_dates=["date"])
    log = log.merge(nr, on=["date", "ticker"], how="left").sort_values(["ticker", "date"])
    pos = positions(log)
    ret = {tk: g["next_ret"].values for tk, g in log.groupby("ticker")}
    R = port_ret(pos, ret)
    sh = sharpe(R)
    par = {}
    for tk, p in pos.items():
        prev = np.concatenate([[0], p[:-1]])
        par[tk] = dict(T=len(p), n_long=int(p.sum()), n_entries=int(((p == 1) & (prev == 0)).sum()),
                       ends_long=bool(p[-1] == 1), turnover=int(np.abs(p - prev).sum()))
    rng = np.random.default_rng(SEED)
    sims, sim_ret = [], []
    for _ in range(n_sim):
        rp = {tk: random_matched(q["T"], q["n_long"], q["n_entries"], q["ends_long"], rng)
              if q["n_long"] > 0 else np.zeros(q["T"], int) for tk, q in par.items()}
        rr = port_ret(rp, ret)
        sims.append(sharpe(rr)); sim_ret.append(rr.mean() * 252)
    sims = np.array(sims)
    pct = float((sims < sh).mean() * 100)
    pval = float((sims >= sh).mean())
    # deflated Sharpe: trials = every distinct strategy variant evaluated on the test period
    bm = pd.read_csv(f"{results_dir}/backtest_metrics.csv", index_col=0)
    se = pd.read_csv(f"{results_dir}/backtest_sensitivity.csv")
    trial_sr = [bm.loc[k, "sharpe"] for k in bm.index if k.startswith("Decision engine") or k.startswith("Naive")]
    for _, r in se.iterrows():
        if (r["kind"] == "cost_bps" and str(r["value"]) != "5") or \
           (r["kind"] == "prob_quantiles" and r["value"] != "0.7/0.3"):
            trial_sr.append(r["sharpe"])
    N = len(trial_sr)
    sr_d = R.mean() / R.std(ddof=1)                       # per-period (daily) units, as in the paper
    v = np.var(np.array(trial_sr) / np.sqrt(252), ddof=1)
    g = np.euler_gamma
    sr0 = np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * np.e)))
    sk, ku = stats.skew(R), stats.kurtosis(R, fisher=False)   # raw (non-excess) kurtosis
    z = (sr_d - sr0) * np.sqrt(len(R) - 1) / np.sqrt(1 - sk * sr_d + (ku - 1) / 4 * sr_d ** 2)
    out = dict(
        sharpe_reproduced=float(sh), n_days=int(len(R)),
        exposure=float(np.mean([q["n_long"] / q["T"] for q in par.values()])),
        turnover_per_stock_per_year=float(np.mean([q["turnover"] for q in par.values()]) / (len(R) / 252)),
        random_sharpe_mean=float(sims.mean()), random_sharpe_sd=float(sims.std(ddof=1)),
        random_sharpe_p5=float(np.percentile(sims, 5)), random_sharpe_p50=float(np.percentile(sims, 50)),
        random_sharpe_p95=float(np.percentile(sims, 95)),
        engine_percentile_in_random=pct, p_random_ge_engine=pval,
        random_ann_return_mean=float(np.mean(sim_ret)),
        n_trials_for_dsr=int(N), trial_sharpe_sd_annualised=float(np.sqrt(v * 252)),
        sr0_annualised=float(sr0 * np.sqrt(252)), deflated_sharpe_prob=float(stats.norm.cdf(z)),
        skew=float(sk), kurtosis=float(ku), n_random=int(n_sim))
    pd.Series(sims, name="random_matched_sharpe").to_csv(f"{out_dir}/random_matched_sharpe_{tag}.csv", index=False)
    json.dump(out, open(f"{out_dir}/exposure_matched_benchmark_{tag}.json", "w"), indent=1)
    price_checks(data_path, out_dir)
    return out


def price_checks(data_path=DATA, out_dir=OUT):
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values(["ticker", "date"])
    rows = []
    for tk, g in df.groupby("ticker"):
        r = g["close"].pct_change().dropna()
        bad_ohlc = int(((g.high < g[["open", "close"]].max(axis=1) - 1e-9) | (g.low > g[["open", "close"]].min(axis=1) + 1e-9)).sum())
        gaps = g["date"].diff().dt.days.dropna()
        rows.append(dict(ticker=tk, n_days=len(g), first_close=g.close.iloc[0], max_abs_daily_return=r.abs().max(),
                         date_of_max=str(g["date"].iloc[1:][r.abs().values == r.abs().max()].iloc[0].date()),
                         n_abs_ret_gt_10pct=int((r.abs() > .10).sum()), ohlc_inconsistent_rows=bad_ohlc,
                         zero_volume_days=int((g.volume <= 0).sum()), duplicate_dates=int(g.date.duplicated().sum()),
                         longest_calendar_gap_days=int(gaps.max())))
    pc = pd.DataFrame(rows)
    pc.to_csv(f"{out_dir}/price_checks.csv", index=False)
    return pc


def main():
    summary = {run: analyse(d, DATA, OUT, N_SIM, run) for run, d in RUNS.items()}
    json.dump(summary, open(f"{OUT}/exposure_matched_benchmark.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(pd.read_csv(f"{OUT}/price_checks.csv").to_string())


if __name__ == "__main__":
    main()
