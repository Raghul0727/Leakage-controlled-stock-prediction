#!/usr/bin/env python3
"""
Second-source price check.  Compares the study's closing prices with two independent
downloads and reports, per stock, how far they disagree.  Needs internet.

    python verify_prices.py                       # uses data/sp500_10stocks_daily_2013_2018.csv
    python verify_prices.py --data results_wf_sp500/prices_used.csv

Sources:  (1) Yahoo Finance via yfinance, auto_adjust=False  -> split-adjusted, NOT dividend-adjusted close
          (2) Stooq daily CSV (https://stooq.com)              -> second, independent vendor
Daily RETURNS are compared (not levels) so that a constant dividend-adjustment factor does not matter.
A stock passes if the 99th percentile of |return difference| is below 0.1% and no day differs by > 1%.
Writes price_verification.csv.
"""
import argparse, io, urllib.request
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sp500_10stocks_daily_2013_2018.csv")
ap.add_argument("--out", default="price_verification.csv")
ap.add_argument("--stooq-dir", default=None, help="folder with Stooq CSV exports named <ticker>.us.csv (Date,Open,High,Low,Close,Volume) if the web download fails")
a = ap.parse_args()
df = pd.read_csv(a.data, parse_dates=["date"]).sort_values(["ticker", "date"])


def yahoo(tk, d0, d1):
    import yfinance as yf
    d = yf.download(tk, start=d0, end=d1, auto_adjust=False, progress=False)
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    return d["close"].rename("yahoo")


def stooq(tk, d0, d1):
    if a.stooq_dir:
        t = pd.read_csv(f"{a.stooq_dir}/{tk.lower()}.us.csv", parse_dates=["Date"])
        return t.set_index("Date")["Close"].rename("stooq")
    sym = tk.lower().replace("-", ".") + (".us" if not tk.endswith(".NS") else "")
    url = f"https://stooq.com/q/d/l/?s={sym}&d1={d0.replace('-', '')}&d2={d1.replace('-', '')}&i=d"
    t = pd.read_csv(io.StringIO(urllib.request.urlopen(url, timeout=30).read().decode()), parse_dates=["Date"])
    return t.set_index("Date")["Close"].rename("stooq")


rows = []
for tk, g in df.groupby("ticker"):
    s = g.set_index("date")["close"].rename("study")
    d0, d1 = str(s.index.min().date()), str((s.index.max() + pd.Timedelta(days=3)).date())
    for name, fn in (("yahoo", yahoo), ("stooq", stooq)):
        try:
            o = fn(tk, d0, d1)
            j = pd.concat([s, o], axis=1, join="inner").dropna()
            rd = (j["study"].pct_change() - j[name].pct_change()).abs().dropna()
            rows.append(dict(ticker=tk, source=name, n_common_days=len(j), share_days_matched=len(j) / len(s),
                             p99_abs_return_diff=rd.quantile(.99), max_abs_return_diff=rd.max(),
                             n_days_diff_gt_1pct=int((rd > .01).sum()),
                             dates_diff_gt_1pct=";".join(str(x.date()) for x in rd[rd > .01].index),
                             passed=bool(rd.quantile(.99) < .001 and rd.max() < .01)))
        except Exception as e:
            rows.append(dict(ticker=tk, source=name, error=str(e)[:80]))
out = pd.DataFrame(rows); out.to_csv(a.out, index=False)
print(out.drop(columns=[c for c in ["error"] if c in out and out[c].isna().all()]).to_string())
