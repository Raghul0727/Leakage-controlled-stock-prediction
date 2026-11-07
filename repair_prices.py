#!/usr/bin/env python3
"""
Repair the four corrupted closing prices found by verify_prices.py and show old vs new values.
Needs internet (yfinance).  Usage:
    python repair_prices.py                 # report only
    python repair_prices.py --apply         # also write data/sp500_10stocks_daily_2013_2018_repaired.csv
After applying, re-run the whole pipeline on the repaired file:
    python xai_stock_pipeline.py --data data/sp500_10stocks_daily_2013_2018_repaired.csv --out results_repaired
Known evidence: AAPL 2017-08-01 should be about 150.05 (Reuters/Fortune report the close of 1 Aug 2017); the file shows 158.59.
XOM 2014-07-28 is a near-copy of the 2014-07-30 row. JNJ 2017-08-22 and KO 2014-10-29 are not externally confirmed.
"""
import argparse, pandas as pd, yfinance as yf
ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); ap.add_argument("--data", default="data/sp500_10stocks_daily_2013_2018.csv"); a = ap.parse_args()
fix = [("AAPL", "2017-08-01"), ("JNJ", "2017-08-22"), ("KO", "2014-10-29"), ("XOM", "2014-07-28")]
df = pd.read_csv(a.data, parse_dates=["date"])
for tk, dt in fix:
    d0 = pd.Timestamp(dt); y = yf.download(tk, start=d0 - pd.Timedelta(days=5), end=d0 + pd.Timedelta(days=5), auto_adjust=False, progress=False)
    y.columns = [c[0].lower() for c in y.columns]; y = y.reset_index().rename(columns={"Date": "date"})
    row = y[y.date == d0]
    old = df[(df.ticker == tk) & (df.date == d0)]
    print(tk, dt, "\n study:", old[["open", "high", "low", "close", "volume"]].round(3).values.tolist(), "\n yahoo:", row[["open", "high", "low", "close", "volume"]].round(3).values.tolist())
    if a.apply and len(row):
        # Yahoo serves split-adjusted history on TODAY'S scale (e.g. AAPL is 4x lower after the 2020 split): rescale to the file's scale
        nb = y[y.date != d0].merge(df[df.ticker == tk][["date", "close"]], on="date", suffixes=("_y", "_f"))
        ratio = float((nb.close_f / nb.close_y).median()) if len(nb) else 1.0
        k = float(round(ratio)) if round(ratio) >= 1 and abs(ratio - round(ratio)) < 0.03 * round(ratio) else 1.0
        print("  scale factor applied:", k)
        for c in ["open", "high", "low", "close"]: df.loc[(df.ticker == tk) & (df.date == d0), c] = float(row[c].iloc[0]) * k
        df.loc[(df.ticker == tk) & (df.date == d0), "volume"] = float(row["volume"].iloc[0]) / k
if a.apply:
    out = a.data.replace(".csv", "_repaired.csv"); df.to_csv(out, index=False); print("wrote", out)
