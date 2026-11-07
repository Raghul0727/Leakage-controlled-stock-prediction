#!/usr/bin/env python3
"""Create the cleaned wide-panel files used in the paper: complete history from the start date,
stocks with known unadjusted corporate actions removed, common calendar. Offline."""
import pandas as pd
def clean(src, out, complete_by, drop):
    d = pd.read_csv(src, parse_dates=["date"])
    first = d.groupby("ticker").date.min()
    keep = [t for t in first[first <= complete_by].index if t not in drop]
    d = d[d.ticker.isin(keep)]
    c = d.groupby("date").ticker.nunique()
    d = d[d.date.isin(c[c == len(keep)].index)]
    d.to_csv(out, index=False); print(out, len(keep), "stocks,", d.date.nunique(), "dates")
clean("data/stocks_2010_2025.csv", "data/us2010_clean.csv", "2010-01-15", {"WY"})
clean("data/nifty50_2010_2025.csv", "data/nifty_clean.csv", "2010-01-15", {"NESTLEIND.NS", "TMPV.NS"})
