#!/usr/bin/env python3
"""Offline: apply the four verified price repairs to the ten-stock file (values from repair_prices.py / Yahoo Finance).
Yahoo's AAPL values are on today's post-2020-split scale (4-for-1), so they are rescaled (prices x4, volume /4) to match the file."""
import pandas as pd
SRC = "data/sp500_10stocks_daily_2013_2018.csv"; OUT = "data/sp500_10stocks_daily_2013_2018_repaired.csv"
fix = {  # ticker, date: (open, high, low, close, volume, scale)  scale = factor from Yahoo's scale to this file's scale
 ("AAPL", "2017-08-01"): (37.275, 37.555, 37.103, 37.513, 141474400, 4.0),
 ("JNJ", "2017-08-22"): (133.27, 134.79, 132.9, 134.7, 4150300, 1.0),
 ("KO", "2014-10-29"): (40.65, 41.03, 40.52, 40.96, 25761800, 1.0),
 ("XOM", "2014-07-28"): (103.29, 104.52, 102.83, 104.37, 9197800, 1.0)}
d = pd.read_csv(SRC, parse_dates=["date"])
for (tk, dt), (o, h, l, c, v, k) in fix.items():
    m = (d.ticker == tk) & (d.date == dt); assert m.sum() == 1
    d.loc[m, ["open", "high", "low", "close"]] = [o * k, h * k, l * k, c * k]; d.loc[m, "volume"] = int(round(v / k))
d.to_csv(OUT, index=False)
for (tk, dt) in fix:
    g = d[d.ticker == tk].sort_values("date").reset_index(drop=True); i = g.index[g.date == dt][0]
    r = g.close.pct_change()
    print(tk, dt, "close", round(g.close[i], 2), "| returns around:", [f"{x:+.1%}" for x in r[i - 1:i + 3]])
print("wrote", OUT)
