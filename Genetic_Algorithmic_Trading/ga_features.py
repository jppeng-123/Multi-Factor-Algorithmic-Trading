"""Factor-Trading.ipynb"""


from __future__ import annotations

import os
import math
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats as sp_stats
from scipy.special import gammaln
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import yfinance as yf
from arch import arch_model
from massive import RESTClient

import numba as nb
from numba import njit, prange

from sklearn.base import clone
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler






API_KEY = "API KEY"
client = RESTClient(API_KEY)



"""Initialization, data generation and key matrices formation"""


tickers =  [# --- Your original 50 ---
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","ORCL",
    "ADBE","CRM","INTC","AMD","QCOM","CSCO","AMAT","TXN","NOW","NFLX",
    "PEP","COST","WMT","HD","LOW","MCD","SBUX","NKE","DIS","TMO",
    "UNH","JNJ","PFE","ABBV","MRK","LLY","CVX","XOM","COP","JPM",
    "BAC","WFC","MS","GS","V","MA","KO","CAT","GE","BA",

    # --- Add 50 more (sector-balanced, liquid, long history) ---
    # Financials / Payments / Insurance
    "BRK.B","BLK","C","AXP","SCHW","USB","PNC","TFC","CME","ICE",

    # Healthcare / Pharma / Devices
    "AMGN","GILD","BMY","CVS","CI","HUM","MDT","SYK","BSX","ISRG",

    # Consumer Staples / Retail
    "PG","CL","KMB","MO","PM","EL","MDLZ","KHC","WBA","TGT",

    # Industrials / Defense / Transport
    "HON","LMT","RTX","DE","UNP","UPS","FDX","CSX","MMM","EMR",

    # Energy / Materials
    "SLB","EOG","PXD","OXY","PSX","VLO","LIN","APD","ECL","NEM",

    # Utilities / Real Estate / Telecom
    "NEE","DUK","SO","AEP","EXC","AMT","PLD","CCI","VZ","T",
    "TER"
]

start = '2015-11-28'
end = '2026-01-28'

aggs = []
for ticker in tickers:
  for a in client.list_aggs(
      ticker,
      1,
      "day",
      start,
      end,
      adjusted="false",
      sort="asc",
  ):
      aggs.append(a)











# ============================================================
# Class Packaging (Industry-grade: pure/static methods + backward-compatible aliases)
# ============================================================

class AggsMatrixFactory:
    """
    Pure utilities for:
      - splitting flat Agg list into per-ticker chunks
      - converting chunk -> per-ticker OHLCV DataFrame
      - building aligned matrices (date x ticker)
    Notes:
      - All methods are static to preserve original call signatures.
      - No side-effects, no hidden state: deterministic & testable.
    """

    @staticmethod
    def split_aggs_by_timestamp_jumps(aggs, tickers):
        """
        Split a flat aggs list into ticker chunks by detecting timestamp decreases.

        Rule:
          - Within the same stock, timestamps must be non-decreasing (asc).
          - When timestamp goes backwards (cur_ts < prev_ts), we start a new chunk.
          - Chunks are mapped to tickers in the given order.

        This is ONLY valid if each ticker boundary produces at least one timestamp decrease.
        """
        if not tickers:
            raise ValueError("tickers is empty.")
        if not aggs:
            return {t: [] for t in tickers}

        cut_points = [0]
        prev_ts = int(aggs[0].timestamp)

        for i in range(1, len(aggs)):
            cur_ts = int(aggs[i].timestamp)
            if cur_ts < prev_ts:
                cut_points.append(i)
            prev_ts = cur_ts

        cut_points.append(len(aggs))

        chunks = [aggs[cut_points[j]:cut_points[j+1]] for j in range(len(cut_points)-1)]

        # Strict correctness check: number of chunks must equal number of tickers
        if len(chunks) != len(tickers):
            raise ValueError(
                f"Timestamp-jump split found {len(chunks)} chunks, but you have {len(tickers)} tickers.\n"
                f"This means timestamp-only splitting cannot uniquely recover your ticker boundaries.\n"
                f"Fix: keep per-ticker lists when collecting (recommended), OR ensure each ticker boundary causes a timestamp drop."
            )

        # Additional strict checks: timestamps within each chunk must be non-decreasing
        for k, chunk in enumerate(chunks):
            ts = [int(x.timestamp) for x in chunk]
            if any(ts[i] < ts[i-1] for i in range(1, len(ts))):
                raise ValueError(
                    f"Chunk {k} (mapped to {tickers[k]}) is not monotone increasing in timestamp. "
                    "So the timestamp-jump assumption is violated."
                )

        return {tickers[i]: chunks[i] for i in range(len(tickers))}

    @staticmethod
    def chunk_to_df(chunk, tz="America/New_York"):
        """
        Convert one ticker chunk to a DataFrame indexed by local trading date.
        """
        if len(chunk) == 0:
            df = pd.DataFrame(columns=["open_price", "high", "low", "close_price", "volume", "vwap"])
            df.index.name = "date"
            return df

        idx = pd.to_datetime([int(x.timestamp) for x in chunk], unit="ms", utc=True)
        idx = idx.tz_convert(tz).normalize().tz_localize(None)  # date index

        df = pd.DataFrame(
            {
                "open_price":  [float(x.open) for x in chunk],
                "high":        [float(x.high) for x in chunk],
                "low":         [float(x.low) for x in chunk],
                "close_price": [float(x.close) for x in chunk],
                "volume":      [float(x.volume) for x in chunk],
                "vwap":        [float(x.vwap) if x.vwap is not None else np.nan for x in chunk],
            },
            index=idx,
        )

        # If duplicates for same date, keep last and sort
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.index.name = "date"
        return df

    @staticmethod
    def make_matrix(per_ticker_df, tickers, field):
        """
        Outer-join across tickers by date => NaN where missing.
        """
        mat = pd.concat({t: per_ticker_df[t][field] for t in tickers}, axis=1)
        mat = mat.reindex(columns=tickers).sort_index()
        mat.index.name = "date"
        return mat


class BasicFeatureFactory:
    """
    Pure utilities for basic feature engineering.
    - All methods are static to preserve signature & avoid implicit state.
    - Designed to be imported/reused by other modules without side-effects.
    """

    @staticmethod
    def _align_to_base(base: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
        return x.reindex(index=base.index, columns=base.columns)

    @staticmethod
    def _safe_div(a, b, eps=1e-12):
        return a / (b + eps)

    @staticmethod
    def _rolling_zscore(x: pd.DataFrame, L: int) -> pd.DataFrame:
        mu = x.rolling(L, min_periods=L).mean()
        sd = x.rolling(L, min_periods=L).std()
        return (x - mu) / (sd + 1e-12)

    @staticmethod
    def _rolling_corr(a: pd.DataFrame, b: pd.DataFrame, L: int) -> pd.DataFrame:
        # rolling corr per column
        return a.rolling(L, min_periods=L).corr(b)


# ------------------------------------------------------------
# Backward-compatible function handles (names unchanged)
# ------------------------------------------------------------
split_aggs_by_timestamp_jumps = AggsMatrixFactory.split_aggs_by_timestamp_jumps
chunk_to_df = AggsMatrixFactory.chunk_to_df
make_matrix = AggsMatrixFactory.make_matrix

_align_to_base = BasicFeatureFactory._align_to_base
_safe_div = BasicFeatureFactory._safe_div
_rolling_zscore = BasicFeatureFactory._rolling_zscore
_rolling_corr = BasicFeatureFactory._rolling_corr







# -----------------------------
# MAIN: split using timestamp jumps, then build matrices
# -----------------------------
chunks = split_aggs_by_timestamp_jumps(aggs, tickers)
per_ticker_df = {t: chunk_to_df(chunks[t], tz="America/New_York") for t in tickers}

open_price  = make_matrix(per_ticker_df, tickers, "open_price")
high        = make_matrix(per_ticker_df, tickers, "high")
low         = make_matrix(per_ticker_df, tickers, "low")
close_price = make_matrix(per_ticker_df, tickers, "close_price")
volume      = make_matrix(per_ticker_df, tickers, "volume")
vwap        = make_matrix(per_ticker_df, tickers, "vwap")

print(close_price)
print(close_price.info())

# Get Split Gain Matrix split_gain
splits = []
for ticker in tickers:
  for s in client.list_stocks_splits(
      ticker=ticker,
      execution_date_gte=start,
      limit="100",
      sort="execution_date.desc",
      ):
      splits.append(s)

print(splits)

rows = []
for s in splits:
    rows.append({
        "date": pd.to_datetime(s.execution_date),
        "ticker": s.ticker,
        "split_from": float(s.split_from),
        "split_to": float(s.split_to),
    })

splits_df = pd.DataFrame(rows)

# If no events, return an empty matrix with correct columns
if splits_df.empty:
    split_gain = pd.DataFrame(columns=tickers, dtype=float)
    split_gain.index.name = "date"
else:
    # Filter date window (inclusive)
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    splits_df = splits_df[(splits_df["date"] >= start_dt) & (splits_df["date"] <= end_dt)].copy()

    # -----------------------------
    # 3) Correct math for "split gain" S_t:
    #    split_gain = (split_to / split_from) - 1
    # -----------------------------
    if (splits_df["split_from"] <= 0).any():
        raise ValueError("Found split_from <= 0; cannot compute split ratios safely.")

    splits_df["split_gain"] = (splits_df["split_to"] / splits_df["split_from"]) - 1.0

    # If multiple split-like events occur for the same ticker on the same date,
    # combine multiplicatively in the split factor, then convert back to gain:
    # factor_total = Π (split_to/split_from), gain_total = factor_total - 1
    splits_df["split_factor"] = splits_df["split_to"] / splits_df["split_from"]
    combined = (
        splits_df.groupby(["date", "ticker"], as_index=False)["split_factor"]
        .prod()
    )
    combined["split_gain"] = combined["split_factor"] - 1.0

    # -----------------------------
    # 4) Build the matrix: rows=dates, cols=tickers, NaN = "no action"
    # -----------------------------
    # Create a business-day index, but ensure all event dates are included
    idx = pd.bdate_range(start_dt, end_dt).union(pd.DatetimeIndex(combined["date"].unique())).sort_values()
    idx.name = "date"

    split_gain = pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)
    for r in combined.itertuples(index=False):
        split_gain.loc[r.date, r.ticker] = r.split_gain

# split_gain is your matrix (mostly NaN, numbers only on action dates)
print(split_gain.loc[split_gain.notna().any(axis=1)])

# get dividend matrix div
dividends = []
for ticker in tickers:
  for d in client.list_stocks_dividends(
      ticker=ticker,
      ex_dividend_date_gte=start,
      limit="100",
      sort="ticker.asc",
      ):
      dividends.append(d)

print(dividends)

rows = []
for d in dividends:
    ex_date = getattr(d, "ex_dividend_date", None)
    cash_amt = getattr(d, "cash_amount", None)

    # skip malformed records safely
    if ex_date is None or cash_amt is None:
        continue

    rows.append({
        "date": pd.to_datetime(ex_date),     # ex-dividend date is the event date for returns
        "ticker": d.ticker,
        "cash_per_share": float(cash_amt),   # EXACT cash_amount as you requested
    })

div_df = pd.DataFrame(rows)

# -----------------------------
# 3) Build dividend matrix: rows=dates, cols=tickers, NaN = "no action"
# -----------------------------
start_dt = pd.to_datetime(start)
end_dt   = pd.to_datetime(end)

if div_df.empty:
    div = pd.DataFrame(columns=tickers, dtype=float)
    div.index.name = "date"
else:
    # filter window (inclusive)
    div_df = div_df[(div_df["date"] >= start_dt) & (div_df["date"] <= end_dt)].copy()

    # If multiple dividends happen same ticker on same ex-date (rare), cash effects add.
    combined = (
        div_df.groupby(["date", "ticker"], as_index=False)["cash_per_share"]
        .sum()
    )

    # Build a business-day index, but ensure all ex-dates included
    idx = pd.bdate_range(start_dt, end_dt).union(pd.DatetimeIndex(combined["date"].unique())).sort_values()
    idx.name = "date"

    div = pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)
    for r in combined.itertuples(index=False):
        div.loc[r.date, r.ticker] = r.cash_per_share

# div is your dividend-per-share matrix (mostly NaN)
print(div.loc[div.notna().any(axis=1)].head(30))

# Log Return Matrix
idx = close_price.index
cols = close_price.columns

split_gain = split_gain.reindex(index=idx, columns=cols)
div        = div.reindex(index=idx, columns=cols)

# -----------------------------
# 1) Build F_t and D_t for operations
#    You said B_t is not provided -> set B_t = 0
#    F_t = 1 + B_t + S_t = 1 + S_t (and if no split, S_t is NaN -> treat as 0)
# -----------------------------
F = 1.0 + split_gain.fillna(0.0)     # share factor
D = div.fillna(0.0)                  # cash dividend (0 if none)

# -----------------------------
# 2) Compute log return matrix using:
#    1 + R_t = (F_t * P_t + D_t) / P_{t-1}
#    logret_t = log(1 + R_t)
# -----------------------------
P_t   = close_price
P_tm1 = close_price.shift(1)

gross = (F * P_t + D) / P_tm1
logret = np.log(gross)

# -----------------------------
# 3) Clean up obvious non-finite values (first row, missing prices, etc.)
# -----------------------------
logret = logret.replace([np.inf, -np.inf], np.nan)

# Optional: if you want to enforce NaN wherever either P_t or P_{t-1} missing
logret = logret.where(P_t.notna() & P_tm1.notna())

# logret is your corrected log return matrix
print(logret.head())
print(logret.info())



"""Basic Features"""

# ============================================================
# INPUTS you already have (DataFrames: index=dates, cols=tickers)
#   open_price, high, low, close_price, volume, vwap, logret
# All should already be aligned to same index/columns (or we will align below).
# ============================================================

# -----------------------------
# 0) Align matrices + helpers
# -----------------------------
tickers = list(close_price.columns)

# Derived basics
eps = 1e-12
r = logret
abs_r = r.abs()

logV = np.log(volume.replace(0, np.nan))
dlogV = logV.diff(1)

# Range proxies
range_abs = (high - low)
range_log = np.log(_safe_div(high, low, eps=eps))          # log(H/L)
range_rel = _safe_div(range_abs, close_price.shift(1), eps=eps)  # (H-L)/C_{t-1}

# Dollar volume
dollar_vol = close_price * volume
log_dollar_vol = np.log(dollar_vol.replace(0, np.nan))


# ============================================================
# 1) Build the 24 base features (DataFrames)
# ============================================================


features = {}

# ---- A) Momentum (6)
features["MOM_3"]  = r.rolling(3,  min_periods=3).sum()
features["MOM_5"]  = r.rolling(5,  min_periods=5).sum()
features["MOM_10"] = r.rolling(10, min_periods=10).sum()
features["MOM_20"] = r.rolling(20, min_periods=20).sum()
features["MOM_60"] = r.rolling(60, min_periods=60).sum()
features["ACCEL_5_20"] = features["MOM_5"] - features["MOM_20"]

# ---- B) Reversal (4)
features["REV_1"] = -r.shift(1)  # -r_{t-1}
features["REV_3"] = -r.rolling(3, min_periods=3).sum()
features["REV_5"] = -r.rolling(5, min_periods=5).sum()

# GAP_REV: -log(O_t / C_{t-1})
features["GAP_REV"] = -np.log(_safe_div(open_price, close_price.shift(1), eps=eps))

# ---- C) Volatility / risk (6)
features["RV_10"] = r.rolling(10, min_periods=10).std()
features["RV_20"] = r.rolling(20, min_periods=20).std()
features["RV_60"] = r.rolling(60, min_periods=60).std()

rv10 = features["RV_10"]
rv60 = features["RV_60"]
features["VOL_SHOCK_10_60"] = _safe_div(rv10 - rv60, rv60, eps=eps)

# vol-of-vol: std of RV_20 over last 20 days divided by mean RV_20 (scale-free)
rv20 = features["RV_20"]
features["VOL_OF_VOL_20"] = _safe_div(rv20.rolling(20, min_periods=20).std(),
                                      rv20.rolling(20, min_periods=20).mean(),
                                      eps=eps)

# range-based risk proxy (pick one; you asked "RANGE_1", we implement log(H/L))
features["RANGE_1"] = range_log

# ---- D) Volume / liquidity proxies (5)
features["LVOL_Z_20"] = _rolling_zscore(logV, 20)           # z(log V_t; 20)
features["DVOL"] = log_dollar_vol                           # log(C_t * V_t)

# Amihud proxy: |r_t| / $volume_t  (use dollar volume, not raw volume)
features["AMIHUD"] = _safe_div(abs_r, dollar_vol, eps=eps)

# Volume trend: mean(logV;5) - mean(logV;20)
features["VOL_TREND"] = logV.rolling(5, min_periods=5).mean() - logV.rolling(20, min_periods=20).mean()

# Price-volume correlation: corr(r, ΔlogV; 20)
features["PV_CORR_20"] = _rolling_corr(r, dlogV, 20)

# ---- E) VWAP / close pressure (3)
features["CLOSE_VWAP"] = _safe_div(close_price - vwap, close_price, eps=eps)

features["CLOSE_VWAP_RANGE"] = _safe_div(close_price - vwap, range_abs, eps=eps)

# CLV = ((C-L) - (H-C)) / (H-L) = (2C - H - L)/(H-L)
features["CLV"] = _safe_div((2.0 * close_price - high - low), range_abs, eps=eps)


# ============================================================
# 2) Pack into a single 3D-like structure:
#    dict[name] -> DataFrame (date x ticker)
#    (Optional) quick checks
# ============================================================
expected = 24
if len(features) != expected:
    raise RuntimeError(f"Expected {expected} features, got {len(features)}")

# Ensure alignment & float dtype
for k in list(features.keys()):
    features[k] = features[k].reindex(index=close_price.index, columns=tickers).astype(float)

# If you want a single MultiIndex DataFrame for storage/ML:
# rows: date, cols: (feature, ticker)
features_panel = pd.concat(features, axis=1)  # columns become MultiIndex: (feature, ticker)

print("Built features:", list(features.keys()))
print("features_panel shape:", features_panel.shape)  # (n_dates, 24 * n_tickers)




