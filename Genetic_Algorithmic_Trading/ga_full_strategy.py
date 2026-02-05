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






API_KEY = "API_KEY"
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


























# ============================================================
# ROUTE A — Walk-forward factor mining + strict holdout backtest
# Engineering-grade, regime-robust screening workflow
# (PACKAGED into class; logic/signatures/names unchanged)
# ============================================================

# -----------------------------
# 0) Global parameters (ALL tunable here)
# -----------------------------

# ---- A) Signal timing ----
DELAY = 1
PRED_HORIZON = 5  # forward return horizon used for IC & backtest target

# ---- B) Outer walk-forward schedule (INDUSTRY DEFAULTS) ----
TRAIN_LOOKBACK_DAYS = 252 * 3
REBALANCE_EVERY = "1M"      # "1M", "3M", "63B" (business days) etc.
BACKTEST_DAYS = 63          # ≈ 1 quarter of trading days
MIN_TRAIN_OBS = 252 * 2

# ---- C) Inner split for screening inside each training window ----
IS_RATIO = 0.70

# ---- D) Normalization & IC ----
CS_STANDARDIZE = "zscore"   # "rank" or "zscore"
IC_METHOD = "pearson"       # "spearman" or "pearson"

# ---- E) Gates (wide-in strict-out) ----
FULL_N_MIN = 250
OOS_N_MIN  = 120

REQUIRE_OOS_SIGN_CONSISTENCY = True
OOS_MEAN_ABS_MIN = 0.0005

USE_ROLLING_STABILITY_FILTER = True
STABILITY_WINDOW_FOR_CHECK = 252
ROLL_MEAN_ABS_THRESHOLD = 0.0005
ROLL_CONSECUTIVE_DAYS = 252

# ---- F) Robust preprocessing (industry default for zscore+pearson) ----
AUTO_ROBUST_ZPEARSON = True
CS_WINSOR_Q = 0.01
Z_CLIP = 3.0

AUTO_WINSORIZE_TARGET_FOR_ZPEARSON = False
TARGET_WINSOR_Q = 0.01

# ---- G) Mean(IC) inference: HAC (Newey-West) ----
USE_HAC_ALWAYS = True
HAC_LAGS_PRIMARY = None      # None => auto=min(max(1,2H),20)
HAC_LAGS_SENSITIVITY = "auto"

# ---- H) Redundancy pruning ----
CORR_THRESHOLD = 0.85
MAX_SELECTED = 60

PRUNE_CORR_SCOPE = "oos"     # "oos" / "full" / "recent"
RECENT_WINDOW_DAYS = 504

PRUNE_CORR_ON = "exposure"   # "exposure" or "ic"
PRUNE_VECTOR_SAMPLE_STEP = 2

# ---- I) Portfolio construction in the holdout backtest window ----
USE_DOLLAR_NEUTRAL = True
MIN_CS_ASSETS_FOR_PORT = 30

# ---- J) Acceleration ----
USE_NUMBA = True



# ============================================================
# 3) Fast daily IC (Numba) + Newey-West tstat (Numba)
# (keep these kernels as module-level for Numba stability/perf)
# ============================================================

if USE_NUMBA:

    @nb.njit(cache=True)
    def _daily_corr_pearson_numba(S, Y):
        T, N = S.shape
        out = np.empty(T, dtype=np.float64)
        out[:] = np.nan
        for t in range(T):
            n = 0
            s_sum = 0.0
            y_sum = 0.0
            for i in range(N):
                s = S[t, i]
                y = Y[t, i]
                if np.isfinite(s) and np.isfinite(y):
                    n += 1
                    s_sum += s
                    y_sum += y
            if n < 2:
                continue
            s_mu = s_sum / n
            y_mu = y_sum / n

            cov = 0.0
            ss = 0.0
            yy = 0.0
            for i in range(N):
                s = S[t, i]
                y = Y[t, i]
                if np.isfinite(s) and np.isfinite(y):
                    ds = s - s_mu
                    dy = y - y_mu
                    cov += ds * dy
                    ss += ds * ds
                    yy += dy * dy
            if ss <= 0.0 or yy <= 0.0:
                continue
            out[t] = cov / math.sqrt(ss * yy)
        return out

    @nb.njit(cache=True)
    def _classic_tstat_mean_numba(x):
        n = 0
        s = 0.0
        for i in range(x.size):
            v = x[i]
            if np.isfinite(v):
                n += 1
                s += v
        if n < 2:
            return np.nan
        mu = s / n
        ss = 0.0
        for i in range(x.size):
            v = x[i]
            if np.isfinite(v):
                d = v - mu
                ss += d * d
        var = ss / (n - 1)
        if var <= 0:
            return np.nan
        return mu / math.sqrt(var / n)

    @nb.njit(cache=True)
    def _newey_west_tstat_mean_numba(x, lags):
        n = 0
        for i in range(x.size):
            if np.isfinite(x[i]):
                n += 1
        if n < 2:
            return np.nan

        v = np.empty(n, dtype=np.float64)
        k = 0
        for i in range(x.size):
            if np.isfinite(x[i]):
                v[k] = x[i]
                k += 1

        mu = 0.0
        for i in range(n):
            mu += v[i]
        mu /= n

        u = np.empty(n, dtype=np.float64)
        for i in range(n):
            u[i] = v[i] - mu

        L = lags
        if L < 0:
            L = 0
        if L > n - 2:
            L = n - 2

        gamma0 = 0.0
        for i in range(n):
            gamma0 += u[i] * u[i]
        gamma0 /= n

        lrv = gamma0
        for lag in range(1, L + 1):
            gk = 0.0
            for i in range(lag, n):
                gk += u[i] * u[i - lag]
            gk /= n
            wk = 1.0 - lag / (L + 1.0)
            lrv += 2.0 * wk * gk

        if lrv <= 0.0 or (not np.isfinite(lrv)):
            return np.nan

        se = math.sqrt(lrv / n)
        if se <= 0:
            return np.nan
        return mu / se

    @nb.njit(cache=True)
    def _nanaware_corr_1d_numba(a, b, min_pairs):
        n = 0
        sa = 0.0
        sb = 0.0
        for i in range(a.size):
            ai = a[i]
            bi = b[i]
            if np.isfinite(ai) and np.isfinite(bi):
                n += 1
                sa += ai
                sb += bi
        if n < min_pairs or n < 2:
            return np.nan
        ma = sa / n
        mb = sb / n

        cov = 0.0
        va = 0.0
        vb = 0.0
        for i in range(a.size):
            ai = a[i]
            bi = b[i]
            if np.isfinite(ai) and np.isfinite(bi):
                da = ai - ma
                db = bi - mb
                cov += da * db
                va += da * da
                vb += db * db
        if va <= 0.0 or vb <= 0.0:
            return np.nan
        return cov / math.sqrt(va * vb)

else:
    _daily_corr_pearson_numba = None
    _newey_west_tstat_mean_numba = None
    _classic_tstat_mean_numba = None
    _nanaware_corr_1d_numba = None


# ============================================================
# 4) Screening core outputs (unchanged)
# ============================================================

@dataclass
class ScreeningOutputs:
    ic_full: pd.DataFrame
    ic_is: pd.DataFrame
    ic_oos: pd.DataFrame
    qa_table: pd.DataFrame
    selection_table: pd.DataFrame
    candidates_table: pd.DataFrame
    pruned_selected: List[str]
    prune_log: pd.DataFrame
    std_signals_train: Dict[str, np.ndarray]
    train_index: pd.Index
    is_index: pd.Index
    oos_index: pd.Index


# ============================================================
# Class Packaging (Industry-grade + backward-compatible aliases)
# ============================================================

class RouteAEngine:
    """
    Route A workflow packed into a single class (stateless static methods).
    - Keeps all original function names/signatures/logic intact.
    - Exposes global parameters also as class attributes for discoverability.
    - Provides backward-compatible global aliases after class definition.
    """

    # ---- mirror globals as class attrs (discoverable for others) ----
    DELAY = DELAY
    PRED_HORIZON = PRED_HORIZON
    TRAIN_LOOKBACK_DAYS = TRAIN_LOOKBACK_DAYS
    REBALANCE_EVERY = REBALANCE_EVERY
    BACKTEST_DAYS = BACKTEST_DAYS
    MIN_TRAIN_OBS = MIN_TRAIN_OBS
    IS_RATIO = IS_RATIO
    CS_STANDARDIZE = CS_STANDARDIZE
    IC_METHOD = IC_METHOD
    FULL_N_MIN = FULL_N_MIN
    OOS_N_MIN = OOS_N_MIN
    REQUIRE_OOS_SIGN_CONSISTENCY = REQUIRE_OOS_SIGN_CONSISTENCY
    OOS_MEAN_ABS_MIN = OOS_MEAN_ABS_MIN
    USE_ROLLING_STABILITY_FILTER = USE_ROLLING_STABILITY_FILTER
    STABILITY_WINDOW_FOR_CHECK = STABILITY_WINDOW_FOR_CHECK
    ROLL_MEAN_ABS_THRESHOLD = ROLL_MEAN_ABS_THRESHOLD
    ROLL_CONSECUTIVE_DAYS = ROLL_CONSECUTIVE_DAYS
    AUTO_ROBUST_ZPEARSON = AUTO_ROBUST_ZPEARSON
    CS_WINSOR_Q = CS_WINSOR_Q
    Z_CLIP = Z_CLIP
    AUTO_WINSORIZE_TARGET_FOR_ZPEARSON = AUTO_WINSORIZE_TARGET_FOR_ZPEARSON
    TARGET_WINSOR_Q = TARGET_WINSOR_Q
    USE_HAC_ALWAYS = USE_HAC_ALWAYS
    HAC_LAGS_PRIMARY = HAC_LAGS_PRIMARY
    CORR_THRESHOLD = CORR_THRESHOLD
    MAX_SELECTED = MAX_SELECTED
    PRUNE_CORR_SCOPE = PRUNE_CORR_SCOPE
    RECENT_WINDOW_DAYS = RECENT_WINDOW_DAYS
    PRUNE_CORR_ON = PRUNE_CORR_ON
    PRUNE_VECTOR_SAMPLE_STEP = PRUNE_VECTOR_SAMPLE_STEP
    USE_DOLLAR_NEUTRAL = USE_DOLLAR_NEUTRAL
    MIN_CS_ASSETS_FOR_PORT = MIN_CS_ASSETS_FOR_PORT
    USE_NUMBA = USE_NUMBA

    # ============================================================
    # 1) Utilities: alignment & indexing
    # ============================================================

    @staticmethod
    def ensure_aligned(features: dict, logret: pd.DataFrame):
        base = logret.sort_index()
        cols = list(base.columns)
        aligned = {
            name: df.reindex(index=base.index, columns=cols).sort_index().astype(float)
            for name, df in features.items()
        }
        return aligned, base.astype(float)

    @staticmethod
    def valid_index(idx: pd.Index, delay: int, horizon: int) -> pd.Index:
        if delay < 0 or horizon < 0:
            raise ValueError("DELAY and PRED_HORIZON must be >= 0")
        start = int(delay)
        end = len(idx) - int(horizon) if horizon > 0 else len(idx)
        if end <= start:
            raise ValueError("Not enough dates after applying DELAY/PRED_HORIZON.")
        return idx[start:end]

    @staticmethod
    def time_split_index(idx: pd.Index, is_ratio: float):
        n = len(idx)
        cut = int(np.floor(n * is_ratio))
        return idx[:cut], idx[cut:]

    @staticmethod
    def mask_by_years(idx: pd.Index, start_year: int, end_year: int):
        y = pd.Index(idx).year
        return idx[(y >= start_year) & (y <= end_year)]

    @staticmethod
    def safe_df(df):
        df = pd.DataFrame(df).copy()
        return df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def safe_series(s):
        s = pd.Series(s).copy()
        return s.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _date_to_pos(idx: pd.Index, dt) -> int:
        """Nearest position at or after dt (left search)."""
        return int(idx.searchsorted(pd.Timestamp(dt), side="left"))

    @staticmethod
    def _slice_by_pos(idx: pd.Index, l: int, r: int) -> pd.Index:
        l = max(0, l); r = min(len(idx), r)
        return idx[l:r]

    # ============================================================
    # 2) Numpy-based cross-sectional transforms (fast)
    # ============================================================

    @staticmethod
    def cs_rank_centered(X: np.ndarray) -> np.ndarray:
        """
        Row-wise rank pct in (0,1], then center to [-0.5,0.5].
        Note: ranking is O(N log N) per day; use only if needed.
        """
        T, N = X.shape
        out = np.full((T, N), np.nan, dtype=np.float64)
        for t in range(T):
            x = X[t]
            m = np.isfinite(x)
            k = int(m.sum())
            if k <= 1:
                continue
            vals = x[m]
            order = np.argsort(vals, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, k + 1, dtype=np.float64)
            pct = ranks / k
            out[t, np.where(m)[0]] = pct - 0.5
        return out

    @staticmethod
    def cs_zscore(X: np.ndarray) -> np.ndarray:
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, ddof=1, keepdims=True)
        sd = np.where(sd <= 0, np.nan, sd)
        return (X - mu) / sd

    @staticmethod
    def cs_winsorize_quantile(X: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Row-wise winsorize by nanquantile(q) and nanquantile(1-q).
        Returns winsorized X and mask of changed entries.
        """
        if q is None or q <= 0:
            mask = np.zeros_like(X, dtype=bool)
            return X, mask
        lo = np.nanquantile(X, q, axis=1, keepdims=True)
        hi = np.nanquantile(X, 1 - q, axis=1, keepdims=True)
        Xw = np.clip(X, lo, hi)
        mask = np.isfinite(X) & (X != Xw)
        return Xw, mask

    @staticmethod
    def standardize_signal_np(
        S: np.ndarray,
        cs_standardize: str,
        ic_method: str,
        auto_robust_zpearson: bool,
        winsor_q: float,
        z_clip: float
    ) -> Tuple[np.ndarray, dict]:
        diag = {}
        finite = np.isfinite(S)
        diag["coverage_rate"] = float(finite.mean())
        diag["avg_cs_n"] = float(finite.sum(axis=1).mean())

        if cs_standardize == "rank":
            Z = cs_rank_centered(S)
            diag["winsor_rate"] = 0.0
            diag["clip_rate"] = 0.0
            return Z, diag

        if cs_standardize == "zscore":
            if auto_robust_zpearson and ic_method == "pearson":
                Sw, mask_w = cs_winsorize_quantile(S, winsor_q)
                Z = cs_zscore(Sw)
                if z_clip is not None:
                    mask_c = np.isfinite(Z) & (np.abs(Z) > abs(z_clip))
                    Z = np.clip(Z, -abs(z_clip), abs(z_clip))
                else:
                    mask_c = np.zeros_like(Z, dtype=bool)

                denom = max(1, int(np.isfinite(S).sum()))
                denom2 = max(1, int(np.isfinite(Z).sum()))
                diag["winsor_rate"] = float(mask_w.sum() / denom)
                diag["clip_rate"] = float(mask_c.sum() / denom2)
                return Z, diag
            else:
                Z = cs_zscore(S)
                diag["winsor_rate"] = 0.0
                diag["clip_rate"] = 0.0
                return Z, diag

        raise ValueError("CS_STANDARDIZE must be 'rank' or 'zscore'.")

    # ============================================================
    # 3) IC + Newey-West inference
    # ============================================================

    @staticmethod
    def daily_ic_np(S_std: np.ndarray, Y: np.ndarray, method: str) -> np.ndarray:
        if method == "pearson":
            if USE_NUMBA:
                return _daily_corr_pearson_numba(S_std, Y)
            out = []
            for t in range(S_std.shape[0]):
                s = S_std[t]
                y = Y[t]
                m = np.isfinite(s) & np.isfinite(y)
                if m.sum() < 2:
                    out.append(np.nan); continue
                out.append(np.corrcoef(s[m], y[m])[0, 1])
            return np.array(out, dtype=float)

        if method == "spearman":
            Sr = cs_rank_centered(S_std) + 0.5
            Yr = cs_rank_centered(Y) + 0.5
            if USE_NUMBA:
                return _daily_corr_pearson_numba(Sr, Yr)
            out = []
            for t in range(Sr.shape[0]):
                s = Sr[t]; y = Yr[t]
                m = np.isfinite(s) & np.isfinite(y)
                if m.sum() < 2:
                    out.append(np.nan); continue
                out.append(np.corrcoef(s[m], y[m])[0, 1])
            return np.array(out, dtype=float)

        raise ValueError("IC_METHOD must be 'pearson' or 'spearman'.")

    # HAC lags = min(2*H, 20)
    @staticmethod
    def choose_primary_hac_lag(pred_horizon: int, n: int) -> int:
        if HAC_LAGS_PRIMARY is not None:
            L = int(HAC_LAGS_PRIMARY)
        else:
            L = int(min(max(1, 2 * int(pred_horizon)), 20))
        return int(max(0, min(L, n - 2))) if n >= 2 else 0

    @staticmethod
    def classic_tstat_mean(x: np.ndarray) -> float:
        if USE_NUMBA:
            return float(_classic_tstat_mean_numba(x))
        s = pd.Series(x).dropna().values.astype(float)
        if len(s) < 2:
            return np.nan
        mu = s.mean()
        sd = s.std(ddof=1)
        if sd <= 0:
            return np.nan
        return float(mu / (sd / np.sqrt(len(s))))

    @staticmethod
    def newey_west_tstat_mean(x: np.ndarray, lags: int) -> float:
        if USE_NUMBA:
            return float(_newey_west_tstat_mean_numba(x, int(lags)))
        s = pd.Series(x).dropna().values.astype(float)
        n = len(s)
        if n < 2:
            return np.nan
        mu = s.mean()
        u = s - mu
        L = int(max(0, min(lags, n - 2)))
        gamma0 = (u @ u) / n
        lrv = gamma0
        for k in range(1, L + 1):
            gk = (u[k:] @ u[:-k]) / n
            wk = 1.0 - k / (L + 1.0)
            lrv += 2.0 * wk * gk
        if lrv <= 0:
            return np.nan
        se = np.sqrt(lrv / n)
        return float(mu / se) if se > 0 else np.nan

    @staticmethod
    def ic_summary_np(ic: np.ndarray, pred_horizon: int) -> dict:
        x = ic[np.isfinite(ic)]
        n = int(x.size)
        if n < 2:
            return {"N": n, "IC_mean": np.nan, "IC_std": np.nan, "IC_IR": np.nan,
                    "IC_pos_pct": np.nan, "t_classic": np.nan, "t_hac_primary": np.nan,
                    "hac_lag_primary": 0}

        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        ir = float(mu / sd) if np.isfinite(sd) and sd > 0 else np.nan
        pos = float(np.mean(x > 0)) * 100.0

        t_cls = classic_tstat_mean(ic)

        if USE_HAC_ALWAYS:
            Lp = choose_primary_hac_lag(pred_horizon, n)
            t_hac = newey_west_tstat_mean(ic, Lp)
        else:
            Lp = 0
            t_hac = np.nan

        return {"N": n, "IC_mean": mu, "IC_std": sd, "IC_IR": ir,
                "IC_pos_pct": pos, "t_classic": float(t_cls),
                "t_hac_primary": float(t_hac), "hac_lag_primary": int(Lp)}

    @staticmethod
    def rolling_mean_np(x: np.ndarray, window: int) -> np.ndarray:
        s = pd.Series(x)
        return s.rolling(window, min_periods=window).mean().values

    @staticmethod
    def long_zero_check(roll_mean: np.ndarray, threshold: float, consecutive_days: int) -> bool:
        z = np.where(np.isfinite(roll_mean), np.abs(roll_mean) < threshold, False)
        max_run = 0
        cur = 0
        for v in z:
            if v:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        return bool(max_run >= consecutive_days)

    # ============================================================
    # 4) Screening core: run on a given training window index
    # ============================================================

    @staticmethod
    def _prune_corr_scope_index(valid_idx: pd.Index, is_idx: pd.Index, oos_idx: pd.Index) -> pd.Index:
        if PRUNE_CORR_SCOPE == "full":
            return valid_idx
        if PRUNE_CORR_SCOPE == "oos":
            return oos_idx
        if PRUNE_CORR_SCOPE == "recent":
            w = int(RECENT_WINDOW_DAYS)
            return valid_idx[-w:] if len(valid_idx) > w else valid_idx
        raise ValueError("PRUNE_CORR_SCOPE must be 'full'/'oos'/'recent'.")

    @staticmethod
    def _vectorize_exposure_np(S_std: np.ndarray, sample_step: int = 1) -> np.ndarray:
        if sample_step is None or sample_step <= 1:
            return S_std.reshape(-1)
        return S_std[::sample_step].reshape(-1)

    @staticmethod
    def _vectorize_ic_np(ic: np.ndarray, sample_step: int = 1) -> np.ndarray:
        if sample_step is None or sample_step <= 1:
            return ic.astype(float)
        return ic[::sample_step].astype(float)

    @staticmethod
    def nanaware_corr_1d(a: np.ndarray, b: np.ndarray, min_pairs: int = 200) -> float:
        if USE_NUMBA:
            return float(_nanaware_corr_1d_numba(a.astype(np.float64), b.astype(np.float64), int(min_pairs)))
        m = np.isfinite(a) & np.isfinite(b)
        k = int(m.sum())
        if k < min_pairs:
            return np.nan
        av = a[m]; bv = b[m]
        sa = av.std(ddof=1); sb = bv.std(ddof=1)
        if sa <= 0 or sb <= 0:
            return np.nan
        return float(np.corrcoef(av, bv)[0, 1])

    @staticmethod
    def greedy_corr_prune(
        candidates: pd.DataFrame,
        vecs: Dict[str, np.ndarray],
        corr_threshold: float,
        score_col: str,
        max_selected: Optional[int] = None,
        min_pairs: int = 200
    ):
        ordered = candidates.sort_values(score_col, ascending=False).index.tolist()
        kept = []
        dropped = []

        for f in ordered:
            if max_selected is not None and len(kept) >= int(max_selected):
                break

            ok = True
            reason = None
            for g in kept:
                c = nanaware_corr_1d(vecs[f], vecs[g], min_pairs=min_pairs)
                if np.isfinite(c) and abs(c) > corr_threshold:
                    ok = False
                    reason = (g, c)
                    break

            if ok:
                kept.append(f)
            else:
                dropped.append({"factor": f, "dropped_because_of": reason[0], "corr": reason[1]})

        prune_log = pd.DataFrame(dropped).set_index("factor") if dropped else pd.DataFrame(
            columns=["dropped_because_of", "corr"]
        )
        return kept, prune_log

    @staticmethod
    def screen_and_select_factors(
        features_np: Dict[str, np.ndarray],
        logret_np: np.ndarray,
        idx_all: pd.Index,
        y_all: np.ndarray,
        train_idx: pd.Index,
        delay: int = DELAY,
        pred_horizon: int = PRED_HORIZON,
        is_ratio: float = IS_RATIO,
        cs_standardize: str = CS_STANDARDIZE,
        ic_method: str = IC_METHOD
    ) -> ScreeningOutputs:

        valid_idx = train_idx
        is_idx, oos_idx = time_split_index(valid_idx, is_ratio)

        pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)
        pos_valid = pos_map.reindex(valid_idx).values.astype(int)
        Y = y_all[pos_valid, :]

        if (AUTO_WINSORIZE_TARGET_FOR_ZPEARSON and cs_standardize == "zscore" and ic_method == "pearson"):
            Yw, _ = cs_winsorize_quantile(Y, TARGET_WINSOR_Q)
            Y = Yw

        ic_series = {}
        std_signals = {}
        qa_rows = []

        for name, X in features_np.items():
            S = X[pos_valid - delay, :]

            S_std, diag = standardize_signal_np(
                S, cs_standardize, ic_method,
                auto_robust_zpearson=AUTO_ROBUST_ZPEARSON,
                winsor_q=CS_WINSOR_Q,
                z_clip=Z_CLIP
            )

            ic = daily_ic_np(S_std, Y, ic_method)

            ic_series[name] = ic
            std_signals[name] = S_std

            qa_rows.append({
                "factor": name,
                "coverage_rate": diag["coverage_rate"],
                "avg_cs_n": diag["avg_cs_n"],
                "winsor_rate": diag["winsor_rate"],
                "clip_rate": diag["clip_rate"]
            })

        qa_table = pd.DataFrame(qa_rows).set_index("factor")

        full_rows = []
        is_rows = []
        oos_rows = []

        local_pos = pd.Series(np.arange(len(valid_idx)), index=valid_idx)
        m_is = local_pos.reindex(is_idx).values.astype(int)
        m_oos = local_pos.reindex(oos_idx).values.astype(int)

        for f, ic in ic_series.items():
            sm_full = ic_summary_np(ic, pred_horizon)
            sm_is   = ic_summary_np(ic[m_is], pred_horizon) if len(m_is) > 0 else ic_summary_np(np.array([]), pred_horizon)
            sm_oos  = ic_summary_np(ic[m_oos], pred_horizon) if len(m_oos) > 0 else ic_summary_np(np.array([]), pred_horizon)

            full_rows.append({"factor": f, **sm_full})
            is_rows.append({"factor": f, **sm_is})
            oos_rows.append({"factor": f, **sm_oos})

        ic_full = pd.DataFrame(full_rows).set_index("factor")
        ic_is   = pd.DataFrame(is_rows).set_index("factor")
        ic_oos  = pd.DataFrame(oos_rows).set_index("factor")

        ic_full = ic_full.sort_values(["t_hac_primary", "IC_IR"], ascending=False)
        ic_is = ic_is.reindex(ic_full.index)
        ic_oos = ic_oos.reindex(ic_full.index)
        qa_table = qa_table.reindex(ic_full.index)

        rolling_means = {}
        if USE_ROLLING_STABILITY_FILTER and (STABILITY_WINDOW_FOR_CHECK is not None):
            w = int(STABILITY_WINDOW_FOR_CHECK)
            for f, ic in ic_series.items():
                rolling_means[f] = rolling_mean_np(ic, w)

        sel_rows = []
        for f in ic_full.index:
            full_n = ic_full.loc[f, "N"]
            full_mu = ic_full.loc[f, "IC_mean"]

            oos_n = ic_oos.loc[f, "N"]
            oos_mu = ic_oos.loc[f, "IC_mean"]

            gate_full_n = (np.isfinite(full_n) and full_n >= FULL_N_MIN)
            gate_oos_n  = (np.isfinite(oos_n) and oos_n >= OOS_N_MIN)

            sign_ok = True
            if REQUIRE_OOS_SIGN_CONSISTENCY and gate_oos_n and np.isfinite(full_mu) and np.isfinite(oos_mu):
                sign_ok = (np.sign(full_mu) == np.sign(oos_mu)) or (oos_mu == 0.0)

            oos_mean_ok = True
            if (OOS_MEAN_ABS_MIN is not None) and gate_oos_n and np.isfinite(oos_mu):
                oos_mean_ok = (abs(oos_mu) >= OOS_MEAN_ABS_MIN)

            stable_ok = True
            if USE_ROLLING_STABILITY_FILTER and (STABILITY_WINDOW_FOR_CHECK is not None):
                rm = rolling_means.get(f, None)
                if rm is not None:
                    stable_ok = (not long_zero_check(rm, ROLL_MEAN_ABS_THRESHOLD, ROLL_CONSECUTIVE_DAYS))

            passed = gate_full_n and gate_oos_n and sign_ok and oos_mean_ok and stable_ok

            score_num = (
                (ic_oos.loc[f, "t_hac_primary"] if np.isfinite(ic_oos.loc[f, "t_hac_primary"]) else -1e9) * 1_000_000
                + (abs(oos_mu) if np.isfinite(oos_mu) else -1e9) * 10_000
                + (ic_full.loc[f, "t_hac_primary"] if np.isfinite(ic_full.loc[f, "t_hac_primary"]) else -1e9)
            )

            sel_rows.append({
                "factor": f,
                "pass": passed,
                "full_N_ok": gate_full_n,
                "oos_N_ok": gate_oos_n,
                "sign_ok": sign_ok,
                "oos_mean_ok": oos_mean_ok,
                "stable_ok": stable_ok,
                "score_num": score_num,
                "full_mean": full_mu,
                "oos_mean": oos_mu,
                "full_t_hac": ic_full.loc[f, "t_hac_primary"],
                "oos_t_hac": ic_oos.loc[f, "t_hac_primary"],
                "full_t_classic": ic_full.loc[f, "t_classic"],
                "oos_t_classic": ic_oos.loc[f, "t_classic"]
            })

        selection_table = pd.DataFrame(sel_rows).set_index("factor").reindex(ic_full.index)
        candidates = selection_table[selection_table["pass"]].copy().sort_values("score_num", ascending=False)

        corr_scope_idx = _prune_corr_scope_index(valid_idx, is_idx, oos_idx)
        corr_local = pd.Series(np.arange(len(valid_idx)), index=valid_idx).reindex(corr_scope_idx).values.astype(int)

        vecs = {}
        if PRUNE_CORR_ON == "exposure":
            for f in candidates.index:
                S_std = std_signals[f]
                vecs[f] = _vectorize_exposure_np(S_std[corr_local], sample_step=PRUNE_VECTOR_SAMPLE_STEP)
        elif PRUNE_CORR_ON == "ic":
            for f in candidates.index:
                ic = ic_series[f]
                vecs[f] = _vectorize_ic_np(ic[corr_local], sample_step=PRUNE_VECTOR_SAMPLE_STEP)
        else:
            raise ValueError("PRUNE_CORR_ON must be 'exposure' or 'ic'")

        kept, prune_log = greedy_corr_prune(
            candidates=candidates,
            vecs=vecs,
            corr_threshold=CORR_THRESHOLD,
            score_col="score_num",
            max_selected=MAX_SELECTED,
            min_pairs=200
        )

        return ScreeningOutputs(
            ic_full=ic_full, ic_is=ic_is, ic_oos=ic_oos,
            qa_table=qa_table,
            selection_table=selection_table,
            candidates_table=candidates,
            pruned_selected=kept,
            prune_log=prune_log,
            std_signals_train=std_signals,
            train_index=valid_idx,
            is_index=is_idx,
            oos_index=oos_idx
        )

    # ============================================================
    # 5) Holdout backtest on the reserved backtest window (no leakage)
    # ============================================================

    @staticmethod
    def build_composite_signal_on_window(
        features_np: Dict[str, np.ndarray],
        logret_np: np.ndarray,
        idx_all: pd.Index,
        window_idx: pd.Index,
        y_all: np.ndarray,
        selected: List[str],
        delay: int = DELAY,
        pred_horizon: int = PRED_HORIZON,
        cs_standardize: str = CS_STANDARDIZE,
        ic_method: str = IC_METHOD,
        weight_mode: str = "equal"
    ) -> Tuple[np.ndarray, np.ndarray]:

        pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)
        pos_win = pos_map.reindex(window_idx).values.astype(int)

        Y = y_all[pos_win, :]

        W = np.ones(len(selected), dtype=np.float64)
        W = W / max(1, W.sum())

        Sbar = np.zeros((len(window_idx), logret_np.shape[1]), dtype=np.float64)
        Sbar[:] = 0.0

        for w, f in zip(W, selected):
            X = features_np[f]
            S = X[pos_win - delay, :]
            S_std, _ = standardize_signal_np(
                S, cs_standardize, ic_method,
                auto_robust_zpearson=AUTO_ROBUST_ZPEARSON,
                winsor_q=CS_WINSOR_Q,
                z_clip=Z_CLIP
            )
            Sbar += w * S_std

        if cs_standardize == "zscore":
            Sbar = cs_zscore(Sbar)
            if Z_CLIP is not None:
                Sbar = np.clip(Sbar, -abs(Z_CLIP), abs(Z_CLIP))
        else:
            Sbar = cs_rank_centered(Sbar)

        return Sbar, Y

    @staticmethod
    def portfolio_return_from_signal(Sbar: np.ndarray, Y: np.ndarray) -> np.ndarray:
        T, N = Sbar.shape
        rp = np.full(T, np.nan, dtype=np.float64)

        for t in range(T):
            s = Sbar[t]
            y = Y[t]
            m = np.isfinite(s) & np.isfinite(y)
            k = int(m.sum())
            if k < MIN_CS_ASSETS_FOR_PORT:
                continue

            ss = s[m]
            yy = y[m]

            if USE_DOLLAR_NEUTRAL:
                ss = ss - np.mean(ss)

            denom = np.sum(np.abs(ss))
            if denom <= 0 or (not np.isfinite(denom)):
                continue
            w = ss / denom
            rp[t] = float(np.sum(w * yy))

        return rp

    @staticmethod
    def backtest_holdout_window(
        features_np: Dict[str, np.ndarray],
        logret_np: np.ndarray,
        idx_all: pd.Index,
        window_idx: pd.Index,
        y_all: np.ndarray,
        selected: List[str],
    ) -> dict:
        if len(selected) == 0:
            return {"ret": pd.Series(index=window_idx, dtype=float), "summary": {}}

        Sbar, Y = build_composite_signal_on_window(
            features_np, logret_np, idx_all, window_idx, y_all, selected,
            delay=DELAY, pred_horizon=PRED_HORIZON,
            cs_standardize=CS_STANDARDIZE, ic_method=IC_METHOD
        )

        rp = portfolio_return_from_signal(Sbar, Y)
        sr = pd.Series(rp, index=window_idx)

        sm = ic_summary_np(rp, PRED_HORIZON)
        return {"ret": sr, "summary": sm}

    # ============================================================
    # 6) Walk-forward orchestrator (Route A)
    # ============================================================

    @staticmethod
    def _build_rebalance_dates(idx: pd.Index, every: str) -> List[pd.Timestamp]:
        idx = pd.DatetimeIndex(idx)
        start = idx[0]
        end = idx[-1]

        cal = pd.date_range(start=start, end=end, freq=every)
        if len(cal) == 0:
            return [idx[0]]

        dates = []
        for dt in cal:
            pos = idx.searchsorted(dt, side="left")
            if pos < len(idx):
                dates.append(idx[pos])
        dates = sorted(set(dates))
        return dates

    @staticmethod
    def forward_sum_strict_np(logret_np: np.ndarray, H: int) -> np.ndarray:
        T, N = logret_np.shape
        y = np.full((T, N), np.nan, dtype=np.float64)
        if H <= 0:
            return y

        A = np.roll(logret_np, shift=-1, axis=0)
        A[-1, :] = np.nan

        csum = np.nancumsum(np.where(np.isfinite(A), A, 0.0), axis=0)
        ccnt = np.cumsum(np.isfinite(A).astype(np.int32), axis=0)

        for t in range(0, T - H):
            t2 = t + H - 1
            s = csum[t2, :] - (csum[t - 1, :] if t > 0 else 0.0)
            n = ccnt[t2, :] - (ccnt[t - 1, :] if t > 0 else 0)
            y[t, :] = np.where(n == H, s, np.nan)
        return y

    @staticmethod
    def run_walkforward_routeA(
        features: Dict[str, pd.DataFrame],
        logret: pd.DataFrame,
        verbose: bool = True
    ) -> dict:

        features, logret = ensure_aligned(features, logret)
        idx_all = logret.index

        idx_valid_global = valid_index(idx_all, DELAY, PRED_HORIZON)

        logret_np = logret.values.astype(np.float64)
        features_np = {k: v.values.astype(np.float64) for k, v in features.items()}
        y_all = forward_sum_strict_np(logret_np, PRED_HORIZON)

        reb_dates = _build_rebalance_dates(idx_valid_global, REBALANCE_EVERY)

        all_slice_stats = []
        all_holdout_returns = []

        selected_by_slice = {}
        screen_reports = {}

        for t0 in reb_dates:
            t0 = pd.Timestamp(t0)

            p0 = _date_to_pos(idx_valid_global, t0)
            if p0 <= 0:
                continue

            train_end = p0
            train_start = max(0, train_end - int(TRAIN_LOOKBACK_DAYS))
            train_idx = _slice_by_pos(idx_valid_global, train_start, train_end)

            if len(train_idx) < max(MIN_TRAIN_OBS, FULL_N_MIN + OOS_N_MIN):
                continue

            test_start = p0
            test_end = min(len(idx_valid_global), test_start + int(BACKTEST_DAYS))
            holdout_idx = _slice_by_pos(idx_valid_global, test_start, test_end)
            if len(holdout_idx) < 2:
                continue

            scr = screen_and_select_factors(
                features_np=features_np,
                logret_np=logret_np,
                idx_all=idx_all,
                y_all=y_all,
                train_idx=train_idx,
                delay=DELAY,
                pred_horizon=PRED_HORIZON,
                is_ratio=IS_RATIO,
                cs_standardize=CS_STANDARDIZE,
                ic_method=IC_METHOD
            )

            selected = scr.pruned_selected
            selected_by_slice[t0] = selected
            screen_reports[t0] = scr

            bt = backtest_holdout_window(
                features_np=features_np,
                logret_np=logret_np,
                idx_all=idx_all,
                window_idx=holdout_idx,
                y_all=y_all,
                selected=selected
            )

            r = bt["ret"]
            sm = bt["summary"]

            holdout_mean = sm.get("IC_mean", np.nan)
            holdout_tnw  = sm.get("t_hac_primary", np.nan)
            if not ((np.isfinite(holdout_mean) and (holdout_mean > 0.0)) or (np.isfinite(holdout_tnw) and (holdout_tnw > 0.0))):
                if verbose:
                    print(f"[WF][skip] {t0.date()} holdout gate fail | mean={holdout_mean} | tNW={holdout_tnw}")
                continue

            row = {
                "rebalance_date": t0,
                "train_start": train_idx[0],
                "train_end": train_idx[-1],
                "holdout_start": holdout_idx[0],
                "holdout_end": holdout_idx[-1],
                "n_selected": len(selected),
            }
            for k, v in sm.items():
                row[f"holdout_{k}"] = v

            all_slice_stats.append(row)
            all_holdout_returns.append(r)

            if verbose:
                print(f"[WF] {t0.date()} | train={train_idx[0].date()}..{train_idx[-1].date()} "
                      f"| holdout={holdout_idx[0].date()}..{holdout_idx[-1].date()} "
                      f"| selected={len(selected)} | holdout_mean={sm.get('IC_mean', np.nan):.6f} "
                      f"| holdout_tNW={sm.get('t_hac_primary', np.nan):.3f}")

        if len(all_holdout_returns) > 0:
            ret_all = pd.concat(all_holdout_returns).sort_index()
            ret_all = ret_all.groupby(ret_all.index).mean()
        else:
            ret_all = pd.Series(dtype=float)

        slice_stats = pd.DataFrame(all_slice_stats).set_index("rebalance_date") if all_slice_stats else pd.DataFrame()

        out = {
            "holdout_returns": ret_all,
            "slice_stats": slice_stats,
            "selected_by_slice": selected_by_slice,
            "screen_reports": screen_reports,
            "config": {
                "DELAY": DELAY, "PRED_HORIZON": PRED_HORIZON,
                "TRAIN_LOOKBACK_DAYS": TRAIN_LOOKBACK_DAYS,
                "REBALANCE_EVERY": REBALANCE_EVERY,
                "BACKTEST_DAYS": BACKTEST_DAYS,
                "IS_RATIO": IS_RATIO,
                "CS_STANDARDIZE": CS_STANDARDIZE,
                "IC_METHOD": IC_METHOD,
                "GATES": {
                    "FULL_N_MIN": FULL_N_MIN, "OOS_N_MIN": OOS_N_MIN,
                    "REQUIRE_OOS_SIGN_CONSISTENCY": REQUIRE_OOS_SIGN_CONSISTENCY,
                    "OOS_MEAN_ABS_MIN": OOS_MEAN_ABS_MIN,
                    "USE_ROLLING_STABILITY_FILTER": USE_ROLLING_STABILITY_FILTER,
                    "STABILITY_WINDOW_FOR_CHECK": STABILITY_WINDOW_FOR_CHECK,
                    "ROLL_MEAN_ABS_THRESHOLD": ROLL_MEAN_ABS_THRESHOLD,
                    "ROLL_CONSECUTIVE_DAYS": ROLL_CONSECUTIVE_DAYS,
                },
                "ROBUST": {
                    "AUTO_ROBUST_ZPEARSON": AUTO_ROBUST_ZPEARSON,
                    "CS_WINSOR_Q": CS_WINSOR_Q,
                    "Z_CLIP": Z_CLIP,
                },
                "HAC": {
                    "USE_HAC_ALWAYS": USE_HAC_ALWAYS,
                    "HAC_LAGS_PRIMARY": HAC_LAGS_PRIMARY,
                },
                "PRUNE": {
                    "CORR_THRESHOLD": CORR_THRESHOLD,
                    "MAX_SELECTED": MAX_SELECTED,
                    "PRUNE_CORR_SCOPE": PRUNE_CORR_SCOPE,
                    "PRUNE_CORR_ON": PRUNE_CORR_ON,
                    "PRUNE_VECTOR_SAMPLE_STEP": PRUNE_VECTOR_SAMPLE_STEP,
                },
            }
        }
        return out

    # ============================================================
    # 7) Report-level plotting (only the most important)
    # ============================================================

    @staticmethod
    def set_presentation_style():
        mpl.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "savefig.dpi": 200,
            "figure.dpi": 120,
        })

    @staticmethod
    def legend_below(ax, ncol=4, fontsize=11, frameon=False, yshift=-0.20):
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, yshift), ncol=ncol,
                  fontsize=fontsize, frameon=frameon)

    @staticmethod
    def finalize_figure(fig, bottom=0.22):
        fig.tight_layout()
        fig.subplots_adjust(bottom=bottom)

    @staticmethod
    def plot_walkforward_report(wf: dict, save_dir: Optional[str] = None):
        set_presentation_style()

        ret = wf.get("holdout_returns", pd.Series(dtype=float)).dropna()
        stats = wf.get("slice_stats", pd.DataFrame())

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(13, 6))
        eq = ret.cumsum()
        ax.plot(eq.index, eq.values, linewidth=2.0, label="Holdout cumulative (H-day return units)")
        ax.axhline(0, linewidth=1.0)
        ax.set_title("Walk-forward holdout equity curve (strict holdout windows)")
        ax.set_xlabel("date")
        ax.set_ylabel("cumulative return")
        legend_below(ax, ncol=1, yshift=-0.22)
        finalize_figure(fig, bottom=0.26)
        if save_dir:
            fig.savefig(os.path.join(save_dir, "wf_equity_curve.png"), bbox_inches="tight")
        plt.show()

        if len(ret) >= 60:
            fig, ax = plt.subplots(figsize=(13, 6))
            rm = ret.rolling(63, min_periods=63).mean()
            ax.plot(rm.index, rm.values, linewidth=2.0, label="rolling mean (63d)")
            ax.axhline(0, linewidth=1.0)
            ax.set_title("Holdout rolling mean (63 trading days)")
            ax.set_xlabel("date")
            ax.set_ylabel("rolling mean return")
            legend_below(ax, ncol=1, yshift=-0.22)
            finalize_figure(fig, bottom=0.26)
            if save_dir:
                fig.savefig(os.path.join(save_dir, "wf_rolling_mean.png"), bbox_inches="tight")
            plt.show()

        if stats is not None and len(stats) > 0:
            if "holdout_t_hac_primary" in stats.columns:
                fig, ax = plt.subplots(figsize=(13, 5))
                ax.bar(stats.index.astype(str), stats["holdout_t_hac_primary"].values)
                ax.axhline(0, linewidth=1.0)
                ax.set_title("Per-slice holdout Newey-West t-stat (mean return)")
                ax.set_xlabel("rebalance date")
                ax.set_ylabel("t_hac_primary")
                ax.tick_params(axis="x", rotation=45)
                finalize_figure(fig, bottom=0.28)
                if save_dir:
                    fig.savefig(os.path.join(save_dir, "wf_slice_tstat.png"), bbox_inches="tight")
                plt.show()

            fig, ax = plt.subplots(figsize=(13, 5))
            ax.bar(stats.index.astype(str), stats["n_selected"].values)
            ax.set_title("Per-slice number of selected factors (after gates + corr prune)")
            ax.set_xlabel("rebalance date")
            ax.set_ylabel("n_selected")
            ax.tick_params(axis="x", rotation=45)
            finalize_figure(fig, bottom=0.28)
            if save_dir:
                fig.savefig(os.path.join(save_dir, "wf_slice_n_selected.png"), bbox_inches="tight")
            plt.show()

            if ("holdout_IC_mean" in stats.columns) and ("holdout_t_hac_primary" in stats.columns):
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.scatter(stats["holdout_IC_mean"].values, stats["holdout_t_hac_primary"].values, s=45, alpha=0.9)
                ax.axhline(0, linewidth=1.0)
                ax.axvline(0, linewidth=1.0)
                ax.set_title("Per-slice holdout mean vs t-stat")
                ax.set_xlabel("holdout mean return")
                ax.set_ylabel("holdout t_hac_primary")
                finalize_figure(fig, bottom=0.10)
                if save_dir:
                    fig.savefig(os.path.join(save_dir, "wf_slice_scatter_mean_vs_t.png"), bbox_inches="tight")
                plt.show()

        print("[done] Walk-forward report plots generated.")


# ------------------------------------------------------------
# Backward-compatible global aliases (names unchanged)
# ------------------------------------------------------------

ensure_aligned = RouteAEngine.ensure_aligned
valid_index = RouteAEngine.valid_index
time_split_index = RouteAEngine.time_split_index
mask_by_years = RouteAEngine.mask_by_years
safe_df = RouteAEngine.safe_df
safe_series = RouteAEngine.safe_series
_date_to_pos = RouteAEngine._date_to_pos
_slice_by_pos = RouteAEngine._slice_by_pos

cs_rank_centered = RouteAEngine.cs_rank_centered
cs_zscore = RouteAEngine.cs_zscore
cs_winsorize_quantile = RouteAEngine.cs_winsorize_quantile
standardize_signal_np = RouteAEngine.standardize_signal_np

daily_ic_np = RouteAEngine.daily_ic_np
choose_primary_hac_lag = RouteAEngine.choose_primary_hac_lag
classic_tstat_mean = RouteAEngine.classic_tstat_mean
newey_west_tstat_mean = RouteAEngine.newey_west_tstat_mean
ic_summary_np = RouteAEngine.ic_summary_np
rolling_mean_np = RouteAEngine.rolling_mean_np
long_zero_check = RouteAEngine.long_zero_check

_prune_corr_scope_index = RouteAEngine._prune_corr_scope_index
_vectorize_exposure_np = RouteAEngine._vectorize_exposure_np
_vectorize_ic_np = RouteAEngine._vectorize_ic_np
nanaware_corr_1d = RouteAEngine.nanaware_corr_1d
greedy_corr_prune = RouteAEngine.greedy_corr_prune

screen_and_select_factors = RouteAEngine.screen_and_select_factors
build_composite_signal_on_window = RouteAEngine.build_composite_signal_on_window
portfolio_return_from_signal = RouteAEngine.portfolio_return_from_signal
backtest_holdout_window = RouteAEngine.backtest_holdout_window

_build_rebalance_dates = RouteAEngine._build_rebalance_dates
forward_sum_strict_np = RouteAEngine.forward_sum_strict_np
run_walkforward_routeA = RouteAEngine.run_walkforward_routeA

set_presentation_style = RouteAEngine.set_presentation_style
legend_below = RouteAEngine.legend_below
finalize_figure = RouteAEngine.finalize_figure
plot_walkforward_report = RouteAEngine.plot_walkforward_report


# ============================================================
# 8) Example usage (Route A) 
# ============================================================

wf = run_walkforward_routeA(features, logret, verbose=True)
print(wf["slice_stats"].head(10))
plot_walkforward_report(wf, save_dir=None)






















# ============================================================
# LASSO FILTER LAYER — plugs into your Route A wf output
# No leakage | Strict NaN forward returns | Purged+Embargo KFold by date
# (CLASS-WRAPPED per your checklist ONLY; math/IO/names unchanged)
# ============================================================



class LassoFilterLayer:
    # -----------------------------
    # Tunables
    # -----------------------------
    LASSO_ALPHAS = np.logspace(-7, -2, 80)

    N_SPLITS = 5
    RANDOM_STATE = 42

    TARGET_CS_RANK = True

    # [CHANGED #5.2-3] auto later: max(30, ceil(0.3*N))
    REQUIRE_DATE_MIN_OBS = None

    DROP_EMPTY_DATES = True

    USE_HAC = True
    HAC_LAGS = None   # None => auto = min(max(1,2H),20)

    LASSO_NONZERO_EPS = 0.0
    MAX_COEF_TO_PLOT = 30

    FIG_DPI = 200

    # -----------------------------
    # Make feature processing consistent with Route A
    # -----------------------------
    FEATURE_CS_STANDARDIZE = "zscore"
    FEATURE_ROBUST_Z = True
    FEATURE_WINSOR_Q = 0.01
    FEATURE_ZCLIP = 3.0

    # [CHANGED #5.2-1/2] do NOT fill 0
    FILL_MISSING_EXPOSURE_WITH_ZERO = False

    # [UNCHANGED but required by you #5.2-4]
    DROP_ALLZERO_ROWS = True

    LASSO_PURGE_DAYS = None
    LASSO_EMBARGO_DAYS = None

    # [CHANGED #5] light gate
    LASSO_T_HAC_GATE = 0

    # --------- NUMBA/Numpy acceleration hooks (NO logic change) ----------
    try:
        from numba import njit
        _NUMBA_OK = True
    except Exception:
        _NUMBA_OK = False

        def njit(*args, **kwargs):
            def _wrap(fn):
                return fn
            return _wrap

    # ============================================================
    # 1) Plot style helpers (matplotlib only; legend below)
    # ============================================================

    @classmethod
    def set_presentation_style(cls):
        FIG_DPI = cls.FIG_DPI
        mpl.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "savefig.dpi": FIG_DPI,
            "figure.dpi": 120,
        })

    @staticmethod
    def legend_below(ax, ncol=3, fontsize=11, frameon=False, yshift=-0.22):
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, yshift), ncol=ncol,
                  fontsize=fontsize, frameon=frameon)

    @staticmethod
    def finalize_figure(fig, bottom=0.22):
        fig.tight_layout()
        fig.subplots_adjust(bottom=bottom)

    @staticmethod
    def _savefig(fig, name: str, save_dir: Optional[str]):
        if save_dir is None:
            return
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, name), bbox_inches="tight")

    # ============================================================
    # 2) Core utilities (alignment + strict forward returns)
    # ============================================================

    @staticmethod
    @njit(cache=True)
    def _nw_tstat_numba(v: np.ndarray, lags: int) -> float:
        n = v.size
        if n < 2:
            return np.nan
        mu = 0.0
        for i in range(n):
            mu += v[i]
        mu /= n

        gamma0 = 0.0
        for i in range(n):
            u = v[i] - mu
            gamma0 += u * u
        gamma0 /= n

        L = int(lags)
        if L < 0:
            L = 0
        if L > n - 2:
            L = n - 2

        lrv = gamma0
        for k in range(1, L + 1):
            gk = 0.0
            for i in range(k, n):
                gk += (v[i] - mu) * (v[i - k] - mu)
            gk /= n
            wk = 1.0 - k / (L + 1.0)
            lrv += 2.0 * wk * gk

        if (not np.isfinite(lrv)) or lrv <= 0.0:
            return np.nan
        se = np.sqrt(lrv / n)
        if (not np.isfinite(se)) or se <= 0.0:
            return np.nan
        return mu / se

    @staticmethod
    @njit(cache=True)
    def _daily_rankic_sorted_numba(date_ns: np.ndarray, pred: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = date_ns.size
        out_date = np.empty(n, dtype=np.int64)
        out_ic = np.empty(n, dtype=np.float64)
        out_k = 0

        i = 0
        while i < n:
            d = date_ns[i]
            j = i + 1
            while j < n and date_ns[j] == d:
                j += 1

            # count finite
            m = 0
            for t in range(i, j):
                pv = pred[t]
                yv = y[t]
                if np.isfinite(pv) and np.isfinite(yv):
                    m += 1

            if m >= 3:
                p = np.empty(m, dtype=np.float64)
                yy = np.empty(m, dtype=np.float64)
                kk = 0
                for t in range(i, j):
                    pv = pred[t]
                    yv = y[t]
                    if np.isfinite(pv) and np.isfinite(yv):
                        p[kk] = pv
                        yy[kk] = yv
                        kk += 1

                # uniqueness gate (match your old behavior: constant -> ic=0)
                pmin = p[0]; pmax = p[0]
                ymin = yy[0]; ymax = yy[0]
                for t in range(1, m):
                    if p[t] < pmin: pmin = p[t]
                    if p[t] > pmax: pmax = p[t]
                    if yy[t] < ymin: ymin = yy[t]
                    if yy[t] > ymax: ymax = yy[t]

                if pmin == pmax or ymin == ymax:
                    ic = 0.0
                else:
                    # ranks with average ties (Spearman)
                    ordp = np.argsort(p)
                    ordy = np.argsort(yy)
                    rp = np.empty(m, dtype=np.float64)
                    ry = np.empty(m, dtype=np.float64)

                    ii = 0
                    while ii < m:
                        jj = ii
                        v0 = p[ordp[ii]]
                        while (jj + 1) < m and p[ordp[jj + 1]] == v0:
                            jj += 1
                        r = 0.5 * (ii + jj) + 1.0
                        for k in range(ii, jj + 1):
                            rp[ordp[k]] = r
                        ii = jj + 1

                    ii = 0
                    while ii < m:
                        jj = ii
                        v0 = yy[ordy[ii]]
                        while (jj + 1) < m and yy[ordy[jj + 1]] == v0:
                            jj += 1
                        r = 0.5 * (ii + jj) + 1.0
                        for k in range(ii, jj + 1):
                            ry[ordy[k]] = r
                        ii = jj + 1

                    # Pearson corr of ranks
                    mrp = 0.0; mry = 0.0
                    for t in range(m):
                        mrp += rp[t]
                        mry += ry[t]
                    mrp /= m
                    mry /= m

                    v1 = 0.0; v2 = 0.0; cov = 0.0
                    for t in range(m):
                        a = rp[t] - mrp
                        b = ry[t] - mry
                        cov += a * b
                        v1 += a * a
                        v2 += b * b

                    if v1 <= 0.0 or v2 <= 0.0:
                        ic = 0.0
                    else:
                        ic = cov / np.sqrt(v1 * v2)

                out_date[out_k] = d
                out_ic[out_k] = ic
                out_k += 1

            i = j

        return out_date[:out_k], out_ic[:out_k]

    @staticmethod
    def ensure_aligned(features: Dict[str, pd.DataFrame], logret: pd.DataFrame):
        base = logret.sort_index()
        cols = list(base.columns)
        aligned = {
            name: df.reindex(index=base.index, columns=cols).sort_index().astype(float)
            for name, df in features.items()
        }
        return aligned, base.astype(float)

    @staticmethod
    def forward_sum_strict(logret: pd.DataFrame, horizon: int) -> pd.DataFrame:
        # numpy strict forward sum (same semantics as your pandas rolling version)
        lr = logret.values.astype(np.float64)
        T, N = lr.shape
        h = int(horizon)

        out = np.full((T, N), np.nan, dtype=np.float64)
        if h <= 0 or T == 0:
            return pd.DataFrame(out, index=logret.index, columns=logret.columns)

        finite = np.isfinite(lr).astype(np.int32)
        lr0 = np.where(np.isfinite(lr), lr, 0.0)

        csum = np.vstack([np.zeros((1, N), dtype=np.float64), np.cumsum(lr0, axis=0)])
        ccnt = np.vstack([np.zeros((1, N), dtype=np.int32), np.cumsum(finite, axis=0)])

        # y[t] = sum_{k=1..h} lr[t+k]  (strict)
        # window indices: [t+1, t+h]
        last_t = T - h - 1
        if last_t >= 0:
            s = csum[(1 + h):(1 + h + last_t + 1)] - csum[1:(1 + last_t + 1)]
            c = ccnt[(1 + h):(1 + h + last_t + 1)] - ccnt[1:(1 + last_t + 1)]
            good = (c == h)
            out[:(last_t + 1)] = np.where(good, s, np.nan)

        return pd.DataFrame(out, index=logret.index, columns=logret.columns)

    @staticmethod
    def cs_rank_centered_df(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, pct=True, method="average")
        return r - 0.5

    @staticmethod
    def _cs_zscore_df(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=1)
        sd = df.std(axis=1, ddof=1).replace(0.0, np.nan)
        return df.sub(mu, axis=0).div(sd, axis=0)

    @staticmethod
    def _cs_winsorize_df(df: pd.DataFrame, q: float) -> pd.DataFrame:
        if q is None or q <= 0:
            return df

        # numpy nanquantile (much faster than per-row pandas quantile; same linear interpolation intent)
        X = df.values.astype(np.float64, copy=False)
        lo = np.nanquantile(X, q, axis=1)
        hi = np.nanquantile(X, 1 - q, axis=1)
        Xw = np.minimum(np.maximum(X, lo[:, None]), hi[:, None])
        return pd.DataFrame(Xw, index=df.index, columns=df.columns)

    @classmethod
    def cs_standardize_features_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        FEATURE_CS_STANDARDIZE = cls.FEATURE_CS_STANDARDIZE
        FEATURE_ROBUST_Z = cls.FEATURE_ROBUST_Z
        FEATURE_WINSOR_Q = cls.FEATURE_WINSOR_Q
        FEATURE_ZCLIP = cls.FEATURE_ZCLIP

        if FEATURE_CS_STANDARDIZE == "rank":
            return cls.cs_rank_centered_df(df)

        if FEATURE_CS_STANDARDIZE == "zscore":
            # numpy fast path (logic identical to winsor -> zscore -> clip)
            x = df
            if FEATURE_ROBUST_Z:
                x = cls._cs_winsorize_df(x, FEATURE_WINSOR_Q)
            # zscore
            X = x.values.astype(np.float64, copy=False)
            mu = np.nanmean(X, axis=1)
            sd = np.nanstd(X, axis=1, ddof=1)
            sd = np.where(np.isfinite(sd) & (sd > 0.0), sd, np.nan)
            Z = (X - mu[:, None]) / sd[:, None]
            if FEATURE_ZCLIP is not None:
                zc = float(abs(FEATURE_ZCLIP))
                Z = np.clip(Z, -zc, zc)
            return pd.DataFrame(Z, index=df.index, columns=df.columns)

        raise ValueError("FEATURE_CS_STANDARDIZE must be 'zscore' or 'rank'")

    @classmethod
    def daily_rankic(cls, pred: pd.Series, y_raw: pd.Series) -> pd.Series:
        # fast Spearman-by-date (keeps exact behavior: drop NaN/inf rows; date groups sorted)
        pred_v = pred.values.astype(np.float64, copy=False)
        y_v = y_raw.values.astype(np.float64, copy=False)

        dates = pred.index.get_level_values(0).to_numpy()
        dts = pd.to_datetime(dates).values.astype("datetime64[ns]")
        date_ns = dts.view(np.int64)

        # enforce sort by date key (matches groupby(sort=True))
        if date_ns.size > 1 and np.any(date_ns[1:] < date_ns[:-1]):
            ord0 = np.argsort(date_ns, kind="mergesort")
            date_ns = date_ns[ord0]
            pred_v = pred_v[ord0]
            y_v = y_v[ord0]

        out_date_ns, out_ic = cls._daily_rankic_sorted_numba(date_ns, pred_v, y_v)
        ds = pd.to_datetime(out_date_ns)
        return pd.Series(out_ic, index=pd.Index(ds, name="date"), name="rank_ic")

    @staticmethod
    def tstat_classic_mean(x: pd.Series) -> float:
        v = x.dropna().values.astype(np.float64)
        v = v[np.isfinite(v)]
        n = v.size
        if n < 2:
            return np.nan
        mu = float(v.mean())
        sd = float(v.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return np.nan
        return float(mu / (sd / np.sqrt(n)))

    @classmethod
    def tstat_newey_west_mean(cls, x: pd.Series, lags: int) -> float:
        v = x.dropna().values.astype(np.float64)
        v = v[np.isfinite(v)]
        n = v.size
        if n < 2:
            return np.nan
        L = int(max(0, min(int(lags), n - 2)))
        return float(cls._nw_tstat_numba(v, L))

    # [CHANGED #5.1-3 mirror] HAC lags = min(2*H, 20)
    @staticmethod
    def _auto_hac_lags(horizon: int, n_dates: int) -> int:
        L = int(min(max(1, 2 * int(horizon)), 20))
        return int(max(0, min(L, n_dates - 2))) if n_dates >= 2 else 0

    # ============================================================
    # 3) Build long panel (date,ticker) for a slice
    # ============================================================

    @classmethod
    def build_long_panel_slice(
        cls,
        features: Dict[str, pd.DataFrame],
        logret: pd.DataFrame,
        selected: List[str],
        train_idx: pd.Index,
        horizon: int,
        delay: int,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray, pd.Series]:

        TARGET_CS_RANK = cls.TARGET_CS_RANK
        REQUIRE_DATE_MIN_OBS = cls.REQUIRE_DATE_MIN_OBS
        DROP_EMPTY_DATES = cls.DROP_EMPTY_DATES
        FILL_MISSING_EXPOSURE_WITH_ZERO = cls.FILL_MISSING_EXPOSURE_WITH_ZERO
        DROP_ALLZERO_ROWS = cls.DROP_ALLZERO_ROWS

        features, logret = cls.ensure_aligned(features, logret)
        train_idx = pd.Index(train_idx).intersection(logret.index)
        if len(train_idx) == 0:
            raise ValueError("train_idx empty after intersecting logret.index")

        y_raw_df = cls.forward_sum_strict(logret, horizon=horizon)
        y_raw_w = y_raw_df.reindex(train_idx)

        if TARGET_CS_RANK:
            y_rank_w = cls.cs_rank_centered_df(y_raw_w)
        else:
            y_rank_w = y_raw_w.copy()

        # --- numpy fast path build (same output, less pandas/groupby overhead) ---
        cols = logret.columns
        N = int(len(cols))
        T = int(len(train_idx))
        K = int(len(selected))

        # build y vectors first
        y_raw_mat = y_raw_w.values.astype(np.float64, copy=False)
        y_rank_mat = y_rank_w.values.astype(np.float64, copy=False)

        # features into (T,N,K)
        X3 = np.empty((T, N, K), dtype=np.float64)
        X3[:] = np.nan

        base_index = logret.index
        pos = base_index.get_indexer(train_idx)

        for kk, f in enumerate(selected):
            if f not in features:
                raise KeyError(f"Selected factor '{f}' missing from features dict.")

            full = features[f].values.astype(np.float64, copy=False)  # (T_all,N)
            take = pos - int(delay)

            Xf = np.full((T, N), np.nan, dtype=np.float64)
            good = take >= 0
            if np.any(good):
                Xf[good, :] = full[take[good], :]

            Xf_df = pd.DataFrame(Xf, index=train_idx, columns=cols)
            Xf_df = cls.cs_standardize_features_df(Xf_df)

            # [CHANGED #5.2-1/2] do NOT fill 0
            if FILL_MISSING_EXPOSURE_WITH_ZERO:
                Xf_df = Xf_df.fillna(0.0)

            X3[:, :, kk] = Xf_df.values.astype(np.float64, copy=False)

        X2 = X3.reshape(T * N, K)

        y_raw_1d = y_raw_mat.reshape(T * N)
        y_rank_1d = y_rank_mat.reshape(T * N)

        mask = np.isfinite(y_raw_1d) & np.isfinite(y_rank_1d)

        # [CHANGED #5.2-1] mask: each row features must be all finite (no fill-0)
        mask = mask & np.all(np.isfinite(X2), axis=1)

        if DROP_ALLZERO_ROWS:
            row_abs_sum = np.sum(np.abs(X2), axis=1)
            mask = mask & (row_abs_sum > 0)

        mask_mat = mask.reshape(T, N)
        cs_count_by_date = pd.Series(mask_mat.sum(axis=1).astype(int), index=pd.Index(train_idx, name="date"))

        # [CHANGED #5.2-3] REQUIRE_DATE_MIN_OBS = max(30, ceil(0.3*N))
        date_min_obs = int(max(30, math.ceil(0.3 * float(N)))) if REQUIRE_DATE_MIN_OBS is None else int(REQUIRE_DATE_MIN_OBS)

        if date_min_obs is not None and date_min_obs > 0:
            good_dates_bool = (cs_count_by_date.values.astype(int) >= int(date_min_obs))
            if good_dates_bool.size:
                mask_mat = mask_mat & good_dates_bool[:, None]
                mask = mask_mat.reshape(T * N)
                cs_count_by_date = pd.Series(mask_mat.sum(axis=1).astype(int), index=pd.Index(train_idx, name="date"))

        X2 = X2[mask]
        y_raw_1d = y_raw_1d[mask]
        y_rank_1d = y_rank_1d[mask]

        keep_idx = np.nonzero(mask)[0]
        date_pos = (keep_idx // N).astype(np.int64)
        tick_pos = (keep_idx % N).astype(np.int64)

        dates_rep = train_idx.values[date_pos]
        tick_rep = cols.values[tick_pos]
        idx_long = pd.MultiIndex.from_arrays([dates_rep, tick_rep], names=["date", "ticker"])

        X_long = pd.DataFrame(X2, index=idx_long, columns=selected)
        y_raw_long = pd.Series(y_raw_1d, index=idx_long, name="y_raw")
        y_rank_long = pd.Series(y_rank_1d, index=idx_long, name="y_rank")

        if DROP_EMPTY_DATES:
            # no groupby; use cs_count_by_date already computed on train_idx grid
            keep_dates = cs_count_by_date[cs_count_by_date > 0].index
            X_long = X_long.loc[(keep_dates, slice(None)), :]
            y_raw_long = y_raw_long.loc[(keep_dates, slice(None))]
            y_rank_long = y_rank_long.loc[(keep_dates, slice(None))]
            cs_count_by_date = cs_count_by_date.reindex(keep_dates).fillna(0).astype(int)

        groups = X_long.index.get_level_values(0).to_numpy()
        return X_long, y_rank_long, y_raw_long, groups, cs_count_by_date

    # ============================================================
    # 4) CV helpers (LASSO Purged+Embargo KFold by date)
    # ============================================================

    class LassoPurgedEmbargoKFold:
        def __init__(self, n_splits: int = 5, purge_days: int = 0, embargo_days: int = 0):
            self.n_splits = int(n_splits)
            self.purge_days = int(max(0, purge_days))
            self.embargo_days = int(max(0, embargo_days))

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

        def split(self, X, y=None, groups=None):
            if groups is None:
                raise ValueError("LassoPurgedEmbargoKFold requires 'groups' (dates per row).")

            g = pd.to_datetime(pd.Index(groups))
            g_idx = pd.Index(g)

            dates = pd.Index(pd.to_datetime(pd.unique(g_idx))).sort_values()
            n_dates = int(len(dates))
            if n_dates < self.n_splits:
                raise ValueError(f"Not enough unique dates ({n_dates}) for n_splits={self.n_splits}")

            fold_sizes = np.full(self.n_splits, n_dates // self.n_splits, dtype=int)
            fold_sizes[: (n_dates % self.n_splits)] += 1

            cur = 0
            for k in range(self.n_splits):
                start = cur
                stop = cur + int(fold_sizes[k])
                cur = stop

                test_dates = dates[start:stop]
                if len(test_dates) == 0:
                    continue

                drop_start = max(0, start - self.purge_days)
                drop_end = min(n_dates, stop + self.embargo_days)

                train_dates = dates[:drop_start].append(dates[drop_end:])

                te_mask = np.asarray(g_idx.isin(test_dates), dtype=bool)
                tr_mask = np.asarray(g_idx.isin(train_dates), dtype=bool)

                tr_idx = np.where(tr_mask)[0]
                te_idx = np.where(te_mask)[0]
                if tr_idx.size == 0 or te_idx.size == 0:
                    continue

                yield tr_idx, te_idx

    @classmethod
    def _make_lasso(cls, alpha: float):
        RANDOM_STATE = cls.RANDOM_STATE
        try:
            return Lasso(alpha=float(alpha), fit_intercept=True, max_iter=200_000, random_state=RANDOM_STATE)
        except TypeError:
            return Lasso(alpha=float(alpha), fit_intercept=True, max_iter=200_000)

    @classmethod
    def purged_cv_alpha_search_by_rankic(
        cls,
        X: pd.DataFrame,
        y_rank: pd.Series,
        y_raw: pd.Series,
        groups: np.ndarray,
        alphas: np.ndarray,
        n_splits: int,
        purge_days: int,
        embargo_days: int,
    ) -> Tuple[float, pd.DataFrame]:

        splitter = cls.LassoPurgedEmbargoKFold(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)
        rows = []

        best_alpha = None
        best_mean = -np.inf
        best_ir = -np.inf

        # precompute folds once (same logic, less overhead)
        folds = list(splitter.split(X, y_rank, groups=groups))

        # numpy views (same data, less pandas slicing overhead)
        Xv = X.values.astype(np.float64, copy=False)
        yv = y_rank.values.astype(np.float64, copy=False)

        for a in alphas:
            pred_arr = np.full(Xv.shape[0], np.nan, dtype=np.float64)

            for tr, te in folds:
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xtr = scaler.fit_transform(Xv[tr])
                Xte = scaler.transform(Xv[te])

                mdl = cls._make_lasso(float(a))
                mdl.fit(Xtr, yv[tr])
                pred_arr[te] = mdl.predict(Xte)

            pred = pd.Series(pred_arr, index=X.index, dtype=float)

            ic_daily = cls.daily_rankic(pred, y_raw)
            mu = float(ic_daily.mean()) if len(ic_daily) else -np.inf
            sd = float(ic_daily.std(ddof=1)) if len(ic_daily) > 1 else np.nan
            ir = float(mu / sd) if np.isfinite(sd) and sd > 0 else np.nan

            rows.append({
                "alpha": float(a),
                "cv_rankic_mean": mu,
                "cv_rankic_std": sd,
                "cv_rankic_ir": ir,
                "cv_n_days": int(len(ic_daily)),
            })

            better = False
            if np.isfinite(mu) and (mu > best_mean + 1e-12):
                better = True
            elif np.isfinite(mu) and abs(mu - best_mean) <= 1e-12:
                if np.isfinite(ir) and (ir > best_ir + 1e-12):
                    better = True
                elif (not np.isfinite(best_ir)) and np.isfinite(ir):
                    better = True
                elif (np.isfinite(ir) and np.isfinite(best_ir) and abs(ir - best_ir) <= 1e-12):
                    if best_alpha is None or float(a) < float(best_alpha):
                        better = True

            if better:
                best_mean = mu
                best_ir = ir if np.isfinite(ir) else best_ir
                best_alpha = float(a)

        cv_table = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)

        if best_alpha is None:
            best_alpha = float(np.min(alphas))

        return float(best_alpha), cv_table

    @classmethod
    def purged_oos_predictions(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray,
        pipe: Pipeline,
        n_splits: int,
        purge_days: int,
        embargo_days: int,
    ) -> pd.Series:
        splitter = cls.LassoPurgedEmbargoKFold(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)

        folds = list(splitter.split(X, y, groups=groups))
        Xv = X.values.astype(np.float64, copy=False)
        yv = y.values.astype(np.float64, copy=False)

        pred_arr = np.full(Xv.shape[0], np.nan, dtype=np.float64)
        for tr, te in folds:
            mdl = clone(pipe)
            mdl.fit(Xv[tr], yv[tr])
            pred_arr[te] = mdl.predict(Xv[te])

        pred = pd.Series(pred_arr, index=X.index, dtype=float)
        return pred

    # ============================================================
    # 5) Result container
    # ============================================================

    @dataclass
    class LassoSliceResult:
        rebalance_date: pd.Timestamp
        alpha: float
        cv_table: pd.DataFrame
        coef: pd.Series
        intercept: float
        pred_oos: pd.Series
        rankic_daily_oos: pd.Series
        ic_mean: float
        ic_std: float
        ic_ir: float
        t_classic: float
        t_hac: float
        nonzero_factors: List[str]
        nonzero_weights: pd.Series
        n_rows: int
        n_dates: int
        n_factors_in: int
        n_factors_nonzero: int
        cs_med: float
        cs_p10: float
        cs_p90: float
        purge_days: int
        embargo_days: int

    # ============================================================
    # 6) Slice runner
    # ============================================================

    @classmethod
    def run_lasso_on_rebalance(
        cls,
        t0: pd.Timestamp,
        selected: List[str],
        train_idx: pd.Index,
        features: Dict[str, pd.DataFrame],
        logret: pd.DataFrame,
        horizon: int,
        delay: int,
        verbose: bool = True,
        plot_report: bool = False,
        save_dir: Optional[str] = None,
    ) -> "LassoSliceResult":

        LASSO_ALPHAS = cls.LASSO_ALPHAS
        N_SPLITS = cls.N_SPLITS
        USE_HAC = cls.USE_HAC
        HAC_LAGS = cls.HAC_LAGS
        LASSO_NONZERO_EPS = cls.LASSO_NONZERO_EPS
        FEATURE_CS_STANDARDIZE = cls.FEATURE_CS_STANDARDIZE
        FEATURE_ROBUST_Z = cls.FEATURE_ROBUST_Z
        FILL_MISSING_EXPOSURE_WITH_ZERO = cls.FILL_MISSING_EXPOSURE_WITH_ZERO
        LASSO_PURGE_DAYS = cls.LASSO_PURGE_DAYS
        LASSO_EMBARGO_DAYS = cls.LASSO_EMBARGO_DAYS

        if selected is None or len(selected) == 0:
            raise ValueError(f"[{t0}] selected factors empty. Nothing to run.")

        X_long, y_rank_long, y_raw_long, groups, cs_count_by_date = cls.build_long_panel_slice(
            features=features,
            logret=logret,
            selected=selected,
            train_idx=train_idx,
            horizon=horizon,
            delay=delay,
        )

        n_dates = int(pd.Index(groups).nunique())
        n_rows = int(X_long.shape[0])
        n_factors_in = int(X_long.shape[1])

        hac_lags = cls._auto_hac_lags(horizon, n_dates) if HAC_LAGS is None else int(HAC_LAGS)

        auto_pe = int(max(0, (int(horizon) - 1) + int(delay)))
        purge_days = int(auto_pe if LASSO_PURGE_DAYS is None else int(LASSO_PURGE_DAYS))
        embargo_days = int(auto_pe if LASSO_EMBARGO_DAYS is None else int(LASSO_EMBARGO_DAYS))

        best_alpha, cv_table = cls.purged_cv_alpha_search_by_rankic(
            X=X_long,
            y_rank=y_rank_long,
            y_raw=y_raw_long,
            groups=groups,
            alphas=LASSO_ALPHAS,
            n_splits=N_SPLITS,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", cls._make_lasso(best_alpha)),
        ])

        pred_oos = cls.purged_oos_predictions(
            X_long, y_rank_long, groups, pipe,
            n_splits=N_SPLITS, purge_days=purge_days, embargo_days=embargo_days
        )

        pipe.fit(X_long, y_rank_long)
        core = pipe.named_steps["model"]
        coef = pd.Series(core.coef_, index=X_long.columns, name="coef_std")
        coef = coef.sort_values(key=lambda s: s.abs(), ascending=False)

        ic_daily = cls.daily_rankic(pred_oos, y_raw_long)
        ic_mean = float(ic_daily.mean()) if len(ic_daily) else np.nan
        ic_std = float(ic_daily.std(ddof=1)) if len(ic_daily) > 1 else np.nan
        ic_ir = float(ic_mean / ic_std) if np.isfinite(ic_std) and ic_std > 0 else np.nan

        t_cls = cls.tstat_classic_mean(ic_daily) if len(ic_daily) else np.nan
        t_hac = cls.tstat_newey_west_mean(ic_daily, lags=hac_lags) if (USE_HAC and len(ic_daily)) else np.nan

        nz = coef.loc[lambda s: s.abs() > float(LASSO_NONZERO_EPS)]
        nonzero_factors = nz.index.tolist()
        nonzero_weights = nz.copy()
        if nonzero_weights.abs().sum() > 0:
            nonzero_weights = nonzero_weights / nonzero_weights.abs().sum()

        cs_vals = cs_count_by_date.values.astype(float)
        cs_med = float(np.nanmedian(cs_vals)) if cs_vals.size else np.nan
        cs_p10 = float(np.nanpercentile(cs_vals, 10)) if cs_vals.size else np.nan
        cs_p90 = float(np.nanpercentile(cs_vals, 90)) if cs_vals.size else np.nan

        res = cls.LassoSliceResult(
            rebalance_date=pd.Timestamp(t0),
            alpha=float(best_alpha),
            cv_table=cv_table,
            coef=coef,
            intercept=float(getattr(core, "intercept_", np.nan)),
            pred_oos=pred_oos,
            rankic_daily_oos=ic_daily,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            t_classic=float(t_cls) if np.isfinite(t_cls) else np.nan,
            t_hac=float(t_hac) if np.isfinite(t_hac) else np.nan,
            nonzero_factors=nonzero_factors,
            nonzero_weights=nonzero_weights,
            n_rows=n_rows,
            n_dates=n_dates,
            n_factors_in=n_factors_in,
            n_factors_nonzero=int(len(nonzero_factors)),
            cs_med=cs_med,
            cs_p10=cs_p10,
            cs_p90=cs_p90,
            purge_days=int(purge_days),
            embargo_days=int(embargo_days),
        )

        if verbose:
            print("============================================================")
            print(f"LASSO FILTER | rebalance={pd.Timestamp(t0).date()}")
            print(f"CV: LassoPurgedEmbargoKFold by DATE (purge={purge_days}, embargo={embargo_days})")
            print("CV objective: maximize RankIC mean (Spearman) on RAW forward returns")
            print("Feature X:", f"CS-{FEATURE_CS_STANDARDIZE}" + ("(winsor+clip)" if (FEATURE_CS_STANDARDIZE=="zscore" and FEATURE_ROBUST_Z) else ""))
            print("Missing X:", "fill 0 (neutral)" if FILL_MISSING_EXPOSURE_WITH_ZERO else "drop rows with NaN X (all-finite row mask)")
            print("Target y for fit:", "CS-rank(forward sum)" if cls.TARGET_CS_RANK else "forward sum")
            print(f"Forward return: strict forward_sum(H={horizon}) of logret[t+1..t+H] (NaN window => NaN)")
            print("------------------------------------------------------------")
            print(f"Rows: {res.n_rows:,} | #dates: {res.n_dates} | #factors_in: {res.n_factors_in}")
            print(f"CS count p10/med/p90: {res.cs_p10:.0f}/{res.cs_med:.0f}/{res.cs_p90:.0f} (after strict y + exposure policy)")
            print(f"Best alpha: {res.alpha:.6g}")
            print(f"OOS RankIC mean/std/IR: {res.ic_mean:.6f} / {res.ic_std:.6f} / {res.ic_ir:.3f}")
            print(f"t_classic: {res.t_classic:.3f} | t_HAC(NeweyWest): {res.t_hac:.3f}")
            print(f"Nonzero factors: {res.n_factors_nonzero}")
            if res.n_factors_nonzero > 0:
                print("Top nonzero normalized weights:")
                print(res.nonzero_weights.head(20))
            print("Top coefficients (std-X):")
            print(res.coef.head(20))
            print("============================================================")

        if plot_report:
            cls.plot_lasso_slice_report(res, save_dir=save_dir)

        return res

    # ============================================================
    # 7) Plotting — slice + walk-forward summary
    # ============================================================

    @classmethod
    def plot_lasso_slice_report(cls, res: "LassoSliceResult", save_dir: Optional[str] = None):
        MAX_COEF_TO_PLOT = cls.MAX_COEF_TO_PLOT

        cls.set_presentation_style()

        cv = res.cv_table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cv["alpha"].values, cv["cv_rankic_mean"].values, marker="o", linewidth=2.0, label="CV RankIC mean")
        ax.axvline(res.alpha, linewidth=2.0, label=f"best alpha={res.alpha:.2g}")
        ax.set_xscale("log")
        ax.set_title(f"[{res.rebalance_date.date()}] PurgedEmbargo CV: alpha vs RankIC mean")
        ax.set_xlabel("alpha (log scale)")
        ax.set_ylabel("CV RankIC mean")
        cls.legend_below(ax, ncol=2, yshift=-0.22)
        cls.finalize_figure(fig, bottom=0.26)
        cls._savefig(fig, f"lasso_{res.rebalance_date.date()}_alpha_cv.png", save_dir)
        plt.show()

        coef = res.coef.dropna()
        top = coef.head(MAX_COEF_TO_PLOT)
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(top.index.astype(str), top.values)
        ax.axhline(0, linewidth=1.0)
        ax.set_title(f"[{res.rebalance_date.date()}] LASSO coefficients (top {len(top)} by |coef|)")
        ax.set_xlabel("factor")
        ax.set_ylabel("coef (std-X)")
        ax.tick_params(axis="x", rotation=40)
        cls.finalize_figure(fig, bottom=0.30)
        cls._savefig(fig, f"lasso_{res.rebalance_date.date()}_coef_top.png", save_dir)
        plt.show()

        ic = res.rankic_daily_oos.dropna()
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(ic.index, ic.values, linewidth=1.2, alpha=0.9, label="daily RankIC (OOS)")
        ax.axhline(0, linewidth=1.0)
        ax.set_title(f"[{res.rebalance_date.date()}] OOS Daily RankIC (PurgedEmbargo CV prediction)")
        ax.set_xlabel("date")
        ax.set_ylabel("RankIC")
        cls.legend_below(ax, ncol=1, yshift=-0.22)
        cls.finalize_figure(fig, bottom=0.26)
        cls._savefig(fig, f"lasso_{res.rebalance_date.date()}_rankic_daily.png", save_dir)
        plt.show()

    @classmethod
    def plot_walkforward_lasso_report(cls, wf_lasso: dict, save_dir: Optional[str] = None):
        cls.set_presentation_style()

        st = wf_lasso.get("slice_stats", pd.DataFrame()).copy()
        if st is None or len(st) == 0:
            print("[skip] empty wf_lasso['slice_stats']")
            return

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(st.index.astype(str), st["ic_mean"].values, marker="o", linewidth=2.0, label="IC mean (OOS RankIC)")
        ax.axhline(0, linewidth=1.0)
        ax.set_title("Walk-forward LASSO: RankIC mean by rebalance date")
        ax.set_xlabel("rebalance date")
        ax.set_ylabel("RankIC mean")
        ax.tick_params(axis="x", rotation=45)
        cls.legend_below(ax, ncol=1, yshift=-0.28)
        cls.finalize_figure(fig, bottom=0.32)
        cls._savefig(fig, "wf_lasso_ic_mean_by_slice.png", save_dir)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(st.index.astype(str), st["t_hac"].values, marker="o", linewidth=2.0, label="t_HAC (Newey-West)")
        ax.axhline(0, linewidth=1.0)
        ax.set_title("Walk-forward LASSO: HAC t-stat by rebalance date")
        ax.set_xlabel("rebalance date")
        ax.set_ylabel("t_HAC")
        ax.tick_params(axis="x", rotation=45)
        cls.legend_below(ax, ncol=1, yshift=-0.28)
        cls.finalize_figure(fig, bottom=0.32)
        cls._savefig(fig, "wf_lasso_t_hac_by_slice.png", save_dir)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(st.index.astype(str), st["n_factors_nonzero"].values, marker="o", linewidth=2.0, label="#nonzero factors")
        ax.set_title("Walk-forward LASSO: sparsity (#nonzero) by rebalance date")
        ax.set_xlabel("rebalance date")
        ax.set_ylabel("#nonzero")
        ax.tick_params(axis="x", rotation=45)
        cls.legend_below(ax, ncol=1, yshift=-0.28)
        cls.finalize_figure(fig, bottom=0.32)
        cls._savefig(fig, "wf_lasso_nonzero_count_by_slice.png", save_dir)
        plt.show()

    # ============================================================
    # 8) Main entry — plugs into your wf dict
    # ============================================================

    @classmethod
    def run_walkforward_lasso_layer(
        cls,
        wf: dict,
        features: Dict[str, pd.DataFrame],
        logret: pd.DataFrame,
        verbose: bool = True,
        plot_each_slice: bool = False,
        save_dir: Optional[str] = None,
    ) -> dict:

        LASSO_T_HAC_GATE = cls.LASSO_T_HAC_GATE
        N_SPLITS = cls.N_SPLITS
        LASSO_ALPHAS = cls.LASSO_ALPHAS
        USE_HAC = cls.USE_HAC
        HAC_LAGS = cls.HAC_LAGS
        REQUIRE_DATE_MIN_OBS = cls.REQUIRE_DATE_MIN_OBS
        FEATURE_CS_STANDARDIZE = cls.FEATURE_CS_STANDARDIZE
        FEATURE_ROBUST_Z = cls.FEATURE_ROBUST_Z
        FEATURE_WINSOR_Q = cls.FEATURE_WINSOR_Q
        FEATURE_ZCLIP = cls.FEATURE_ZCLIP
        FILL_MISSING_EXPOSURE_WITH_ZERO = cls.FILL_MISSING_EXPOSURE_WITH_ZERO
        DROP_ALLZERO_ROWS = cls.DROP_ALLZERO_ROWS
        LASSO_PURGE_DAYS = cls.LASSO_PURGE_DAYS
        LASSO_EMBARGO_DAYS = cls.LASSO_EMBARGO_DAYS
        TARGET_CS_RANK = cls.TARGET_CS_RANK

        if "selected_by_slice" not in wf or wf["selected_by_slice"] is None:
            raise KeyError("wf missing 'selected_by_slice' (expected from Route A).")
        if "screen_reports" not in wf or wf["screen_reports"] is None:
            raise KeyError("wf missing 'screen_reports' (expected from Route A).")

        cfg = wf.get("config", {})
        horizon = int(cfg.get("PRED_HORIZON", 5))
        delay = int(cfg.get("DELAY", 1))

        selected_by_slice = wf["selected_by_slice"]
        screen_reports = wf["screen_reports"]

        rebs = sorted([pd.Timestamp(k) for k in selected_by_slice.keys()])

        results: List[LassoFilterLayer.LassoSliceResult] = []
        for t0 in rebs:
            sel = selected_by_slice.get(t0, None)
            scr = screen_reports.get(t0, None)
            if scr is None or not hasattr(scr, "train_index"):
                raise KeyError(f"screen_reports[{t0}] missing or has no train_index.")
            train_idx = scr.train_index

            if sel is None or len(sel) == 0:
                if verbose:
                    print(f"[skip] {t0.date()} selection empty")
                continue

            res = cls.run_lasso_on_rebalance(
                t0=t0,
                selected=list(sel),
                train_idx=train_idx,
                features=features,
                logret=logret,
                horizon=horizon,
                delay=delay,
                verbose=verbose,
                plot_report=plot_each_slice,
                save_dir=save_dir,
            )

            # [CHANGED #5.2-5] light gate: t_hac >= 0
            if not (np.isfinite(res.t_hac) and (res.t_hac >= float(LASSO_T_HAC_GATE))):
                if verbose:
                    print(f"[skip] {t0.date()} LASSO t_hac gate fail: t_hac={res.t_hac} < {LASSO_T_HAC_GATE}")
                continue

            results.append(res)

        rows = []
        for r in results:
            rows.append({
                "rebalance_date": r.rebalance_date,
                "alpha": r.alpha,
                "purge_days": r.purge_days,
                "embargo_days": r.embargo_days,
                "n_rows": r.n_rows,
                "n_dates": r.n_dates,
                "n_factors_in": r.n_factors_in,
                "n_factors_nonzero": r.n_factors_nonzero,
                "cs_p10": r.cs_p10,
                "cs_med": r.cs_med,
                "cs_p90": r.cs_p90,
                "ic_mean": r.ic_mean,
                "ic_std": r.ic_std,
                "ic_ir": r.ic_ir,
                "t_classic": r.t_classic,
                "t_hac": r.t_hac,
            })
        slice_stats = pd.DataFrame(rows).set_index("rebalance_date") if rows else pd.DataFrame()

        return {
            "slice_results": results,
            "slice_stats": slice_stats,
            "config_used": {
                "HORIZON": horizon,
                "DELAY": delay,
                "TARGET_CS_RANK": TARGET_CS_RANK,
                "N_SPLITS": N_SPLITS,
                "LASSO_ALPHAS": (float(LASSO_ALPHAS.min()), float(LASSO_ALPHAS.max()), int(len(LASSO_ALPHAS))),
                "USE_HAC": USE_HAC,
                "HAC_LAGS": HAC_LAGS if HAC_LAGS is not None else "auto(min(max(1,2H),20))",
                "REQUIRE_DATE_MIN_OBS": "auto=max(30, ceil(0.3*N))" if REQUIRE_DATE_MIN_OBS is None else REQUIRE_DATE_MIN_OBS,
                "STRICT_FORWARD_RETURNS": "rolling(min_periods=H) on logret.shift(-1)",
                "FEATURE_CS_STANDARDIZE": FEATURE_CS_STANDARDIZE,
                "FEATURE_ROBUST_Z": FEATURE_ROBUST_Z,
                "FEATURE_WINSOR_Q": FEATURE_WINSOR_Q,
                "FEATURE_ZCLIP": FEATURE_ZCLIP,
                "FILL_MISSING_EXPOSURE_WITH_ZERO": FILL_MISSING_EXPOSURE_WITH_ZERO,
                "DROP_ALLZERO_ROWS": DROP_ALLZERO_ROWS,
                "LASSO_PURGE_DAYS": LASSO_PURGE_DAYS if LASSO_PURGE_DAYS is not None else "auto=(H-1)+delay",
                "LASSO_EMBARGO_DAYS": LASSO_EMBARGO_DAYS if LASSO_EMBARGO_DAYS is not None else "auto=(H-1)+delay",
                "ALPHA_SELECTION": "PurgedEmbargo CV maximize RankIC mean (Spearman) on RAW forward returns",
                "T_HAC_GATE": LASSO_T_HAC_GATE,
            }
        }





wf_lasso = LassoFilterLayer.run_walkforward_lasso_layer(
    wf, features, logret,
    verbose=True, plot_each_slice=False, save_dir=None
)
print(wf_lasso["slice_stats"].head(10))
LassoFilterLayer.plot_walkforward_lasso_report(wf_lasso, save_dir=None)
















# ============================================================
# GA LAYER (WF CONNECTED) — Industry-grade GP/GA on LASSO-selected terminals
# - Outer: reuse wf slice schedule + real OOS holdout
# - Inner: within each slice train window, split GA_train / GA_val (time-ordered)
#          with purge gap >= (H-1)+DELAY
# - Fitness: Newey-West t-stat of daily IC mean (primary) - lambda_complexity * complexity
# - Neutralization: Barra-style daily cross-sectional regression (beta/size/industry) -> residual
# - Stores ALL per-slice artifacts + per-generation logs (no cutting)
# ============================================================



_NUMBA_AVAILABLE = True

# ============================================================
# 0) GA Tunables (safe professional defaults)
# ============================================================

GA_SEED_BASE = 7

GA_VAL_RATIO = 0.20                     # last 20% as GA_val (time-ordered)
GA_PURGE_GAP_DAYS = None                # None => auto = (H-1)+DELAY  (industry standard)

# ---- US-noise-friendly search budget (with stronger parsimony) ----
GA_POP_SIZE = 384                       # was 256
GA_N_GEN = 180                          # was 120; early-stop prevents overrun
GA_TOURNAMENT_K = 5
GA_ELITE_FRAC = 0.06                    # slightly higher elite pool for diversity after de-dup
GA_CROSSOVER_P = 0.55
GA_MUTATION_P = 0.35
GA_REPRO_P = 0.10                       # plain copy probability (diversity balance)

# ---- Parsimony / overfit control ----
GA_MAX_DEPTH = 3                        # was 4; US XSec gets noisy fast with deep trees
GA_MIN_DEPTH = 1

GA_P_CONST = 0.06                       # was 0.10; reduce constant overfit
GA_CONST_RANGE = (-1.5, 1.5)            # tighter than (-2,2)

# Rolling windows
GA_ROLL_WINDOWS = [5, 10, 20, 60]       # keep; US names/vol regimes ok

# ---- rolling missing tolerance ----
GA_TS_MIN_FRAC = 0.80                   # require >= ceil(w * frac) valid points in rolling window

# Complexity penalty (parsimony pressure)
GA_LAMBDA_COMPLEXITY = 0.022            # was 0.015; push simpler formulas in US
GA_ROLLING_COMPLEXITY_BONUS = 2.0       # extra penalty for ts_* ops

# ---- NEW: train-val gap & sign flip penalties (anti-overfit) ----
GA_LAMBDA_GAP = 0.15                    # penalize (train_t - val_t) if positive
GA_SIGN_FLIP_PENALTY = 1.0              # penalize sign flip between train/val IC mean
GA_ELITE_UNIQUE = True                  # prevent elite clone collapse

# Early stopping on GA_val objective
GA_EARLYSTOP_PATIENCE = 25
GA_EARLYSTOP_MIN_DELTA = 1e-3

# validate more than 1 elite each generation (stability)
GA_VAL_TOPM = 16                        # was 8

# more stable ridge in neutralization
GA_NEUTRALIZE_RIDGE = 1e-4

# effective sample guards for IC/HAC
try:
    _MIN_PORT = int(MIN_CS_ASSETS_FOR_PORT)
except Exception:
    _MIN_PORT = 50
GA_MIN_CS_N = max(50, _MIN_PORT)
GA_MIN_T_EFF = 60                       # keep for train/val (HAC needs length)

# ---- NEW: per-slice Top-N outputs ----
GA_TOPN_PER_SLICE = 10
GA_TOPN_POOL_MAX = 300                  # cap candidate pool re-eval for speed
GA_TOPN_MIN_VAL_T = -np.inf             # you can set 1.0/1.5 to be stricter later

# safety asserts
GA_ASSERT_ALIGNMENT = True

# Signal post-processing (reuse your robust zscore settings)
GA_CS_STANDARDIZE = CS_STANDARDIZE
GA_IC_METHOD = IC_METHOD


def ga_to_date_index(idx: pd.Index) -> pd.Index:
    dt = pd.to_datetime(idx)
    return pd.Index(dt.date, name="date")


def ga_spy_aggs_to_close_series(spy_aggs: list, target_index: pd.Index, tz: str = "America/New_York") -> pd.Series:
    if len(spy_aggs) == 0:
        raise ValueError("spy_aggs is empty.")
    ts = np.array([a.timestamp for a in spy_aggs], dtype=np.int64)
    close = np.array([a.close for a in spy_aggs], dtype=np.float64)
    dt = pd.to_datetime(ts, unit="ms", utc=True).tz_convert(tz)
    spy_dates = pd.Index(dt.date, name="date")
    spy_close = pd.Series(close, index=spy_dates, name="SPY_close").sort_index()

    tgt_dates = ga_to_date_index(target_index)
    aligned = spy_close.reindex(tgt_dates).astype(float)
    aligned.index = target_index
    return aligned


# ============================================================
# 1) Strict time split with purge gap (inner GA split)
# ============================================================

def ga_time_split_with_purge(idx: pd.Index, val_ratio: float, purge_gap: int) -> Tuple[pd.Index, pd.Index]:
    idx = pd.DatetimeIndex(idx)
    n = len(idx)
    if n < 10:
        return idx, idx[:0]

    purge_gap = int(max(0, purge_gap))

    cut = int(np.floor(n * (1.0 - float(val_ratio))))
    cut = max(2, min(cut, n - 2))

    # purge gap removes boundary dates to prevent leakage from overlapping labels (H-day forward ret)
    tr_end = max(0, cut - purge_gap)
    va_start = min(n, cut + purge_gap)

    tr = idx[:tr_end]
    va = idx[va_start:]

    return pd.Index(tr), pd.Index(va)


# ============================================================
# 2) Barra-style daily cross-sectional neutralization
# ============================================================

_IND_DUMMY_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, Dict[int, int]]] = {}

def _build_industry_dummies(ind_codes: np.ndarray, max_groups: int = 40) -> Tuple[np.ndarray, Dict[int, int]]:
    ind_codes = np.asarray(ind_codes).astype(int)
    valid = ind_codes[ind_codes >= 0]
    if valid.size == 0:
        return np.zeros((ind_codes.size, 0), dtype=np.float64), {}

    vc = pd.Series(valid).value_counts()
    keep = vc.index[:max_groups].tolist()

    mapping = {int(code): j for j, code in enumerate(keep)}
    G = len(keep)
    D = np.zeros((ind_codes.size, G), dtype=np.float64)
    for i, c in enumerate(ind_codes):
        j = mapping.get(int(c), None)
        if j is not None:
            D[i, j] = 1.0
    return D, mapping

def _get_industry_dummies_cached(industry_codes: np.ndarray, max_groups: int = 40) -> Tuple[np.ndarray, Dict[int, int]]:
    ind = np.asarray(industry_codes).astype(int)
    key = (ind.dtype.str, int(ind.size), ind.tobytes(), int(max_groups))
    hit = _IND_DUMMY_CACHE.get(key, None)
    if hit is not None:
        return hit
    out = _build_industry_dummies(ind, max_groups=max_groups)
    _IND_DUMMY_CACHE[key] = out
    return out

@njit(cache=True)
def _neutralize_barra_numba(
    S: np.ndarray,
    beta: np.ndarray,
    ln_mcap: np.ndarray,
    D2: np.ndarray,
    use_beta: bool,
    use_size: bool,
    use_ind: bool,
    ridge: float,
    min_cs_n: int,
) -> np.ndarray:
    T, N = S.shape
    out = np.empty((T, N), dtype=np.float64)
    out[:] = np.nan

    G2 = D2.shape[1] if use_ind else 0
    for t in range(T):
        idxs = np.empty(N, dtype=np.int64)
        k = 0
        for i in range(N):
            y = S[t, i]
            if not np.isfinite(y):
                continue
            if use_beta and (not np.isfinite(beta[t, i])):
                continue
            if use_size and (not np.isfinite(ln_mcap[t, i])):
                continue
            idxs[k] = i
            k += 1

        if k < min_cs_n:
            continue

        p = 1 + (1 if use_beta else 0) + (1 if use_size else 0) + (G2 if (use_ind and G2 > 0) else 0)

        X = np.empty((k, p), dtype=np.float64)
        yy = np.empty(k, dtype=np.float64)

        for r in range(k):
            ii = idxs[r]
            yy[r] = S[t, ii]
            X[r, 0] = 1.0

        col = 1
        if use_beta:
            for r in range(k):
                ii = idxs[r]
                X[r, col] = beta[t, ii]
            col += 1

        if use_size:
            for r in range(k):
                ii = idxs[r]
                X[r, col] = ln_mcap[t, ii]
            col += 1

        if use_ind and G2 > 0:
            for g in range(G2):
                for r in range(k):
                    ii = idxs[r]
                    X[r, col + g] = D2[ii, g]
            col += G2

        XtX = X.T @ X
        for d in range(p):
            XtX[d, d] += ridge
        Xty = X.T @ yy

        b = np.linalg.solve(XtX, Xty)
        resid = yy - X @ b

        for r in range(k):
            out[t, idxs[r]] = resid[r]

    return out


def neutralize_signal_barra_style(
    S: np.ndarray,
    beta: Optional[np.ndarray] = None,
    ln_mcap: Optional[np.ndarray] = None,
    industry_codes: Optional[np.ndarray] = None,
    ridge: float = 1e-6,
    min_cs_n: int = 50,
    max_industries: int = 40,
) -> np.ndarray:
    T, N = S.shape
    out = np.full_like(S, np.nan, dtype=np.float64)

    use_beta = beta is not None
    use_size = ln_mcap is not None
    use_ind  = industry_codes is not None

    if not (use_beta or use_size or use_ind):
        return S.copy()

    if use_ind:
        D, _ = _get_industry_dummies_cached(industry_codes, max_groups=max_industries)
        if D.shape[1] >= 2:
            D2 = D[:, 1:].astype(np.float64)
        else:
            D2 = np.zeros((N, 0), dtype=np.float64)
        G2 = D2.shape[1]
    else:
        D2 = np.zeros((N, 0), dtype=np.float64)
        G2 = 0

    if _NUMBA_AVAILABLE:
        try:
            beta_use = beta.astype(np.float64) if use_beta else np.zeros_like(S, dtype=np.float64)
            size_use = ln_mcap.astype(np.float64) if use_size else np.zeros_like(S, dtype=np.float64)
            return _neutralize_barra_numba(
                S.astype(np.float64),
                beta_use,
                size_use,
                D2,
                bool(use_beta),
                bool(use_size),
                bool(use_ind and (G2 > 0)),
                float(ridge),
                int(min_cs_n),
            )
        except Exception:
            pass

    for t in range(T):
        y = S[t]
        m = np.isfinite(y)
        if use_beta:
            m = m & np.isfinite(beta[t])
        if use_size:
            m = m & np.isfinite(ln_mcap[t])

        idx = np.where(m)[0]
        k = idx.size
        if k < int(min_cs_n):
            continue

        cols = []
        cols.append(np.ones(k, dtype=np.float64))
        if use_beta:
            cols.append(beta[t, idx].astype(np.float64))
        if use_size:
            cols.append(ln_mcap[t, idx].astype(np.float64))
        if use_ind and G2 > 0:
            Di = D2[idx, :]
            if Di.shape[1] > 0:
                cols.append(Di.astype(np.float64))

        X = np.column_stack(cols)
        yy = y[idx].astype(np.float64)

        p = X.shape[1]
        XtX = X.T @ X
        XtX.flat[::p+1] += float(ridge)
        Xty = X.T @ yy
        try:
            b = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        resid = yy - X @ b
        out[t, idx] = resid

    return out


# ============================================================
# 3) Rolling ops (robust)
# ============================================================

def ts_mean_strict_np(X: np.ndarray, window: int) -> np.ndarray:
    T, N = X.shape
    w = int(window)
    out = np.full((T, N), np.nan, dtype=np.float64)
    if w <= 1:
        return X.copy().astype(np.float64)

    finite = np.isfinite(X).astype(np.int32)
    X0 = np.where(np.isfinite(X), X, 0.0)

    csum = np.cumsum(X0, axis=0)
    ccnt = np.cumsum(finite, axis=0)

    sum_w = csum[w-1:] - np.vstack([np.zeros((1, N)), csum[:-w]])
    cnt_w = ccnt[w-1:] - np.vstack([np.zeros((1, N), dtype=np.int32), ccnt[:-w]])

    min_req = int(np.ceil(w * float(GA_TS_MIN_FRAC)))
    min_req = max(1, min_req)

    denom = np.where(cnt_w > 0, cnt_w.astype(np.float64), np.nan)
    mean = sum_w / denom
    out[w-1:] = np.where(cnt_w >= min_req, mean, np.nan)
    return out


def ts_std_strict_np(X: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    T, N = X.shape
    w = int(window)
    out = np.full((T, N), np.nan, dtype=np.float64)
    if w <= 1:
        return np.zeros((T, N), dtype=np.float64)

    finite = np.isfinite(X).astype(np.int32)
    X0 = np.where(np.isfinite(X), X, 0.0)
    X02 = X0 * X0

    csum = np.cumsum(X0, axis=0)
    csum2 = np.cumsum(X02, axis=0)
    ccnt = np.cumsum(finite, axis=0)

    sum_w = csum[w-1:] - np.vstack([np.zeros((1, N)), csum[:-w]])
    sum2_w = csum2[w-1:] - np.vstack([np.zeros((1, N)), csum2[:-w]])
    cnt_w = ccnt[w-1:] - np.vstack([np.zeros((1, N), dtype=np.int32), ccnt[:-w]])

    min_req = int(np.ceil(w * float(GA_TS_MIN_FRAC)))
    min_req = max(ddof + 2, min_req)

    cntf = cnt_w.astype(np.float64)
    denom = cntf - float(ddof)

    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sum2_w - (sum_w * sum_w) / np.where(cntf > 0, cntf, np.nan)) / np.where(denom > 0, denom, np.nan)

    var = np.where(np.isfinite(var) & (var >= 0.0), var, np.nan)

    good = (cnt_w >= min_req) & np.isfinite(var) & (var >= 0.0)
    sd = np.sqrt(np.where(good, var, np.nan))

    out[w-1:] = sd
    return out


def safe_div(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    B2 = np.where(np.isfinite(B) & (np.abs(B) > eps), B, np.nan)
    return A / B2


# ============================================================
# 4) GP Tree representation
# ============================================================

@dataclass
class GPNode:
    op: str
    val: Any = None
    kids: Tuple["GPNode", ...] = ()

def gp_clone(n: GPNode) -> GPNode:
    return GPNode(n.op, n.val, tuple(gp_clone(k) for k in n.kids))

def gp_size(n: GPNode) -> int:
    return 1 + sum(gp_size(k) for k in n.kids)

def gp_depth(n: GPNode) -> int:
    if not n.kids:
        return 1
    return 1 + max(gp_depth(k) for k in n.kids)

def gp_to_string(n: GPNode) -> str:
    if n.op == "term":
        return str(n.val)
    if n.op == "const":
        return f"{float(n.val):.6g}"
    if len(n.kids) == 1:
        return f"{n.op}({gp_to_string(n.kids[0])})"
    if len(n.kids) == 2:
        return f"({gp_to_string(n.kids[0])} {n.op} {gp_to_string(n.kids[1])})"
    return f"{n.op}(" + ",".join(gp_to_string(k) for k in n.kids) + ")"

def gp_key(n: GPNode) -> str:
    k = getattr(n, "_gp_key", None)
    if k is not None:
        return k
    k = gp_to_string(n)
    try:
        setattr(n, "_gp_key", k)
    except Exception:
        pass
    return k

def gp_complexity(n: GPNode) -> float:
    s = float(gp_size(n))
    st = gp_key(n)
    roll_cnt = st.count("ts_mean_") + st.count("ts_std_")
    return s + GA_ROLLING_COMPLEXITY_BONUS * float(roll_cnt)


# ============================================================
# 5) GP Random generation + genetic operators
# ============================================================

_BIN_OPS = ["+", "-", "*", "/"]
_UNARY_OPS = ["neg", "abs", "tanh", "sign"]
_TS_OPS = []

def _init_ts_ops():
    global _TS_OPS
    _TS_OPS = []
    for w in GA_ROLL_WINDOWS:
        _TS_OPS.append(f"ts_mean_{int(w)}")
        _TS_OPS.append(f"ts_std_{int(w)}")

_init_ts_ops()

def gp_random_terminal(rng: np.random.Generator, terminals: List[str]) -> GPNode:
    if (rng.random() < GA_P_CONST) and (len(terminals) >= 1):
        c = rng.uniform(GA_CONST_RANGE[0], GA_CONST_RANGE[1])
        return GPNode("const", float(c), ())
    t = terminals[int(rng.integers(0, len(terminals)))]
    return GPNode("term", t, ())

def gp_random_tree(rng: np.random.Generator, terminals: List[str], max_depth: int) -> GPNode:
    if max_depth <= 1:
        return gp_random_terminal(rng, terminals)

    p = rng.random()
    if p < 0.25:
        return gp_random_terminal(rng, terminals)

    # slightly reduce ts-op share to avoid overfit
    fam = rng.choice(["bin", "unary", "ts"], p=[0.58, 0.22, 0.20])
    if fam == "bin":
        op = _BIN_OPS[int(rng.integers(0, len(_BIN_OPS)))]
        a = gp_random_tree(rng, terminals, max_depth - 1)
        b = gp_random_tree(rng, terminals, max_depth - 1)
        return GPNode(op, None, (a, b))

    if fam == "unary":
        op = _UNARY_OPS[int(rng.integers(0, len(_UNARY_OPS)))]
        a = gp_random_tree(rng, terminals, max_depth - 1)
        return GPNode(op, None, (a,))

    op = _TS_OPS[int(rng.integers(0, len(_TS_OPS)))]
    a = gp_random_tree(rng, terminals, max_depth - 1)
    return GPNode(op, None, (a,))

def gp_collect_paths(n: GPNode, path=()) -> List[Tuple[Tuple[int, ...], GPNode]]:
    out = [(path, n)]
    for i, k in enumerate(n.kids):
        out.extend(gp_collect_paths(k, path + (i,)))
    return out

def gp_replace_at(n: GPNode, path: Tuple[int, ...], new_sub: GPNode) -> GPNode:
    if len(path) == 0:
        return gp_clone(new_sub)
    i = path[0]
    kids = list(n.kids)
    kids[i] = gp_replace_at(kids[i], path[1:], new_sub)
    return GPNode(n.op, n.val, tuple(kids))

def gp_crossover(rng: np.random.Generator, a: GPNode, b: GPNode, max_depth: int) -> Tuple[GPNode, GPNode]:
    pa = gp_collect_paths(a)
    pb = gp_collect_paths(b)
    path_a, sub_a = pa[int(rng.integers(0, len(pa)))]
    path_b, sub_b = pb[int(rng.integers(0, len(pb)))]

    na = gp_replace_at(a, path_a, sub_b)
    nb = gp_replace_at(b, path_b, sub_a)

    if gp_depth(na) > max_depth:
        na = gp_clone(a)
    if gp_depth(nb) > max_depth:
        nb = gp_clone(b)
    return na, nb

def gp_mutate(rng: np.random.Generator, n: GPNode, terminals: List[str], max_depth: int) -> GPNode:
    paths = gp_collect_paths(n)
    path, sub = paths[int(rng.integers(0, len(paths)))]

    mtype = rng.choice(["subtree", "point", "const"], p=[0.55, 0.35, 0.10])

    if mtype == "subtree":
        new_sub = gp_random_tree(rng, terminals, max_depth=max(1, max_depth - len(path)))
        out = gp_replace_at(n, path, new_sub)
        if gp_depth(out) > max_depth:
            return gp_clone(n)
        return out

    if mtype == "const":
        if sub.op == "const":
            c = float(sub.val)
            c2 = c + rng.normal(0.0, 0.25)
            out = gp_replace_at(n, path, GPNode("const", float(c2), ()))
            return out
        new_sub = GPNode("const", float(rng.uniform(*GA_CONST_RANGE)), ())
        out = gp_replace_at(n, path, new_sub)
        if gp_depth(out) > max_depth:
            return gp_clone(n)
        return out

    if sub.op == "term":
        t = terminals[int(rng.integers(0, len(terminals)))]
        out = gp_replace_at(n, path, GPNode("term", t, ()))
        return out

    if sub.op in _BIN_OPS:
        op = _BIN_OPS[int(rng.integers(0, len(_BIN_OPS)))]
        out = gp_replace_at(n, path, GPNode(op, None, sub.kids))
        return out

    if sub.op in _UNARY_OPS:
        op = _UNARY_OPS[int(rng.integers(0, len(_UNARY_OPS)))]
        out = gp_replace_at(n, path, GPNode(op, None, sub.kids))
        return out

    if isinstance(sub.op, str) and (sub.op.startswith("ts_mean_") or sub.op.startswith("ts_std_")):
        op = _TS_OPS[int(rng.integers(0, len(_TS_OPS)))]
        out = gp_replace_at(n, path, GPNode(op, None, sub.kids))
        return out

    return gp_clone(n)


# ============================================================
# 6) GP evaluation on a slice (vectorized) with caching
# ============================================================

def gp_eval(
    n: GPNode,
    terminal_bank: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
) -> np.ndarray:
    key = gp_key(n)
    if key in cache:
        return cache[key]

    if n.op == "term":
        out = terminal_bank[str(n.val)]
        cache[key] = out
        return out

    if n.op == "const":
        any_term = next(iter(terminal_bank.values()))
        out = np.full_like(any_term, float(n.val), dtype=np.float64)
        cache[key] = out
        return out

    if n.op in _UNARY_OPS:
        A = gp_eval(n.kids[0], terminal_bank, cache)
        if n.op == "neg":
            out = -A
        elif n.op == "abs":
            out = np.abs(A)
        elif n.op == "tanh":
            out = np.tanh(A)
        elif n.op == "sign":
            out = np.sign(A)
        else:
            out = A
        cache[key] = out
        return out

    if isinstance(n.op, str) and n.op.startswith("ts_mean_"):
        w = int(n.op.split("_")[-1])
        A = gp_eval(n.kids[0], terminal_bank, cache)
        out = ts_mean_strict_np(A, w)
        cache[key] = out
        return out

    if isinstance(n.op, str) and n.op.startswith("ts_std_"):
        w = int(n.op.split("_")[-1])
        A = gp_eval(n.kids[0], terminal_bank, cache)
        out = ts_std_strict_np(A, w, ddof=1)
        cache[key] = out
        return out

    if n.op in _BIN_OPS:
        A = gp_eval(n.kids[0], terminal_bank, cache)
        B = gp_eval(n.kids[1], terminal_bank, cache)
        if n.op == "+":
            out = A + B
        elif n.op == "-":
            out = A - B
        elif n.op == "*":
            out = A * B
        elif n.op == "/":
            out = safe_div(A, B)
        else:
            out = A
        cache[key] = out
        return out

    out = gp_eval(n.kids[0], terminal_bank, cache) if n.kids else next(iter(terminal_bank.values()))
    cache[key] = out
    return out


# ============================================================
# 7) Fitness evaluation (IC -> NW tstat) + neutralization
# ============================================================

@dataclass
class GPFitness:
    obj: float
    t_hac: float
    ic_mean: float
    ic_ir: float
    N: int
    complexity: float
    formula: str


def _index_filter_in_order(base: pd.Index, allowed: pd.Index) -> pd.Index:
    m = base.isin(allowed)
    return base[m]


def gp_score_on_index(
    node: GPNode,
    slice_idx: pd.Index,
    idx_all: pd.Index,
    pos_map: pd.Series,
    y_all: np.ndarray,
    terminal_bank_full: Dict[str, np.ndarray],
    sub_idx: pd.Index,
    risk_beta_full: Optional[np.ndarray] = None,
    risk_lnmcap_full: Optional[np.ndarray] = None,
    industry_codes: Optional[np.ndarray] = None,
    lambda_complexity: float = GA_LAMBDA_COMPLEXITY,
    min_t_eff_override: Optional[int] = None,
) -> GPFitness:

    slice_idx = pd.Index(slice_idx)
    sub_idx = pd.Index(sub_idx)
    sub2 = _index_filter_in_order(sub_idx, slice_idx)

    slice_pos = pd.Series(np.arange(len(slice_idx)), index=slice_idx)
    loc = slice_pos.reindex(sub2).values
    loc = loc[np.isfinite(loc)].astype(int)

    if loc.size < 3:
        return GPFitness(-np.inf, np.nan, np.nan, np.nan, int(loc.size), np.inf, gp_key(node))

    cache = {}
    S_raw_full = gp_eval(node, terminal_bank_full, cache)
    S_raw = S_raw_full[loc, :]

    if (risk_beta_full is not None) or (risk_lnmcap_full is not None) or (industry_codes is not None):
        b = risk_beta_full[loc, :] if risk_beta_full is not None else None
        s = risk_lnmcap_full[loc, :] if risk_lnmcap_full is not None else None
        S_raw = neutralize_signal_barra_style(
            S_raw,
            beta=b,
            ln_mcap=s,
            industry_codes=industry_codes,
            ridge=float(GA_NEUTRALIZE_RIDGE),
            min_cs_n=int(max(30, _MIN_PORT)),
            max_industries=40
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        S_std, _ = standardize_signal_np(
            S_raw,
            cs_standardize=GA_CS_STANDARDIZE,
            ic_method=GA_IC_METHOD,
            auto_robust_zpearson=AUTO_ROBUST_ZPEARSON,
            winsor_q=CS_WINSOR_Q,
            z_clip=Z_CLIP
        )

    pos_sub = pos_map.reindex(sub2).values
    if GA_ASSERT_ALIGNMENT and (not np.all(np.isfinite(pos_sub))):
        raise ValueError("pos_map.reindex(sub_idx) contains NaN; idx_all/sub_idx alignment broken.")
    pos_sub = pos_sub[np.isfinite(pos_sub)].astype(int)

    if pos_sub.size != loc.size:
        sub2 = _index_filter_in_order(sub2, slice_idx)
        loc = slice_pos.reindex(sub2).values
        loc = loc[np.isfinite(loc)].astype(int)
        pos_sub = pos_map.reindex(sub2).values
        if GA_ASSERT_ALIGNMENT and (not np.all(np.isfinite(pos_sub))):
            raise ValueError("pos_map.reindex(sub_idx) contains NaN after re-align; alignment broken.")
        pos_sub = pos_sub[np.isfinite(pos_sub)].astype(int)

    if pos_sub.size != loc.size or loc.size < 3:
        return GPFitness(-np.inf, np.nan, np.nan, np.nan, int(loc.size), np.inf, gp_key(node))

    Y = y_all[pos_sub, :]

    finite_pair = np.isfinite(S_std) & np.isfinite(Y)
    cs_n = np.sum(finite_pair, axis=1).astype(np.int32)
    good_day = cs_n >= int(GA_MIN_CS_N)

    n_eff = int(np.sum(good_day))
    min_t_eff = int(GA_MIN_T_EFF if (min_t_eff_override is None) else int(min_t_eff_override))
    if n_eff < int(min_t_eff):
        return GPFitness(-np.inf, np.nan, np.nan, np.nan, int(n_eff), float(gp_complexity(node)), gp_key(node))

    ic = daily_ic_np(S_std, Y, GA_IC_METHOD)
    ic = np.where(good_day, ic, np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sm = ic_summary_np(ic, PRED_HORIZON)

    t_hac = sm.get("t_hac_primary", np.nan)
    ic_mean = sm.get("IC_mean", np.nan)
    ic_ir = sm.get("IC_IR", np.nan)
    N = int(sm.get("N", 0))

    comp = float(gp_complexity(node))
    if (not np.isfinite(t_hac)) or (N <= 0):
        obj = -np.inf
    else:
        obj = float(t_hac) - float(lambda_complexity) * comp

    return GPFitness(
        obj=float(obj),
        t_hac=float(t_hac) if np.isfinite(t_hac) else np.nan,
        ic_mean=float(ic_mean) if np.isfinite(ic_mean) else np.nan,
        ic_ir=float(ic_ir) if np.isfinite(ic_ir) else np.nan,
        N=int(N),
        complexity=float(comp),
        formula=gp_key(node)
    )


# ============================================================
# 8) GA evolution within one slice
# ============================================================

@dataclass
class GASliceResult:
    rebalance_date: pd.Timestamp
    terminals: List[str]
    train_idx: pd.Index
    ga_train_idx: pd.Index
    ga_val_idx: pd.Index
    holdout_idx: pd.Index

    best_node: GPNode
    best_formula: str

    best_train: GPFitness
    best_val: GPFitness
    best_holdout: Optional[GPFitness]

    gen_log: pd.DataFrame
    top_bank: pd.DataFrame

    top10_nodes: List[GPNode]
    top10_table: pd.DataFrame


def _val_obj_adjusted(train_fit: GPFitness, val_fit: GPFitness) -> float:
    if (not np.isfinite(val_fit.obj)) or (not np.isfinite(val_fit.t_hac)):
        return -np.inf

    gap_pen = 0.0
    if np.isfinite(train_fit.t_hac) and np.isfinite(val_fit.t_hac):
        gap = float(train_fit.t_hac) - float(val_fit.t_hac)
        if gap > 0:
            gap_pen = GA_LAMBDA_GAP * gap

    flip_pen = 0.0
    if np.isfinite(train_fit.ic_mean) and np.isfinite(val_fit.ic_mean):
        if (train_fit.ic_mean * val_fit.ic_mean) < 0:
            flip_pen = GA_SIGN_FLIP_PENALTY

    return float(val_fit.obj) - float(gap_pen) - float(flip_pen)


def ga_evolve_on_slice(
    t0: pd.Timestamp,
    terminals: List[str],
    slice_idx: pd.Index,
    holdout_idx: pd.Index,
    idx_all: pd.Index,
    y_all: np.ndarray,
    terminal_bank_full: Dict[str, np.ndarray],
    pos_map: pd.Series,
    risk_beta_full: Optional[np.ndarray] = None,
    risk_lnmcap_full: Optional[np.ndarray] = None,
    industry_codes: Optional[np.ndarray] = None,
    seed: int = 0,
    verbose: bool = True,
) -> GASliceResult:

    purge_gap = int((PRED_HORIZON - 1) + DELAY) if GA_PURGE_GAP_DAYS is None else int(GA_PURGE_GAP_DAYS)
    ga_tr, ga_va = ga_time_split_with_purge(slice_idx, GA_VAL_RATIO, purge_gap)

    if len(ga_tr) < 60 or len(ga_va) < 20:
        if verbose:
            print(f"[GA][warn] {pd.Timestamp(t0).date()} small GA split: train={len(ga_tr)} val={len(ga_va)} (purge={purge_gap})")

    rng = np.random.default_rng(int(seed))

    pop: List[GPNode] = []
    for i in range(int(GA_POP_SIZE)):
        depth = int(rng.integers(GA_MIN_DEPTH, GA_MAX_DEPTH + 1))
        pop.append(gp_random_tree(rng, terminals, max_depth=depth))

    best_val_obj = -np.inf  # NOTE: now stores adjusted val objective (anti-overfit)
    best_node = None
    best_train_fit = None
    best_val_fit = None
    no_improve = 0

    gen_rows = []
    top_rows = []

    # ---- NEW: keep unique candidate nodes across generations for Top-10 mining ----
    formula_to_node: Dict[str, GPNode] = {}

    for gen in range(int(GA_N_GEN)):
        fits = []
        for ind in pop:
            f = gp_score_on_index(
                ind, slice_idx, idx_all, pos_map, y_all, terminal_bank_full, ga_tr,
                risk_beta_full=risk_beta_full, risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes, lambda_complexity=GA_LAMBDA_COMPLEXITY
            )
            fits.append(f)

        # rank by train obj
        order = np.argsort([-(f.obj if np.isfinite(f.obj) else -np.inf) for f in fits])

        # ---- NEW: elite de-dup by formula to prevent clone collapse ----
        elite_n_target = max(1, int(np.floor(GA_ELITE_FRAC * len(pop))))
        elites: List[GPNode] = []
        elite_fits: List[GPFitness] = []
        seen = set()
        for i in order:
            fi = fits[int(i)]
            if not np.isfinite(fi.obj):
                continue
            if GA_ELITE_UNIQUE and (fi.formula in seen):
                continue
            elites.append(pop[int(i)])
            elite_fits.append(fi)
            seen.add(fi.formula)
            if len(elites) >= elite_n_target:
                break
        if len(elites) == 0:
            # extremely rare: all invalid -> random restart small
            elites = [pop[int(order[0])]]
            elite_fits = [fits[int(order[0])]]

        # candidate pool update
        for e in elites[:min(12, len(elites))]:
            k = gp_key(e)
            if k not in formula_to_node:
                formula_to_node[k] = gp_clone(e)

        # evaluate top-M elites on GA_val
        topm = int(min(max(1, GA_VAL_TOPM), len(elites)))
        val_fits = []
        val_obj_adj = np.full(topm, -np.inf, dtype=np.float64)

        for j in range(topm):
            vf = gp_score_on_index(
                elites[j], slice_idx, idx_all, pos_map, y_all, terminal_bank_full, ga_va,
                risk_beta_full=risk_beta_full, risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes, lambda_complexity=GA_LAMBDA_COMPLEXITY
            )
            val_fits.append(vf)
            val_obj_adj[j] = _val_obj_adjusted(elite_fits[j], vf)

        j_star = int(np.argmax(val_obj_adj))
        cand = elites[j_star]
        cand_train = elite_fits[j_star]
        val_best = val_fits[j_star]
        val_best_adj = float(val_obj_adj[j_star])

        train_best = elite_fits[0]

        improved = (np.isfinite(val_best_adj) and (val_best_adj > best_val_obj + GA_EARLYSTOP_MIN_DELTA))
        if improved:
            best_val_obj = float(val_best_adj)
            best_node = gp_clone(cand)
            best_train_fit = cand_train
            best_val_fit = val_best
            no_improve = 0
        else:
            no_improve += 1

        gen_rows.append({
            "gen": gen,
            "elite_train_obj": float(train_best.obj) if np.isfinite(train_best.obj) else np.nan,
            "elite_train_t_hac": train_best.t_hac,
            "elite_train_ic_mean": train_best.ic_mean,
            "elite_train_ic_ir": train_best.ic_ir,
            "elite_train_N": train_best.N,
            "elite_complexity": train_best.complexity,
            "elite_val_obj": float(val_best.obj) if np.isfinite(val_best.obj) else np.nan,
            "elite_val_obj_adj": float(val_best_adj) if np.isfinite(val_best_adj) else np.nan,  # NEW
            "elite_val_t_hac": val_best.t_hac,
            "elite_val_ic_mean": val_best.ic_mean,
            "elite_val_ic_ir": val_best.ic_ir,
            "elite_val_N": val_best.N,
            "best_val_obj_sofar": float(best_val_obj) if np.isfinite(best_val_obj) else np.nan,  # now adj
            "no_improve": int(no_improve),
            "elite_formula": train_best.formula
        })

        # store top bank (top 10 each gen; now elites are unique already)
        topk = min(10, len(elites))
        for j in range(topk):
            top_rows.append({
                "gen": gen,
                "rank": j,
                "train_obj": elite_fits[j].obj,
                "train_t_hac": elite_fits[j].t_hac,
                "train_ic_mean": elite_fits[j].ic_mean,
                "complexity": elite_fits[j].complexity,
                "formula": elite_fits[j].formula
            })

        if verbose and (gen % 10 == 0 or gen == GA_N_GEN - 1):
            print(f"[GA] {pd.Timestamp(t0).date()} gen={gen:03d} | "
                  f"train_tNW={train_best.t_hac:.3f} val_tNW={val_best.t_hac:.3f} | "
                  f"val_obj_adj_best={best_val_obj:.4f} | no_improve={no_improve}")

        if no_improve >= int(GA_EARLYSTOP_PATIENCE):
            if verbose:
                print(f"[GA][earlystop] {pd.Timestamp(t0).date()} stop at gen={gen} (patience={GA_EARLYSTOP_PATIENCE})")
            break

        def tournament_pick() -> GPNode:
            kk = int(GA_TOURNAMENT_K)
            cand_idx = rng.integers(0, len(pop), size=kk)
            best_i = None
            best_o = -np.inf
            for ii in cand_idx:
                oo = fits[int(ii)].obj
                oo = oo if np.isfinite(oo) else -np.inf
                if oo > best_o:
                    best_o = oo
                    best_i = int(ii)
            return pop[int(best_i)] if best_i is not None else pop[int(rng.integers(0, len(pop)))]

        new_pop: List[GPNode] = []
        new_pop.extend([gp_clone(e) for e in elites])

        while len(new_pop) < len(pop):
            u = rng.random()
            if u < GA_CROSSOVER_P:
                p1 = tournament_pick()
                p2 = tournament_pick()
                c1, c2 = gp_crossover(rng, gp_clone(p1), gp_clone(p2), max_depth=GA_MAX_DEPTH)
                new_pop.append(c1)
                if len(new_pop) < len(pop):
                    new_pop.append(c2)
            elif u < GA_CROSSOVER_P + GA_MUTATION_P:
                p = tournament_pick()
                m = gp_mutate(rng, gp_clone(p), terminals, max_depth=GA_MAX_DEPTH)
                new_pop.append(m)
            else:
                p = tournament_pick()
                new_pop.append(gp_clone(p))

        pop = new_pop[:len(pop)]

    if best_node is None:
        if verbose:
            print(f"[GA][fallback] {pd.Timestamp(t0).date()} no valid val best; fallback to last-gen best train")
        fits = []
        for ind in pop:
            f = gp_score_on_index(
                ind, slice_idx, idx_all, pos_map, y_all, terminal_bank_full, ga_tr,
                risk_beta_full=risk_beta_full, risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes, lambda_complexity=GA_LAMBDA_COMPLEXITY
            )
            fits.append(f)
        order = np.argsort([-(f.obj if np.isfinite(f.obj) else -np.inf) for f in fits])
        best_node = gp_clone(pop[int(order[0])])
        best_train_fit = fits[int(order[0])]
        best_val_fit = gp_score_on_index(
            best_node, slice_idx, idx_all, pos_map, y_all, terminal_bank_full, ga_va,
            risk_beta_full=risk_beta_full, risk_lnmcap_full=risk_lnmcap_full,
            industry_codes=industry_codes, lambda_complexity=GA_LAMBDA_COMPLEXITY
        )

    gen_log = pd.DataFrame(gen_rows).set_index("gen") if gen_rows else pd.DataFrame()
    top_bank = pd.DataFrame(top_rows) if top_rows else pd.DataFrame()

    # ============================================================
    # ---- NEW: Build per-slice Top-10 UNIQUE formulas (for signal building)
    # Ranking by adjusted val objective + stability proxy from top_bank
    # ============================================================

    top10_nodes: List[GPNode] = []
    top10_table = pd.DataFrame()

    if len(formula_to_node) > 0:
        # stability proxy from top_bank
        if (top_bank is not None) and (len(top_bank) > 0):
            stab = top_bank.groupby("formula").agg(
                n_gen=("gen", "nunique"),
                n_rows=("gen", "size"),
                max_train_obj=("train_obj", "max"),
            )
        else:
            stab = pd.DataFrame(index=list(formula_to_node.keys()), data={"n_gen": 0, "n_rows": 0, "max_train_obj": np.nan})

        # pool cap: pick most promising formulas by max_train_obj + n_gen
        stab2 = stab.copy()
        if "max_train_obj" not in stab2.columns:
            stab2["max_train_obj"] = np.nan
        if "n_gen" not in stab2.columns:
            stab2["n_gen"] = 0

        stab2["pool_score"] = stab2["max_train_obj"].fillna(-np.inf) + 0.05 * stab2["n_gen"].fillna(0).astype(float)
        pool = stab2.sort_values("pool_score", ascending=False).head(int(GA_TOPN_POOL_MAX)).index.tolist()

        cand_rows = []
        for formula in pool:
            node = formula_to_node.get(formula, None)
            if node is None:
                continue

            trf = gp_score_on_index(
                node, slice_idx, idx_all, pos_map, y_all, terminal_bank_full, ga_tr,
                risk_beta_full=risk_beta_full, risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes, lambda_complexity=GA_LAMBDA_COMPLEXITY
            )
            vaf = gp_score_on_index(
                node, slice_idx, idx_all, pos_map, y_all, terminal_bank_full, ga_va,
                risk_beta_full=risk_beta_full, risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes, lambda_complexity=GA_LAMBDA_COMPLEXITY
            )
            if (not np.isfinite(vaf.t_hac)) or (float(vaf.t_hac) < float(GA_TOPN_MIN_VAL_T)):
                continue

            adj = _val_obj_adjusted(trf, vaf)
            cand_rows.append({
                "formula": formula,
                "complexity": float(trf.complexity),
                "train_obj": float(trf.obj) if np.isfinite(trf.obj) else np.nan,
                "train_t_hac": trf.t_hac,
                "train_ic_mean": trf.ic_mean,
                "train_N": trf.N,
                "val_obj": float(vaf.obj) if np.isfinite(vaf.obj) else np.nan,
                "val_obj_adj": float(adj) if np.isfinite(adj) else np.nan,
                "val_t_hac": vaf.t_hac,
                "val_ic_mean": vaf.ic_mean,
                "val_N": vaf.N,
                "n_gen": int(stab.loc[formula, "n_gen"]) if (formula in stab.index and "n_gen" in stab.columns) else 0,
                "n_rows": int(stab.loc[formula, "n_rows"]) if (formula in stab.index and "n_rows" in stab.columns) else 0,
                "max_train_obj": float(stab.loc[formula, "max_train_obj"]) if (formula in stab.index and "max_train_obj" in stab.columns) else np.nan,
            })

        if len(cand_rows) > 0:
            dfc = pd.DataFrame(cand_rows)
            # sort: primary adj val obj, then stability, then simpler
            dfc = dfc.sort_values(
                ["val_obj_adj", "n_gen", "val_t_hac", "max_train_obj", "complexity"],
                ascending=[False, False, False, False, True]
            ).reset_index(drop=True)

            dfc["rank"] = np.arange(1, len(dfc) + 1)
            dfc = dfc.head(int(GA_TOPN_PER_SLICE)).copy()

            top10_table = dfc.set_index("rank")
            top10_nodes = [formula_to_node[f] for f in top10_table["formula"].tolist() if f in formula_to_node]

    # ensure best_node included in top10 if missing (rare)
    if best_node is not None and len(top10_nodes) > 0:
        bf = gp_key(best_node)
        if bf not in set(top10_table["formula"].tolist()):
            # append if space, otherwise replace worst
            if len(top10_nodes) < int(GA_TOPN_PER_SLICE):
                top10_nodes.append(gp_clone(best_node))
            else:
                # replace last
                top10_nodes[-1] = gp_clone(best_node)

    return GASliceResult(
        rebalance_date=pd.Timestamp(t0),
        terminals=list(terminals),
        train_idx=pd.Index(slice_idx),
        ga_train_idx=pd.Index(ga_tr),
        ga_val_idx=pd.Index(ga_va),
        holdout_idx=pd.Index(holdout_idx),
        best_node=best_node,
        best_formula=gp_to_string(best_node),
        best_train=best_train_fit,
        best_val=best_val_fit,
        best_holdout=None,
        gen_log=gen_log,
        top_bank=top_bank,
        top10_nodes=top10_nodes,
        top10_table=top10_table
    )

 


# ============================================================
# 9) Prepare terminals: build slice terminal bank
# ============================================================

def build_terminal_bank_for_index(
    features_np: Dict[str, np.ndarray],
    idx_all: pd.Index,
    slice_idx: pd.Index,
    terminals: List[str],
    delay: int,
) -> Dict[str, np.ndarray]:
    pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)
    pos = pos_map.reindex(slice_idx).values

    if GA_ASSERT_ALIGNMENT:
        if not np.all(np.isfinite(pos)):
            raise ValueError("build_terminal_bank_for_index: slice_idx contains dates not in idx_all (pos_map reindex has NaN).")
        pos_int = pos.astype(int)
        if np.min(pos_int) < int(delay):
            raise ValueError("build_terminal_bank_for_index: min(pos) < delay; would cause negative indexing.")
    else:
        pos_int = pos.astype(int)

    bank = {}
    for f in terminals:
        X = features_np[f]
        S = X[pos_int - int(delay), :]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            S_std, _ = standardize_signal_np(
                S,
                cs_standardize=GA_CS_STANDARDIZE,
                ic_method=GA_IC_METHOD,
                auto_robust_zpearson=AUTO_ROBUST_ZPEARSON,
                winsor_q=CS_WINSOR_Q,
                z_clip=Z_CLIP
            )
        bank[f] = S_std
    return bank


# ============================================================
# 10) Rolling beta (strict) for Barra-style risk factor
# ============================================================

@njit(cache=True, parallel=True)
def _rolling_beta_strict_numba(lr: np.ndarray, mm: np.ndarray, window: int) -> np.ndarray:
    T, N = lr.shape
    w = int(window)
    out = np.empty((T, N), dtype=np.float64)
    out[:] = np.nan
    if w <= 1:
        return out

    for j in prange(N):
        for t in range(w - 1, T):
            ok = True
            for k in range(t - w + 1, t + 1):
                if (not np.isfinite(mm[k])) or (not np.isfinite(lr[k, j])):
                    ok = False
                    break
            if not ok:
                continue

            sm_m = 0.0
            sm_r = 0.0
            for k in range(t - w + 1, t + 1):
                sm_m += mm[k]
                sm_r += lr[k, j]
            mean_m = sm_m / w
            mean_r = sm_r / w

            cov = 0.0
            var = 0.0
            for k in range(t - w + 1, t + 1):
                dm = mm[k] - mean_m
                dr = lr[k, j] - mean_r
                cov += dr * dm
                var += dm * dm

            denom = (w - 1)
            if denom <= 0:
                continue
            cov /= denom
            var /= denom
            if np.isfinite(var) and (var != 0.0):
                out[t, j] = cov / var

    return out


def compute_rolling_beta_matrix_strict(
    logret: pd.DataFrame,
    mkt_ret: pd.Series,
    window: int = 60,
    lag_for_trade: int = 1,
) -> np.ndarray:
    idx = logret.index
    m = mkt_ret.reindex(idx).astype(float)

    if lag_for_trade and int(lag_for_trade) > 0:
        lr_df = logret.shift(int(lag_for_trade))
        mm_ser = m.shift(int(lag_for_trade))
    else:
        lr_df = logret
        mm_ser = m

    if _NUMBA_AVAILABLE:
        lr = lr_df.values.astype(np.float64)
        mm = mm_ser.values.astype(np.float64)
        return _rolling_beta_strict_numba(lr, mm, int(window))

    beta = np.full(lr_df.shape, np.nan, dtype=np.float64)

    var_m = mm_ser.rolling(window, min_periods=window).var(ddof=1)
    for j, col in enumerate(lr_df.columns):
        ri = lr_df[col]
        cov = ri.rolling(window, min_periods=window).cov(mm_ser)
        cnt = pd.concat([ri, mm_ser], axis=1).rolling(window, min_periods=window).count().min(axis=1)
        b = cov / var_m
        b = b.where(cnt >= window, np.nan)
        beta[:, j] = b.values.astype(np.float64)

    return beta


# ============================================================
# 11) Orchestrator: run GA on top of wf + wf_lasso, keep ALL artifacts
# ============================================================

def run_walkforward_ga_layer(
    wf: dict,
    wf_lasso: dict,
    features: Dict[str, pd.DataFrame],
    logret: pd.DataFrame,
    spy_close_price: Optional[pd.Series] = None,
    ln_mcap: Optional[pd.DataFrame] = None,
    industry_codes: Optional[np.ndarray] = None,
    beta_window: int = 60,
    verbose: bool = True,
) -> dict:

    features, logret = ensure_aligned(features, logret)
    idx_all = logret.index
    idx_valid_global = valid_index(idx_all, DELAY, PRED_HORIZON)

    logret_np = logret.values.astype(np.float64)
    features_np = {k: v.values.astype(np.float64) for k, v in features.items()}
    y_all = forward_sum_strict_np(logret_np, PRED_HORIZON)

    pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)

    beta_mat = None
    if spy_close_price is not None:
        spy_close = spy_close_price.reindex(idx_all).astype(float)
        spy_ret = np.log(spy_close).diff()
        beta_mat = compute_rolling_beta_matrix_strict(
            logret=logret,
            mkt_ret=spy_ret,
            window=int(beta_window),
            lag_for_trade=DELAY
        )

    ln_mcap_mat = None
    if ln_mcap is not None:
        ln_mcap_aligned = ln_mcap.reindex(index=idx_all, columns=logret.columns).astype(float)
        ln_mcap_mat = ln_mcap_aligned.values.astype(np.float64)

    lasso_results: List[Any] = wf_lasso.get("slice_results", [])
    lasso_by_date = {pd.Timestamp(r.rebalance_date): r for r in lasso_results}

    screen_reports = wf.get("screen_reports", {})
    slice_stats = wf.get("slice_stats", pd.DataFrame())

    rebs = sorted([pd.Timestamp(k) for k in screen_reports.keys()])

    # ✅ FIX: build selected_by_slice map (Route A output) for terminal fallback
    _sel_raw = wf.get("selected_by_slice", {}) or {}
    selected_by_date = {}
    if isinstance(_sel_raw, dict):
        for k, v in _sel_raw.items():
            try:
                selected_by_date[pd.Timestamp(k)] = v
            except Exception:
                # keep best effort: if key is weird, ignore
                pass

    all_feature_names = list(features_np.keys())

    def _unique_preserve(seq):
        out = []
        seen = set()
        for x in seq:
            if x is None:
                continue
            s = str(x)
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _sanitize_terminals(seq):
        # keep only factors that exist in features_np
        out = []
        seen = set()
        for x in seq:
            s = str(x)
            if s in seen:
                continue
            if s not in features_np:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _get_selected_pool(t0, scr_obj):
        # 1) prefer wf["selected_by_slice"][t0]
        pool = selected_by_date.get(pd.Timestamp(t0), None)

        # 2) fallback: maybe screen report has an attribute
        if pool is None:
            for attr in ["selected_by_slice", "selected_factors", "selected", "features_selected"]:
                if hasattr(scr_obj, attr):
                    try:
                        pool = getattr(scr_obj, attr)
                        break
                    except Exception:
                        pass

        # normalize
        if pool is None:
            pool = all_feature_names
        if isinstance(pool, (pd.Index, np.ndarray)):
            pool = pool.tolist()
        if not isinstance(pool, (list, tuple)):
            pool = list(pool) if hasattr(pool, "__iter__") else all_feature_names

        pool = _sanitize_terminals(_unique_preserve(pool))
        if len(pool) == 0:
            pool = _sanitize_terminals(all_feature_names)
        return pool

    ga_slice_results: List[GASliceResult] = []
    ga_rows = []

    for t0 in rebs:
        scr = screen_reports.get(t0, None)
        if scr is None or not hasattr(scr, "train_index"):
            continue

        train_idx = pd.Index(scr.train_index).intersection(idx_valid_global)
        if len(train_idx) < 200:
            continue

        if (slice_stats is not None) and (len(slice_stats) > 0) and (t0 in slice_stats.index):
            hs = pd.Timestamp(slice_stats.loc[t0, "holdout_start"])
            he = pd.Timestamp(slice_stats.loc[t0, "holdout_end"])
            holdout_idx = pd.Index(idx_valid_global[(idx_valid_global >= hs) & (idx_valid_global <= he)])
        else:
            p0 = _date_to_pos(idx_valid_global, t0)
            holdout_idx = _slice_by_pos(idx_valid_global, p0, min(len(idx_valid_global), p0 + int(BACKTEST_DAYS)))

        if len(holdout_idx) < 10:
            continue

        # ✅ FIX: terminals selection w/ fallback (NO MORE SKIP)
        K_MIN = 3
        K_FILL = 5  # follow your screenshot example (补齐到 5；至少>=3)

        selected_pool = _get_selected_pool(t0, scr)

        lres = lasso_by_date.get(pd.Timestamp(t0), None)

        terminals_source = "lasso_nonzero"
        terminals_raw = []

        if (lres is None) or (not hasattr(lres, "nonzero_factors")):
            # missing lasso slice -> fallback to selected_by_slice pool
            terminals_source = "fallback_selected_by_slice"
            terminals_raw = selected_pool[:K_FILL]
        else:
            terminals_raw = list(getattr(lres, "nonzero_factors", []))

        terminals = _sanitize_terminals(_unique_preserve(terminals_raw))

        # if too few terminals after lasso, top-up from selected_pool
        if len(terminals) < K_MIN:
            terminals_source = "lasso_nonzero+topup_selected_by_slice" if terminals_source == "lasso_nonzero" else terminals_source
            for f in selected_pool:
                if f not in terminals:
                    terminals.append(f)
                if len(terminals) >= max(K_MIN, K_FILL):
                    break

        # absolute last resort: top up from ALL features (guarantee coverage)
        if len(terminals) < K_MIN:
            terminals_source = terminals_source + "+lastresort_all_features"
            for f in all_feature_names:
                if f not in terminals:
                    terminals.append(f)
                if len(terminals) >= max(K_MIN, K_FILL):
                    break

        # still protect against pathological case (should not happen)
        if len(terminals) < 1:
            raise ValueError(f"[GA] {t0.date()} terminals empty even after fallback; features list empty?")

        if verbose:
            if (lres is None) or (not hasattr(lres, "nonzero_factors")):
                print(f"[GA][fallback] {t0.date()} missing lasso slice result -> terminals={len(terminals)} source={terminals_source}")
            else:
                if len(terminals_raw) < K_MIN:
                    print(f"[GA][fallback] {t0.date()} too few terminals after lasso: {len(terminals_raw)} -> filled to {len(terminals)} source={terminals_source}")

        bank_train_full = build_terminal_bank_for_index(
            features_np=features_np,
            idx_all=idx_all,
            slice_idx=train_idx,
            terminals=terminals,
            delay=DELAY
        )
        bank_holdout = build_terminal_bank_for_index(
            features_np=features_np,
            idx_all=idx_all,
            slice_idx=holdout_idx,
            terminals=terminals,
            delay=DELAY
        )

        beta_train = beta_mat[pos_map.reindex(train_idx).values.astype(int), :] if beta_mat is not None else None
        beta_hold = beta_mat[pos_map.reindex(holdout_idx).values.astype(int), :] if beta_mat is not None else None

        size_train = ln_mcap_mat[pos_map.reindex(train_idx).values.astype(int), :] if ln_mcap_mat is not None else None
        size_hold  = ln_mcap_mat[pos_map.reindex(holdout_idx).values.astype(int), :] if ln_mcap_mat is not None else None

        seed = int(GA_SEED_BASE + (int(pd.Timestamp(t0).strftime("%Y%m%d")) % 100000))

        res = ga_evolve_on_slice(
            t0=t0,
            terminals=terminals,
            slice_idx=train_idx,
            holdout_idx=holdout_idx,
            idx_all=idx_all,
            y_all=y_all,
            terminal_bank_full=bank_train_full,
            pos_map=pos_map,
            risk_beta_full=beta_train,
            risk_lnmcap_full=size_train,
            industry_codes=industry_codes,
            seed=seed,
            verbose=verbose
        )

        hold_min_t = int(max(20, min(40, len(holdout_idx))))
        hold_fit = gp_score_on_index(
            res.best_node,
            slice_idx=holdout_idx,
            idx_all=idx_all,
            pos_map=pos_map,
            y_all=y_all,
            terminal_bank_full=bank_holdout,
            sub_idx=holdout_idx,
            risk_beta_full=beta_hold,
            risk_lnmcap_full=size_hold,
            industry_codes=industry_codes,
            lambda_complexity=GA_LAMBDA_COMPLEXITY,
            min_t_eff_override=hold_min_t
        )
        res.best_holdout = hold_fit

        if (res.top10_nodes is not None) and (len(res.top10_nodes) > 0):
            hrows = []
            for rnk, nd in enumerate(res.top10_nodes, start=1):
                hf = gp_score_on_index(
                    nd,
                    slice_idx=holdout_idx,
                    idx_all=idx_all,
                    pos_map=pos_map,
                    y_all=y_all,
                    terminal_bank_full=bank_holdout,
                    sub_idx=holdout_idx,
                    risk_beta_full=beta_hold,
                    risk_lnmcap_full=size_hold,
                    industry_codes=industry_codes,
                    lambda_complexity=GA_LAMBDA_COMPLEXITY,
                    min_t_eff_override=hold_min_t
                )
                hrows.append({
                    "rank": int(rnk),
                    "formula": hf.formula,
                    "holdout_obj": float(hf.obj) if np.isfinite(hf.obj) else np.nan,
                    "holdout_t_hac": hf.t_hac,
                    "holdout_ic_mean": hf.ic_mean,
                    "holdout_ic_ir": hf.ic_ir,
                    "holdout_N": hf.N,
                })
            hdf = pd.DataFrame(hrows).set_index("rank") if len(hrows) > 0 else pd.DataFrame()
            if (res.top10_table is not None) and (len(res.top10_table) > 0) and (len(hdf) > 0):
                tmp = res.top10_table.copy()
                tmp = tmp.merge(
                    hdf.reset_index(drop=False)[["rank","formula","holdout_t_hac","holdout_ic_mean","holdout_ic_ir","holdout_N"]],
                    how="left", on=["rank","formula"]
                )
                tmp = tmp.set_index("rank")
                res.top10_table = tmp

        ga_slice_results.append(res)

        ga_rows.append({
            "rebalance_date": pd.Timestamp(t0),
            "n_terminals": int(len(terminals)),
            "train_start": train_idx[0],
            "train_end": train_idx[-1],
            "ga_train_n": int(len(res.ga_train_idx)),
            "ga_val_n": int(len(res.ga_val_idx)),
            "holdout_start": holdout_idx[0],
            "holdout_end": holdout_idx[-1],
            "best_formula": res.best_formula,
            "train_t_hac": res.best_train.t_hac,
            "val_t_hac": res.best_val.t_hac,
            "holdout_t_hac": hold_fit.t_hac,
            "train_ic_mean": res.best_train.ic_mean,
            "val_ic_mean": res.best_val.ic_mean,
            "holdout_ic_mean": hold_fit.ic_mean,
            "complexity": res.best_train.complexity,
            "val_obj": res.best_val.obj,
            "top10_formulas": (res.top10_table["formula"].tolist() if (res.top10_table is not None and len(res.top10_table)>0) else []),

            # ✅ FIX: add transparent provenance (doesn't change any existing names)
            "terminals_source": terminals_source,
            "terminals_lasso_k": int(len(terminals_raw)) if terminals_source.startswith("lasso") else np.nan,
        })

        if verbose:
            print("==========================================================")
            print(f"[GA WF CONNECTED] rebalance={pd.Timestamp(t0).date()}")
            print(f"terminals(K)={len(terminals)} | max_depth={GA_MAX_DEPTH} | roll={GA_ROLL_WINDOWS}")
            print(f"inner split: ga_train={len(res.ga_train_idx)}  ga_val={len(res.ga_val_idx)}  purge={(PRED_HORIZON-1)+DELAY if GA_PURGE_GAP_DAYS is None else GA_PURGE_GAP_DAYS}")
            print(f"best: train tNW={res.best_train.t_hac:.3f} | val tNW={res.best_val.t_hac:.3f} | holdout tNW={hold_fit.t_hac:.3f}")
            print(f"best formula: {res.best_formula[:160]}{'...' if len(res.best_formula)>160 else ''}")
            if (res.top10_table is not None) and (len(res.top10_table) > 0):
                print(f"top10 ready: n={len(res.top10_table)} (unique, ranked by val_obj_adj, with holdout QA columns if available)")
            print("==========================================================")

    ga_stats = pd.DataFrame(ga_rows).set_index("rebalance_date") if ga_rows else pd.DataFrame()

    return {
        "slice_results": ga_slice_results,
        "slice_stats": ga_stats,
        "config_used": {
            "GA_POP_SIZE": GA_POP_SIZE,
            "GA_N_GEN": GA_N_GEN,
            "GA_MAX_DEPTH": GA_MAX_DEPTH,
            "GA_VAL_RATIO": GA_VAL_RATIO,
            "GA_PURGE_GAP_DAYS": GA_PURGE_GAP_DAYS if GA_PURGE_GAP_DAYS is not None else "auto=(H-1)+DELAY",
            "GA_ROLL_WINDOWS": list(GA_ROLL_WINDOWS),
            "GA_LAMBDA_COMPLEXITY": GA_LAMBDA_COMPLEXITY,
            "GA_ROLLING_COMPLEXITY_BONUS": GA_ROLLING_COMPLEXITY_BONUS,
            "GA_LAMBDA_GAP": GA_LAMBDA_GAP,
            "GA_SIGN_FLIP_PENALTY": GA_SIGN_FLIP_PENALTY,
            "GA_ELITE_UNIQUE": GA_ELITE_UNIQUE,
            "GA_TOPN_PER_SLICE": GA_TOPN_PER_SLICE,
            "FITNESS": "Newey-West tstat(mean daily IC) - lambda*complexity",
            "VAL_SELECTION": "val_obj_adj = val_obj - gap_penalty - signflip_penalty (anti-overfit)",
            "NEUTRALIZATION": "CS regression residual vs beta/ln_mcap/industry if provided",
            "BETA_WINDOW": beta_window,
            "SEED_BASE": GA_SEED_BASE,
            "EARLYSTOP": {
                "PATIENCE": GA_EARLYSTOP_PATIENCE,
                "MIN_DELTA": GA_EARLYSTOP_MIN_DELTA,
            }
        }
    }


# ============================================================
# -------------- CLASS PACKAGING (NO LOGIC CHANGE) -----------
# ============================================================

class GA_LAYER_WF_CONNECTED:
    GA_SEED_BASE = GA_SEED_BASE
    GA_VAL_RATIO = GA_VAL_RATIO
    GA_PURGE_GAP_DAYS = GA_PURGE_GAP_DAYS
    GA_POP_SIZE = GA_POP_SIZE
    GA_N_GEN = GA_N_GEN
    GA_TOURNAMENT_K = GA_TOURNAMENT_K
    GA_ELITE_FRAC = GA_ELITE_FRAC
    GA_CROSSOVER_P = GA_CROSSOVER_P
    GA_MUTATION_P = GA_MUTATION_P
    GA_REPRO_P = GA_REPRO_P
    GA_MAX_DEPTH = GA_MAX_DEPTH
    GA_MIN_DEPTH = GA_MIN_DEPTH
    GA_P_CONST = GA_P_CONST
    GA_CONST_RANGE = GA_CONST_RANGE
    GA_ROLL_WINDOWS = GA_ROLL_WINDOWS
    GA_TS_MIN_FRAC = GA_TS_MIN_FRAC
    GA_LAMBDA_COMPLEXITY = GA_LAMBDA_COMPLEXITY
    GA_ROLLING_COMPLEXITY_BONUS = GA_ROLLING_COMPLEXITY_BONUS
    GA_LAMBDA_GAP = GA_LAMBDA_GAP
    GA_SIGN_FLIP_PENALTY = GA_SIGN_FLIP_PENALTY
    GA_ELITE_UNIQUE = GA_ELITE_UNIQUE
    GA_EARLYSTOP_PATIENCE = GA_EARLYSTOP_PATIENCE
    GA_EARLYSTOP_MIN_DELTA = GA_EARLYSTOP_MIN_DELTA
    GA_VAL_TOPM = GA_VAL_TOPM
    GA_NEUTRALIZE_RIDGE = GA_NEUTRALIZE_RIDGE
    GA_MIN_CS_N = GA_MIN_CS_N
    GA_MIN_T_EFF = GA_MIN_T_EFF
    GA_TOPN_PER_SLICE = GA_TOPN_PER_SLICE
    GA_TOPN_POOL_MAX = GA_TOPN_POOL_MAX
    GA_TOPN_MIN_VAL_T = GA_TOPN_MIN_VAL_T
    GA_ASSERT_ALIGNMENT = GA_ASSERT_ALIGNMENT
    GA_CS_STANDARDIZE = GA_CS_STANDARDIZE
    GA_IC_METHOD = GA_IC_METHOD

    GPNode = GPNode
    GPFitness = GPFitness
    GASliceResult = GASliceResult

    ga_to_date_index = staticmethod(ga_to_date_index)
    ga_spy_aggs_to_close_series = staticmethod(ga_spy_aggs_to_close_series)
    ga_time_split_with_purge = staticmethod(ga_time_split_with_purge)

    _build_industry_dummies = staticmethod(_build_industry_dummies)
    _get_industry_dummies_cached = staticmethod(_get_industry_dummies_cached)
    neutralize_signal_barra_style = staticmethod(neutralize_signal_barra_style)

    ts_mean_strict_np = staticmethod(ts_mean_strict_np)
    ts_std_strict_np = staticmethod(ts_std_strict_np)
    safe_div = staticmethod(safe_div)

    gp_clone = staticmethod(gp_clone)
    gp_size = staticmethod(gp_size)
    gp_depth = staticmethod(gp_depth)
    gp_to_string = staticmethod(gp_to_string)
    gp_key = staticmethod(gp_key)
    gp_complexity = staticmethod(gp_complexity)

    gp_random_terminal = staticmethod(gp_random_terminal)
    gp_random_tree = staticmethod(gp_random_tree)
    gp_collect_paths = staticmethod(gp_collect_paths)
    gp_replace_at = staticmethod(gp_replace_at)
    gp_crossover = staticmethod(gp_crossover)
    gp_mutate = staticmethod(gp_mutate)

    gp_eval = staticmethod(gp_eval)
    gp_score_on_index = staticmethod(gp_score_on_index)

    _val_obj_adjusted = staticmethod(_val_obj_adjusted)
    ga_evolve_on_slice = staticmethod(ga_evolve_on_slice)

    build_terminal_bank_for_index = staticmethod(build_terminal_bank_for_index)
    compute_rolling_beta_matrix_strict = staticmethod(compute_rolling_beta_matrix_strict)

    run_walkforward_ga_layer = staticmethod(run_walkforward_ga_layer)



ga_engine = GA_LAYER_WF_CONNECTED()


# ============================================================
# 12) Example usage (GA layer)
# ============================================================

spy = []
for a in client.list_aggs(
    "SPY",
    1,
    "day",
    start,
    end,
    adjusted="true",
    sort="asc",
    limit=120,
):
    spy.append(a)

print(spy)
target_index = logret.index

spy_close_price = ga_spy_aggs_to_close_series(
    spy_aggs=spy,
    target_index=target_index,
    tz="America/New_York"
)

ga_wf = run_walkforward_ga_layer(
    wf=wf,
    wf_lasso=wf_lasso,
    features=features,
    logret=logret,
    spy_close_price=spy_close_price,
    ln_mcap=None,
    industry_codes=None,
    beta_window=60,
    verbose=True
)

print(ga_wf["slice_stats"].head(10))























# ============================================================
# PREP BLOCK AS A CLASS
# EGARCH + Budget + Signal Bank + Standardization
# Fully runnable given: ga_wf / features / logret / spy_aggs
# Requires external funcs in your project: gp_eval, gp_key, ga_spy_aggs_to_close_series
# Optional: compute_rolling_beta_matrix_strict
# ============================================================


_HAS_JOBLIB = True
_HAS_NUMBA = True
_HAS_ARCH = True





# ============================================================
# Specs / Config
# ============================================================

@dataclass
class EGARCHSpec:
    p: int = 1
    o: int = 1
    q: int = 1
    dist: str = "t"
    mean: str = "Constant"
    rescale: bool = False
    H: int = 5
    sims: int = 4000
    jobs: int = 8
    min_obs: int = 252
    min_var: float = 1e-10
    sigma_cap_daily: float = 2.0
    sigma_floor_daily: float = 1e-6
    logh_min: float = np.log((1e-4) ** 2)
    seed: int = 123


@dataclass(frozen=True)
class BudgetSpecA:
    H: int = 5
    kappa: float = 1.0
    lam: float = 0.35
    eps: float = 1e-12
    sigma_cap_daily: float = 2.0
    min_eligible: int = 30
    fallback_equal: bool = True


@dataclass
class PrepConfig:
    GA_BETA_WINDOW: int = 60
    GA_BETA_MIN_PERIODS: int = 60
    GA_SPY_MIN_NONNAN_RATIO: float = 0.90

    GA_MIN_CS_N: int = 30
    GA_FILL_MISSING_EXPOSURE_WITH_ZERO: bool = False

    # 你要求“之前 shift 过了” => 默认 True，避免 double-delay
    GA_TERMINALS_ALREADY_DELAYED: bool = True

    # default horizon/delay if ga_wf config_used missing
    PRED_HORIZON: int = 5
    DELAY: int = 1

    # timezone for spy aggs alignment
    SPY_TZ: str = "America/New_York"


# ============================================================
# EGARCH simulation kernel
# ============================================================

def _E_abs_std_t(nu: float) -> float:
    nu = float(nu)
    if nu <= 2.0:
        nu = 2.05
    logE = (
        np.log(2.0)
        + 0.5 * np.log(nu - 2.0)
        + gammaln((nu + 1.0) / 2.0)
        - 0.5 * np.log(np.pi)
        - np.log(nu - 1.0)
        - gammaln(nu / 2.0)
    )
    return float(np.exp(logE))


def _std_t_draws(rng: np.random.Generator, nu: float, shape: Tuple[int, int]) -> np.ndarray:
    nu = float(nu)
    if nu <= 2.0:
        nu = 2.05
    z = rng.standard_t(df=nu, size=shape)
    z = z * np.sqrt((nu - 2.0) / nu)
    return z.astype(np.float64, copy=False)


if _HAS_NUMBA:
    @njit(cache=True)
    def _egarch_simulate_var_paths(
        z_draws: np.ndarray,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
        Eabs: float,
        logh0: float,
        z0: float,
        logh_min: float,
        logh_max: float,
    ) -> np.ndarray:
        S, H = z_draws.shape
        out = np.empty((S, H), dtype=np.float64)
        for s in range(S):
            logh_prev = logh0
            z_prev = z0
            for k in range(H):
                logh = omega + alpha * (abs(z_prev) - Eabs) + gamma * z_prev + beta * logh_prev
                if logh < logh_min:
                    logh = logh_min
                elif logh > logh_max:
                    logh = logh_max
                out[s, k] = np.exp(logh)
                z_prev = z_draws[s, k]
                logh_prev = logh
        return out
else:
    def _egarch_simulate_var_paths(
        z_draws, omega, alpha, gamma, beta, Eabs, logh0, z0, logh_min, logh_max
    ):
        S, H = z_draws.shape
        out = np.empty((S, H), dtype=np.float64)
        for s in range(S):
            logh_prev = logh0
            z_prev = z0
            for k in range(H):
                logh = omega + alpha * (abs(z_prev) - Eabs) + gamma * z_prev + beta * logh_prev
                logh = min(max(logh, logh_min), logh_max)
                out[s, k] = np.exp(logh)
                z_prev = z_draws[s, k]
                logh_prev = logh
        return out


# ============================================================
# Main Class
# ============================================================

class PrepBlockPipeline:
    """
    One-stop PREP pipeline:
      1) EGARCH walk-forward vol forecast
      2) Budget allocation (Stage A)
      3) Signal bank build from GA WF results (GPNode)
      4) CS z-score standardization

    External dependencies expected in your project:
      - gp_eval(node, terminal_bank, cache) -> ndarray (T,N)
      - gp_key(node) -> stable string key
      - ga_spy_aggs_to_close_series(spy_aggs, target_index, tz=...) -> Series aligned to target_index
      - optional compute_rolling_beta_matrix_strict(logret, mkt_ret, window, lag_for_trade)
    """

    def __init__(
        self,
        config: Optional[PrepConfig] = None,
        gp_eval_func=None,
        gp_key_func=None,
        spy_aggs_to_close_func=None,
        rolling_beta_strict_func=None,
    ):
        self.cfg = config or PrepConfig()

        # resolve external funcs (prefer explicit injection, else global)
        self.gp_eval = gp_eval_func or globals().get("gp_eval", None)
        self.gp_key  = gp_key_func  or globals().get("gp_key", None)

        self.spy_to_close = spy_aggs_to_close_func or globals().get("ga_spy_aggs_to_close_series", None)
        self.beta_strict  = rolling_beta_strict_func or globals().get("compute_rolling_beta_matrix_strict", None)

        if self.gp_eval is None:
            raise RuntimeError("Missing gp_eval. Please pass gp_eval_func=... or define gp_eval in globals.")
        if self.gp_key is None:
            raise RuntimeError("Missing gp_key. Please pass gp_key_func=... or define gp_key in globals.")
        if self.spy_to_close is None:
            raise RuntimeError("Missing ga_spy_aggs_to_close_series. Please provide spy_aggs_to_close_func or define it.")

        # storage for results (optional)
        self.egarch_out = None
        self.budget_out = None
        self.signal_bank = None

    # ========================================================
    # 0) Utilities
    # ========================================================

    @staticmethod
    def _to_dt_index(idx: pd.Index) -> pd.DatetimeIndex:
        out = pd.to_datetime(idx)
        if getattr(out, "tz", None) is not None:
            out = out.tz_convert(None)
        return pd.DatetimeIndex(out).sort_values()

    @staticmethod
    def ga_ensure_aligned(features: Dict[str, pd.DataFrame], logret: pd.DataFrame):
        if not isinstance(logret, pd.DataFrame):
            raise TypeError("logret must be a DataFrame")
        idx = pd.Index(pd.to_datetime(logret.index)).sort_values()
        cols = pd.Index(logret.columns)

        logret2 = logret.copy()
        logret2.index = pd.to_datetime(logret2.index)
        logret2 = logret2.sort_index()
        logret2 = logret2.reindex(index=idx, columns=cols)

        feats2 = {}
        for k, df in features.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"features[{k}] must be DataFrame")
            tmp = df.copy()
            tmp.index = pd.to_datetime(tmp.index)
            tmp = tmp.sort_index()
            tmp = tmp.reindex(index=idx, columns=cols)
            feats2[k] = tmp

        return feats2, logret2

    @staticmethod
    def ga_close_to_logret(close: pd.Series) -> pd.Series:
        close = close.astype(float)
        lr = np.log(close).diff()
        lr.name = "logret"
        return lr

    @staticmethod
    def ga_valid_index(idx_all: pd.Index, delay: int, horizon: int, beta_window: int = 60) -> pd.Index:
        idx = pd.DatetimeIndex(pd.to_datetime(idx_all)).sort_values()
        T = len(idx)
        d = int(delay); h = int(horizon); w = int(beta_window)
        if T == 0:
            return pd.Index(idx)
        left = max(d, d + w - 1)
        right = T - h
        if right < left:
            return pd.Index(idx[:0])
        return pd.Index(idx[left:right + 1])

    @staticmethod
    def ga_rolling_beta_matrix_legacy(r_stk_all: np.ndarray, r_mkt_all: np.ndarray, window: int, minp: int):
        r_stk_all = np.asarray(r_stk_all, dtype=np.float64)
        r_mkt_all = np.asarray(r_mkt_all, dtype=np.float64).reshape(-1)
        T, N = r_stk_all.shape
        w = int(window)
        out = np.full((T, N), np.nan, dtype=np.float64)
        if w <= 1 or T < w:
            return out

        for t in range(w - 1, T):
            mwin = r_mkt_all[t - w + 1:t + 1]
            if not np.all(np.isfinite(mwin)):
                continue
            vm = np.var(mwin, ddof=1)
            if not np.isfinite(vm) or vm <= 0:
                continue
            mm = np.mean(mwin)
            dm = mwin - mm
            for j in range(N):
                xwin = r_stk_all[t - w + 1:t + 1, j]
                if not np.all(np.isfinite(xwin)):
                    continue
                mx = np.mean(xwin)
                dx = xwin - mx
                cov = np.sum(dx * dm) / (w - 1)
                out[t, j] = cov / vm
        return out

    def ga_cs_standardize_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.astype(float)
        cnt = x.notna().sum(axis=1)
        min_cs = int(max(10, self.cfg.GA_MIN_CS_N))
        good = cnt >= min_cs
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        if not good.any():
            return out
        xg = x.loc[good]
        mu = xg.mean(axis=1)
        sd = xg.std(axis=1, ddof=1).replace(0.0, np.nan)
        out.loc[good] = xg.sub(mu, axis=0).div(sd, axis=0)
        return out

    @staticmethod
    def ga_neutralize_cs_intercept_beta_fallback_demean(S: np.ndarray, beta: np.ndarray, min_cs_n: int = 50) -> np.ndarray:
        S = np.asarray(S, dtype=np.float64)
        beta = np.asarray(beta, dtype=np.float64)
        T, N = S.shape
        out = np.full_like(S, np.nan, dtype=np.float64)

        for t in range(T):
            y = S[t]
            b = beta[t]
            m = np.isfinite(y) & np.isfinite(b)
            if int(np.sum(m)) < int(min_cs_n):
                continue
            yy = y[m]
            bb = b[m]
            X = np.column_stack([np.ones_like(bb), bb])
            XtX = X.T @ X
            XtX.flat[::3] += 1e-8
            Xty = X.T @ yy
            try:
                coef = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
            resid = yy - X @ coef
            out[t, np.where(m)[0]] = resid
        return out

    def _call_spy_to_close(self, spy_aggs, target_index):
        # support both positional and keyword signatures
        try:
            return self.spy_to_close(spy_aggs=spy_aggs, target_index=target_index, tz=self.cfg.SPY_TZ)
        except TypeError:
            return self.spy_to_close(spy_aggs, target_index, self.cfg.SPY_TZ)

    # ========================================================
    # 1) EGARCH walk-forward
    # ========================================================

    @staticmethod
    def _next_trading_days(index: pd.DatetimeIndex, start: pd.Timestamp, H: int) -> pd.DatetimeIndex:
        if start not in index:
            return pd.DatetimeIndex([])
        i0 = index.get_loc(start)
        return index[i0:i0 + H]

    def _fit_and_forecast_one_ticker(self, y_fit_pct: np.ndarray, spec: EGARCHSpec, seed: int):
        if y_fit_pct.size < spec.min_obs:
            return None, f"too_short(n={y_fit_pct.size})"
        if np.var(y_fit_pct) < spec.min_var:
            return None, "degenerate_var"

        if not _HAS_ARCH:
            x = y_fit_pct.astype(np.float64)
            lam = 0.94
            v = np.nanvar(x)
            for r in x[-spec.min_obs:]:
                if not np.isfinite(r):
                    continue
                v = lam * v + (1.0 - lam) * (r * r)
            sig = np.sqrt(max(v, 1e-12)) / 100.0
            sig = float(np.clip(sig, spec.sigma_floor_daily, spec.sigma_cap_daily))
            return (np.full(spec.H, sig, dtype=np.float64)), "fallback_ewma"

        try:
            am = arch_model(
                y_fit_pct,
                mean=spec.mean,
                vol="EGARCH",
                p=spec.p, o=spec.o, q=spec.q,
                dist=spec.dist,
                rescale=spec.rescale
            )
            res = am.fit(disp="off")
        except Exception as e:
            return None, f"fit_fail({type(e).__name__})"

        params = res.params
        try:
            omega = float(params["omega"])
            alpha = float(params["alpha[1]"])
            gamma = float(params["gamma[1]"])
            beta  = float(params["beta[1]"])
            nu    = float(params["nu"]) if "nu" in params.index else 8.0
        except Exception:
            return None, "param_missing"

        sig_last = float(res.conditional_volatility[-1])
        if (not np.isfinite(sig_last)) or (sig_last <= 0):
            return None, "bad_cond_vol"

        std_resid = res.std_resid
        z0 = float(std_resid[-1]) if (std_resid is not None and np.isfinite(std_resid[-1])) else 0.0
        logh0 = np.log(sig_last * sig_last)

        Eabs = _E_abs_std_t(nu)

        sigma_cap_pct = spec.sigma_cap_daily * 100.0
        sigma_floor_pct = spec.sigma_floor_daily * 100.0
        logh_min = float(spec.logh_min)
        logh_max = float(np.log((sigma_cap_pct ** 2)))

        rng = np.random.default_rng(seed)
        z_draws = _std_t_draws(rng, nu=nu, shape=(spec.sims, spec.H))

        h_paths = _egarch_simulate_var_paths(
            z_draws, omega, alpha, gamma, beta, Eabs, logh0, z0, logh_min, logh_max
        )

        h_mean = np.mean(h_paths, axis=0)
        if not np.all(np.isfinite(h_mean)):
            return None, "nonfinite_hmean"

        sigma_pct = np.sqrt(np.maximum(h_mean, 0.0))
        if (np.max(sigma_pct) > sigma_cap_pct) or (not np.all(np.isfinite(sigma_pct))):
            return None, "sigma_exploded"
        if np.max(sigma_pct) < sigma_floor_pct:
            return None, "sigma_degenerate"

        return (sigma_pct / 100.0).astype(np.float64, copy=False), None

    def _run_stage_egarch(self, logret: pd.DataFrame, row: pd.Series, spec: EGARCHSpec, stage_seed: int):
        fit_start = pd.to_datetime(row["train_start"])
        fit_end   = pd.to_datetime(row["train_end"])
        fcast_start = pd.to_datetime(row["holdout_start"])

        f_dates = self._next_trading_days(logret.index, fcast_start, spec.H)
        cols = logret.columns

        if len(f_dates) < spec.H:
            dv = pd.DataFrame(np.nan, index=f_dates, columns=cols)
            sigma_H = pd.Series(np.nan, index=cols, name="sigma_H")
            meta = {
                "fit_start": fit_start, "fit_end": fit_end,
                "holdout_start": fcast_start,
                "fcast_start": fcast_start, "fcast_end": (f_dates[-1] if len(f_dates) > 0 else pd.NaT),
                "eligible": int(logret.shape[1]),
                "fit_ok": 0,
                "nan_ratio_in_vol": 1.0,
                "sigmaH_min": np.nan,
                "sigmaH_max": np.nan,
            }
            return dv, sigma_H, meta

        fit_df = logret.loc[fit_start:fit_end]
        N = len(cols)

        out = np.full((spec.H, N), np.nan, dtype=np.float64)
        ok = np.zeros(N, dtype=bool)

        def _one(j: int):
            c = cols[j]
            x = fit_df[c].dropna().astype(float).values
            y = 100.0 * x
            seed = (stage_seed * 1_000_003 + j * 10_007) & 0xFFFFFFFF
            sigma_path, err = self._fit_and_forecast_one_ticker(y, spec, seed=seed)
            return j, sigma_path, err

        if _HAS_JOBLIB and spec.jobs and spec.jobs > 1:
            results = Parallel(n_jobs=spec.jobs, backend="loky")(delayed(_one)(j) for j in range(N))
        else:
            results = [_one(j) for j in range(N)]

        for j, sigma_path, err in results:
            if sigma_path is not None:
                out[:, j] = sigma_path
                ok[j] = True

        dv = pd.DataFrame(out, index=f_dates, columns=cols)

        finite = np.isfinite(out)
        col_all_nan = (~finite).all(axis=0)

        sigma_H_vals = np.sqrt(np.nansum(np.where(finite, out * out, 0.0), axis=0))
        sigma_H_vals[col_all_nan] = np.nan
        sigma_H_vals[~np.isfinite(sigma_H_vals)] = np.nan
        sigma_H = pd.Series(sigma_H_vals, index=cols, name="sigma_H")

        nan_ratio = float(np.mean(~np.isfinite(out)))
        meta = {
            "fit_start": fit_start, "fit_end": fit_end,
            "holdout_start": fcast_start,
            "fcast_start": f_dates[0], "fcast_end": f_dates[-1],
            "eligible": int(N),
            "fit_ok": int(ok.sum()),
            "nan_ratio_in_vol": nan_ratio,
            "sigmaH_min": float(np.nanmin(sigma_H_vals)) if np.any(np.isfinite(sigma_H_vals)) else np.nan,
            "sigmaH_max": float(np.nanmax(sigma_H_vals)) if np.any(np.isfinite(sigma_H_vals)) else np.nan,
        }
        return dv, sigma_H, meta

    def egarch_walkforward_forecast(self, logret: pd.DataFrame, slice_stats: pd.DataFrame, spec: Optional[EGARCHSpec] = None, verbose: bool = True):
        if spec is None:
            spec = EGARCHSpec()

        logret = logret.copy()
        logret.index = pd.to_datetime(logret.index)
        logret = logret.sort_index()

        slice_stats = slice_stats.copy()
        slice_stats.index = pd.to_datetime(slice_stats.index)
        slice_stats = slice_stats.sort_index()

        if verbose:
            print(
                f"[EGARCH] T={logret.shape[0]}, N={logret.shape[1]}, H={spec.H}, sims={spec.sims}, "
                f"jobs={spec.jobs}, joblib={_HAS_JOBLIB}, numba={_HAS_NUMBA}, arch={_HAS_ARCH}"
            )

        daily_vol_by_stage: Dict[pd.Timestamp, pd.DataFrame] = {}
        sigmaH_rows, meta_rows = [], []
        base_seed = int(spec.seed)

        for i, (reb, row) in enumerate(slice_stats.iterrows(), start=1):
            if verbose:
                ho_start = pd.to_datetime(row["holdout_start"])
                f_dates = self._next_trading_days(logret.index, ho_start, spec.H)
                f_end = f_dates[-1] if len(f_dates) > 0 else pd.NaT
                print(f"[EGARCH] {i}/{len(slice_stats)} reb={reb.date()} -> fcast {ho_start.date()}..{(f_end.date() if pd.notna(f_end) else None)}")

            stage_seed = (base_seed + 97 * i) & 0xFFFFFFFF
            dv, sigma_H, meta = self._run_stage_egarch(logret, row, spec, stage_seed=stage_seed)

            daily_vol_by_stage[reb] = dv
            sigmaH_rows.append(sigma_H.rename(reb))
            meta_rows.append({
                "fit_start": meta["fit_start"],
                "fit_end": meta["fit_end"],
                "holdout_start": meta["holdout_start"],
                "fcast_start": meta["fcast_start"],
                "fcast_end": meta["fcast_end"],
                "eligible": meta["eligible"],
                "fit_ok": meta["fit_ok"],
                "nan_ratio_in_vol": meta["nan_ratio_in_vol"],
                "sigmaH_min": meta["sigmaH_min"],
                "sigmaH_max": meta["sigmaH_max"],
            })

        sigma_H_df = pd.DataFrame(sigmaH_rows)
        sigma_H_df.index.name = "rebalance_date"
        stage_meta = pd.DataFrame(meta_rows, index=slice_stats.index)
        stage_meta.index.name = "rebalance_date"

        out = {"stage_meta": stage_meta, "daily_vol_by_stage": daily_vol_by_stage, "sigma_H": sigma_H_df}
        self.egarch_out = out
        return out

    # ========================================================
    # 2) Budget (Stage A)
    # ========================================================

    @staticmethod
    def robust_median_mad(x: np.ndarray):
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if x.size == 0:
            return np.nan, np.nan, 0
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return med, 1.4826 * mad, int(x.size)

    def risk_index_from_vol_row(self, vol_row: np.ndarray, spec: BudgetSpecA):
        x = vol_row.astype(float, copy=False)
        x = np.where(np.isfinite(x), np.clip(x, 0.0, spec.sigma_cap_daily), np.nan)
        mu, s, n = self.robust_median_mad(x)
        if (n < spec.min_eligible) or (not np.isfinite(mu)) or (mu <= 0):
            return np.nan, mu, s, n, np.nan
        cv = s / (mu + spec.eps) if np.isfinite(s) else 0.0
        return mu * (1.0 + spec.lam * cv), mu, s, n, cv

    @staticmethod
    def normalize_to_one(w: np.ndarray, eps: float = 0.0):
        s = np.nansum(w)
        if not np.isfinite(s) or s <= eps:
            return None
        return w / s

    def budget_alloc_stage_A(self, dv_stage: pd.DataFrame, spec: BudgetSpecA):
        if not isinstance(dv_stage, pd.DataFrame):
            raise TypeError("dv_stage must be a pandas DataFrame")

        idx = dv_stage.index
        T = len(idx)

        R = np.full(T, np.nan, dtype=float)
        mu = np.full(T, np.nan, dtype=float)
        s  = np.full(T, np.nan, dtype=float)
        cv = np.full(T, np.nan, dtype=float)
        n  = np.zeros(T, dtype=int)

        X = dv_stage.to_numpy(dtype=float)
        for t in range(T):
            Rt, mut, st, nt, cvt = self.risk_index_from_vol_row(X[t, :], spec)

            # BUGFIX: cv / n 不再写反
            R[t]  = Rt
            mu[t] = mut
            s[t]  = st
            n[t]  = int(nt) if np.isfinite(nt) else 0
            cv[t] = cvt

        raw = np.where(np.isfinite(R) & (R > 0), R ** (-spec.kappa), np.nan)
        frac = self.normalize_to_one(raw)

        if frac is None:
            if not spec.fallback_equal:
                raise ValueError("All days invalid; cannot allocate without fallback_equal.")
            frac = np.ones(T, dtype=float) / float(T)

        frac_sum = float(np.sum(frac))
        if (not np.isfinite(frac_sum)) or (abs(frac_sum - 1.0) > 1e-10):
            frac = frac / np.sum(frac)

        day_frac = pd.Series(frac, index=idx, name="day_frac")
        diag = pd.DataFrame(
            {"R_t": R, "mu_median": mu, "s_mad_scaled": s, "cv": cv, "eligible_n": n},
            index=idx,
        )
        return day_frac, diag

    def build_budget_dict_all_stages_A(self, daily_vol_by_stage: dict, stage_meta: pd.DataFrame = None, specA: BudgetSpecA = BudgetSpecA(), verbose: bool = True):
        budget_frac_by_stage, diag_by_stage, rows = {}, {}, []

        for reb, dv_stage in daily_vol_by_stage.items():
            day_frac, diag = self.budget_alloc_stage_A(dv_stage, specA)

            s1 = float(day_frac.sum())
            if abs(s1 - 1.0) > 1e-10:
                raise AssertionError(f"[BudgetA] Stage {reb}: day_frac sum != 1 (got {s1})")

            budget_frac_by_stage[reb] = day_frac
            diag_by_stage[reb] = diag

            row = {
                "rebalance_date": reb,
                "n_days": int(len(day_frac)),
                "sum_frac": float(day_frac.sum()),
                "min_frac": float(day_frac.min()),
                "max_frac": float(day_frac.max()),
                "mean_R": float(np.nanmean(diag["R_t"].values)) if np.isfinite(np.nanmean(diag["R_t"].values)) else np.nan,
                "min_R": float(np.nanmin(diag["R_t"].values)) if np.isfinite(np.nanmin(diag["R_t"].values)) else np.nan,
                "max_R": float(np.nanmax(diag["R_t"].values)) if np.isfinite(np.nanmax(diag["R_t"].values)) else np.nan,
                "min_eligible_n": int(np.min(diag["eligible_n"].values)) if len(diag) else 0,
            }
            if stage_meta is not None and reb in stage_meta.index:
                for c in ["fit_start","fit_end","holdout_start","fcast_start","fcast_end","eligible","fit_ok"]:
                    if c in stage_meta.columns:
                        row[c] = stage_meta.loc[reb, c]
            rows.append(row)

            if verbose:
                print(f"[BudgetA] stage={reb} days={len(day_frac)} sum={s1:.12f} min={day_frac.min():.4f} max={day_frac.max():.4f} minEligible={row['min_eligible_n']}")

        stage_summary = pd.DataFrame(rows).set_index("rebalance_date").sort_index()
        out = (budget_frac_by_stage, diag_by_stage, stage_summary)
        self.budget_out = out
        return out

    # ========================================================
    # 3) Signal bank build (from GA WF)
    # ========================================================

    @staticmethod
    def _as_norm_ts(x) -> Optional[pd.Timestamp]:
        if x is None:
            return None
        ts = pd.Timestamp(x)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.normalize()

    @staticmethod
    def _index_norm_ts(idx: pd.Index) -> pd.DatetimeIndex:
        out = pd.to_datetime(idx)
        if getattr(out, "tz", None) is not None:
            out = out.tz_convert(None)
        return pd.DatetimeIndex(out).normalize()

    def _infer_horizon_delay_from_ga_wf(self, ga_wf: dict) -> Tuple[int, int]:
        cfg = (ga_wf or {}).get("config_used", {}) or {}
        H = int(cfg.get("PRED_HORIZON", cfg.get("HORIZON", self.cfg.PRED_HORIZON)))
        D = int(cfg.get("DELAY", self.cfg.DELAY))
        return H, D

    def _build_stage_dates_from_ga_wf(self, ga_wf: dict) -> List[pd.Timestamp]:
        if ga_wf is None:
            return []
        ss = ga_wf.get("slice_stats", None)
        if isinstance(ss, pd.DataFrame) and len(ss) > 0:
            return sorted([pd.Timestamp(x) for x in ss.index])
        sr = ga_wf.get("slice_results", None)
        if isinstance(sr, list) and len(sr) > 0:
            out = []
            for r in sr:
                if hasattr(r, "rebalance_date"):
                    out.append(pd.Timestamp(r.rebalance_date))
            return sorted(list(set(out)))
        return []

    def _build_holdout_index_for_stage(self, idx_valid: pd.Index, stage_dates: List[pd.Timestamp], t0: pd.Timestamp) -> pd.Index:
        t0 = pd.Timestamp(t0)
        stage_dates_sorted = sorted([pd.Timestamp(x) for x in stage_dates])
        if t0 not in stage_dates_sorted:
            stage_dates_sorted = sorted(list(set(stage_dates_sorted + [t0])))
        i = stage_dates_sorted.index(t0)
        next_t0 = stage_dates_sorted[i + 1] if (i + 1 < len(stage_dates_sorted)) else None
        if next_t0 is None:
            return pd.Index(idx_valid[idx_valid >= t0])
        next_t0 = pd.Timestamp(next_t0)
        return pd.Index(idx_valid[(idx_valid >= t0) & (idx_valid < next_t0)])

    def _compute_terminals_std_fullsample(self, features: Dict[str, pd.DataFrame], base_factor_names: List[str], idx_all: pd.Index) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for f in base_factor_names:
            Xf = features[f].reindex(index=idx_all).astype(float)
            Xf_std = self.ga_cs_standardize_features_df(Xf)
            if self.cfg.GA_FILL_MISSING_EXPOSURE_WITH_ZERO:
                Xf_std = Xf_std.fillna(0.0)
            out[f] = Xf_std
        return out

    def _eval_formula_signal_fullsample(
        self,
        node,
        X_std_map: Dict[str, pd.DataFrame],
        base_names: List[str],
        idx_all: pd.Index,
        beta_all: Optional[np.ndarray],
        neutralize_beta: bool,
        keep_raw: bool,
    ) -> Dict[str, pd.DataFrame]:
        # terminal_bank: name -> (T,N) ndarray
        terminal_bank = {name: X_std_map[name].values.astype(np.float64) for name in base_names}
        cache = {}
        S = self.gp_eval(node, terminal_bank, cache)
        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 2:
            raise ValueError(f"gp_eval must return 2D (T,N), got shape={S.shape}")

        T, N = S.shape
        if T != len(idx_all):
            raise ValueError(f"Signal T mismatch: {T} vs len(idx_all)={len(idx_all)}")

        cols = X_std_map[base_names[0]].columns
        if len(cols) != N:
            raise ValueError(f"Signal N mismatch: {N} vs columns={len(cols)}")

        S_raw_df = pd.DataFrame(S, index=idx_all, columns=cols)

        if neutralize_beta:
            if beta_all is None:
                raise ValueError("neutralize_beta=True but beta_all is None.")
            beta_all = np.asarray(beta_all, dtype=np.float64)
            if beta_all.shape != S.shape:
                raise ValueError(f"beta_all shape {beta_all.shape} must match signal shape {S.shape}")

            S_neut = self.ga_neutralize_cs_intercept_beta_fallback_demean(S, beta_all, min_cs_n=int(self.cfg.GA_MIN_CS_N))
            out = {"signal": pd.DataFrame(S_neut, index=idx_all, columns=cols)}
            if keep_raw:
                out["raw"] = S_raw_df
            return out

        out = {"signal": S_raw_df}
        if keep_raw:
            out["raw"] = S_raw_df
        return out

    def build_ga_signal_bank(
        self,
        ga_wf: dict,
        features: Dict[str, pd.DataFrame],
        logret: pd.DataFrame,
        spy_aggs: list,
        neutralize_beta: bool = True,
        keep_raw: bool = False,
        verbose: bool = True,
        build_live: bool = True,
    ) -> dict:
        features_aligned, logret_aligned = self.ga_ensure_aligned(features, logret)
        idx_all = logret_aligned.index

        horizon, delay = self._infer_horizon_delay_from_ga_wf(ga_wf)

        idx_valid = self.ga_valid_index(
            idx_all,
            delay=int(delay),
            horizon=int(horizon),
            beta_window=int(self.cfg.GA_BETA_WINDOW)
        )

        stage_dates = self._build_stage_dates_from_ga_wf(ga_wf)
        stage_dates = [pd.Timestamp(d) for d in stage_dates if pd.Timestamp(d) in idx_valid]
        if len(stage_dates) == 0:
            raise ValueError("No stage dates found in valid index. Check ga_wf alignment.")

        # ---- beta_all ----
        beta_all = None
        if neutralize_beta:
            spy_close = self._call_spy_to_close(spy_aggs, idx_all)
            spy_lr = self.ga_close_to_logret(spy_close).reindex(idx_all).astype(float)

            cov = float(np.isfinite(spy_lr.values).mean())
            if cov < self.cfg.GA_SPY_MIN_NONNAN_RATIO:
                raise ValueError(f"SPY mkt_ret non-null ratio={cov:.2f} < {self.cfg.GA_SPY_MIN_NONNAN_RATIO}.")

            # try strict first
            if self.beta_strict is not None:
                try:
                    beta_all = self.beta_strict(
                        logret=logret_aligned,
                        mkt_ret=spy_lr,
                        window=int(self.cfg.GA_BETA_WINDOW),
                        lag_for_trade=int(delay)
                    )
                except Exception:
                    beta_all = None

            # fallback legacy + align lag_for_trade=delay
            if beta_all is None:
                beta_raw = self.ga_rolling_beta_matrix_legacy(
                    logret_aligned.values.astype(np.float64),
                    spy_lr.values.astype(np.float64),
                    window=int(self.cfg.GA_BETA_WINDOW),
                    minp=int(self.cfg.GA_BETA_MIN_PERIODS),
                )
                if int(delay) > 0:
                    d = int(delay)
                    beta_all = np.full_like(beta_raw, np.nan, dtype=np.float64)
                    beta_all[d:, :] = beta_raw[:-d, :]
                else:
                    beta_all = beta_raw

        # ---- slice_results mapping ----
        slice_results = ga_wf.get("slice_results", [])
        sr_by_date = {}
        for r in slice_results:
            if hasattr(r, "rebalance_date"):
                sr_by_date[pd.Timestamp(r.rebalance_date)] = r

        by_slice: Dict[pd.Timestamp, dict] = {}

        for t0 in stage_dates:
            t0 = pd.Timestamp(t0)
            holdout_idx = self._build_holdout_index_for_stage(idx_valid, stage_dates, t0)
            if len(holdout_idx) == 0:
                continue

            sr = sr_by_date.get(t0, None)
            if sr is None:
                if verbose:
                    print(f"[signal_bank][skip] {t0.date()} missing ga_wf['slice_results'] entry")
                continue

            terminals = list(getattr(sr, "terminals", []))
            terminals = [x for x in terminals if x in features_aligned]
            if len(terminals) == 0:
                if verbose:
                    print(f"[signal_bank][skip] {t0.date()} terminals empty after filtering.")
                continue

            # node_map: gp_key(node) -> node
            node_map = {}
            for n in (getattr(sr, "top10_nodes", []) or []):
                node_map[str(self.gp_key(n))] = n
            if hasattr(sr, "best_node") and sr.best_node is not None:
                node_map.setdefault(str(self.gp_key(sr.best_node)), sr.best_node)

            formulas = []
            if hasattr(sr, "top10_table") and isinstance(sr.top10_table, pd.DataFrame) and ("formula" in sr.top10_table.columns):
                tmp = [str(x) for x in sr.top10_table["formula"].tolist()]
                formulas = [f for f in tmp if f in node_map]
            if len(formulas) == 0:
                formulas = list(node_map.keys())

            if len(formulas) == 0:
                if verbose:
                    print(f"[signal_bank][skip] {t0.date()} formulas empty.")
                continue

            # terminals standardized
            X_std_map = self._compute_terminals_std_fullsample(features_aligned, terminals, idx_all)

            # shift policy: avoid double-delay by default
            if (not bool(self.cfg.GA_TERMINALS_ALREADY_DELAYED)) and int(delay) != 0:
                d = int(delay)
                for name in terminals:
                    X_std_map[name] = X_std_map[name].shift(d)

            sig_map: Dict[str, pd.DataFrame] = {}
            raw_map: Optional[Dict[str, pd.DataFrame]] = {} if keep_raw else None

            for fstr in formulas:
                node = node_map.get(str(fstr), None)
                if node is None:
                    continue

                res = self._eval_formula_signal_fullsample(
                    node=node,
                    X_std_map=X_std_map,
                    base_names=terminals,
                    idx_all=idx_all,
                    beta_all=beta_all,
                    neutralize_beta=bool(neutralize_beta),
                    keep_raw=bool(keep_raw),
                )

                sig_map[str(fstr)] = res["signal"].reindex(holdout_idx)
                if keep_raw and raw_map is not None and ("raw" in res):
                    raw_map[str(fstr)] = res["raw"].reindex(holdout_idx)

            if len(sig_map) == 0:
                if verbose:
                    print(f"[signal_bank][skip] {t0.date()} no signals built.")
                continue

            comp_df = pd.concat(sig_map.values(), axis=0, keys=list(sig_map.keys())).groupby(level=1).mean()

            pack = {
                "holdout_index": pd.Index(holdout_idx),
                "terminals": list(terminals),
                "signals": sig_map,
                "composite": comp_df,
            }
            if keep_raw:
                pack["raw_signals"] = raw_map
                if raw_map is not None and len(raw_map) > 0:
                    pack["composite_raw"] = pd.concat(raw_map.values(), axis=0, keys=list(raw_map.keys())).groupby(level=1).mean()
                else:
                    pack["composite_raw"] = None

            by_slice[t0] = pack

            if verbose:
                print(f"[signal_bank] {t0.date()} holdout={holdout_idx[0].date()}..{holdout_idx[-1].date()} | terminals={len(terminals)} | formulas={len(sig_map)}")

        # ---- live build ----
        live = None
        if build_live and len(by_slice) > 0:
            t_last = sorted(by_slice.keys())[-1]
            terminals = by_slice[t_last]["terminals"]
            formulas = list(by_slice[t_last]["signals"].keys())
            idx_live = pd.Index(idx_valid[idx_valid >= pd.Timestamp(t_last)])

            sr = sr_by_date.get(pd.Timestamp(t_last), None)
            node_map = {}
            if sr is not None:
                for n in (getattr(sr, "top10_nodes", []) or []):
                    node_map[str(self.gp_key(n))] = n
                if hasattr(sr, "best_node") and sr.best_node is not None:
                    node_map.setdefault(str(self.gp_key(sr.best_node)), sr.best_node)

            X_std_map = self._compute_terminals_std_fullsample(features_aligned, terminals, idx_all)
            if (not bool(self.cfg.GA_TERMINALS_ALREADY_DELAYED)) and int(delay) != 0:
                d = int(delay)
                for name in terminals:
                    X_std_map[name] = X_std_map[name].shift(d)

            sig_map: Dict[str, pd.DataFrame] = {}
            raw_map: Optional[Dict[str, pd.DataFrame]] = {} if keep_raw else None

            for fstr in formulas:
                node = node_map.get(str(fstr), None)
                if node is None:
                    continue
                res = self._eval_formula_signal_fullsample(
                    node=node,
                    X_std_map=X_std_map,
                    base_names=terminals,
                    idx_all=idx_all,
                    beta_all=beta_all,
                    neutralize_beta=bool(neutralize_beta),
                    keep_raw=bool(keep_raw),
                )
                sig_map[str(fstr)] = res["signal"].reindex(idx_live)
                if keep_raw and raw_map is not None and ("raw" in res):
                    raw_map[str(fstr)] = res["raw"].reindex(idx_live)

            if len(sig_map) > 0:
                comp_df = pd.concat(sig_map.values(), axis=0, keys=list(sig_map.keys())).groupby(level=1).mean()
                live = {
                    "active_stage": pd.Timestamp(t_last),
                    "index": pd.Index(idx_live),
                    "terminals": list(terminals),
                    "signals": sig_map,
                    "composite": comp_df,
                }
                if keep_raw:
                    live["raw_signals"] = raw_map
                    if raw_map is not None and len(raw_map) > 0:
                        live["composite_raw"] = pd.concat(raw_map.values(), axis=0, keys=list(raw_map.keys())).groupby(level=1).mean()
                    else:
                        live["composite_raw"] = None

        # ---- helpers (closures) ----
        def get_active_stage(dt: Any) -> Optional[pd.Timestamp]:
            dt = self._as_norm_ts(dt)
            keys = sorted([self._as_norm_ts(k) for k in by_slice.keys()])
            if len(keys) == 0:
                return None
            ks = [k for k in keys if k <= dt]
            return ks[-1] if ks else None

        def get_active_slice(dt: Any) -> Optional[dict]:
            t0 = get_active_stage(dt)
            return None if t0 is None else by_slice.get(pd.Timestamp(t0), None)

        def get_active_composite(dt: Any, prefer_live: bool = True) -> Optional[pd.Series]:
            dt2 = self._as_norm_ts(dt)
            if prefer_live and live is not None:
                df2 = live["composite"].copy()
                df2.index = self._index_norm_ts(df2.index)
                if dt2 in df2.index:
                    return df2.loc[dt2]
            sl = get_active_slice(dt2)
            if sl is None:
                return None
            df2 = sl["composite"].copy()
            df2.index = self._index_norm_ts(df2.index)
            if dt2 not in df2.index:
                return None
            return df2.loc[dt2]

        def get_active_formula_row(dt: Any, formula_str: str, prefer_live: bool = True) -> Optional[pd.Series]:
            dt2 = self._as_norm_ts(dt)
            formula_str = str(formula_str)
            if prefer_live and live is not None and formula_str in live["signals"]:
                df2 = live["signals"][formula_str].copy()
                df2.index = self._index_norm_ts(df2.index)
                if dt2 in df2.index:
                    return df2.loc[dt2]
            sl = get_active_slice(dt2)
            if sl is None or formula_str not in sl["signals"]:
                return None
            df2 = sl["signals"][formula_str].copy()
            df2.index = self._index_norm_ts(df2.index)
            if dt2 not in df2.index:
                return None
            return df2.loc[dt2]

        meta = {
            "horizon": int(horizon),
            "delay": int(delay),
            "beta_window": int(self.cfg.GA_BETA_WINDOW),
            "beta_min_periods": int(self.cfg.GA_BETA_MIN_PERIODS),
            "neutralize_beta": bool(neutralize_beta),
            "keep_raw": bool(keep_raw),
            "terminals_already_delayed": bool(self.cfg.GA_TERMINALS_ALREADY_DELAYED),
            "idx_all_start": pd.Timestamp(idx_all[0]),
            "idx_all_end": pd.Timestamp(idx_all[-1]),
            "idx_valid_start": pd.Timestamp(idx_valid[0]) if len(idx_valid) else pd.NaT,
            "idx_valid_end": pd.Timestamp(idx_valid[-1]) if len(idx_valid) else pd.NaT,
        }

        out = {
            "by_slice": by_slice,
            "live": live,
            "helpers": {
                "get_active_stage": get_active_stage,
                "get_active_slice": get_active_slice,
                "get_active_composite": get_active_composite,
                "get_active_formula_row": get_active_formula_row,
            },
            "meta": meta,
        }
        self.signal_bank = out
        return out

    # ========================================================
    # 4) Standardization
    # ========================================================

    @staticmethod
    def cs_zscore_pure(df: pd.DataFrame, min_cs_n: int = 30, ddof: int = 1) -> pd.DataFrame:
        x = df.astype(float)
        cnt = x.notna().sum(axis=1)
        good = cnt >= int(min_cs_n)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        if not good.any():
            return out
        xg = x.loc[good]
        mu = xg.mean(axis=1)
        sd = xg.std(axis=1, ddof=ddof).replace(0.0, np.nan)
        out.loc[good] = xg.sub(mu, axis=0).div(sd, axis=0)
        return out

    def _standardize_one_block_pure_z(self, block: dict, min_cs_n: int, ddof: int) -> None:
        if not isinstance(block, dict):
            return
        sigs = block.get("signals", None)
        if not isinstance(sigs, dict) or len(sigs) == 0:
            block["composite"] = None
            return

        for fml, df in list(sigs.items()):
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                sigs[fml] = self.cs_zscore_pure(df, min_cs_n=min_cs_n, ddof=ddof)

        dfs = [df for df in sigs.values() if isinstance(df, pd.DataFrame) and len(df) > 0]
        if len(dfs) == 0:
            block["composite"] = None
            return

        idx = dfs[0].index
        cols = dfs[0].columns
        for df in dfs[1:]:
            idx = idx.intersection(df.index)
            cols = cols.intersection(df.columns)
        if len(idx) == 0 or len(cols) == 0:
            block["composite"] = None
            return

        acc, k = None, 0
        for df in dfs:
            x = df.reindex(index=idx, columns=cols).astype(float)
            acc = x.copy() if acc is None else acc.add(x, fill_value=np.nan)
            k += 1
        block["composite"] = acc / float(k) if (acc is not None and k > 0) else None

    @staticmethod
    def _is_stage_mapping(d: dict) -> bool:
        if not isinstance(d, dict) or len(d) == 0:
            return False
        for v in d.values():
            if isinstance(v, dict) and ("signals" in v):
                return True
        return False

    def zscore_signal_bank_inplace(self, signal_bank: dict, min_cs_n: int = 30, ddof: int = 1, verbose: bool = True) -> dict:
        if not isinstance(signal_bank, dict):
            raise ValueError("signal_bank must be a dict.")

        by_slice = signal_bank.get("by_slice", {})
        if isinstance(by_slice, dict):
            for t0, block in by_slice.items():
                self._standardize_one_block_pure_z(block, min_cs_n=min_cs_n, ddof=ddof)
                if verbose:
                    comp = block.get("composite", None) if isinstance(block, dict) else None
                    comp_ok = isinstance(comp, pd.DataFrame) and len(comp) > 0
                    print(f"[z_bank] by_slice {pd.Timestamp(t0).date()} | zscore | composite={'OK' if comp_ok else 'None'}")

        live = signal_bank.get("live", None)
        live_is_mapping = self._is_stage_mapping(live) if isinstance(live, dict) else False

        if isinstance(live, dict) and ("signals" in live) and (not live_is_mapping):
            self._standardize_one_block_pure_z(live, min_cs_n=min_cs_n, ddof=ddof)
            if verbose:
                t0_live = live.get("active_stage", "live")
                try:
                    t0_live = pd.Timestamp(t0_live).date()
                except Exception:
                    pass
                comp = live.get("composite", None)
                comp_ok = isinstance(comp, pd.DataFrame) and len(comp) > 0
                print(f"[z_bank] live(single) {t0_live} | zscore | composite={'OK' if comp_ok else 'None'}")

        elif live_is_mapping:
            for t0, block in live.items():
                if not (isinstance(block, dict) and ("signals" in block)):
                    continue
                self._standardize_one_block_pure_z(block, min_cs_n=min_cs_n, ddof=ddof)
                if verbose:
                    comp = block.get("composite", None)
                    comp_ok = isinstance(comp, pd.DataFrame) and len(comp) > 0
                    print(f"[z_bank] live(map) {pd.Timestamp(t0).date()} | zscore | composite={'OK' if comp_ok else 'None'}")

        return signal_bank

    # ========================================================
    # Convenience: run all
    # ========================================================

    def run_all(
        self,
        ga_wf: dict,
        features: Dict[str, pd.DataFrame],
        logret: pd.DataFrame,
        spy_aggs: list,
        egarch_spec: Optional[EGARCHSpec] = None,
        budget_spec: Optional[BudgetSpecA] = None,
        verbose: bool = True,
    ):
        egarch_spec = egarch_spec or EGARCHSpec(H=self.cfg.PRED_HORIZON, sims=4000, jobs=8, seed=123, sigma_cap_daily=2.0, min_obs=252)
        budget_spec = budget_spec or BudgetSpecA(H=self.cfg.PRED_HORIZON, kappa=1.0, lam=0.35, sigma_cap_daily=2.0, min_eligible=30, fallback_equal=True)

        out1 = self.egarch_walkforward_forecast(logret=logret, slice_stats=ga_wf["slice_stats"], spec=egarch_spec, verbose=verbose)
        stage_meta = out1["stage_meta"]
        daily_vol_by_stage = out1["daily_vol_by_stage"]

        out2 = self.build_budget_dict_all_stages_A(daily_vol_by_stage=daily_vol_by_stage, stage_meta=stage_meta, specA=budget_spec, verbose=verbose)

        out3 = self.build_ga_signal_bank(
            ga_wf=ga_wf, features=features, logret=logret, spy_aggs=spy_aggs,
            neutralize_beta=True, keep_raw=False, verbose=verbose, build_live=True
        )

        out3 = self.zscore_signal_bank_inplace(out3, min_cs_n=self.cfg.GA_MIN_CS_N, ddof=1, verbose=verbose)

        return {
            "egarch": out1,
            "budget": {
                "budget_frac_by_stage": out2[0],
                "budget_diag_by_stage": out2[1],
                "budget_stage_summary": out2[2],
            },
            "signal_bank": out3,
        }


# ============================================================
# Example usage 
# ============================================================

# 1) init pipeline
pipe = PrepBlockPipeline(
    config=PrepConfig(
        GA_BETA_WINDOW=60,
        GA_BETA_MIN_PERIODS=60,
        GA_SPY_MIN_NONNAN_RATIO=0.90,
        GA_MIN_CS_N=30,
        GA_FILL_MISSING_EXPOSURE_WITH_ZERO=False,
        GA_TERMINALS_ALREADY_DELAYED=True,   # 你要求：之前shift过了
        PRED_HORIZON=5,
        DELAY=1,
        SPY_TZ="America/New_York",
    ),
    # 如果你不传，类会自动从 globals() 找 gp_eval/gp_key/ga_spy_aggs_to_close_series
    # gp_eval_func=gp_eval,
    # gp_key_func=gp_key,
    # spy_aggs_to_close_func=ga_spy_aggs_to_close_series,
    # rolling_beta_strict_func=compute_rolling_beta_matrix_strict,
)

# 2) run egarch
spec = EGARCHSpec(H=5, sims=4000, jobs=8, seed=123, sigma_cap_daily=2.0, min_obs=252)
out = pipe.egarch_walkforward_forecast(logret=logret, slice_stats=ga_wf["slice_stats"], spec=spec, verbose=True)

stage_meta = out["stage_meta"]
daily_vol_by_stage = out["daily_vol_by_stage"]
sigma_H = out["sigma_H"]

active_reb = stage_meta.index[-1]
print("active stage:", active_reb)
print(stage_meta.loc[active_reb])

dv_active = daily_vol_by_stage[active_reb]
print(dv_active[["AAPL","MSFT","NVDA","CRM"]])
print("sigma_H (reb):")
print(sigma_H.loc[active_reb, ["AAPL","MSFT","NVDA","CRM"]])

# 3) budget
specA = BudgetSpecA(H=5, kappa=1.0, lam=0.35, sigma_cap_daily=2.0, min_eligible=30, fallback_equal=True)
budget_frac_by_stage, budget_diag_by_stage, budget_stage_summary = pipe.build_budget_dict_all_stages_A(
    daily_vol_by_stage=daily_vol_by_stage,
    stage_meta=stage_meta,
    specA=specA,
    verbose=True
)

active_reb = stage_meta.index[-1]
print("\nActive stage:", active_reb)
print("Daily budget fractions (must sum to 1):")
print(budget_frac_by_stage[active_reb])
print("sum =", budget_frac_by_stage[active_reb].sum())

# 4) signal bank + zscore
signal_bank = pipe.build_ga_signal_bank(
    ga_wf=ga_wf,
    features=features,
    logret=logret,
    spy_aggs=spy,
    neutralize_beta=True,
    keep_raw=False,
    verbose=True,
    build_live=True
)
signal_bank = pipe.zscore_signal_bank_inplace(signal_bank, min_cs_n=30, ddof=1, verbose=True)
print("\n[OK] signal_bank built + standardized (CS zscore).")

t0_first = sorted(signal_bank["by_slice"].keys())[0]
first_formula = list(signal_bank["by_slice"][t0_first]["signals"].keys())[0]
print("First slice:", pd.Timestamp(t0_first).date())
print("First formula:", first_formula)
print(signal_bank["by_slice"][t0_first]["signals"][first_formula].iloc[:5, :5])

dt_latest = pd.Timestamp(logret.index.max())
t_active = signal_bank["helpers"]["get_active_stage"](dt_latest)
row = signal_bank["helpers"]["get_active_composite"](dt_latest, prefer_live=True)
print("\nLatest date:", dt_latest, "active_stage:", None if t_active is None else t_active.date())
print(row.iloc[:10] if row is not None else "No composite row found.")
















''' BackTesting Framework '''

# ============================================================
# 0) Inputs (keep as-is)
# ============================================================

target_index = logret.index   # e.g., close.index / idx_valid / the index you want to align to

# 3) Use your existing function to get aligned SPY close (index matches target_index exactly)
spy_close_price = ga_spy_aggs_to_close_series(
    spy_aggs=spy,
    target_index=target_index,
    tz="America/New_York"
)


# ============================================================
# FULL CODE (PARAMS MOVED TO TOP; ONLY PARAM CHANGES)
# ============================================================




# ============================================================
# 0) PARAMS (EDIT HERE ONLY)
#   - Tuned for: ~100-stock US small universe + 5D GA factor
#   - More realistic frictions + more robust against overfit
# ============================================================

CFG = {
    # ---------- Engine / Portfolio ----------
    "engine": {
        "init_cash": 1e8,
        "hold_days": 5,           # locked by implementation
        "n_pick": 15,             # more diversified for ~100-stock universe
        "tau_softmax": 1.15,      # less extreme concentration / more robust
        "fee_per_share": 0.001,   # modern fee scale (impact should dominate)
        "impact_bps": 0.0012,     # 12 bps for small universe + daily trading + softmax
    },

    # ---------- Sample Space ----------
    "sample_space": {
        "amo_window": 20,             # smoother liquidity filter
        "amo_threshold": 2e7,         # ~$20m/day dollar volume
        "min_listed_days": 120,       # avoid IPO/new listing noise
        "require_price_positive": True,
    },

    # ---------- Backtest wrapper ----------
    "wrapper": {
        "strict_tminus1": True,          # keep: no look-ahead
        "liquidation_buffer_days": 15,   # more buffer for maturity + sell fail
        "min_cs_n": 50,                  # stricter gate for ~100-stock universe
        "start": None,                   # auto infer
        "end": None,                     # auto infer
    },

    # ---------- Reporting ----------
    "reporting": {
        "trading_days": 252,
    },
}


# ============================================================
# 0) Utilities
# ============================================================

def _to_datetime_index(df_or_ser):
    obj = df_or_ser.copy()
    obj.index = pd.to_datetime(obj.index)
    return obj.sort_index()

def _annualize_from_nav(nav: pd.Series, trading_days=252):
    r = nav.pct_change().fillna(0.0)
    if len(nav) < 3:
        return dict(ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, max_dd=np.nan)
    max_dd = float((nav / nav.cummax() - 1.0).min())
    ann_vol = float(r.std(ddof=1) * np.sqrt(trading_days)) if len(nav) > 10 else np.nan
    years = len(nav) / trading_days
    ann_ret = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    sharpe = ann_ret / ann_vol if (np.isfinite(ann_vol) and ann_vol > 0) else np.nan
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd)

def _roll_h_stats(nav: pd.Series, H=5):
    if len(nav) <= H:
        return dict(mean_h=np.nan, vol_h=np.nan, var_h=np.nan, n_h=0)
    rh = nav / nav.shift(H) - 1.0
    rh = rh.dropna()
    if len(rh) == 0:
        return dict(mean_h=np.nan, vol_h=np.nan, var_h=np.nan, n_h=0)
    return dict(
        mean_h=float(rh.mean()),
        vol_h=float(rh.std(ddof=1)) if len(rh) > 1 else 0.0,
        var_h=float(rh.var(ddof=1)) if len(rh) > 1 else 0.0,
        n_h=int(len(rh)),
    )

def _max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())

def _infer_stage_blocks(signal_bank: dict):
    stages = []
    for t0, blk in signal_bank.get("by_slice", {}).items():
        comp = blk.get("composite", None)
        if comp is None or (not isinstance(comp, pd.DataFrame)) or comp.empty:
            continue

        meta = blk.get("meta", {}) if isinstance(blk, dict) else {}
        hs = meta.get("holdout_start", None)
        he = meta.get("holdout_end", None)
        ts = meta.get("train_start", None)
        te = meta.get("train_end", None)
        sel = meta.get("selected_factors", None)

        comp = _to_datetime_index(comp)
        if hs is None:
            hs = comp.index.min()
        if he is None:
            he = comp.index.max()

        stages.append(dict(
            stage_start=pd.Timestamp(t0),
            train_start=pd.Timestamp(ts) if ts is not None else None,
            train_end=pd.Timestamp(te) if te is not None else None,
            holdout_start=pd.Timestamp(hs),
            holdout_end=pd.Timestamp(he),
            selected_factors=sel,
        ))
    stages = sorted(stages, key=lambda d: d["stage_start"])
    return stages

def _infer_global_backtest_window_from_signal_bank(
    signal_bank: dict,
    close_index: pd.Index,
    hold_days: int,
    liquidation_buffer_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    stages = _infer_stage_blocks(signal_bank)
    if len(stages) == 0:
        raise ValueError("Cannot infer window: signal_bank has no usable stages.")

    hs_list = [pd.Timestamp(s["holdout_start"]) for s in stages if s.get("holdout_start") is not None]
    he_list = [pd.Timestamp(s["holdout_end"]) for s in stages if s.get("holdout_end") is not None]
    if len(hs_list) == 0 or len(he_list) == 0:
        raise ValueError("Cannot infer window: stages missing holdout_start/holdout_end.")

    earliest_hs = min(hs_list)
    latest_he = max(he_list)

    close_index = pd.to_datetime(pd.Index(close_index)).sort_values()
    if len(close_index) == 0:
        raise ValueError("close_df.index is empty.")

    global_start = earliest_hs
    if global_start < close_index.min():
        global_start = close_index.min()
    if global_start > close_index.max():
        raise ValueError("Inferred global_start is after last available close date.")

    need_extra = int(hold_days) + max(int(liquidation_buffer_days), 0)
    after = close_index[close_index > latest_he]
    if len(after) > 0 and need_extra > 0:
        extra = after[:need_extra]
        global_end = extra.max()
    else:
        global_end = latest_he

    if global_end > close_index.max():
        global_end = close_index.max()
    if global_end < global_start:
        raise ValueError(f"Inferred global_end({global_end}) < global_start({global_start}).")

    return pd.Timestamp(global_start), pd.Timestamp(global_end), pd.Timestamp(earliest_hs), pd.Timestamp(latest_he)

def _pick_book_frac(budget_frac_by_stage, stage_start, hold_days=5):
    if budget_frac_by_stage is None:
        return np.full(hold_days, 1.0 / hold_days, dtype=np.float64)

    key = pd.Timestamp(stage_start)
    v = None
    if isinstance(budget_frac_by_stage, dict):
        if key in budget_frac_by_stage:
            v = budget_frac_by_stage[key]
        elif stage_start in budget_frac_by_stage:
            v = budget_frac_by_stage[stage_start]

    if v is None:
        return np.full(hold_days, 1.0 / hold_days, dtype=np.float64)

    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.shape[0] != hold_days:
        raise ValueError(f"budget_frac_by_stage[{stage_start}] length != hold_days({hold_days})")
    s = float(np.nansum(arr))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("book_frac sum invalid")
    return (arr / s).astype(np.float64)

def _align_close_volume(close_df: pd.DataFrame,
                        volume_df: pd.DataFrame | None = None,
                        amount_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    close_df = _to_datetime_index(close_df).astype(float)
    if volume_df is not None:
        volume_df = _to_datetime_index(volume_df).reindex_like(close_df).astype(float)
    if amount_df is not None:
        amount_df = _to_datetime_index(amount_df).reindex_like(close_df).astype(float)
    return close_df, volume_df, amount_df


# ============================================================
# 1) Sample Space builder (AMO + listing days) -> 0/1 matrix
# ============================================================

def build_sample_space_amo_listing(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame | None = None,
    amount_df: pd.DataFrame | None = None,
    amo_window: int = 10,
    amo_threshold: float = 1e7,
    min_listed_days: int = 60,
    require_price_positive: bool = True,
    delay: int = 1,   # industry default = 1 (t-1 known)
) -> pd.DataFrame:
    """
    Industry-grade sample space:
    - Use ONLY information available as of t-delay.
    - Avoid look-ahead by shifting ALL eligibility signals by `delay`.
    """
    close_df = _to_datetime_index(close_df).astype(float)

    if amount_df is None:
        if volume_df is None:
            raise ValueError("Need either amount_df or volume_df to build AMO sample space.")
        volume_df = _to_datetime_index(volume_df).reindex_like(close_df).astype(float)
        amo = close_df * volume_df
    else:
        amo = _to_datetime_index(amount_df).reindex_like(close_df).astype(float)

    amo_bar = amo.rolling(amo_window, min_periods=amo_window).mean()
    if delay is not None and int(delay) > 0:
        amo_bar = amo_bar.shift(int(delay))

    listed_days = close_df.notna().cumsum(axis=0)
    if delay is not None and int(delay) > 0:
        listed_days = listed_days.shift(int(delay))

    if require_price_positive:
        px_ok = (close_df > 0)
    else:
        px_ok = close_df.notna()
    if delay is not None and int(delay) > 0:
        px_ok = px_ok.shift(int(delay))

    cond_amo = amo_bar > amo_threshold
    cond_list = listed_days >= min_listed_days
    cond_px = px_ok

    sample = (cond_amo & cond_list & cond_px).astype(np.uint8)

    # ============================================================
    # [MOD] Reduce subtle look-ahead:
    # Do not use same-day close.notna() to decide eligibility; use as-of(t-delay).
    # Same-day "no price / cannot trade" is handled by ORD_NO_PRICE and pending sells.
    # ============================================================
    if delay is not None and int(delay) > 0:
        known_close = close_df.shift(int(delay))
        sample = sample.where(known_close.notna(), 0)
    else:
        sample = sample.where(close_df.notna(), 0)

    sample = sample.fillna(0).astype(np.uint8)
    return sample


# ============================================================
# 2) Numba: cross-sectional zscore + softmax topN weights
# ============================================================

@nb.njit(cache=True, fastmath=True)
def _count_eligible_finite(x, eligible):
    N = x.shape[0]
    n = 0
    for i in range(N):
        if eligible[i] == 1 and math.isfinite(x[i]):
            n += 1
    return n

@nb.njit(cache=True, fastmath=True)
def _cs_zscore_one_day(x, eligible, min_cs_n):
    N = x.shape[0]
    n = 0
    s = 0.0
    for i in range(N):
        if eligible[i] == 1 and math.isfinite(x[i]):
            s += x[i]
            n += 1
    z = np.zeros(N, dtype=np.float64)
    if n < min_cs_n:
        return z
    mu = s / n

    ss = 0.0
    for i in range(N):
        if eligible[i] == 1 and math.isfinite(x[i]):
            d = x[i] - mu
            ss += d * d
    if n <= 1:
        return z
    var = ss / (n - 1)
    if var <= 0.0 or (not math.isfinite(var)):
        return z
    std = math.sqrt(var)
    for i in range(N):
        if eligible[i] == 1 and math.isfinite(x[i]):
            z[i] = (x[i] - mu) / std
    return z

@nb.njit(cache=True, fastmath=True)
def _softmax_topn_weights(scores, eligible, n_pick, tau):
    N = scores.shape[0]
    idx = np.empty(N, dtype=np.int64)
    val = np.empty(N, dtype=np.float64)
    cnt = 0
    for i in range(N):
        if eligible[i] == 1 and math.isfinite(scores[i]):
            idx[cnt] = i
            val[cnt] = scores[i]
            cnt += 1

    w = np.zeros(N, dtype=np.float64)
    if cnt == 0:
        return w

    k = n_pick
    if k > cnt:
        k = cnt

    for a in range(k):
        best = a
        for b in range(a + 1, cnt):
            if val[b] > val[best]:
                best = b
        if best != a:
            tmpi = idx[a]; idx[a] = idx[best]; idx[best] = tmpi
            tmpv = val[a]; val[a] = val[best]; val[best] = tmpv

    mx = val[0]
    for j in range(1, k):
        if val[j] > mx:
            mx = val[j]

    denom = 0.0
    ex = np.empty(k, dtype=np.float64)
    for j in range(k):
        e = math.exp(tau * (val[j] - mx))
        ex[j] = e
        denom += e
    if denom <= 0.0:
        return w

    for j in range(k):
        w[idx[j]] = ex[j] / denom
    return w

@nb.njit(cache=True, fastmath=True)
def _build_w_trade_stage(
    comp,
    sample_day,
    entry_mask,
    n_pick, tau, min_cs_n,
):
    T, N = comp.shape
    W_trade = np.zeros((T, N), dtype=np.float64)
    allow_entry = np.zeros(T, dtype=np.uint8)

    for t in range(T):
        if entry_mask[t] != 1:
            continue
        elig = sample_day[t]
        x = comp[t]

        # ============================================================
        # [MOD] Prevent "random trading under low sample":
        # If eligible & finite cross-section count < min_cs_n, skip entry on this day.
        # ============================================================
        n_cs = _count_eligible_finite(x, elig)
        if n_cs < min_cs_n:
            continue

        z = _cs_zscore_one_day(x, elig, min_cs_n)
        w = _softmax_topn_weights(z, elig, n_pick, tau)
        W_trade[t] = w
        allow_entry[t] = 1

    return W_trade, allow_entry


# ============================================================
# 3) LOT-level 5D staggered engine (Numba) + LOT lifecycle capture
# ============================================================

ORD_OK          = 0
ORD_INELIGIBLE  = 1
ORD_NO_PRICE    = 2
ORD_TOO_SMALL   = 3
ORD_NO_CASH     = 4

LOT_EXIT_NORMAL = 0
LOT_EXIT_DELAY  = 1

@nb.njit(cache=True, fastmath=True)
def _bt_stage_5d_lots(
    price,
    sample_day,
    W_trade,
    allow_entry,
    book_frac,
    hold_days,
    init_cash,
    fee_per_share,
    impact_bps,
    n_pick,
):
    T, N = price.shape
    price_ok = np.zeros((T, N), dtype=np.uint8)
    for t in range(T):
        for i in range(N):
            price_ok[t, i] = 0 if math.isnan(price[t, i]) else 1

    last_valid_price = np.zeros(N, dtype=np.float64)

    cash = init_cash
    cash_series = np.empty(T, dtype=np.float64)

    pos_book = np.zeros((hold_days, N), dtype=np.int64)
    pos_total = np.zeros(N, dtype=np.int64)
    pos_out = np.zeros((T, N), dtype=np.int64)

    max_lots = T * n_pick + 10
    lot_ticker   = np.empty(max_lots, dtype=np.int64)
    lot_shares   = np.empty(max_lots, dtype=np.int64)
    lot_entry_t  = np.empty(max_lots, dtype=np.int64)
    lot_entry_px = np.empty(max_lots, dtype=np.float64)
    lot_entry_cps= np.empty(max_lots, dtype=np.float64)
    lot_book     = np.empty(max_lots, dtype=np.int8)
    lot_open     = np.zeros(max_lots, dtype=np.uint8)

    lot_maturity_t   = np.empty(max_lots, dtype=np.int64)
    lot_exit_t       = -np.ones(max_lots, dtype=np.int64)
    lot_exit_px      = np.zeros(max_lots, dtype=np.float64)
    lot_exit_pps     = np.zeros(max_lots, dtype=np.float64)
    lot_fail_days    = np.zeros(max_lots, dtype=np.int64)
    lot_exit_reason  = np.zeros(max_lots, dtype=np.int8)

    lot_cnt = 0

    max_mature_per_day = n_pick + 5
    sched = -np.ones((T, max_mature_per_day), dtype=np.int64)
    sched_cnt = np.zeros(T, dtype=np.int64)

    pending = np.empty(max_lots, dtype=np.int64)
    pending_cnt = 0

    max_trades = T * (n_pick * 2 + 20)
    tr_t    = np.empty(max_trades, dtype=np.int64)
    tr_i    = np.empty(max_trades, dtype=np.int64)
    tr_side = np.empty(max_trades, dtype=np.int8)
    tr_sh   = np.empty(max_trades, dtype=np.int64)
    tr_px   = np.empty(max_trades, dtype=np.float64)
    tr_cps  = np.empty(max_trades, dtype=np.float64)
    tr_cf   = np.empty(max_trades, dtype=np.float64)
    tr_book = np.empty(max_trades, dtype=np.int8)
    tr_lot  = np.empty(max_trades, dtype=np.int64)
    tr_entry_t  = np.empty(max_trades, dtype=np.int64)
    tr_entry_px = np.empty(max_trades, dtype=np.float64)
    tr_pnl      = np.empty(max_trades, dtype=np.float64)
    tr_cnt = 0

    max_orders = T * (n_pick + 10)
    ord_t      = np.empty(max_orders, dtype=np.int64)
    ord_i      = np.empty(max_orders, dtype=np.int64)
    ord_book   = np.empty(max_orders, dtype=np.int8)
    ord_tw     = np.empty(max_orders, dtype=np.float64)
    ord_budget = np.empty(max_orders, dtype=np.float64)
    ord_des_sh = np.empty(max_orders, dtype=np.int64)
    ord_fill_sh= np.empty(max_orders, dtype=np.int64)
    ord_status = np.empty(max_orders, dtype=np.int8)
    ord_cnt = 0

    max_sell_fail = T * (n_pick + 10)
    sf_t   = np.empty(max_sell_fail, dtype=np.int64)
    sf_i   = np.empty(max_sell_fail, dtype=np.int64)
    sf_lot = np.empty(max_sell_fail, dtype=np.int64)
    sf_cnt = 0

    tmp_idx = np.empty(N, dtype=np.int64)

    nav = np.empty(T, dtype=np.float64)
    nav_open = np.empty(T, dtype=np.float64)

    for t in range(T):
        for i in range(N):
            if price_ok[t, i] == 1:
                last_valid_price[i] = price[t, i]

        # ===========================
        # 1) SELL phase (scheduled maturity)
        # ===========================
        sc = sched_cnt[t]
        for k in range(sc):
            lid = sched[t, k]
            if lid < 0 or lot_open[lid] != 1:
                continue
            i = lot_ticker[lid]
            sh = lot_shares[lid]
            if sh <= 0:
                lot_open[lid] = 0
                continue

            if price_ok[t, i] == 1:
                p = price[t, i]
                pps = p * (1.0 - impact_bps) - fee_per_share
                if pps < 0.0:
                    pps = 0.0
                proceeds = sh * pps
                cash += proceeds
                pnl = proceeds - sh * lot_entry_cps[lid]

                b = lot_book[lid]
                pos_book[b, i] -= sh
                pos_total[i] -= sh

                lot_open[lid] = 0
                lot_shares[lid] = 0

                lot_exit_t[lid] = t
                lot_exit_px[lid] = p
                lot_exit_pps[lid] = pps
                lot_exit_reason[lid] = LOT_EXIT_NORMAL

                if tr_cnt < max_trades:
                    tr_t[tr_cnt] = t
                    tr_i[tr_cnt] = i
                    tr_side[tr_cnt] = -1
                    tr_sh[tr_cnt] = sh
                    tr_px[tr_cnt] = p
                    tr_cps[tr_cnt] = pps
                    tr_cf[tr_cnt] = proceeds
                    tr_book[tr_cnt] = b
                    tr_lot[tr_cnt] = lid
                    tr_entry_t[tr_cnt] = lot_entry_t[lid]
                    tr_entry_px[tr_cnt] = lot_entry_px[lid]
                    tr_pnl[tr_cnt] = pnl
                    tr_cnt += 1
            else:
                if pending_cnt < max_lots:
                    pending[pending_cnt] = lid
                    pending_cnt += 1
                lot_fail_days[lid] += 1
                if sf_cnt < max_sell_fail:
                    sf_t[sf_cnt] = t
                    sf_i[sf_cnt] = i
                    sf_lot[sf_cnt] = lid
                    sf_cnt += 1

        # ===========================
        # 1b) SELL phase (pending)
        # ===========================
        if pending_cnt > 0:
            new_pending = np.empty(max_lots, dtype=np.int64)
            new_cnt = 0
            for kk in range(pending_cnt):
                lid = pending[kk]
                if lid < 0 or lot_open[lid] != 1:
                    continue
                i = lot_ticker[lid]
                sh = lot_shares[lid]
                if sh <= 0:
                    lot_open[lid] = 0
                    continue

                if price_ok[t, i] == 1:
                    p = price[t, i]
                    pps = p * (1.0 - impact_bps) - fee_per_share
                    if pps < 0.0:
                        pps = 0.0
                    proceeds = sh * pps
                    cash += proceeds
                    pnl = proceeds - sh * lot_entry_cps[lid]

                    b = lot_book[lid]
                    pos_book[b, i] -= sh
                    pos_total[i] -= sh

                    lot_open[lid] = 0
                    lot_shares[lid] = 0

                    lot_exit_t[lid] = t
                    lot_exit_px[lid] = p
                    lot_exit_pps[lid] = pps
                    lot_exit_reason[lid] = LOT_EXIT_DELAY

                    if tr_cnt < max_trades:
                        tr_t[tr_cnt] = t
                        tr_i[tr_cnt] = i
                        tr_side[tr_cnt] = -1
                        tr_sh[tr_cnt] = sh
                        tr_px[tr_cnt] = p
                        tr_cps[tr_cnt] = pps
                        tr_cf[tr_cnt] = proceeds
                        tr_book[tr_cnt] = b
                        tr_lot[tr_cnt] = lid
                        tr_entry_t[tr_cnt] = lot_entry_t[lid]
                        tr_entry_px[tr_cnt] = lot_entry_px[lid]
                        tr_pnl[tr_cnt] = pnl
                        tr_cnt += 1
                else:
                    new_pending[new_cnt] = lid
                    new_cnt += 1
                    lot_fail_days[lid] += 1
                    if sf_cnt < max_sell_fail:
                        sf_t[sf_cnt] = t
                        sf_i[sf_cnt] = i
                        sf_lot[sf_cnt] = lid
                        sf_cnt += 1
            pending = new_pending
            pending_cnt = new_cnt

        # ===========================
        # 1c) NAV open snapshot (pre-buy)
        # ===========================
        nav_open_t = cash
        for i in range(N):
            sh = pos_total[i]
            if sh > 0:
                nav_open_t += sh * last_valid_price[i]
        nav_open[t] = nav_open_t

        # ===========================
        # 2) BUY phase
        # ===========================
        b_today = t % hold_days

        if allow_entry[t] == 1:
            equity = nav_open_t

            tranche_budget = equity * book_frac[b_today]
            if tranche_budget > cash:
                tranche_budget = cash
            if tranche_budget < 0.0:
                tranche_budget = 0.0

            m = 0
            for i in range(N):
                if W_trade[t, i] > 0.0:
                    tmp_idx[m] = i
                    m += 1

            for a in range(m):
                best = a
                for b in range(a + 1, m):
                    if W_trade[t, tmp_idx[b]] > W_trade[t, tmp_idx[best]]:
                        best = b
                if best != a:
                    tmp = tmp_idx[a]; tmp_idx[a] = tmp_idx[best]; tmp_idx[best] = tmp

            fill_sh = np.zeros(m, dtype=np.int64)
            cps_arr = np.zeros(m, dtype=np.float64)
            ord_ptr = np.empty(m, dtype=np.int64)
            for k in range(m):
                ord_ptr[k] = -1

            cash_rem = cash
            for k in range(m):
                i = tmp_idx[k]
                w = W_trade[t, i]
                target_dol = tranche_budget * w

                if ord_cnt < max_orders:
                    ord_t[ord_cnt] = t
                    ord_i[ord_cnt] = i
                    ord_book[ord_cnt] = b_today
                    ord_tw[ord_cnt] = w
                    ord_budget[ord_cnt] = target_dol
                    ord_des_sh[ord_cnt] = 0
                    ord_fill_sh[ord_cnt] = 0
                    ord_status[ord_cnt] = ORD_OK
                    ord_ptr[k] = ord_cnt
                    ord_cnt += 1

                if sample_day[t, i] != 1:
                    if ord_ptr[k] >= 0:
                        ord_status[ord_ptr[k]] = ORD_INELIGIBLE
                    continue
                if price_ok[t, i] != 1:
                    if ord_ptr[k] >= 0:
                        ord_status[ord_ptr[k]] = ORD_NO_PRICE
                    continue

                p = price[t, i]
                cps = p * (1.0 + impact_bps) + fee_per_share
                if cps <= 0.0 or (not math.isfinite(cps)):
                    if ord_ptr[k] >= 0:
                        ord_status[ord_ptr[k]] = ORD_NO_PRICE
                    continue
                cps_arr[k] = cps

                des = int(math.floor(target_dol / cps))
                if ord_ptr[k] >= 0:
                    ord_des_sh[ord_ptr[k]] = des

                if des <= 0:
                    if ord_ptr[k] >= 0:
                        ord_status[ord_ptr[k]] = ORD_TOO_SMALL
                    continue

                cost = des * cps
                if cost > cash_rem:
                    des = int(math.floor(cash_rem / cps))
                    if des <= 0:
                        if ord_ptr[k] >= 0:
                            ord_status[ord_ptr[k]] = ORD_NO_CASH
                        continue
                    cost = des * cps

                fill_sh[k] = des
                cash_rem -= cost
                if ord_ptr[k] >= 0:
                    ord_fill_sh[ord_ptr[k]] = des

            progressed = True
            iter_cap = 5 * m
            it = 0
            while progressed and it < iter_cap:
                progressed = False
                it += 1
                for k in range(m):
                    if fill_sh[k] <= 0:
                        continue
                    cps = cps_arr[k]
                    if cps <= 0.0:
                        continue
                    if cash_rem >= cps:
                        cash_rem -= cps
                        fill_sh[k] += 1
                        progressed = True
                        if ord_ptr[k] >= 0:
                            ord_fill_sh[ord_ptr[k]] = fill_sh[k]

            for k in range(m):
                sh = fill_sh[k]
                if sh <= 0:
                    continue
                i = tmp_idx[k]
                p = price[t, i]
                cps = cps_arr[k]
                if cps <= 0.0:
                    continue

                cost = sh * cps
                cash -= cost

                lid = lot_cnt
                if lid >= max_lots:
                    continue
                lot_ticker[lid] = i
                lot_shares[lid] = sh
                lot_entry_t[lid] = t
                lot_entry_px[lid] = p
                lot_entry_cps[lid] = cps
                lot_book[lid] = b_today
                lot_open[lid] = 1

                mt = t + hold_days
                lot_maturity_t[lid] = mt

                lot_cnt += 1

                if mt < T:
                    c = sched_cnt[mt]
                    if c < max_mature_per_day:
                        sched[mt, c] = lid
                        sched_cnt[mt] = c + 1

                pos_book[b_today, i] += sh
                pos_total[i] += sh

                if tr_cnt < max_trades:
                    tr_t[tr_cnt] = t
                    tr_i[tr_cnt] = i
                    tr_side[tr_cnt] = 1
                    tr_sh[tr_cnt] = sh
                    tr_px[tr_cnt] = p
                    tr_cps[tr_cnt] = cps
                    tr_cf[tr_cnt] = -cost
                    tr_book[tr_cnt] = b_today
                    tr_lot[tr_cnt] = lid
                    tr_entry_t[tr_cnt] = -1
                    tr_entry_px[tr_cnt] = 0.0
                    tr_pnl[tr_cnt] = 0.0
                    tr_cnt += 1

        # ===========================
        # 3) NAV close snapshot (post-buy)
        # ===========================
        nav_t = cash
        for i in range(N):
            sh = pos_total[i]
            if sh > 0:
                nav_t += sh * last_valid_price[i]
        nav[t] = nav_t
        cash_series[t] = cash
        for i in range(N):
            pos_out[t, i] = pos_total[i]

    return (
        nav_open, nav, cash_series, pos_out,
        tr_t, tr_i, tr_side, tr_sh, tr_px, tr_cps, tr_cf, tr_book, tr_lot, tr_entry_t, tr_entry_px, tr_pnl, tr_cnt,
        ord_t, ord_i, ord_book, ord_tw, ord_budget, ord_des_sh, ord_fill_sh, ord_status, ord_cnt,
        sf_t, sf_i, sf_lot, sf_cnt,
        lot_ticker, lot_book, lot_entry_t, lot_entry_px, lot_entry_cps, lot_maturity_t, lot_exit_t, lot_exit_px, lot_exit_pps, lot_fail_days, lot_exit_reason, lot_cnt,
    )


# ============================================================
# 4) Latest status helper
# ============================================================

def _compute_latest_status(signal_bank, by_stage_report, picks_df, dt_latest):
    dt_latest = pd.Timestamp(dt_latest)
    stages = _infer_stage_blocks(signal_bank)

    in_holdout = None
    in_training = []
    for s in stages:
        hs, he = s["holdout_start"], s["holdout_end"]
        if dt_latest >= hs and dt_latest <= he:
            in_holdout = s
        ts, te = s["train_start"], s["train_end"]
        if ts is not None and te is not None:
            if dt_latest >= ts and dt_latest <= te:
                in_training.append(s)

    region = "training_or_gap"
    active_stage = None
    selected_factors = None
    if in_holdout is not None:
        region = "backtest_holdout"
        active_stage = in_holdout["stage_start"]
        selected_factors = in_holdout.get("selected_factors", None)

    picks = []
    if region == "backtest_holdout" and picks_df is not None and (not picks_df.empty):
        trade_dates = picks_df.index.get_level_values(0)
        cand = trade_dates[trade_dates <= dt_latest]
        if len(cand) > 0:
            last_td = cand.max()
            sub = picks_df.loc[last_td].reset_index()
            sub = sub.sort_values("weight", ascending=False).head(10)
            picks = sub.to_dict("records")
            dt_used = pd.Timestamp(last_td)
        else:
            dt_used = dt_latest
    else:
        dt_used = dt_latest

    overlap = []
    for s in in_training:
        overlap.append(dict(stage_start=s["stage_start"], train_window=(s["train_start"], s["train_end"])) )

    active_metrics = None
    if active_stage is not None and active_stage in by_stage_report:
        active_metrics = by_stage_report[active_stage]["metrics"]

    return dict(
        date=dt_used,
        region=region,
        active_stage=active_stage,
        selected_factors=selected_factors,
        picks=picks,
        training_overlap=overlap,
        active_stage_metrics=active_metrics,
    )


# ============================================================
# 5) Main backtest wrapper (WF stages) - STRICT + rich outputs
# ============================================================

def run_backtest_ga_wf_softmax_5d(
    close_df: pd.DataFrame,
    signal_bank: dict,
    start: str | None = None,
    end: str | None = None,
    volume_df: pd.DataFrame | None = None,
    amount_df: pd.DataFrame | None = None,
    spy_close: pd.Series | None = None,
    budget_frac_by_stage: dict | None = None,
    params: dict | None = None,

    sample_space: pd.DataFrame | None = None,
    amo_window: int = 10,
    amo_threshold: float = 1e7,
    min_listed_days: int = 60,

    min_cs_n: int = 30,
    liquidation_buffer_days: int = 10,
    strict_tminus1: bool = True,

    barra_expo: pd.DataFrame | None = None,
    factor_returns: pd.DataFrame | None = None,
):
    P = dict(
        init_cash=1e8,
        hold_days=5,
        n_pick=10,
        tau_softmax=1.4,
        fee_per_share=0.005,
        impact_bps=4e-4,
    )
    if params is not None:
        P.update(params)

    H = int(P["hold_days"])
    if H != 5:
        raise ValueError("This implementation is locked to hold_days=5 per spec.")

    close_df, volume_df, amount_df = _align_close_volume(close_df, volume_df, amount_df)

    close_index = close_df.index
    start_is_auto = (start is None) or (isinstance(start, str) and start.lower() == "auto")
    end_is_auto   = (end is None) or (isinstance(end, str) and end.lower() == "auto")

    if start_is_auto or end_is_auto:
        g_start, g_end, earliest_hs, latest_he = _infer_global_backtest_window_from_signal_bank(
            signal_bank=signal_bank,
            close_index=close_index,
            hold_days=int(P["hold_days"]),
            liquidation_buffer_days=int(liquidation_buffer_days),
        )
        if start_is_auto:
            start = g_start
        else:
            start = pd.Timestamp(start)

        if end_is_auto:
            end = g_end
        else:
            end = pd.Timestamp(end)

        inferred_window = dict(
            inferred_start=pd.Timestamp(g_start),
            inferred_end_with_liquidation=pd.Timestamp(g_end),
            earliest_holdout_start=pd.Timestamp(earliest_hs),
            latest_holdout_end=pd.Timestamp(latest_he),
        )
    else:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        inferred_window = None

    close_seg = close_df.loc[start:end].copy()
    stocks = close_seg.columns
    if len(stocks) == 0:
        raise ValueError("No tickers in close_df within [start,end].")

    if sample_space is None:
        sample_space = build_sample_space_amo_listing(
            close_df=close_df,
            volume_df=volume_df,
            amount_df=amount_df,
            amo_window=amo_window,
            amo_threshold=amo_threshold,
            min_listed_days=min_listed_days,
            require_price_positive=True,
        )
    sample_space = _to_datetime_index(sample_space).reindex_like(close_df).fillna(0).astype(np.uint8)

    if spy_close is not None:
        spy_close = _to_datetime_index(spy_close.to_frame("SPY"))["SPY"].astype(float)
    else:
        spy_close = None

    stage_blocks = _infer_stage_blocks(signal_bank)
    if len(stage_blocks) == 0:
        raise ValueError("signal_bank['by_slice'] has no usable stages with composite.")

    all_trades = []
    all_orders = []
    all_sell_fail = []
    all_roundtrips = []
    all_picks = []
    all_lots = []
    by_stage = {}
    nav_global = []
    nav_open_global = []
    daily_global = []
    exec_daily_global = []

    cash_carry = float(P["init_cash"])

    for sb in stage_blocks:
        stage_start = sb["stage_start"]
        holdout_start = sb["holdout_start"]
        holdout_end = sb["holdout_end"]

        if holdout_end < start or holdout_start > end:
            continue

        blk = signal_bank["by_slice"].get(stage_start, None)
        if blk is None:
            continue
        comp_df = blk.get("composite", None)
        if comp_df is None or comp_df.empty:
            continue

        comp_df = _to_datetime_index(comp_df).reindex(columns=stocks).astype(float)

        hs = max(holdout_start, start)
        he = min(holdout_end, end)

        holdout_idx = comp_df.index[(comp_df.index >= hs) & (comp_df.index <= he)]
        if len(holdout_idx) < 2:
            continue

        base_start = holdout_idx.min()
        prev_days = close_df.index[close_df.index < base_start]
        prev_day = prev_days.max() if (strict_tminus1 and len(prev_days) > 0) else base_start

        after = close_df.index[close_df.index > he]
        need_extra = H + max(int(liquidation_buffer_days), 0)
        if len(after) > 0:
            extra = after[:need_extra]
            he_buf = min(extra.max(), end)
        else:
            he_buf = he

        stage_dates = close_df.index[(close_df.index >= prev_day) & (close_df.index <= he_buf)]
        stage_dates = pd.to_datetime(stage_dates)
        if len(stage_dates) < (H + 2):
            continue

        close_stage = close_df.reindex(index=stage_dates, columns=stocks).astype(float)
        sample_stage_raw = sample_space.reindex(index=stage_dates, columns=stocks).astype(np.uint8)
        comp_stage_raw = comp_df.reindex(index=stage_dates).astype(float)

        # ============================================================
        # Factor/alpha signals are already delayed upstream (inside signal_bank),
        # and sample_space already enforces delay in build_sample_space_amo_listing.
        # No additional shift is applied here.
        # ============================================================
        sample_stage = sample_stage_raw
        comp_stage = comp_stage_raw

        T = len(stage_dates)
        entry_mask = np.zeros(T, dtype=np.uint8)
        for t in range(T):
            dt = pd.Timestamp(stage_dates[t])
            if dt < hs or dt > he:
                continue
            entry_mask[t] = 1

        price = close_stage.to_numpy(dtype=np.float64)
        sample_day = sample_stage.to_numpy(dtype=np.uint8)
        comp = comp_stage.to_numpy(dtype=np.float64)

        W_trade, allow_entry = _build_w_trade_stage(
            comp=comp,
            sample_day=sample_day,
            entry_mask=entry_mask,
            n_pick=int(P["n_pick"]),
            tau=float(P["tau_softmax"]),
            min_cs_n=int(min_cs_n),
        )

        # picks
        for t in range(T):
            if allow_entry[t] != 1:
                continue
            td = pd.Timestamp(stage_dates[t])
            w = W_trade[t]
            pick_idx = np.where(w > 0)[0]
            if pick_idx.size:
                elig = sample_day[t]
                z = _cs_zscore_one_day(comp[t], elig, int(min_cs_n))
                for j in pick_idx:
                    all_picks.append(dict(
                        stage_start=pd.Timestamp(stage_start),
                        trade_date=td,
                        ticker=stocks[j],
                        zscore=float(z[j]),
                        weight=float(w[j]),
                    ))

        book_frac = _pick_book_frac(budget_frac_by_stage, stage_start, hold_days=H)

        out = _bt_stage_5d_lots(
            price=price,
            sample_day=sample_day,
            W_trade=W_trade,
            allow_entry=allow_entry,
            book_frac=book_frac,
            hold_days=H,
            init_cash=cash_carry,
            fee_per_share=float(P["fee_per_share"]),
            impact_bps=float(P["impact_bps"]),
            n_pick=int(P["n_pick"]),
        )

        (
            nav_open_arr, nav_arr, cash_series, pos_out,
            tr_t, tr_i, tr_side, tr_sh, tr_px, tr_cps, tr_cf, tr_book, tr_lot, tr_entry_t, tr_entry_px, tr_pnl, tr_cnt,
            ord_t, ord_i, ord_book, ord_tw, ord_budget, ord_des_sh, ord_fill_sh, ord_status, ord_cnt,
            sf_t, sf_i, sf_lot, sf_cnt,
            lot_ticker, lot_book, lot_entry_t, lot_entry_px, lot_entry_cps, lot_maturity_t, lot_exit_t, lot_exit_px, lot_exit_pps, lot_fail_days, lot_exit_reason, lot_cnt,
        ) = out

        nav_open_s = pd.Series(nav_open_arr, index=stage_dates, name="NAV_OPEN")
        nav_s = pd.Series(nav_arr, index=stage_dates, name="NAV")

        mtm_price = close_stage.ffill()
        pos_df = pd.DataFrame(pos_out, index=stage_dates, columns=stocks).astype(np.int64)
        hv_df = pos_df.mul(mtm_price).astype(float)
        w_df = hv_df.div(nav_s, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cash_s = pd.Series(cash_series, index=stage_dates, name="CASH").astype(float)

        daily_stage = dict(
            stage_start=pd.Timestamp(stage_start),
            pos=pos_df,
            holdings_value=hv_df,
            weights=w_df,
            cash=cash_s,
            mtm_price=mtm_price,
            nav=nav_s,
            nav_open=nav_open_s,
        )
        daily_global.append(daily_stage)

        # trades df
        stage_trade_recs = []
        for k in range(int(tr_cnt)):
            dt = pd.Timestamp(stage_dates[int(tr_t[k])])
            i = int(tr_i[k])
            ticker = stocks[i]
            side = "BUY" if int(tr_side[k]) == 1 else "SELL"
            stage_trade_recs.append(dict(
                stage_start=pd.Timestamp(stage_start),
                date=dt,
                ticker=ticker,
                side=side,
                shares=int(tr_sh[k]),
                price=float(tr_px[k]),
                cost_or_proceeds_per_sh=float(tr_cps[k]),
                cash_flow=float(tr_cf[k]),
                book=int(tr_book[k]),
                lot_id=int(tr_lot[k]),
                entry_date=(pd.Timestamp(stage_dates[int(tr_entry_t[k])]) if int(tr_entry_t[k]) >= 0 else pd.NaT),
                entry_price=float(tr_entry_px[k]) if int(tr_entry_t[k]) >= 0 else np.nan,
                realized_pnl=float(tr_pnl[k]) if side == "SELL" else 0.0,
            ))
        trades_df = pd.DataFrame(stage_trade_recs)

        # orders df
        stage_order_recs = []
        for k in range(int(ord_cnt)):
            dt = pd.Timestamp(stage_dates[int(ord_t[k])])
            i = int(ord_i[k])
            stage_order_recs.append(dict(
                stage_start=pd.Timestamp(stage_start),
                date=dt,
                ticker=stocks[i],
                book=int(ord_book[k]),
                target_weight=float(ord_tw[k]),
                target_dollar=float(ord_budget[k]),
                desired_shares=int(ord_des_sh[k]),
                filled_shares=int(ord_fill_sh[k]),
                status=int(ord_status[k]),
            ))
        orders_df = pd.DataFrame(stage_order_recs)

        # sell fail df
        sf_recs = []
        for k in range(int(sf_cnt)):
            dt = pd.Timestamp(stage_dates[int(sf_t[k])])
            i = int(sf_i[k])
            sf_recs.append(dict(
                stage_start=pd.Timestamp(stage_start),
                date=dt,
                ticker=stocks[i],
                lot_id=int(sf_lot[k]),
                reason="NO_CLOSE_PRICE",
            ))
        sell_fail_df = pd.DataFrame(sf_recs)

        # lots df
        lot_recs = []
        for lid in range(int(lot_cnt)):
            i = int(lot_ticker[lid])
            ticker = stocks[i]
            ent_t = int(lot_entry_t[lid])
            mat_t = int(lot_maturity_t[lid])
            ex_t  = int(lot_exit_t[lid])
            lot_recs.append(dict(
                stage_start=pd.Timestamp(stage_start),
                lot_id=int(lid),
                ticker=ticker,
                book=int(lot_book[lid]),
                entry_date=pd.Timestamp(stage_dates[ent_t]),
                entry_px=float(lot_entry_px[lid]),
                entry_cost_per_sh=float(lot_entry_cps[lid]),
                shares=int(trades_df[trades_df["lot_id"]==lid]["shares"].iloc[0]) if not trades_df.empty else np.nan,
                maturity_date=(pd.Timestamp(stage_dates[mat_t]) if mat_t < len(stage_dates) else pd.NaT),
                exit_date=(pd.Timestamp(stage_dates[ex_t]) if ex_t >= 0 else pd.NaT),
                exit_px=float(lot_exit_px[lid]) if ex_t >= 0 else np.nan,
                exit_proceeds_per_sh=float(lot_exit_pps[lid]) if ex_t >= 0 else np.nan,
                exit_reason=("NORMAL" if int(lot_exit_reason[lid])==LOT_EXIT_NORMAL else "DELAY") if ex_t >= 0 else "OPEN",
                fail_days_after_maturity=int(lot_fail_days[lid]),
                delay_days=(ex_t - mat_t) if (ex_t >= 0 and mat_t >= 0) else np.nan,
            ))
        lots_df = pd.DataFrame(lot_recs)

        # roundtrips
        if not trades_df.empty:
            roundtrips_df = trades_df[trades_df["side"] == "SELL"].copy()
            if not roundtrips_df.empty:
                roundtrips_df["holding_days"] = (roundtrips_df["date"] - roundtrips_df["entry_date"]).dt.days
                buy_map = trades_df[trades_df["side"]=="BUY"].set_index("lot_id")[["cost_or_proceeds_per_sh","shares"]]
                roundtrips_df["entry_cost_per_sh"] = roundtrips_df["lot_id"].map(buy_map["cost_or_proceeds_per_sh"])
                roundtrips_df["entry_notional"] = roundtrips_df["shares"] * roundtrips_df["entry_cost_per_sh"]
                roundtrips_df["rt_return"] = roundtrips_df["realized_pnl"] / roundtrips_df["entry_notional"]
        else:
            roundtrips_df = pd.DataFrame()

        if not trades_df.empty:
            all_trades.append(trades_df)
        if not orders_df.empty:
            all_orders.append(orders_df)
        if not sell_fail_df.empty:
            all_sell_fail.append(sell_fail_df)
        if not roundtrips_df.empty:
            all_roundtrips.append(roundtrips_df)
        if not lots_df.empty:
            all_lots.append(lots_df)

        # daily execution summary
        exec_day = []
        dates_only = pd.Index(stage_dates)
        for dt in dates_only:
            od = orders_df[orders_df["date"]==dt] if not orders_df.empty else pd.DataFrame()
            td = trades_df[trades_df["date"]==dt] if not trades_df.empty else pd.DataFrame()
            sf = sell_fail_df[sell_fail_df["date"]==dt] if not sell_fail_df.empty else pd.DataFrame()

            exec_day.append(dict(
                stage_start=pd.Timestamp(stage_start),
                date=pd.Timestamp(dt),
                n_orders=int(len(od)),
                n_order_filled=int((od["filled_shares"]>0).sum()) if not od.empty else 0,
                n_order_failed=int((od["filled_shares"]<=0).sum()) if not od.empty else 0,
                fail_ineligible=int((od["status"]==ORD_INELIGIBLE).sum()) if not od.empty else 0,
                fail_no_price=int((od["status"]==ORD_NO_PRICE).sum()) if not od.empty else 0,
                fail_too_small=int((od["status"]==ORD_TOO_SMALL).sum()) if not od.empty else 0,
                fail_no_cash=int((od["status"]==ORD_NO_CASH).sum()) if not od.empty else 0,
                n_trades=int(len(td)),
                n_buys=int((td["side"]=="BUY").sum()) if not td.empty else 0,
                n_sells=int((td["side"]=="SELL").sum()) if not td.empty else 0,
                realized_pnl_sell=float(td.loc[td["side"]=="SELL","realized_pnl"].sum()) if not td.empty else 0.0,
                n_sell_fail_events=int(len(sf)),
            ))
        exec_daily_df = pd.DataFrame(exec_day)
        exec_daily_global.append(exec_daily_df)

        # stage metrics over holdout only
        nav_hold_close = nav_s.loc[hs:he].copy()
        nav_hold_open  = nav_open_s.loc[hs:he].copy()
        if len(nav_hold_close) < 2:
            continue

        start_nav_report = float(nav_hold_open.iloc[0])
        end_nav_report   = float(nav_hold_close.iloc[-1])
        total_rt_report  = float(end_nav_report / start_nav_report - 1.0)

        nav_report = nav_hold_close.copy()
        nav_report.iloc[0] = start_nav_report

        ann = _annualize_from_nav(nav_report)
        rollH = _roll_h_stats(nav_report, H=H)

        idx_total_rt = np.nan
        hedged_total_rt = np.nan
        avg_daily_excess = np.nan
        if spy_close is not None:
            idx_stage = spy_close.loc[nav_hold_close.index.min():nav_hold_close.index.max()].reindex(nav_hold_close.index).astype(float)
            idx_ret = idx_stage.pct_change().fillna(0.0)
            strat_ret = nav_report.pct_change().fillna(0.0)
            idx_total_rt = float(idx_stage.iloc[-1] / idx_stage.iloc[0] - 1.0)
            net_ret = strat_ret - idx_ret
            hedged_curve = (1 + net_ret).cumprod()
            hedged_total_rt = float(hedged_curve.iloc[-1] - 1.0)
            avg_daily_excess = float(net_ret.mean())

        meta = blk.get("meta", {}) if isinstance(blk, dict) else {}
        selected_factors = meta.get("selected_factors", None)

        stage_payload = dict(
            stage_start=pd.Timestamp(stage_start),
            train_window=(sb.get("train_start", None), sb.get("train_end", None)),
            holdout_window=(pd.Timestamp(hs), pd.Timestamp(he)),
            liquidation_end=pd.Timestamp(stage_dates.max()),
            selected_factors=selected_factors,
            params=P.copy(),
            book_frac=book_frac.copy(),
            nav_open_holdout=nav_hold_open,
            nav_close_holdout=nav_hold_close,
            metrics=dict(
                start_nav=start_nav_report,
                end_nav=end_nav_report,
                total_return=total_rt_report,
                ann_return=ann["ann_ret"],
                ann_vol=ann["ann_vol"],
                sharpe=ann["sharpe"],
                max_drawdown=_max_drawdown(nav_hold_close),
                index_total_return=idx_total_rt,
                hedged_total_return=hedged_total_rt,
                avg_daily_excess=avg_daily_excess,
                trade_window_h_mean=rollH["mean_h"],
                trade_window_h_vol=rollH["vol_h"],
                trade_window_h_var=rollH["var_h"],
                trade_window_h_n=rollH["n_h"],
            ),
        )

        if barra_expo is not None and isinstance(barra_expo.index, pd.MultiIndex) and barra_expo.index.nlevels >= 2:
            w_long = w_df.stack().rename("weight")
            w_long.index = w_long.index.set_names(["date","ticker"])
            expo = barra_expo.copy()
            expo.index = expo.index.set_names(["date","ticker"])
            common = w_long.index.intersection(expo.index)
            if len(common) > 0:
                w_c = w_long.loc[common]
                e_c = expo.loc[common]
                port_expo = e_c.mul(w_c.values, axis=0).groupby(level=0).sum()
                stage_payload["factor_exposure"] = dict(
                    daily=port_expo,
                    mean=port_expo.mean().sort_values(ascending=False),
                    std=port_expo.std(ddof=1).sort_values(ascending=False),
                )

        by_stage[pd.Timestamp(stage_start)] = stage_payload

        nav_global.append(nav_hold_close)
        nav_open_global.append(nav_hold_open)

        cash_carry = float(nav_s.iloc[-1])

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    orders_all = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
    sell_fail_all = pd.concat(all_sell_fail, ignore_index=True) if all_sell_fail else pd.DataFrame()
    roundtrips_all = pd.concat(all_roundtrips, ignore_index=True) if all_roundtrips else pd.DataFrame()
    lots_all = pd.concat(all_lots, ignore_index=True) if all_lots else pd.DataFrame()
    exec_daily_all = pd.concat(exec_daily_global, ignore_index=True) if exec_daily_global else pd.DataFrame()

    picks_df = pd.DataFrame(all_picks) if len(all_picks) else pd.DataFrame()
    if not picks_df.empty:
        picks_df["trade_date"] = pd.to_datetime(picks_df["trade_date"])
        picks_df = picks_df.sort_values(["trade_date","ticker"]).set_index(["trade_date","ticker"])

    if len(nav_global) == 0:
        raise ValueError("No stage produced NAV; check date ranges and signal_bank coverage.")

    nav_all_close = pd.concat(nav_global).sort_index().loc[start:end]
    nav_all_open  = pd.concat(nav_open_global).sort_index().loc[start:end]

    start_nav_report = float(nav_all_open.iloc[0]) if len(nav_all_open) else float(nav_all_close.iloc[0])
    end_nav_report   = float(nav_all_close.iloc[-1])

    nav_report = nav_all_close.copy()
    nav_report.iloc[0] = start_nav_report

    ann = _annualize_from_nav(nav_report)
    overall = dict(
        start_nav=start_nav_report,
        end_nav=end_nav_report,
        total_return=float(end_nav_report / start_nav_report - 1.0),
        ann_return=ann["ann_ret"],
        ann_vol=ann["ann_vol"],
        sharpe=ann["sharpe"],
        max_drawdown=_max_drawdown(nav_all_close),
    )
    if spy_close is not None:
        idx_aligned = spy_close.loc[nav_all_close.index.min():nav_all_close.index.max()].reindex(nav_all_close.index).astype(float)
        idx_ret = idx_aligned.pct_change().fillna(0.0)
        strat_ret = nav_report.pct_change().fillna(0.0)
        net_ret = strat_ret - idx_ret
        overall.update(
            index_total_return=float(idx_aligned.iloc[-1] / idx_aligned.iloc[0] - 1.0),
            hedged_total_return=float((1 + net_ret).cumprod().iloc[-1] - 1.0),
            avg_daily_excess=float(net_ret.mean()),
        )

    latest_status = _compute_latest_status(
        signal_bank=signal_bank,
        by_stage_report=by_stage,
        picks_df=picks_df,
        dt_latest=end,
    )

    bt_bank = dict(
        params=P.copy(),
        strict_tminus1=bool(strict_tminus1),
        nav=nav_all_close,
        nav_open=nav_all_open,
        nav_report=nav_report,
        overall=overall,
        by_stage=by_stage,
        picks_df=picks_df,

        blotter=dict(
            trades=trades_all,
            orders=orders_all,
            sell_fail=sell_fail_all,
            roundtrips=roundtrips_all,
            lots=lots_all,
        ),

        daily=daily_global,
        exec_daily=exec_daily_all,

        latest_status=latest_status,
        sample_space=sample_space,

        inferred_window=inferred_window,
    )

    if barra_expo is not None and isinstance(barra_expo, pd.DataFrame) and (not barra_expo.empty):
        bt_bank["barra_expo"] = barra_expo.copy()
    if factor_returns is not None and isinstance(factor_returns, pd.DataFrame) and (not factor_returns.empty):
        bt_bank["factor_returns"] = _to_datetime_index(factor_returns).copy()

    return bt_bank


# ============================================================
# 6) Post analysis: stage stats + significance + factor attribution
# ============================================================

def _newey_west_mean_test(x: pd.Series, lags: int | None = None):
    """
    H0: E[x] <= 0
    H1: E[x] > 0
    Newey-West HAC standard error for sample mean; asymptotic normal one-sided p.
    """
    s = pd.Series(x).dropna().astype(float)
    n = int(len(s))
    if n < 10:
        return dict(n=n, lags=lags if lags is not None else 0, t_stat=np.nan, p_one_sided=np.nan, mean=np.nan, std=np.nan)

    x = s.to_numpy(dtype=float)
    mu = float(np.mean(x))

    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(1, min(lags, n - 1))

    u = x - mu
    gamma0 = float(np.dot(u, u) / n)

    s_hac = gamma0
    for L in range(1, lags + 1):
        wL = 1.0 - (L / (lags + 1.0))
        cov = float(np.dot(u[L:], u[:-L]) / n)
        s_hac += 2.0 * wL * cov

    if not np.isfinite(s_hac) or s_hac <= 0:
        return dict(n=n, lags=lags, t_stat=np.nan, p_one_sided=np.nan, mean=mu, std=float(np.std(x, ddof=1)))

    se = math.sqrt(s_hac / n)
    t_stat = mu / se if se > 0 else np.nan

    if sp_stats is not None:
        p_one_sided = float(sp_stats.norm.sf(t_stat))
    else:
        p_one_sided = float(0.5 * (1.0 - math.erf(t_stat / math.sqrt(2.0))))

    return dict(
        n=n,
        lags=int(lags),
        t_stat=float(t_stat),
        p_one_sided=float(p_one_sided),
        mean=float(mu),
        std=float(np.std(x, ddof=1)),
    )

def build_backtest_results(bt_bank: dict, trading_days=252) -> dict:
    by_stage = bt_bank.get("by_stage", {})
    if not by_stage:
        return {}

    recs = []
    for k, v in sorted(by_stage.items(), key=lambda x: x[0]):
        m = v["metrics"]
        hs, he = v["holdout_window"]
        recs.append(dict(
            stage_start=pd.Timestamp(v["stage_start"]),
            holdout_start=pd.Timestamp(hs),
            holdout_end=pd.Timestamp(he),
            liquidation_end=pd.Timestamp(v["liquidation_end"]),
            total_return=float(m["total_return"]),
            ann_return=float(m["ann_return"]),
            ann_vol=float(m["ann_vol"]) if m["ann_vol"] is not None else np.nan,
            sharpe=float(m["sharpe"]) if m["sharpe"] is not None else np.nan,
            max_drawdown=float(m["max_drawdown"]),
            trade_window_h_mean=float(m["trade_window_h_mean"]),
            trade_window_h_vol=float(m["trade_window_h_vol"]),
            trade_window_h_var=float(m["trade_window_h_var"]),
            trade_window_h_n=int(m["trade_window_h_n"]),
            index_total_return=float(m.get("index_total_return", np.nan)),
            hedged_total_return=float(m.get("hedged_total_return", np.nan)),
            avg_daily_excess=float(m.get("avg_daily_excess", np.nan)),
        ))
    stage_df = pd.DataFrame(recs).sort_values("stage_start").reset_index(drop=True)

    x = stage_df["total_return"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    t_stat = np.nan
    p_one_sided = np.nan
    n = int(len(x))
    if n >= 2:
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        if sd > 0:
            t_stat = mu / (sd / math.sqrt(n))
            if sp_stats is not None:
                p_one_sided = float(1.0 - sp_stats.t.cdf(t_stat, df=n-1))
            else:
                p_one_sided = float(1.0 - 0.5*(1.0 + math.erf(t_stat / math.sqrt(2.0))))

    nav = bt_bank.get("nav_report", bt_bank["nav"]).astype(float)
    r = nav.pct_change().dropna()

    nw = _newey_west_mean_test(r, lags=None)

    factor_summary = None
    factor_attrib = None

    if "barra_expo" in bt_bank:
        expo = bt_bank["barra_expo"]
        if isinstance(expo.index, pd.MultiIndex) and expo.index.nlevels >= 2:
            wt_list = []
            for d in bt_bank.get("daily", []):
                w = d["weights"].copy()
                w.index = pd.to_datetime(w.index)
                w_long = w.stack().rename("weight")
                w_long.index = w_long.index.set_names(["date","ticker"])
                wt_list.append(w_long)
            if wt_list:
                wt_all = pd.concat(wt_list).groupby(level=[0,1]).last()
                expo2 = expo.copy()
                expo2.index = expo2.index.set_names(["date","ticker"])
                common = wt_all.index.intersection(expo2.index)
                if len(common) > 0:
                    wt_c = wt_all.loc[common]
                    expo_c = expo2.loc[common]
                    port_expo = expo_c.mul(wt_c.values, axis=0).groupby(level=0).sum()

                    factor_summary = dict(
                        portfolio_factor_exposure=port_expo,
                        factor_exposure_mean=port_expo.mean().sort_values(ascending=False),
                        factor_exposure_std=port_expo.std(ddof=1).sort_values(ascending=False),
                    )

                    if "factor_returns" in bt_bank:
                        fr = bt_bank["factor_returns"].copy()
                        fr = fr.reindex(port_expo.index).dropna(how="all")
                        pe = port_expo.reindex(fr.index).fillna(0.0)

                        contrib = pe * fr
                        total_factor_return = contrib.sum(axis=1)

                        factor_attrib = dict(
                            factor_returns=fr,
                            portfolio_exposure=pe,
                            daily_factor_contrib_return=contrib,
                            daily_total_factor_return=total_factor_return,
                            contrib_mean=contrib.mean().sort_values(ascending=False),
                            contrib_std=contrib.std(ddof=1).sort_values(ascending=False),
                        )

    backtest_results = dict(
        stage_table=stage_df,
        stage_return_test=dict(
            n=n,
            t_stat=t_stat,
            p_one_sided=p_one_sided,
            mean=float(np.mean(x)) if n else np.nan,
            std=float(np.std(x, ddof=1)) if n > 1 else np.nan,
        ),
        daily_return_test=dict(
            n=int(nw["n"]),
            lags=int(nw["lags"]),
            t_stat=float(nw["t_stat"]),
            p_one_sided=float(nw["p_one_sided"]),
            mean=float(nw["mean"]),
            std=float(nw["std"]),
            method="newey_west_hac_mean",
        ),
        series=dict(
            daily_return=r,
            drawdown=(bt_bank["nav"] / bt_bank["nav"].cummax() - 1.0),
        ),
        execution=dict(
            exec_daily=bt_bank.get("exec_daily", pd.DataFrame()),
            lots=bt_bank.get("blotter", {}).get("lots", pd.DataFrame()),
        ),
        factor_summary=factor_summary,
        factor_attribution=factor_attrib,
    )
    return backtest_results


# ============================================================
# 7) Plotting (matplotlib only, legend below)  -- UPDATED
# ============================================================

def _set_presentation_style():
    import matplotlib.ticker as mticker  # noqa: F401
    mpl.rcParams.update({
        "figure.dpi": 170,
        "savefig.dpi": 260,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "font.size": 11.5,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.alpha": 0.22,

        "legend.frameon": False,
        "legend.fontsize": 10.0,

        "lines.linewidth": 2.6,
        "lines.markersize": 5.0,
    })

    # Rich, vivid, presentation-grade color cycle (matplotlib-native)
    palette = (
        list(plt.get_cmap("Set2").colors) +
        list(plt.get_cmap("Dark2").colors) +
        list(plt.get_cmap("tab10").colors) +
        list(plt.get_cmap("tab20").colors)
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette)

def _legend_below(ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    if labels is None or len(labels) == 0:
        return 0.05

    n = len(labels)
    ncol = max(1, int(ncol))
    rows = int(math.ceil(n / ncol))

    yoff = -0.16 - 0.07 * max(rows - 1, 0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, yoff),
        ncol=ncol,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.5,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    bottom = 0.10 + 0.06 * rows
    bottom = min(max(bottom, 0.10), 0.32)
    return bottom

def plot_nav(bt_bank: dict, spy_close: pd.Series | None = None, title="NAV vs Benchmark"):
    import matplotlib.dates as mdates
    _set_presentation_style()
    nav = bt_bank["nav"].astype(float)

    fig = plt.figure(figsize=(12.2, 5.2))
    ax = fig.add_subplot(111)

    ax.plot(nav.index, nav.values, label="Strategy NAV (Close)")
    ax.fill_between(nav.index, nav.values, np.nanmin(nav.values), alpha=0.08)

    if spy_close is not None:
        spy = _to_datetime_index(spy_close.to_frame("SPY"))["SPY"].reindex(nav.index).ffill()
        spy_nav = spy / spy.iloc[0] * nav.iloc[0]
        ax.plot(spy_nav.index, spy_nav.values, label="SPY (scaled)", linestyle="--")

    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")

    loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    bottom = _legend_below(ax, ncol=2)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig

def plot_drawdown(bt_results: dict, title="Drawdown"):
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    _set_presentation_style()
    dd = bt_results["series"]["drawdown"].astype(float)

    fig = plt.figure(figsize=(12.2, 4.1))
    ax = fig.add_subplot(111)

    ax.plot(dd.index, dd.values, label="Drawdown")
    ax.fill_between(dd.index, dd.values, 0.0, alpha=0.12)

    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    bottom = _legend_below(ax, ncol=1)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig

def plot_stage_returns(bt_results: dict, title="Stage Total Returns"):
    import matplotlib.ticker as mticker
    _set_presentation_style()
    stage_df = bt_results["stage_table"].copy()
    x = np.arange(len(stage_df))
    y = stage_df["total_return"].astype(float).values

    fig = plt.figure(figsize=(12.2, 4.9))
    ax = fig.add_subplot(111)

    pos = y.copy()
    neg = -y.copy()
    pos[pos < 0] = 0.0
    neg[neg < 0] = 0.0
    max_pos = float(np.nanmax(pos)) if np.any(np.isfinite(pos)) else 0.0
    max_neg = float(np.nanmax(neg)) if np.any(np.isfinite(neg)) else 0.0
    max_pos = max(max_pos, 1e-12)
    max_neg = max(max_neg, 1e-12)

    cmap_pos = plt.get_cmap("Blues")
    cmap_neg = plt.get_cmap("Reds")
    bar_colors = []
    for v in y:
        if not np.isfinite(v):
            bar_colors.append((0.7, 0.7, 0.7, 0.9))
        elif v >= 0:
            bar_colors.append(cmap_pos(0.35 + 0.55 * (v / max_pos)))
        else:
            bar_colors.append(cmap_neg(0.35 + 0.55 * ((-v) / max_neg)))

    ax.bar(x, y, color=bar_colors, edgecolor="white", linewidth=0.7, label="Stage Total Return")
    ax.axhline(0.0, linewidth=1.1, alpha=0.55)

    ax.set_title(title, pad=10)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Total Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax.set_xticks(x)
    ax.set_xticklabels([str(pd.Timestamp(d).date()) for d in stage_df["stage_start"]], rotation=45, ha="right")

    bottom = _legend_below(ax, ncol=1)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig

def plot_stage_return_hist(bt_results: dict, title="Stage Return Distribution"):
    import matplotlib.ticker as mticker
    _set_presentation_style()
    stage_df = bt_results["stage_table"]
    x = stage_df["total_return"].astype(float).values
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        raise ValueError("No finite stage returns to plot.")

    bins = int(max(18, min(60, np.sqrt(n) * 6)))

    fig = plt.figure(figsize=(9.8, 4.7))
    ax = fig.add_subplot(111)

    ax.hist(x, bins=bins, density=True, alpha=0.85, edgecolor="white", linewidth=0.6, label="Stage Returns (density)")
    ax.axvline(0.0, linewidth=1.2, alpha=0.75, label="0%")
    ax.axvline(float(np.mean(x)), linewidth=1.2, alpha=0.75, linestyle="--", label="Mean")

    ax.set_title(title, pad=10)
    ax.set_xlabel("Stage Total Return")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    bottom = _legend_below(ax, ncol=3)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig

def plot_stage_variance(bt_results: dict, title="Stage Return Variance (Total Return)"):
    _set_presentation_style()
    stage_df = bt_results["stage_table"].copy()
    y = stage_df["total_return"].astype(float).values
    mu = np.nanmean(y)
    v = (y - mu) ** 2

    fig = plt.figure(figsize=(12.2, 4.6))
    ax = fig.add_subplot(111)

    cmap = plt.get_cmap("viridis")
    vv = v.copy()
    vv[~np.isfinite(vv)] = 0.0
    vmax = float(np.max(vv)) if len(vv) else 1.0
    vmax = max(vmax, 1e-12)
    colors = [cmap(0.25 + 0.70 * (float(val) / vmax)) for val in vv]

    ax.bar(np.arange(len(v)), v, color=colors, edgecolor="white", linewidth=0.7, label="(r - mean)^2")
    ax.set_title(title, pad=10)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Squared Deviation")

    ax.set_xticks(np.arange(len(stage_df)))
    ax.set_xticklabels([str(pd.Timestamp(d).date()) for d in stage_df["stage_start"]], rotation=45, ha="right")

    bottom = _legend_below(ax, ncol=1)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig

def plot_exec_failures(bt_results: dict, title="Execution Failures (Daily)"):
    import matplotlib.dates as mdates
    _set_presentation_style()
    exec_df = bt_results.get("execution", {}).get("exec_daily", pd.DataFrame())
    if exec_df is None or exec_df.empty:
        raise ValueError("No exec_daily found.")

    df = exec_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    fig = plt.figure(figsize=(12.2, 4.6))
    ax = fig.add_subplot(111)

    ax.plot(df.index, df["n_order_failed"].values, label="Order Failures (count)")
    ax.plot(df.index, df["n_sell_fail_events"].values, label="Sell Fail Events (count)", linestyle="--")
    ax.fill_between(df.index, df["n_order_failed"].values, 0.0, alpha=0.08)

    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")

    loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    bottom = _legend_below(ax, ncol=2)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig

def plot_factor_exposure(bt_results: dict, top_k=12, title="Portfolio Factor Exposure (Top factors)"):
    import matplotlib.dates as mdates
    _set_presentation_style()
    fs = bt_results.get("factor_summary", None)
    if fs is None:
        raise ValueError("No factor_summary found. Provide barra_expo.")

    port_expo = fs["portfolio_factor_exposure"]
    mean_abs = port_expo.abs().mean().sort_values(ascending=False)
    cols = list(mean_abs.head(top_k).index)
    X = port_expo[cols].copy()

    fig = plt.figure(figsize=(12.2, 5.4))
    ax = fig.add_subplot(111)

    arr = X.T.values
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 1e-12)

    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, pad=10)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)

    idx = X.index
    if len(idx) >= 2:
        ax.set_xticks([0, len(idx)//2, max(len(idx)-1, 0)])
        ax.set_xticklabels([str(idx[0].date()), str(idx[len(idx)//2].date()), str(idx[-1].date())])
    else:
        ax.set_xticks([0])
        ax.set_xticklabels([str(idx[0].date())])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Exposure")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig

def plot_factor_contribution(bt_results: dict, top_k=10, title="Factor Contribution (Cumulative Return)"):
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    _set_presentation_style()
    fa = bt_results.get("factor_attribution", None)
    if fa is None:
        raise ValueError("No factor_attribution found. Provide factor_returns + barra_expo.")

    contrib = fa["daily_factor_contrib_return"].copy()
    mean_abs = contrib.abs().mean().sort_values(ascending=False)
    cols = list(mean_abs.head(top_k).index)
    c = contrib[cols].fillna(0.0).cumsum()

    fig = plt.figure(figsize=(12.2, 5.1))
    ax = fig.add_subplot(111)
    for col in cols:
        ax.plot(c.index, c[col].values, label=str(col))

    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return Contribution")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    bottom = _legend_below(ax, ncol=3)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig


# ============================================================
# 8) Printing (professional summary)
# ============================================================

def print_backtest_summary(bt_bank: dict, bt_results: dict | None = None):
    o = bt_bank["overall"]
    print("=== Backtest Summary ===")
    print(f"Strict t-1 (no look-ahead): {bt_bank.get('strict_tminus1', True)}")
    print(f"Start NAV          : {o['start_nav']:.2f}")
    print(f"End NAV            : {o['end_nav']:.2f}")
    print(f"Total Return       : {o['total_return']:.2%}")
    print(f"Annualized Return  : {o['ann_return']:.2%}")
    print(f"Annualized Vol     : {o['ann_vol']:.2%}")
    print(f"Sharpe             : {o['sharpe']:.2f}")
    print(f"Max Drawdown       : {o['max_drawdown']:.2%}")

    if "index_total_return" in o:
        print("=== Benchmark Comparison ===")
        print(f"Index Total Return : {o['index_total_return']:.2%}")
        print(f"Hedged Total Return: {o['hedged_total_return']:.2%}")
        print(f"Avg Daily Excess   : {o['avg_daily_excess']:.4%}")

    ls = bt_bank["latest_status"]
    print("\n=== Latest Status ===")
    print("date:", ls["date"], "| region:", ls["region"], "| active_stage:", ls["active_stage"])
    if ls.get("selected_factors", None) is not None:
        print("selected_factors:", ls["selected_factors"])

    if bt_results is not None and bt_results:
        st = bt_results["stage_return_test"]
        dt = bt_results["daily_return_test"]
        print("\n=== Significance Tests (H1: mean > 0) ===")
        print(f"[Stage total return] n={st['n']}  t={st['t_stat']:.3f}  p(one-sided)={st['p_one_sided']:.4g}  mean={st['mean']:.6g}  std={st['std']:.6g}")
        print(f"[Daily return | HAC] n={dt['n']}  lags={dt.get('lags', np.nan)}  t={dt['t_stat']:.3f}  p(one-sided)={dt['p_one_sided']:.4g}  mean={dt['mean']:.6g}  std={dt['std']:.6g}")

        ex = bt_results.get("execution", {}).get("exec_daily", pd.DataFrame())
        if ex is not None and (not ex.empty):
            print("\n=== Execution Overview ===")
            print("Total order failed:", int(ex["n_order_failed"].sum()))
            print("Total sell fail events:", int(ex["n_sell_fail_events"].sum()))
            print("Total realized pnl (SELL days):", float(ex["realized_pnl_sell"].sum()))

        lots = bt_results.get("execution", {}).get("lots", pd.DataFrame())
        if lots is not None and (not lots.empty):
            d = lots["delay_days"].dropna()
            if len(d):
                print("\n=== Lot Delayed-Sell Stats ===")
                print("Avg delay days:", float(d.mean()))
                print("Max delay days:", float(d.max()))
                print("Delayed ratio:", float((lots["exit_reason"]=="DELAY").mean()))


# ============================================================
# 9) Best/Worst stage report
# ============================================================

def _pick_best_worst_stage(bt_bank: dict):
    by_stage = bt_bank.get("by_stage", {})
    if not by_stage:
        raise ValueError("bt_bank['by_stage'] is empty.")
    items = list(by_stage.items())
    items_sorted = sorted(items, key=lambda kv: float(kv[1]["metrics"]["total_return"]))
    worst = items_sorted[0][0]
    best  = items_sorted[-1][0]
    return best, worst

def _build_window_bt_from_stage(bt_bank: dict, stage_key: pd.Timestamp):
    st = bt_bank["by_stage"][pd.Timestamp(stage_key)]
    hs, he = st["holdout_window"]

    nav_close = bt_bank["nav"].loc[hs:he].copy()
    nav_open = st.get("nav_open_holdout", None)
    if nav_open is not None and len(nav_open.loc[hs:he]):
        start_nav = float(nav_open.loc[hs:he].iloc[0])
    else:
        start_nav = float(nav_close.iloc[0])

    nav_report = nav_close.copy()
    nav_report.iloc[0] = start_nav

    win = dict(
        stage_start=pd.Timestamp(stage_key),
        holdout_start=pd.Timestamp(hs),
        holdout_end=pd.Timestamp(he),
        nav=nav_close,
        nav_report=nav_report,
        overall=dict(
            start_nav=start_nav,
            end_nav=float(nav_close.iloc[-1]),
            total_return=float(nav_close.iloc[-1] / start_nav - 1.0),
            max_drawdown=_max_drawdown(nav_close),
        )
    )
    return win

def print_stage_window_report(win: dict, spy_close: pd.Series | None = None):
    print("\n==============================")
    print(f"=== Window Report | stage_start={win['stage_start'].date()} ===")
    print(f"Holdout: {win['holdout_start'].date()} .. {win['holdout_end'].date()}")
    print(f"Start NAV: {win['overall']['start_nav']:.2f}")
    print(f"End NAV  : {win['overall']['end_nav']:.2f}")
    print(f"Total RT : {win['overall']['total_return']:.2%}")
    print(f"Max DD   : {win['overall']['max_drawdown']:.2%}")
    if spy_close is not None:
        spy = _to_datetime_index(spy_close.to_frame("SPY"))["SPY"].reindex(win["nav"].index).ffill()
        idx_rt = float(spy.iloc[-1] / spy.iloc[0] - 1.0) if len(spy) else np.nan
        print(f"Index RT : {idx_rt:.2%}")

def plot_nav_window(win: dict, spy_close: pd.Series | None = None, title="Window NAV vs Benchmark"):
    import matplotlib.dates as mdates
    _set_presentation_style()
    nav = win["nav"].astype(float)

    fig = plt.figure(figsize=(12.2, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(nav.index, nav.values, label="Strategy NAV (Close)")
    ax.fill_between(nav.index, nav.values, np.nanmin(nav.values), alpha=0.08)

    if spy_close is not None:
        spy = _to_datetime_index(spy_close.to_frame("SPY"))["SPY"].reindex(nav.index).ffill()
        spy_nav = spy / spy.iloc[0] * nav.iloc[0]
        ax.plot(spy_nav.index, spy_nav.values, label="SPY (scaled)", linestyle="--")

    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")

    loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    bottom = _legend_below(ax, ncol=2)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig

def plot_drawdown_window(win: dict, title="Window Drawdown"):
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    _set_presentation_style()
    nav = win["nav"].astype(float)
    dd = nav / nav.cummax() - 1.0

    fig = plt.figure(figsize=(12.2, 4.1))
    ax = fig.add_subplot(111)
    ax.plot(dd.index, dd.values, label="Drawdown")
    ax.fill_between(dd.index, dd.values, 0.0, alpha=0.12)

    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    bottom = _legend_below(ax, ncol=1)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig

def plot_window_daily_return_hist(win: dict, title="Window Daily Return Distribution"):
    import matplotlib.ticker as mticker
    _set_presentation_style()
    r = win["nav_report"].pct_change().dropna()
    x = r.values
    n = len(x)
    if n == 0:
        raise ValueError("No daily returns in window.")
    bins = int(max(25, min(80, np.sqrt(n) * 8)))

    fig = plt.figure(figsize=(9.8, 4.7))
    ax = fig.add_subplot(111)
    ax.hist(x, bins=bins, density=True, alpha=0.85, edgecolor="white", linewidth=0.6, label="Daily Returns (density)")
    ax.axvline(0.0, linewidth=1.2, alpha=0.75, label="0%")
    ax.axvline(float(np.mean(x)), linewidth=1.2, alpha=0.75, linestyle="--", label="Mean")

    ax.set_title(title, pad=10)
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    bottom = _legend_below(ax, ncol=3)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig


# ============================================================
# 11) CLASS WRAPPER (packaged as a class; original functions/calls unchanged)
# ============================================================

class BacktestFramework:
    """
    Industry-grade packaging:
    - Keep all original functions at module level (compatible with your existing usage)
    - Also expose them on the class as staticmethods (namespace management)
    - Keep CFG/constants on class attributes
    """
    CFG = CFG

    # constants
    ORD_OK = ORD_OK
    ORD_INELIGIBLE = ORD_INELIGIBLE
    ORD_NO_PRICE = ORD_NO_PRICE
    ORD_TOO_SMALL = ORD_TOO_SMALL
    ORD_NO_CASH = ORD_NO_CASH
    LOT_EXIT_NORMAL = LOT_EXIT_NORMAL
    LOT_EXIT_DELAY = LOT_EXIT_DELAY

    # utils
    _to_datetime_index = staticmethod(_to_datetime_index)
    _annualize_from_nav = staticmethod(_annualize_from_nav)
    _roll_h_stats = staticmethod(_roll_h_stats)
    _max_drawdown = staticmethod(_max_drawdown)
    _infer_stage_blocks = staticmethod(_infer_stage_blocks)
    _infer_global_backtest_window_from_signal_bank = staticmethod(_infer_global_backtest_window_from_signal_bank)
    _pick_book_frac = staticmethod(_pick_book_frac)
    _align_close_volume = staticmethod(_align_close_volume)

    # sample space
    build_sample_space_amo_listing = staticmethod(build_sample_space_amo_listing)

    # numba kernels (keep as attributes for completeness)
    _count_eligible_finite = staticmethod(_count_eligible_finite)
    _cs_zscore_one_day = staticmethod(_cs_zscore_one_day)
    _softmax_topn_weights = staticmethod(_softmax_topn_weights)
    _build_w_trade_stage = staticmethod(_build_w_trade_stage)
    _bt_stage_5d_lots = staticmethod(_bt_stage_5d_lots)

    # main run + analytics
    _compute_latest_status = staticmethod(_compute_latest_status)
    run_backtest_ga_wf_softmax_5d = staticmethod(run_backtest_ga_wf_softmax_5d)
    _newey_west_mean_test = staticmethod(_newey_west_mean_test)
    build_backtest_results = staticmethod(build_backtest_results)

    # plotting
    _set_presentation_style = staticmethod(_set_presentation_style)
    _legend_below = staticmethod(_legend_below)
    plot_nav = staticmethod(plot_nav)
    plot_drawdown = staticmethod(plot_drawdown)
    plot_stage_returns = staticmethod(plot_stage_returns)
    plot_stage_return_hist = staticmethod(plot_stage_return_hist)
    plot_stage_variance = staticmethod(plot_stage_variance)
    plot_exec_failures = staticmethod(plot_exec_failures)
    plot_factor_exposure = staticmethod(plot_factor_exposure)
    plot_factor_contribution = staticmethod(plot_factor_contribution)

    # printing + stage window
    print_backtest_summary = staticmethod(print_backtest_summary)
    _pick_best_worst_stage = staticmethod(_pick_best_worst_stage)
    _build_window_bt_from_stage = staticmethod(_build_window_bt_from_stage)
    print_stage_window_report = staticmethod(print_stage_window_report)
    plot_nav_window = staticmethod(plot_nav_window)
    plot_drawdown_window = staticmethod(plot_drawdown_window)
    plot_window_daily_return_hist = staticmethod(plot_window_daily_return_hist)


# ============================================================
# 10) USAGE (copy/paste) -- keep your original usage structure
# ============================================================

sample_space = build_sample_space_amo_listing(
    close_df=close_price,
    volume_df=volume,
    amo_window=CFG["sample_space"]["amo_window"],
    amo_threshold=CFG["sample_space"]["amo_threshold"],
    min_listed_days=CFG["sample_space"]["min_listed_days"],
    require_price_positive=CFG["sample_space"]["require_price_positive"],
    delay=1
)

bt_bank = run_backtest_ga_wf_softmax_5d(
    close_df=close_price,
    signal_bank=signal_bank,
    start=CFG["wrapper"]["start"],
    end=CFG["wrapper"]["end"],
    volume_df=volume,
    spy_close=spy_close_price,
    budget_frac_by_stage=budget_frac_by_stage,
    sample_space=sample_space,

    amo_window=CFG["sample_space"]["amo_window"],
    amo_threshold=CFG["sample_space"]["amo_threshold"],
    min_listed_days=CFG["sample_space"]["min_listed_days"],

    min_cs_n=CFG["wrapper"]["min_cs_n"],
    strict_tminus1=CFG["wrapper"]["strict_tminus1"],
    liquidation_buffer_days=CFG["wrapper"]["liquidation_buffer_days"],

    barra_expo=None,
    factor_returns=None,

    params=CFG["engine"],
)

bt_results = build_backtest_results(
    bt_bank,
    trading_days=CFG["reporting"]["trading_days"],
)
bt_bank["backtest_results"] = bt_results

print_backtest_summary(bt_bank, bt_results)

fig1 = plot_nav(bt_bank, spy_close=spy_close_price)
fig2 = plot_drawdown(bt_results)
fig3 = plot_stage_returns(bt_results)
fig4 = plot_stage_return_hist(bt_results)
fig5 = plot_stage_variance(bt_results)
fig6 = plot_exec_failures(bt_results)

best_stage, worst_stage = _pick_best_worst_stage(bt_bank)

best_win = _build_window_bt_from_stage(bt_bank, best_stage)
worst_win = _build_window_bt_from_stage(bt_bank, worst_stage)

print_stage_window_report(best_win, spy_close=spy_close_price)
bw1 = plot_nav_window(best_win, spy_close=spy_close_price, title=f"BEST Window NAV | stage={best_stage.date()}")
bw2 = plot_drawdown_window(best_win, title=f"BEST Window Drawdown | stage={best_stage.date()}")
bw3 = plot_window_daily_return_hist(best_win, title=f"BEST Window Daily Return Dist | stage={best_stage.date()}")

print_stage_window_report(worst_win, spy_close=spy_close_price)
ww1 = plot_nav_window(worst_win, spy_close=spy_close_price, title=f"WORST Window NAV | stage={worst_stage.date()}")
ww2 = plot_drawdown_window(worst_win, title=f"WORST Window Drawdown | stage={worst_stage.date()}")
ww3 = plot_window_daily_return_hist(worst_win, title=f"WORST Window Daily Return Dist | stage={worst_stage.date()}")

plt.show()

bt_bank["blotter"]["orders"]
bt_bank["blotter"]["trades"]
bt_bank["blotter"]["sell_fail"]
bt_bank["blotter"]["lots"]
bt_bank["exec_daily"]
bt_bank["daily"]
bt_bank["backtest_results"]["stage_table"]
bt_bank["inferred_window"]









