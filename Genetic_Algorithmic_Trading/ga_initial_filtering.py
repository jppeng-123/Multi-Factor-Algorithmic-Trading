
'''Initial Filtering'''

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
# 你后面所有调用、以及上面那段 features/logret 产出都能无缝接上
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




