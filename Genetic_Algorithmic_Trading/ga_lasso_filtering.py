''' Lasso Filtering'''

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





