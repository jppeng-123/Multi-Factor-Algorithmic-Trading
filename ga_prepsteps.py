'''GA_Signal_Prepsteps'''


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
# Example usage (与你原来的调用节奏一致，只是换成类)
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




