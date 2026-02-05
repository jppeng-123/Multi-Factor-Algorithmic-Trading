''' Strategy BackTesting '''

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
