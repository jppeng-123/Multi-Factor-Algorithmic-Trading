""" GA_GA_Algorithm  """

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from numba import njit, prange

# Optional imports already used in your broader environment / project.
# Kept here to avoid breaking your existing dependency layout.
from scipy import stats as sp_stats  # noqa: F401
from scipy.special import gammaln  # noqa: F401
from joblib import Parallel, delayed  # noqa: F401

from sklearn.base import clone  # noqa: F401
from sklearn.linear_model import Lasso, Ridge  # noqa: F401
from sklearn.model_selection import GroupKFold  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401

import matplotlib as mpl  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401
from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401

import yfinance as yf  # noqa: F401
from arch import arch_model  # noqa: F401
from massive import RESTClient  # noqa: F401


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# =============================================================================
# Public exports
# =============================================================================

__all__ = [
    "GPNode",
    "GPFitness",
    "GASliceResult",
    "GA_LAYER_WF_CONNECTED",
    "ga_engine",
    "ga_to_date_index",
    "ga_spy_aggs_to_close_series",
    "ga_time_split_with_purge",
    "neutralize_signal_barra_style",
    "ts_mean_strict_np",
    "ts_std_strict_np",
    "safe_div",
    "gp_clone",
    "gp_size",
    "gp_depth",
    "gp_to_string",
    "gp_key",
    "gp_complexity",
    "gp_random_terminal",
    "gp_random_tree",
    "gp_collect_paths",
    "gp_replace_at",
    "gp_crossover",
    "gp_mutate",
    "gp_eval",
    "gp_score_on_index",
    "build_terminal_bank_for_index",
    "compute_rolling_beta_matrix_strict",
    "ga_evolve_on_slice",
    "run_walkforward_ga_layer",
]


# =============================================================================
# Runtime flags
# =============================================================================

_NUMBA_AVAILABLE = True


# =============================================================================
# 0) GA Tunables (safe professional defaults)
# =============================================================================

GA_SEED_BASE = 7

GA_VAL_RATIO = 0.20
GA_PURGE_GAP_DAYS = None

GA_POP_SIZE = 384
GA_N_GEN = 180
GA_TOURNAMENT_K = 5
GA_ELITE_FRAC = 0.06
GA_CROSSOVER_P = 0.55
GA_MUTATION_P = 0.35
GA_REPRO_P = 0.10

GA_MAX_DEPTH = 3
GA_MIN_DEPTH = 1

GA_P_CONST = 0.06
GA_CONST_RANGE = (-1.5, 1.5)

GA_ROLL_WINDOWS = [5, 10, 20, 60]

GA_TS_MIN_FRAC = 0.80

GA_LAMBDA_COMPLEXITY = 0.022
GA_ROLLING_COMPLEXITY_BONUS = 2.0

GA_LAMBDA_GAP = 0.15
GA_SIGN_FLIP_PENALTY = 1.0
GA_ELITE_UNIQUE = True

GA_EARLYSTOP_PATIENCE = 25
GA_EARLYSTOP_MIN_DELTA = 1e-3

GA_VAL_TOPM = 16

GA_NEUTRALIZE_RIDGE = 1e-4

try:
    _MIN_PORT = int(MIN_CS_ASSETS_FOR_PORT)
except Exception:
    _MIN_PORT = 50

GA_MIN_CS_N = max(50, _MIN_PORT)
GA_MIN_T_EFF = 60

GA_TOPN_PER_SLICE = 10
GA_TOPN_POOL_MAX = 300
GA_TOPN_MIN_VAL_T = -np.inf

GA_ASSERT_ALIGNMENT = True

GA_CS_STANDARDIZE = CS_STANDARDIZE
GA_IC_METHOD = IC_METHOD


def _validate_ga_probabilities() -> None:
    total = float(GA_CROSSOVER_P + GA_MUTATION_P + GA_REPRO_P)
    if not np.isclose(total, 1.0, atol=1e-12):
        raise ValueError(
            f"GA probabilities must sum to 1.0, got {total:.12f} "
            f"(crossover={GA_CROSSOVER_P}, mutation={GA_MUTATION_P}, repro={GA_REPRO_P})"
        )


_validate_ga_probabilities()


# =============================================================================
# 1) Date / index utilities
# =============================================================================

def ga_to_date_index(idx: pd.Index) -> pd.Index:
    dt = pd.to_datetime(idx)
    return pd.Index(dt.date, name="date")


def ga_spy_aggs_to_close_series(
    spy_aggs: list,
    target_index: pd.Index,
    tz: str = "America/New_York",
) -> pd.Series:
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


# =============================================================================
# 2) Strict time split with purge gap (inner GA split)
# =============================================================================

def ga_time_split_with_purge(
    idx: pd.Index,
    val_ratio: float,
    purge_gap: int,
) -> Tuple[pd.Index, pd.Index]:
    idx = pd.DatetimeIndex(idx)
    n = len(idx)
    if n < 10:
        return idx, idx[:0]

    purge_gap = int(max(0, purge_gap))

    cut = int(np.floor(n * (1.0 - float(val_ratio))))
    cut = max(2, min(cut, n - 2))

    tr_end = max(0, cut - purge_gap)
    va_start = min(n, cut + purge_gap)

    tr = idx[:tr_end]
    va = idx[va_start:]
    return pd.Index(tr), pd.Index(va)


# =============================================================================
# 3) Barra-style daily cross-sectional neutralization
# =============================================================================

_IND_DUMMY_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, Dict[int, int]]] = {}


def _build_industry_dummies(
    ind_codes: np.ndarray,
    max_groups: int = 40,
) -> Tuple[np.ndarray, Dict[int, int]]:
    ind_codes = np.asarray(ind_codes).astype(int)
    valid = ind_codes[ind_codes >= 0]
    if valid.size == 0:
        return np.zeros((ind_codes.size, 0), dtype=np.float64), {}

    vc = pd.Series(valid).value_counts()
    keep = vc.index[:max_groups].tolist()

    mapping = {int(code): j for j, code in enumerate(keep)}
    g = len(keep)
    d = np.zeros((ind_codes.size, g), dtype=np.float64)

    for i, c in enumerate(ind_codes):
        j = mapping.get(int(c), None)
        if j is not None:
            d[i, j] = 1.0

    return d, mapping


def _get_industry_dummies_cached(
    industry_codes: np.ndarray,
    max_groups: int = 40,
) -> Tuple[np.ndarray, Dict[int, int]]:
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
    s: np.ndarray,
    beta: np.ndarray,
    ln_mcap: np.ndarray,
    d2: np.ndarray,
    use_beta: bool,
    use_size: bool,
    use_ind: bool,
    ridge: float,
    min_cs_n: int,
) -> np.ndarray:
    t_dim, n_dim = s.shape
    out = np.empty((t_dim, n_dim), dtype=np.float64)
    out[:] = np.nan

    g2 = d2.shape[1] if use_ind else 0

    for t in range(t_dim):
        idxs = np.empty(n_dim, dtype=np.int64)
        k = 0

        for i in range(n_dim):
            y = s[t, i]
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

        p = 1 + (1 if use_beta else 0) + (1 if use_size else 0) + (g2 if (use_ind and g2 > 0) else 0)

        x = np.empty((k, p), dtype=np.float64)
        yy = np.empty(k, dtype=np.float64)

        for r in range(k):
            ii = idxs[r]
            yy[r] = s[t, ii]
            x[r, 0] = 1.0

        col = 1
        if use_beta:
            for r in range(k):
                ii = idxs[r]
                x[r, col] = beta[t, ii]
            col += 1

        if use_size:
            for r in range(k):
                ii = idxs[r]
                x[r, col] = ln_mcap[t, ii]
            col += 1

        if use_ind and g2 > 0:
            for g in range(g2):
                for r in range(k):
                    ii = idxs[r]
                    x[r, col + g] = d2[ii, g]
            col += g2

        xtx = x.T @ x
        for d in range(p):
            xtx[d, d] += ridge
        xty = x.T @ yy

        b = np.linalg.solve(xtx, xty)
        resid = yy - x @ b

        for r in range(k):
            out[t, idxs[r]] = resid[r]

    return out


def neutralize_signal_barra_style(
    s: np.ndarray,
    beta: Optional[np.ndarray] = None,
    ln_mcap: Optional[np.ndarray] = None,
    industry_codes: Optional[np.ndarray] = None,
    ridge: float = 1e-6,
    min_cs_n: int = 50,
    max_industries: int = 40,
) -> np.ndarray:
    t_dim, n_dim = s.shape
    out = np.full_like(s, np.nan, dtype=np.float64)

    use_beta = beta is not None
    use_size = ln_mcap is not None
    use_ind = industry_codes is not None

    if not (use_beta or use_size or use_ind):
        return s.copy()

    if use_ind:
        d, _ = _get_industry_dummies_cached(industry_codes, max_groups=max_industries)
        if d.shape[1] >= 2:
            d2 = d[:, 1:].astype(np.float64)
        else:
            d2 = np.zeros((n_dim, 0), dtype=np.float64)
        g2 = d2.shape[1]
    else:
        d2 = np.zeros((n_dim, 0), dtype=np.float64)
        g2 = 0

    if _NUMBA_AVAILABLE:
        try:
            beta_use = beta.astype(np.float64) if use_beta else np.zeros_like(s, dtype=np.float64)
            size_use = ln_mcap.astype(np.float64) if use_size else np.zeros_like(s, dtype=np.float64)
            return _neutralize_barra_numba(
                s.astype(np.float64),
                beta_use,
                size_use,
                d2,
                bool(use_beta),
                bool(use_size),
                bool(use_ind and (g2 > 0)),
                float(ridge),
                int(min_cs_n),
            )
        except Exception:
            logger.exception("Numba neutralization failed; falling back to numpy implementation.")

    for t in range(t_dim):
        y = s[t]
        m = np.isfinite(y)
        if use_beta:
            m = m & np.isfinite(beta[t])
        if use_size:
            m = m & np.isfinite(ln_mcap[t])

        idx = np.where(m)[0]
        k = idx.size
        if k < int(min_cs_n):
            continue

        cols = [np.ones(k, dtype=np.float64)]
        if use_beta:
            cols.append(beta[t, idx].astype(np.float64))
        if use_size:
            cols.append(ln_mcap[t, idx].astype(np.float64))
        if use_ind and g2 > 0:
            di = d2[idx, :]
            if di.shape[1] > 0:
                cols.append(di.astype(np.float64))

        x = np.column_stack(cols)
        yy = y[idx].astype(np.float64)

        p = x.shape[1]
        xtx = x.T @ x
        xtx.flat[:: p + 1] += float(ridge)
        xty = x.T @ yy

        try:
            b = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(xtx, xty, rcond=None)[0]

        resid = yy - x @ b
        out[t, idx] = resid

    return out


# =============================================================================
# 4) Rolling ops (robust)
# =============================================================================

def ts_mean_strict_np(x: np.ndarray, window: int) -> np.ndarray:
    t_dim, n_dim = x.shape
    w = int(window)
    out = np.full((t_dim, n_dim), np.nan, dtype=np.float64)

    if w <= 1:
        return x.copy().astype(np.float64)

    finite = np.isfinite(x).astype(np.int32)
    x0 = np.where(np.isfinite(x), x, 0.0)

    csum = np.cumsum(x0, axis=0)
    ccnt = np.cumsum(finite, axis=0)

    sum_w = csum[w - 1 :] - np.vstack([np.zeros((1, n_dim)), csum[:-w]])
    cnt_w = ccnt[w - 1 :] - np.vstack([np.zeros((1, n_dim), dtype=np.int32), ccnt[:-w]])

    min_req = int(np.ceil(w * float(GA_TS_MIN_FRAC)))
    min_req = max(1, min_req)

    denom = np.where(cnt_w > 0, cnt_w.astype(np.float64), np.nan)
    mean = sum_w / denom
    out[w - 1 :] = np.where(cnt_w >= min_req, mean, np.nan)
    return out


def ts_std_strict_np(x: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    t_dim, n_dim = x.shape
    w = int(window)
    out = np.full((t_dim, n_dim), np.nan, dtype=np.float64)

    if w <= 1:
        return np.zeros((t_dim, n_dim), dtype=np.float64)

    finite = np.isfinite(x).astype(np.int32)
    x0 = np.where(np.isfinite(x), x, 0.0)
    x02 = x0 * x0

    csum = np.cumsum(x0, axis=0)
    csum2 = np.cumsum(x02, axis=0)
    ccnt = np.cumsum(finite, axis=0)

    sum_w = csum[w - 1 :] - np.vstack([np.zeros((1, n_dim)), csum[:-w]])
    sum2_w = csum2[w - 1 :] - np.vstack([np.zeros((1, n_dim)), csum2[:-w]])
    cnt_w = ccnt[w - 1 :] - np.vstack([np.zeros((1, n_dim), dtype=np.int32), ccnt[:-w]])

    min_req = int(np.ceil(w * float(GA_TS_MIN_FRAC)))
    min_req = max(ddof + 2, min_req)

    cntf = cnt_w.astype(np.float64)
    denom = cntf - float(ddof)

    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sum2_w - (sum_w * sum_w) / np.where(cntf > 0, cntf, np.nan)) / np.where(denom > 0, denom, np.nan)

    var = np.where(np.isfinite(var) & (var >= 0.0), var, np.nan)

    good = (cnt_w >= min_req) & np.isfinite(var) & (var >= 0.0)
    sd = np.sqrt(np.where(good, var, np.nan))

    out[w - 1 :] = sd
    return out


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    b2 = np.where(np.isfinite(b) & (np.abs(b) > eps), b, np.nan)
    return a / b2


# =============================================================================
# 5) GP Tree representation
# =============================================================================

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


# =============================================================================
# 6) GP Random generation + genetic operators
# =============================================================================

_BIN_OPS = ["+", "-", "*", "/"]
_UNARY_OPS = ["neg", "abs", "tanh", "sign"]
_TS_OPS: List[str] = []


def _init_ts_ops() -> None:
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


def gp_collect_paths(n: GPNode, path: Tuple[int, ...] = ()) -> List[Tuple[Tuple[int, ...], GPNode]]:
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


def gp_crossover(
    rng: np.random.Generator,
    a: GPNode,
    b: GPNode,
    max_depth: int,
) -> Tuple[GPNode, GPNode]:
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


def gp_mutate(
    rng: np.random.Generator,
    n: GPNode,
    terminals: List[str],
    max_depth: int,
) -> GPNode:
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
            return gp_replace_at(n, path, GPNode("const", float(c2), ()))

        new_sub = GPNode("const", float(rng.uniform(*GA_CONST_RANGE)), ())
        out = gp_replace_at(n, path, new_sub)
        if gp_depth(out) > max_depth:
            return gp_clone(n)
        return out

    if sub.op == "term":
        t = terminals[int(rng.integers(0, len(terminals)))]
        return gp_replace_at(n, path, GPNode("term", t, ()))

    if sub.op in _BIN_OPS:
        op = _BIN_OPS[int(rng.integers(0, len(_BIN_OPS)))]
        return gp_replace_at(n, path, GPNode(op, None, sub.kids))

    if sub.op in _UNARY_OPS:
        op = _UNARY_OPS[int(rng.integers(0, len(_UNARY_OPS)))]
        return gp_replace_at(n, path, GPNode(op, None, sub.kids))

    if isinstance(sub.op, str) and (sub.op.startswith("ts_mean_") or sub.op.startswith("ts_std_")):
        op = _TS_OPS[int(rng.integers(0, len(_TS_OPS)))]
        return gp_replace_at(n, path, GPNode(op, None, sub.kids))

    return gp_clone(n)


# =============================================================================
# 7) GP evaluation on a slice (vectorized) with caching
# =============================================================================

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
        a = gp_eval(n.kids[0], terminal_bank, cache)
        if n.op == "neg":
            out = -a
        elif n.op == "abs":
            out = np.abs(a)
        elif n.op == "tanh":
            out = np.tanh(a)
        elif n.op == "sign":
            out = np.sign(a)
        else:
            out = a
        cache[key] = out
        return out

    if isinstance(n.op, str) and n.op.startswith("ts_mean_"):
        w = int(n.op.split("_")[-1])
        a = gp_eval(n.kids[0], terminal_bank, cache)
        out = ts_mean_strict_np(a, w)
        cache[key] = out
        return out

    if isinstance(n.op, str) and n.op.startswith("ts_std_"):
        w = int(n.op.split("_")[-1])
        a = gp_eval(n.kids[0], terminal_bank, cache)
        out = ts_std_strict_np(a, w, ddof=1)
        cache[key] = out
        return out

    if n.op in _BIN_OPS:
        a = gp_eval(n.kids[0], terminal_bank, cache)
        b = gp_eval(n.kids[1], terminal_bank, cache)

        if n.op == "+":
            out = a + b
        elif n.op == "-":
            out = a - b
        elif n.op == "*":
            out = a * b
        elif n.op == "/":
            out = safe_div(a, b)
        else:
            out = a

        cache[key] = out
        return out

    out = gp_eval(n.kids[0], terminal_bank, cache) if n.kids else next(iter(terminal_bank.values()))
    cache[key] = out
    return out


# =============================================================================
# 8) Fitness evaluation (IC -> NW tstat) + neutralization
# =============================================================================

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

    cache: Dict[str, np.ndarray] = {}
    s_raw_full = gp_eval(node, terminal_bank_full, cache)
    s_raw = s_raw_full[loc, :]

    if (risk_beta_full is not None) or (risk_lnmcap_full is not None) or (industry_codes is not None):
        b = risk_beta_full[loc, :] if risk_beta_full is not None else None
        s = risk_lnmcap_full[loc, :] if risk_lnmcap_full is not None else None
        s_raw = neutralize_signal_barra_style(
            s_raw,
            beta=b,
            ln_mcap=s,
            industry_codes=industry_codes,
            ridge=float(GA_NEUTRALIZE_RIDGE),
            min_cs_n=int(max(30, _MIN_PORT)),
            max_industries=40,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        s_std, _ = standardize_signal_np(
            s_raw,
            cs_standardize=GA_CS_STANDARDIZE,
            ic_method=GA_IC_METHOD,
            auto_robust_zpearson=AUTO_ROBUST_ZPEARSON,
            winsor_q=CS_WINSOR_Q,
            z_clip=Z_CLIP,
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

    y = y_all[pos_sub, :]

    finite_pair = np.isfinite(s_std) & np.isfinite(y)
    cs_n = np.sum(finite_pair, axis=1).astype(np.int32)
    good_day = cs_n >= int(GA_MIN_CS_N)

    n_eff = int(np.sum(good_day))
    min_t_eff = int(GA_MIN_T_EFF if (min_t_eff_override is None) else int(min_t_eff_override))
    if n_eff < int(min_t_eff):
        return GPFitness(-np.inf, np.nan, np.nan, np.nan, int(n_eff), float(gp_complexity(node)), gp_key(node))

    ic = daily_ic_np(s_std, y, GA_IC_METHOD)
    ic = np.where(good_day, ic, np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sm = ic_summary_np(ic, PRED_HORIZON)

    t_hac = sm.get("t_hac_primary", np.nan)
    ic_mean = sm.get("IC_mean", np.nan)
    ic_ir = sm.get("IC_IR", np.nan)
    n_obs = int(sm.get("N", 0))

    comp = float(gp_complexity(node))
    if (not np.isfinite(t_hac)) or (n_obs <= 0):
        obj = -np.inf
    else:
        obj = float(t_hac) - float(lambda_complexity) * comp

    return GPFitness(
        obj=float(obj),
        t_hac=float(t_hac) if np.isfinite(t_hac) else np.nan,
        ic_mean=float(ic_mean) if np.isfinite(ic_mean) else np.nan,
        ic_ir=float(ic_ir) if np.isfinite(ic_ir) else np.nan,
        N=int(n_obs),
        complexity=float(comp),
        formula=gp_key(node),
    )


# =============================================================================
# 9) GA evolution within one slice
# =============================================================================

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
        logger.warning(
            "[GA] %s small inner split: train=%d val=%d purge=%d",
            pd.Timestamp(t0).date(),
            len(ga_tr),
            len(ga_va),
            purge_gap,
        )

    rng = np.random.default_rng(int(seed))

    pop: List[GPNode] = []
    for _ in range(int(GA_POP_SIZE)):
        depth = int(rng.integers(GA_MIN_DEPTH, GA_MAX_DEPTH + 1))
        pop.append(gp_random_tree(rng, terminals, max_depth=depth))

    best_val_obj = -np.inf
    best_node = None
    best_train_fit = None
    best_val_fit = None
    no_improve = 0

    gen_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []

    formula_to_node: Dict[str, GPNode] = {}

    for gen in range(int(GA_N_GEN)):
        fits = []
        for ind in pop:
            f = gp_score_on_index(
                ind,
                slice_idx,
                idx_all,
                pos_map,
                y_all,
                terminal_bank_full,
                ga_tr,
                risk_beta_full=risk_beta_full,
                risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes,
                lambda_complexity=GA_LAMBDA_COMPLEXITY,
            )
            fits.append(f)

        order = np.argsort([-(f.obj if np.isfinite(f.obj) else -np.inf) for f in fits])

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
            elites = [pop[int(order[0])]]
            elite_fits = [fits[int(order[0])]]

        for e in elites[: min(12, len(elites))]:
            k = gp_key(e)
            if k not in formula_to_node:
                formula_to_node[k] = gp_clone(e)

        topm = int(min(max(1, GA_VAL_TOPM), len(elites)))
        val_fits = []
        val_obj_adj = np.full(topm, -np.inf, dtype=np.float64)

        for j in range(topm):
            vf = gp_score_on_index(
                elites[j],
                slice_idx,
                idx_all,
                pos_map,
                y_all,
                terminal_bank_full,
                ga_va,
                risk_beta_full=risk_beta_full,
                risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes,
                lambda_complexity=GA_LAMBDA_COMPLEXITY,
            )
            val_fits.append(vf)
            val_obj_adj[j] = _val_obj_adjusted(elite_fits[j], vf)

        j_star = int(np.argmax(val_obj_adj))
        cand = elites[j_star]
        cand_train = elite_fits[j_star]
        val_best = val_fits[j_star]
        val_best_adj = float(val_obj_adj[j_star])

        train_best = elite_fits[0]

        improved = np.isfinite(val_best_adj) and (val_best_adj > best_val_obj + GA_EARLYSTOP_MIN_DELTA)
        if improved:
            best_val_obj = float(val_best_adj)
            best_node = gp_clone(cand)
            best_train_fit = cand_train
            best_val_fit = val_best
            no_improve = 0
        else:
            no_improve += 1

        gen_rows.append(
            {
                "gen": gen,
                "elite_train_obj": float(train_best.obj) if np.isfinite(train_best.obj) else np.nan,
                "elite_train_t_hac": train_best.t_hac,
                "elite_train_ic_mean": train_best.ic_mean,
                "elite_train_ic_ir": train_best.ic_ir,
                "elite_train_N": train_best.N,
                "elite_complexity": train_best.complexity,
                "elite_val_obj": float(val_best.obj) if np.isfinite(val_best.obj) else np.nan,
                "elite_val_obj_adj": float(val_best_adj) if np.isfinite(val_best_adj) else np.nan,
                "elite_val_t_hac": val_best.t_hac,
                "elite_val_ic_mean": val_best.ic_mean,
                "elite_val_ic_ir": val_best.ic_ir,
                "elite_val_N": val_best.N,
                "best_val_obj_sofar": float(best_val_obj) if np.isfinite(best_val_obj) else np.nan,
                "no_improve": int(no_improve),
                "elite_formula": train_best.formula,
            }
        )

        topk = min(10, len(elites))
        for j in range(topk):
            top_rows.append(
                {
                    "gen": gen,
                    "rank": j,
                    "train_obj": elite_fits[j].obj,
                    "train_t_hac": elite_fits[j].t_hac,
                    "train_ic_mean": elite_fits[j].ic_mean,
                    "complexity": elite_fits[j].complexity,
                    "formula": elite_fits[j].formula,
                }
            )

        if verbose and (gen % 10 == 0 or gen == GA_N_GEN - 1):
            logger.info(
                "[GA] %s gen=%03d | train_tNW=%.3f val_tNW=%.3f | val_obj_adj_best=%.4f | no_improve=%d",
                pd.Timestamp(t0).date(),
                gen,
                train_best.t_hac,
                val_best.t_hac,
                best_val_obj,
                no_improve,
            )

        if no_improve >= int(GA_EARLYSTOP_PATIENCE):
            if verbose:
                logger.info(
                    "[GA][earlystop] %s stop at gen=%d (patience=%d)",
                    pd.Timestamp(t0).date(),
                    gen,
                    GA_EARLYSTOP_PATIENCE,
                )
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

        pop = new_pop[: len(pop)]

    if best_node is None:
        logger.warning(
            "[GA][fallback] %s no valid val-best; fallback to last-gen best train",
            pd.Timestamp(t0).date(),
        )

        fits = []
        for ind in pop:
            f = gp_score_on_index(
                ind,
                slice_idx,
                idx_all,
                pos_map,
                y_all,
                terminal_bank_full,
                ga_tr,
                risk_beta_full=risk_beta_full,
                risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes,
                lambda_complexity=GA_LAMBDA_COMPLEXITY,
            )
            fits.append(f)

        order = np.argsort([-(f.obj if np.isfinite(f.obj) else -np.inf) for f in fits])
        best_node = gp_clone(pop[int(order[0])])
        best_train_fit = fits[int(order[0])]
        best_val_fit = gp_score_on_index(
            best_node,
            slice_idx,
            idx_all,
            pos_map,
            y_all,
            terminal_bank_full,
            ga_va,
            risk_beta_full=risk_beta_full,
            risk_lnmcap_full=risk_lnmcap_full,
            industry_codes=industry_codes,
            lambda_complexity=GA_LAMBDA_COMPLEXITY,
        )

    gen_log = pd.DataFrame(gen_rows).set_index("gen") if gen_rows else pd.DataFrame()
    top_bank = pd.DataFrame(top_rows) if top_rows else pd.DataFrame()

    top10_nodes: List[GPNode] = []
    top10_table = pd.DataFrame()

    if len(formula_to_node) > 0:
        if (top_bank is not None) and (len(top_bank) > 0):
            stab = top_bank.groupby("formula").agg(
                n_gen=("gen", "nunique"),
                n_rows=("gen", "size"),
                max_train_obj=("train_obj", "max"),
            )
        else:
            stab = pd.DataFrame(
                index=list(formula_to_node.keys()),
                data={"n_gen": 0, "n_rows": 0, "max_train_obj": np.nan},
            )

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
                node,
                slice_idx,
                idx_all,
                pos_map,
                y_all,
                terminal_bank_full,
                ga_tr,
                risk_beta_full=risk_beta_full,
                risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes,
                lambda_complexity=GA_LAMBDA_COMPLEXITY,
            )
            vaf = gp_score_on_index(
                node,
                slice_idx,
                idx_all,
                pos_map,
                y_all,
                terminal_bank_full,
                ga_va,
                risk_beta_full=risk_beta_full,
                risk_lnmcap_full=risk_lnmcap_full,
                industry_codes=industry_codes,
                lambda_complexity=GA_LAMBDA_COMPLEXITY,
            )

            if (not np.isfinite(vaf.t_hac)) or (float(vaf.t_hac) < float(GA_TOPN_MIN_VAL_T)):
                continue

            adj = _val_obj_adjusted(trf, vaf)
            cand_rows.append(
                {
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
                    "max_train_obj": (
                        float(stab.loc[formula, "max_train_obj"])
                        if (formula in stab.index and "max_train_obj" in stab.columns)
                        else np.nan
                    ),
                }
            )

        if len(cand_rows) > 0:
            dfc = pd.DataFrame(cand_rows)
            dfc = dfc.sort_values(
                ["val_obj_adj", "n_gen", "val_t_hac", "max_train_obj", "complexity"],
                ascending=[False, False, False, False, True],
            ).reset_index(drop=True)

            dfc["rank"] = np.arange(1, len(dfc) + 1)
            dfc = dfc.head(int(GA_TOPN_PER_SLICE)).copy()

            top10_table = dfc.set_index("rank")
            top10_nodes = [formula_to_node[f] for f in top10_table["formula"].tolist() if f in formula_to_node]

    if best_node is not None and len(top10_nodes) > 0:
        bf = gp_key(best_node)
        if bf not in set(top10_table["formula"].tolist()):
            if len(top10_nodes) < int(GA_TOPN_PER_SLICE):
                top10_nodes.append(gp_clone(best_node))
            else:
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
        top10_table=top10_table,
    )


# =============================================================================
# 10) Prepare terminals: build slice terminal bank
# =============================================================================

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
            raise ValueError("slice_idx contains dates not in idx_all (pos_map reindex has NaN).")
        pos_int = pos.astype(int)
        if np.min(pos_int) < int(delay):
            raise ValueError("min(pos) < delay; would cause negative indexing.")
    else:
        pos_int = pos.astype(int)

    bank: Dict[str, np.ndarray] = {}
    for f in terminals:
        x = features_np[f]
        s = x[pos_int - int(delay), :]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            s_std, _ = standardize_signal_np(
                s,
                cs_standardize=GA_CS_STANDARDIZE,
                ic_method=GA_IC_METHOD,
                auto_robust_zpearson=AUTO_ROBUST_ZPEARSON,
                winsor_q=CS_WINSOR_Q,
                z_clip=Z_CLIP,
            )
        bank[f] = s_std

    return bank


# =============================================================================
# 11) Rolling beta (strict) for Barra-style risk factor
# =============================================================================

@njit(cache=True, parallel=True)
def _rolling_beta_strict_numba(lr: np.ndarray, mm: np.ndarray, window: int) -> np.ndarray:
    t_dim, n_dim = lr.shape
    w = int(window)
    out = np.empty((t_dim, n_dim), dtype=np.float64)
    out[:] = np.nan

    if w <= 1:
        return out

    for j in prange(n_dim):
        for t in range(w - 1, t_dim):
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


# =============================================================================
# 12) Orchestrator: run GA on top of wf + wf_lasso, keep ALL artifacts
# =============================================================================

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
            lag_for_trade=DELAY,
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

    _sel_raw = wf.get("selected_by_slice", {}) or {}
    selected_by_date = {}
    if isinstance(_sel_raw, dict):
        for k, v in _sel_raw.items():
            try:
                selected_by_date[pd.Timestamp(k)] = v
            except Exception:
                continue

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
        pool = selected_by_date.get(pd.Timestamp(t0), None)

        if pool is None:
            for attr in ["selected_by_slice", "selected_factors", "selected", "features_selected"]:
                if hasattr(scr_obj, attr):
                    try:
                        pool = getattr(scr_obj, attr)
                        break
                    except Exception:
                        pass

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
    ga_rows: List[Dict[str, Any]] = []

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

        k_min = 3
        k_fill = 5

        selected_pool = _get_selected_pool(t0, scr)

        lres = lasso_by_date.get(pd.Timestamp(t0), None)

        terminals_source = "lasso_nonzero"
        terminals_raw = []

        if (lres is None) or (not hasattr(lres, "nonzero_factors")):
            terminals_source = "fallback_selected_by_slice"
            terminals_raw = selected_pool[:k_fill]
        else:
            terminals_raw = list(getattr(lres, "nonzero_factors", []))

        terminals = _sanitize_terminals(_unique_preserve(terminals_raw))

        if len(terminals) < k_min:
            if terminals_source == "lasso_nonzero":
                terminals_source = "lasso_nonzero+topup_selected_by_slice"
            for f in selected_pool:
                if f not in terminals:
                    terminals.append(f)
                if len(terminals) >= max(k_min, k_fill):
                    break

        if len(terminals) < k_min:
            terminals_source = terminals_source + "+lastresort_all_features"
            for f in all_feature_names:
                if f not in terminals:
                    terminals.append(f)
                if len(terminals) >= max(k_min, k_fill):
                    break

        if len(terminals) < 1:
            raise ValueError(f"[GA] {t0.date()} terminals empty even after fallback.")

        if verbose:
            if (lres is None) or (not hasattr(lres, "nonzero_factors")):
                logger.info(
                    "[GA][fallback] %s missing lasso slice result -> terminals=%d source=%s",
                    t0.date(),
                    len(terminals),
                    terminals_source,
                )
            elif len(terminals_raw) < k_min:
                logger.info(
                    "[GA][fallback] %s too few terminals after lasso: %d -> filled to %d source=%s",
                    t0.date(),
                    len(terminals_raw),
                    len(terminals),
                    terminals_source,
                )

        bank_train_full = build_terminal_bank_for_index(
            features_np=features_np,
            idx_all=idx_all,
            slice_idx=train_idx,
            terminals=terminals,
            delay=DELAY,
        )
        bank_holdout = build_terminal_bank_for_index(
            features_np=features_np,
            idx_all=idx_all,
            slice_idx=holdout_idx,
            terminals=terminals,
            delay=DELAY,
        )

        beta_train = beta_mat[pos_map.reindex(train_idx).values.astype(int), :] if beta_mat is not None else None
        beta_hold = beta_mat[pos_map.reindex(holdout_idx).values.astype(int), :] if beta_mat is not None else None

        size_train = ln_mcap_mat[pos_map.reindex(train_idx).values.astype(int), :] if ln_mcap_mat is not None else None
        size_hold = ln_mcap_mat[pos_map.reindex(holdout_idx).values.astype(int), :] if ln_mcap_mat is not None else None

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
            verbose=verbose,
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
            min_t_eff_override=hold_min_t,
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
                    min_t_eff_override=hold_min_t,
                )
                hrows.append(
                    {
                        "rank": int(rnk),
                        "formula": hf.formula,
                        "holdout_obj": float(hf.obj) if np.isfinite(hf.obj) else np.nan,
                        "holdout_t_hac": hf.t_hac,
                        "holdout_ic_mean": hf.ic_mean,
                        "holdout_ic_ir": hf.ic_ir,
                        "holdout_N": hf.N,
                    }
                )

            hdf = pd.DataFrame(hrows).set_index("rank") if len(hrows) > 0 else pd.DataFrame()
            if (res.top10_table is not None) and (len(res.top10_table) > 0) and (len(hdf) > 0):
                tmp = res.top10_table.copy()
                tmp = tmp.merge(
                    hdf.reset_index(drop=False)[["rank", "formula", "holdout_t_hac", "holdout_ic_mean", "holdout_ic_ir", "holdout_N"]],
                    how="left",
                    on=["rank", "formula"],
                )
                tmp = tmp.set_index("rank")
                res.top10_table = tmp

        ga_slice_results.append(res)

        ga_rows.append(
            {
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
                "top10_formulas": (
                    res.top10_table["formula"].tolist()
                    if (res.top10_table is not None and len(res.top10_table) > 0)
                    else []
                ),
                "terminals_source": terminals_source,
                "terminals_lasso_k": int(len(terminals_raw)) if terminals_source.startswith("lasso") else np.nan,
            }
        )

        if verbose:
            logger.info("==========================================================")
            logger.info("[GA WF CONNECTED] rebalance=%s", pd.Timestamp(t0).date())
            logger.info("terminals(K)=%d | max_depth=%d | roll=%s", len(terminals), GA_MAX_DEPTH, GA_ROLL_WINDOWS)
            logger.info(
                "inner split: ga_train=%d ga_val=%d purge=%s",
                len(res.ga_train_idx),
                len(res.ga_val_idx),
                (PRED_HORIZON - 1) + DELAY if GA_PURGE_GAP_DAYS is None else GA_PURGE_GAP_DAYS,
            )
            logger.info(
                "best: train tNW=%.3f | val tNW=%.3f | holdout tNW=%.3f",
                res.best_train.t_hac,
                res.best_val.t_hac,
                hold_fit.t_hac,
            )
            logger.info(
                "best formula: %s%s",
                res.best_formula[:160],
                "..." if len(res.best_formula) > 160 else "",
            )
            if (res.top10_table is not None) and (len(res.top10_table) > 0):
                logger.info(
                    "top10 ready: n=%d (unique, ranked by val_obj_adj, with holdout QA columns if available)",
                    len(res.top10_table),
                )
            logger.info("==========================================================")

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
            },
        },
    }


# =============================================================================
# 13) Class packaging
# =============================================================================

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


# =============================================================================
# 14) Optional main guard
# =============================================================================

if __name__ == "__main__":
    logger.info(
        "This module is intended to be imported into your research pipeline. "
        "Wire your client/data loading in a separate runner script or notebook."
    )
