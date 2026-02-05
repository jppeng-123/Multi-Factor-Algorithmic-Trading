import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew, kurtosis
import seaborn as sns
from pathlib import Path
from scipy.stats import t, norm
import statsmodels.api as sm
import networkx as nx
from itertools import combinations
import xgboost as xgb
from numpy.linalg import LinAlgError
import numba as nb
from typing import Dict
import inspect
import math, inspect, warnings
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle
from typing import Tuple
from joblib import Parallel, delayed
from arch import arch_model
from scipy.stats import mstats
from scipy.special import gamma as _gamma
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from numba import njit
import itertools
from pathlib import Path


# 1. 定义训练期起止
start_train = '2020-01-01'
end_train   = '2023-01-01'

# ─── 全局数据库连接 ───────────────────────────────────────────────────────────
CONN = pyodbc.connect("DSN=Trading;UID=sa;PWD=123456")

# ─── 前缀匹配函数 ─────────────────────────────────────────────────────────────
def _add_prefix(code: str) -> str | None:
    sse  = {"600","601","603","605","688","689"}
    szse = {"000","001","002","003","300","301"}
    code = str(code)
    if any(code.startswith(p) for p in sse):  return "sh"+code
    if any(code.startswith(p) for p in szse): return "sz"+code
    return None


# ─── 基础数据加载 ─────────────────────────────────────────────────────────────
def load_open_price(start: str = "2020-01-01",
                    end:   str = "2025-06-13") -> pd.DataFrame:
    sql = f"""
    SELECT symbol, trade_date, [open]
      FROM stock_a_daily
     WHERE trade_date BETWEEN '{start}' AND '{end}'
     ORDER BY trade_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['trade_date'])
    return df.pivot(index='trade_date', columns='symbol', values='open')

def load_high_price(start: str = "2020-01-01",
                    end:   str = "2025-06-13") -> pd.DataFrame:
    sql = f"""
    SELECT symbol, trade_date, high
      FROM stock_a_daily
     WHERE trade_date BETWEEN '{start}' AND '{end}'
     ORDER BY trade_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['trade_date'])
    return df.pivot(index='trade_date', columns='symbol', values='high')

def load_close_price(start: str = "2020-01-01",
                     end:   str = "2025-06-13") -> pd.DataFrame:
    sql = f"""
    SELECT symbol, trade_date, [close]
      FROM stock_a_daily
     WHERE trade_date BETWEEN '{start}' AND '{end}'
     ORDER BY trade_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['trade_date'])
    return df.pivot(index='trade_date', columns='symbol', values='close')

def load_low_price(start: str = "2020-01-01",
                   end:   str = "2025-06-13") -> pd.DataFrame:
    sql = f"""
    SELECT symbol, trade_date, low
      FROM stock_a_daily
     WHERE trade_date BETWEEN '{start}' AND '{end}'
     ORDER BY trade_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['trade_date'])
    return df.pivot(index='trade_date', columns='symbol', values='low')

def load_volume(start: str = "2020-01-01",
                end:   str = "2025-06-13") -> pd.DataFrame:
    sql = f"""
    SELECT symbol, trade_date, volume
      FROM stock_a_daily
     WHERE trade_date BETWEEN '{start}' AND '{end}'
     ORDER BY trade_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['trade_date'])
    return df.pivot(index='trade_date', columns='symbol', values='volume')

def load_amount(start: str = "2020-01-01",
                end:   str = "2025-06-13") -> pd.DataFrame:
    sql = f"""
    SELECT symbol, trade_date, amount
      FROM stock_a_daily
     WHERE trade_date BETWEEN '{start}' AND '{end}'
     ORDER BY trade_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['trade_date'])
    return df.pivot(index='trade_date', columns='symbol', values='amount')

def load_turnover(start: str = "2020-01-01",
                  end:   str = "2025-06-13") -> pd.DataFrame:
    sql = f"""
    SELECT symbol, trade_date, turnover
      FROM stock_a_daily
     WHERE trade_date BETWEEN '{start}' AND '{end}'
     ORDER BY trade_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['trade_date'])
    return df.pivot(index='trade_date', columns='symbol', values='turnover')


def load_market_cap(start: str = "2020-01-01",
                    end:   str = "2025-06-13") -> pd.DataFrame:
    """
    读取 stock_a_share_cap 表中的总市值 (total_mv)，
    返回一个 DataFrame，行索引为 data_date，列为 symbol，值为 total_mv。
    """
    sql = f"""
    SELECT symbol,
           data_date,
           total_mv
      FROM stock_a_share_cap
     WHERE data_date BETWEEN '{start}' AND '{end}'
     ORDER BY data_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['data_date'])
    return df.pivot(index='data_date', columns='symbol', values='total_mv')

def load_market_cap_float(start: str = "2020-01-01",
                          end:   str = "2025-06-13") -> pd.DataFrame:
    """
    读取 stock_a_share_cap 表中的流通市值 (circulating_mv)，
    返回一个 DataFrame，行索引为 data_date，列为 symbol，值为 circulating_mv。
    """
    sql = f"""
    SELECT symbol,
           data_date,
           circulating_mv
      FROM stock_a_share_cap
     WHERE data_date BETWEEN '{start}' AND '{end}'
     ORDER BY data_date, symbol
    """
    df = pd.read_sql(sql, CONN, parse_dates=['data_date'])
    return df.pivot(index='data_date', columns='symbol', values='circulating_mv')

def load_zz1000_index(start: str = "2020-01-01",
                      end:   str = "2025-06-13") -> pd.DataFrame:
    """
    读取 stock_index_zz1000 表中中证1000指数，
    返回 T×1 的 DataFrame，索引为 trade_date，列名 zz1000。
    """
    sql = f"""
    SELECT
      trade_date,
      index_value
    FROM stock_index_zz1000
    WHERE trade_date BETWEEN '{start}' AND '{end}'
    ORDER BY trade_date
    """
    df = pd.read_sql(sql, CONN, parse_dates=['trade_date'])
    # 重命名并设置索引
    df = df.rename(columns={'index_value': 'zz1000'})
    return df.set_index('trade_date')[['zz1000']]

# ─── 分红/送股/拆股加载 ────────────────────────────────────────────────────
def load_dividend(start: str = "2020-01-01",
                  end:   str = "2025-06-13",
                  close_price: pd.DataFrame = None) -> pd.DataFrame:
    sql = f"""
        SELECT symbol, ex_dividend_date, cash_dividend
          FROM stock_dividend_new
         WHERE ex_dividend_date BETWEEN '{start}' AND '{end}'
    """
    df = pd.read_sql(sql, CONN, parse_dates=['ex_dividend_date'])
    df['sym'] = df['symbol'].map(_add_prefix)
    df = df[df['sym'].notna() & df['ex_dividend_date'].isin(close_price.index)]
    mat = df.pivot(index='ex_dividend_date', columns='sym', values='cash_dividend')
    return mat.reindex(index=close_price.index, columns=close_price.columns)

def load_bonus(start: str = "2020-01-01",
               end:   str = "2025-06-13",
               close_price: pd.DataFrame = None) -> pd.DataFrame:
    sql = f"""
        SELECT symbol, ex_dividend_date, bonus_share
          FROM stock_dividend_new
         WHERE ex_dividend_date BETWEEN '{start}' AND '{end}'
    """
    df = pd.read_sql(sql, CONN, parse_dates=['ex_dividend_date'])
    df['sym'] = df['symbol'].map(_add_prefix)
    df = df[df['sym'].notna() & df['ex_dividend_date'].isin(close_price.index)]
    mat = df.pivot(index='ex_dividend_date', columns='sym', values='bonus_share')
    return mat.reindex(index=close_price.index, columns=close_price.columns)

def load_split(start: str = "2020-01-01",
               end:   str = "2025-06-13",
               close_price: pd.DataFrame = None) -> pd.DataFrame:
    sql = f"""
        SELECT symbol, ex_dividend_date, split_share
          FROM stock_dividend_new
         WHERE ex_dividend_date BETWEEN '{start}' AND '{end}'
    """
    df = pd.read_sql(sql, CONN, parse_dates=['ex_dividend_date'])
    df['sym'] = df['symbol'].map(_add_prefix)
    df = df[df['sym'].notna() & df['ex_dividend_date'].isin(close_price.index)]
    mat = df.pivot(index='ex_dividend_date', columns='sym', values='split_share')
    return mat.reindex(index=close_price.index, columns=close_price.columns)

# ─── 对数收益率计算 ─────────────────────────────────────────────────────────
def load_logret(close: pd.DataFrame,
                dividend: pd.DataFrame,
                bonus: pd.DataFrame,
                split: pd.DataFrame) -> pd.DataFrame:
    simple_lr  = np.log(close / close.shift(1))
    adj_factor = 1 + bonus.fillna(0) + split.fillna(0)
    adj_price  = close * adj_factor + dividend.fillna(0)
    ca_lr      = np.log(adj_price / close.shift(1))
    mask       = dividend.notna() | bonus.notna() | split.notna()
    lr         = simple_lr.copy()
    lr[mask]   = ca_lr[mask]
    return lr

# ─── 样本空间构建 ────────────────────────────────────────────────────────────
def load_amt_flag(amount: pd.DataFrame,
                  rolling_win: int = 10,
                  liquidity_thresh: float = 1e8,
                  max_missing: int = 3) -> pd.DataFrame:
    amt_prev = amount.shift(1)
    valid_cnt = amt_prev.notna().rolling(rolling_win).sum()
    avg_zero = amt_prev.fillna(0).rolling(rolling_win).sum() / rolling_win
    min_obs = rolling_win - max_missing
    avg = avg_zero.where(valid_cnt >= min_obs, np.nan)
    return (avg >= liquidity_thresh).astype(int)

def load_list_flag(close: pd.DataFrame,
                   listing_days: int = 60) -> pd.DataFrame:
    traded = close.notna().astype(int)
    days_listed = traded.cumsum()
    return (days_listed >= listing_days).astype(int)

def load_sample_space(amt_flag: pd.DataFrame,
                      list_flag: pd.DataFrame) -> pd.DataFrame:
    return (amt_flag & list_flag).astype(int)


# 辅助函数 ────────────────────────────────────────────────────────────────

# 基础函数

def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x)

def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)

# 截面排序
def rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)

# 时间延迟
def delay(x: pd.DataFrame, d: float) -> pd.DataFrame:
    return x.shift(int(np.floor(d)))

# 时间序列相关与协方差
def correlation(x: pd.DataFrame, y: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    return x.rolling(window=w, min_periods=w).corr(y)

def covariance(x: pd.DataFrame, y: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    return x.rolling(window=w, min_periods=w).cov(y)

# 缩放，使 sum(abs(x)) = a（默认 a = 1）
def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    abs_sum = x.abs().sum(axis=1).replace(0, np.nan)
    return x.mul(a, axis=0).div(abs_sum, axis=0)

# 差分
def delta(x: pd.DataFrame, d: float) -> pd.DataFrame:
    return x - x.shift(int(np.floor(d)))

# 有符号幂
def signedpower(x: pd.DataFrame, a: float) -> pd.DataFrame:
    return np.sign(x) * (x.abs() ** a)

# 线性衰减加权移动平均
def decay_linear(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    weights = np.arange(w, 0, -1, dtype=float)
    norm = weights.sum()
    def _lw(arr: np.ndarray) -> float:
        return float((weights * arr).sum() / norm)
    return x.rolling(window=w, min_periods=w).apply(_lw, raw=True)

# 行业中性化（分组去均值）
def indneutralize(x: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for t in x.index:
        vals = x.loc[t]
        grp  = g.loc[t]
        out.loc[t] = vals - vals.groupby(grp).transform("mean")
    return out

# 时序操作符
def ts_min(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    return x.rolling(window=w, min_periods=w).min()

def ts_max(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    return x.rolling(window=w, min_periods=w).max()

def ts_argmax(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    def _argmax(arr: np.ndarray) -> float:
        return float(np.argmax(arr))
    return x.rolling(window=w, min_periods=w).apply(_argmax, raw=True)

def ts_argmin(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    def _argmin(arr: np.ndarray) -> float:
        return float(np.argmin(arr))
    return x.rolling(window=w, min_periods=w).apply(_argmin, raw=True)

def ts_rank(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    def _rnk(arr: np.ndarray) -> float:
        return np.sum(arr <= arr[-1]) / len(arr)
    return x.rolling(window=w, min_periods=w).apply(_rnk, raw=True)

def ts_sum(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    return x.rolling(window=w, min_periods=w).sum()

def product(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    return x.rolling(window=w, min_periods=w).apply(np.prod, raw=True)

def stddev(x: pd.DataFrame, d: float) -> pd.DataFrame:
    w = int(np.floor(d))
    return x.rolling(window=w, min_periods=w).std()

#---------------------------------------------------------------------------

def load_stock_sector_table(conn: pyodbc.Connection) -> pd.DataFrame:
    sql = """
        SELECT 
            symbol,
            CAST(start_date AS DATE) AS start_date,
            industry_code
        FROM dbo.stock_sector
        WHERE start_date IS NOT NULL
    """
    df = pd.read_sql(sql, conn)
    df['start_date'] = pd.to_datetime(df['start_date'])
    return df


def build_industry_code_df(
    stock_sector_df: pd.DataFrame,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None
) -> pd.DataFrame:
    if start_date is None:
        start_date = stock_sector_df['start_date'].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = pd.to_datetime(datetime.today().date())
    else:
        end_date = pd.to_datetime(end_date)

    pivot_df = stock_sector_df.pivot(
        index='start_date',
        columns='symbol',
        values='industry_code'
    )

    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    reopened = pivot_df.reindex(full_dates).ffill()
    industry_code_df = reopened.fillna('UNK')
    industry_code_df.index.name = 'date'
    return industry_code_df

def load_mapping_df(
    mapping_file_path: str,
    code_col: str = '行业代码',
    level1_col: str = '一级行业名称',
    level2_col: str = '二级行业名称',
    level3_col: str = '三级行业名称'
) -> pd.DataFrame:
    mapping_full = pd.read_excel(mapping_file_path, dtype=str, engine='xlrd')
    mapping_full = mapping_full.rename(columns={
        code_col: 'industry_code',
        level1_col: 'level1',
        level2_col: 'level2',
        level3_col: 'level3'
    })
    mapping_trimmed = mapping_full[['industry_code', 'level1', 'level2', 'level3']].copy()
    mapping_trimmed = mapping_trimmed.set_index('industry_code')
    mapping_trimmed['level1'] = mapping_trimmed['level1'].fillna('UNK')
    mapping_trimmed['level2'] = mapping_trimmed['level2'].fillna('UNK')
    mapping_trimmed['level3'] = mapping_trimmed['level3'].fillna('UNK')
    return mapping_trimmed

def build_sector_and_industry_dfs(
    industry_code_df: pd.DataFrame,
    mapping_df: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    level1_map = mapping_df['level1']
    level2_map = mapping_df['level2']
    level3_map = mapping_df['level3']

    def map_to_level1(code: str) -> str:
        if code == 'UNK' or pd.isna(code):
            return 'UNK'
        return level1_map.get(code, 'UNK')

    def map_to_level2(code: str) -> str:
        if code == 'UNK' or pd.isna(code):
            return 'UNK'
        return level2_map.get(code, 'UNK')

    def map_to_level3(code: str) -> str:
        if code == 'UNK' or pd.isna(code):
            return 'UNK'
        return level3_map.get(code, 'UNK')

    sector_df = industry_code_df.applymap(map_to_level1)
    subindustry_df = industry_code_df.applymap(map_to_level2)
    industry_df = industry_code_df.applymap(map_to_level3)
    return sector_df, subindustry_df, industry_df

def align_factor_df_to_close(factor_df, close_df, default_value="UNK"):
    bare_codes = [col[2:] if isinstance(col, str) and len(col) > 2 else col for col in close_df.columns]
    factor_reindexed = factor_df.reindex(close_df.index, method="ffill")
    aligned = pd.DataFrame(index=close_df.index, columns=close_df.columns)
    for full_col, bare in zip(close_df.columns, bare_codes):
        if bare in factor_reindexed.columns:
            aligned[full_col] = factor_reindexed[bare]
        else:
            aligned[full_col] = default_value
    return aligned


# 一键切片函数
def sl(df): 
    return df.loc[start_train:end_train]

# ─── Alpha 1 ────────────────────────────────────────────────────────────────
def load_alpha1(close_shifted, logret_shifted, sample):
    r    = logret_shifted
    s20  = stddev(r, 20)
    base = s20.where(r < 0, close_shifted)
    p2   = signedpower(base, 2.0)
    k    = ts_argmax(p2, 5)
    alpha = rank(k) - 0.5
    return alpha.where(sample == 1)

# ─── Alpha 2 ────────────────────────────────────────────────────────────────
def load_alpha2(close_shifted, open_shifted, volume_shifted, sample):
    r1 = rank(delta(np.log(volume_shifted), 2))
    r2 = rank((close_shifted - open_shifted) / open_shifted)
    alpha = -correlation(r1, r2, 6)
    return alpha.where(sample == 1)

# ─── Alpha 3 ────────────────────────────────────────────────────────────────
def load_alpha3(open_shifted, volume_shifted, sample):
    alpha = -correlation(rank(open_shifted), rank(volume_shifted), 10)
    return alpha.where(sample == 1)

# ─── Alpha 4 ────────────────────────────────────────────────────────────────
def load_alpha4(low_shifted, sample):
    alpha = -ts_rank(rank(low_shifted), 9)
    return alpha.where(sample == 1)

# ─── Alpha 5 ────────────────────────────────────────────────────────────────
def load_alpha5(open_shifted, close_shifted, vwap_shifted, sample):
    part1 = rank(open_shifted - ts_sum(vwap_shifted, 10) / 10)
    part2 = -abs(rank(close_shifted - vwap_shifted))
    alpha = part1 * part2
    return alpha.where(sample == 1)

# ─── Alpha 6 ────────────────────────────────────────────────────────────────
def load_alpha6(open_shifted, volume_shifted, sample):
    alpha = -correlation(open_shifted, volume_shifted, 10)
    return alpha.where(sample == 1)

# ─── Alpha 7 ────────────────────────────────────────────────────────────────
def load_alpha7(close_shifted, adv20_shifted, volume_shifted, sample):
    d7   = delta(close_shifted, 7)
    core = -ts_rank(abs(d7), 60) * sign(d7)
    alpha = core.where(adv20_shifted < volume_shifted, -1.0)
    return alpha.where(sample == 1)

# ─── Alpha 8 ────────────────────────────────────────────────────────────────
def load_alpha8(open_shifted, logret_shifted, sample):
    term  = ts_sum(open_shifted, 5) * ts_sum(logret_shifted, 5)
    alpha = -rank(term - delay(term, 10))
    return alpha.where(sample == 1)

# ─── Alpha 9 ────────────────────────────────────────────────────────────────
def load_alpha9(close_shifted, sample):
    d1    = delta(close_shifted, 1)
    cond1 = ts_min(d1, 5) > 0
    cond2 = ts_max(d1, 5) < 0
    alpha = d1.where(cond1 | cond2, -d1)
    return alpha.where(sample == 1)

# ─── Alpha 10 ───────────────────────────────────────────────────────────────
def load_alpha10(close_shifted, sample):
    d1    = delta(close_shifted, 1)
    cond1 = ts_min(d1, 4) > 0
    cond2 = ts_max(d1, 4) < 0
    tmp   = d1.where(cond1 | cond2, -d1)
    alpha = rank(tmp)
    return alpha.where(sample == 1)

# ─── Alpha 11 ───────────────────────────────────────────────────────────────
def load_alpha11(close_shifted, vwap_shifted, volume_shifted, sample):
    diff  = vwap_shifted - close_shifted
    part  = rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))
    alpha = part * rank(delta(volume_shifted, 3))
    return alpha.where(sample == 1)

# ─── Alpha 12 ───────────────────────────────────────────────────────────────
def load_alpha12(close_shifted, volume_shifted, sample):
    alpha = sign(delta(volume_shifted, 1)) * (-delta(close_shifted, 1))
    return alpha.where(sample == 1)

# ─── Alpha 13 ───────────────────────────────────────────────────────────────
def load_alpha13(close_shifted, volume_shifted, sample):
    alpha = -rank(covariance(rank(close_shifted), rank(volume_shifted), 5))
    return alpha.where(sample == 1)

# ─── Alpha 14 ───────────────────────────────────────────────────────────────
def load_alpha14(open_shifted, volume_shifted, logret_shifted, sample):
    part1 = -rank(delta(logret_shifted, 3))
    alpha = part1 * correlation(open_shifted, volume_shifted, 10)
    return alpha.where(sample == 1)

# ─── Alpha 15 ───────────────────────────────────────────────────────────────
def load_alpha15(high_shifted, volume_shifted, sample):
    corr  = correlation(rank(high_shifted), rank(volume_shifted), 3)
    alpha = -ts_sum(rank(corr), 3)
    return alpha.where(sample == 1)

# ─── Alpha 16 ───────────────────────────────────────────────────────────────
def load_alpha16(high_shifted, volume_shifted, sample):
    alpha = -rank(covariance(rank(high_shifted), rank(volume_shifted), 5))
    return alpha.where(sample == 1)

# ─── Alpha 17 ───────────────────────────────────────────────────────────────
def load_alpha17(close_shifted, volume_shifted, adv20_shifted, sample):
    part1 = -rank(ts_rank(close_shifted, 10))
    part2 = rank(delta(delta(close_shifted, 1), 1))
    part3 = rank(ts_rank(volume_shifted / adv20_shifted, 5))
    alpha = part1 * part2 * part3
    return alpha.where(sample == 1)

# ─── Alpha 18 ───────────────────────────────────────────────────────────────
def load_alpha18(close_shifted, open_shifted, sample):
    term  = stddev(abs(close_shifted - open_shifted), 5) + (close_shifted - open_shifted)
    corr  = correlation(close_shifted, open_shifted, 10)
    alpha = -rank(term + corr)
    return alpha.where(sample == 1)

# ─── Alpha 19 ───────────────────────────────────────────────────────────────
def load_alpha19(close_shifted, logret_shifted, sample):
    expr  = (close_shifted - delay(close_shifted, 7)) + delta(close_shifted, 7)
    part1 = -sign(expr)
    part2 = 1 + rank(1 + ts_sum(logret_shifted, 250))
    alpha = part1 * part2
    return alpha.where(sample == 1)

# ─── Alpha 20 ───────────────────────────────────────────────────────────────
def load_alpha20(open_shifted, high_shifted, close_shifted, low_shifted, sample):
    r1 = rank(open_shifted - delay(high_shifted, 1))
    r2 = rank(open_shifted - delay(close_shifted, 1))
    r3 = rank(open_shifted - delay(low_shifted, 1))
    alpha = -r1 * r2 * r3
    return alpha.where(sample == 1)

# ─── Alpha 21 ───────────────────────────────────────────────────────────────
def load_alpha21(close_shifted, volume_shifted, adv20_shifted, sample):
    m8   = ts_sum(close_shifted, 8) / 8
    s8   = stddev(close_shifted, 8)
    m2   = ts_sum(close_shifted, 2) / 2
    vr   = volume_shifted / adv20_shifted

    condA = (m8 + s8) < m2
    condB = m2 < (m8 - s8)
    condC = vr >= 1

    val = pd.DataFrame(-1.0, index=close_shifted.index, columns=close_shifted.columns)
    val = val.where(~condB,  1.0)
    val = val.where(~condA, -1.0)
    maskC = (~condA) & (~condB) & condC
    val = val.where(~maskC, 1.0)

    return val.where(sample == 1)

# ─── Alpha 22 ───────────────────────────────────────────────────────────────
def load_alpha22(high_shifted, volume_shifted, close_shifted, sample):
    corr5       = correlation(high_shifted, volume_shifted, 5)
    delta_corr5 = delta(corr5, 5)

    std20   = stddev(close_shifted, 20)
    r_std20 = rank(std20)

    val = -delta_corr5 * r_std20
    return val.where(sample == 1)

# ─── Alpha 23 ───────────────────────────────────────────────────────────────
def load_alpha23(high_shifted, sample):
    m20  = ts_sum(high_shifted, 20) / 20
    cond = m20 < high_shifted

    d2  = delta(high_shifted, 2)
    val = pd.DataFrame(0.0, index=high_shifted.index, columns=high_shifted.columns)
    val = val.where(~cond, -d2)

    return val.where(sample == 1)

# ─── Alpha 24 ───────────────────────────────────────────────────────────────
def load_alpha24(close_shifted, sample):
    m100   = ts_sum(close_shifted, 100) / 100
    d_m100 = delta(m100, 100)
    clk100 = delay(close_shifted, 100)
    ratio  = d_m100.div(clk100)

    tsmin100 = ts_min(close_shifted, 100)
    cond     = ratio <= 0.05

    partA = - (close_shifted - tsmin100)
    partB = - delta(close_shifted, 3)

    val = pd.DataFrame(index=close_shifted.index, columns=close_shifted.columns, dtype=float)
    val = val.where(~cond, partA)
    val = val.fillna(partB)

    return val.where(sample == 1)

# ─── Alpha 25 ───────────────────────────────────────────────────────────────
def load_alpha25(logret_shifted, adv20_shifted, vwap_shifted,
                 high_shifted, close_shifted, sample):
    expr   = ((-1 * logret_shifted) * adv20_shifted) * vwap_shifted * (high_shifted - close_shifted)
    ranked = rank(expr)
    return ranked.where(sample == 1)

# ─── Alpha 26 ───────────────────────────────────────────────────────────────
def load_alpha26(volume_shifted, high_shifted, sample):
    tr_v5  = ts_rank(volume_shifted, 5)
    tr_h5  = ts_rank(high_shifted, 5)

    corr5 = correlation(tr_v5, tr_h5, 5)
    tmax3 = ts_max(corr5, 3)

    val = -tmax3
    return val.where(sample == 1)

# ─── Alpha 27 ───────────────────────────────────────────────────────────────
def load_alpha27(volume_shifted, amount_shifted, sample):
    v1       = volume_shifted
    vw1      = amount_shifted / volume_shifted
    rv       = rank(v1)
    rvw      = rank(vw1)

    corr6   = correlation(rv, rvw, 6)
    m2_corr = ts_sum(corr6, 2) / 2
    r_m2    = rank(m2_corr)

    val = pd.DataFrame(1.0, index=volume_shifted.index, columns=volume_shifted.columns)
    val = val.where(~(r_m2 > 0.5), -1.0)
    return val.where(sample == 1)

# ─── Alpha 28 ───────────────────────────────────────────────────────────────
def load_alpha28(adv20_shifted, low_shifted, high_shifted, close_shifted, sample):
    corr5 = correlation(adv20_shifted, low_shifted, 5)

    mid  = (high_shifted + low_shifted) * 0.5
    expr = corr5 + mid - close_shifted

    scaled = scale(expr)
    return scaled.where(sample == 1)

# ─── Alpha 29 ───────────────────────────────────────────────────────────────
def load_alpha29(close_shifted, logret_shifted, sample):
    neg_delta = -delta(close_shifted, 1)
    r1        = rank(neg_delta)
    r2        = rank(r1)

    tr2   = ts_rank(r2, 2)
    tmin5 = ts_min(tr2, 5)

    sum2     = ts_sum(tmin5, 2)
    log_sum2 = np.log(sum2)
    scaled   = scale(log_sum2)
    ranked   = rank(scaled)

    part1 = ts_min(ranked, 5)
    part2 = ts_rank(-logret_shifted.shift(6), 5)

    val = part1 + part2
    return val.where(sample == 1)

# ─── Alpha 30 ───────────────────────────────────────────────────────────────
def load_alpha30(close_shifted, volume_shifted, sample):
    d1   = close_shifted - delay(close_shifted, 1)
    d2   = delay(close_shifted, 1) - delay(close_shifted, 2)
    d3   = delay(close_shifted, 2) - delay(close_shifted, 3)

    expr   = sign(d1) + sign(d2) + sign(d3)
    r_expr = rank(expr)

    vol5  = ts_sum(volume_shifted, 5)
    vol20 = ts_sum(volume_shifted, 20)

    val = (1.0 - r_expr) * (vol5 / vol20)
    return val.where(sample == 1)

# ─── Alpha 31 ───────────────────────────────────────────────────────────────
def load_alpha31(close_shifted, adv20_shifted, low_shifted, sample):
    d10   = delta(close_shifted, 10)
    r_d10 = rank(d10)
    r_r_d10 = rank(r_d10)
    neg_r_r_d10 = -r_r_d10

    dl = decay_linear(neg_r_r_d10, 10)
    r1 = rank(dl)
    r2 = rank(r1)
    r3 = rank(r2)

    d3      = delta(close_shifted, 3)
    r_neg_d3= rank(-d3)

    corr12       = correlation(adv20_shifted, low_shifted, 12)
    scaled_corr12 = scale(corr12)
    s_corr       = sign(scaled_corr12)

    val = r3 + r_neg_d3 + s_corr
    return val.where(sample == 1)

# ─── Alpha 32 ───────────────────────────────────────────────────────────────
def load_alpha32(close_shifted, vwap_shifted, sample):
    mean7  = ts_sum(close_shifted, 7) / 7
    expr1  = mean7 - close_shifted
    s1     = scale(expr1)

    corr230 = correlation(vwap_shifted, delay(close_shifted, 5), 230)
    s2      = scale(corr230)

    val = s1 + 20.0 * s2
    return val.where(sample == 1)

# ─── Alpha 33 ───────────────────────────────────────────────────────────────
def load_alpha33(open_shifted, close_shifted, sample):
    expr   = (open_shifted / close_shifted) - 1.0
    r_expr = rank(expr)
    return r_expr.where(sample == 1)

# ─── Alpha 34 ───────────────────────────────────────────────────────────────
def load_alpha34(close_shifted, logret_shifted, sample):
    sd2     = stddev(logret_shifted, 2)
    sd5     = stddev(logret_shifted, 5)
    ratio   = sd2.div(sd5)
    r_ratio = rank(ratio)
    part1   = 1.0 - r_ratio

    d1      = delta(close_shifted, 1)
    r_d1    = rank(d1)
    part2   = 1.0 - r_d1

    expr    = part1 + part2
    r_expr  = rank(expr)
    return r_expr.where(sample == 1)

# ─── Alpha 35 ───────────────────────────────────────────────────────────────
def load_alpha35(close_shifted, high_shifted, low_shifted, volume_shifted,
                 logret_shifted, sample):
    tr_vol32   = ts_rank(volume_shifted, 32)

    expr2      = (close_shifted + high_shifted) - low_shifted
    tr_expr2_16= ts_rank(expr2, 16)
    part2      = 1.0 - tr_expr2_16

    tr_ret32 = ts_rank(logret_shifted, 32)
    part3    = 1.0 - tr_ret32

    val = tr_vol32 * part2 * part3
    return val.where(sample == 1)

# ─── Alpha 36 ───────────────────────────────────────────────────────────────
def load_alpha36(close_shifted, open_shifted, volume_shifted, vwap_shifted,
                 adv20_shifted, logret_shifted, sample):
    A         = close_shifted - open_shifted
    B         = delay(volume_shifted, 1)
    corr15    = correlation(A, B, 15)
    r_corr15  = rank(corr15)
    partA     = 2.21 * r_corr15

    r_open_minus_close = rank(open_shifted - close_shifted)
    partB    = 0.7 * r_open_minus_close

    delayed_neg_ret = delay(-logret_shifted, 6)
    tr_delayed_neg_ret_5 = ts_rank(delayed_neg_ret, 5)
    r_tr     = rank(tr_delayed_neg_ret_5)
    partC    = 0.73 * r_tr

    corr_vwap_adv20_6 = correlation(vwap_shifted, adv20_shifted, 6).abs()
    partD   = rank(corr_vwap_adv20_6)

    mean200 = ts_sum(close_shifted, 200) / 200
    exprE   = (mean200 - open_shifted) * (close_shifted - open_shifted)
    partE   = 0.6 * rank(exprE)

    val = partA + partB + partC + partD + partE
    return val.where(sample == 1)

# ─── Alpha 37 ───────────────────────────────────────────────────────────────
def load_alpha37(open_shifted, close_shifted, sample):
    A        = (open_shifted - close_shifted)
    corr200  = correlation(delay(A, 1), close_shifted, 200)
    r_corr200 = rank(corr200)

    r_open_minus_close = rank(open_shifted - close_shifted)

    val = r_corr200 + r_open_minus_close
    return val.where(sample == 1)

# ─── Alpha 38 ───────────────────────────────────────────────────────────────
def load_alpha38(close_shifted, open_shifted, sample):
    tr_close10       = ts_rank(close_shifted, 10)
    r_tr_close10     = rank(tr_close10)
    part1            = -r_tr_close10

    r_close_over_open= rank(close_shifted / open_shifted)

    val = part1 * r_close_over_open
    return val.where(sample == 1)

# ─── Alpha 39 ───────────────────────────────────────────────────────────────
def load_alpha39(close_shifted, volume_shifted, adv20_shifted,
                 logret_shifted, sample):
    ratio_vol_adv20 = volume_shifted.div(adv20_shifted)
    dl_ratio        = decay_linear(ratio_vol_adv20, 9)
    r_dl            = rank(dl_ratio)
    part_inner      = 1.0 - r_dl

    delta7    = delta(close_shifted, 7)
    exprA     = delta7 * part_inner
    r_exprA   = rank(exprA)
    partA     = -r_exprA

    sum_ret250 = ts_sum(logret_shifted, 250)
    r_sum_ret250 = rank(sum_ret250)
    partB     = 1.0 + r_sum_ret250

    val = partA * partB
    return val.where(sample == 1)

# ─── Alpha 40 ───────────────────────────────────────────────────────────────
def load_alpha40(high_shifted, volume_shifted, sample):
    s10    = stddev(high_shifted, 10)
    r_s10  = rank(s10)
    corr10 = correlation(high_shifted, volume_shifted, 10)
    val    = -r_s10 * corr10
    return val.where(sample == 1)

# ─── Alpha 41 ───────────────────────────────────────────────────────────────
def load_alpha41(high_shifted, low_shifted, vwap_shifted, sample):
    geom_hl = (high_shifted * low_shifted).pow(0.5)
    val     = geom_hl - vwap_shifted
    return val.where(sample == 1)

# ─── Alpha 43 ───────────────────────────────────────────────────────────────
def load_alpha43(close_shifted, volume_shifted, adv20_shifted, sample):
    ratio_v = volume_shifted.div(adv20_shifted).replace([np.inf, -np.inf], np.nan)
    tr1     = ts_rank(ratio_v, 20)
    d7      = delta(close_shifted, 7)
    tr2     = ts_rank(-d7, 8)
    val     = tr1 * tr2
    return val.where(sample == 1)

# ─── Alpha 44 ───────────────────────────────────────────────────────────────
def load_alpha44(high_shifted, volume_shifted, sample):
    r_vol     = rank(volume_shifted)
    corr5     = correlation(high_shifted, r_vol, 5)
    val       = -corr5
    return val.where(sample == 1)

# ─── Alpha 45 ───────────────────────────────────────────────────────────────
def load_alpha45(close_shifted, volume_shifted, sample):
    d5        = delay(close_shifted, 5)
    sma20     = ts_sum(d5, 20) / 20
    r_sma20   = rank(sma20)
    corr_c_v2 = correlation(close_shifted, volume_shifted, 2)
    sum5      = ts_sum(close_shifted, 5)
    sum20     = ts_sum(close_shifted, 20)
    corr_sum2 = correlation(sum5, sum20, 2)
    r_corrsum = rank(corr_sum2)
    val       = - (r_sma20 * corr_c_v2 * r_corrsum)
    return val.where(sample == 1)

# ─── Alpha 46 ───────────────────────────────────────────────────────────────
def load_alpha46(close_shifted, sample):
    c    = close_shifted
    c10  = delay(c, 10)
    c20  = delay(c, 20)
    c1   = delay(c, 1)
    term = ((c20 - c10) / 10.0) - ((c10 - c) / 10.0)

    val   = pd.DataFrame(index=c.index, columns=c.columns, dtype=float)
    mask1 = term > 0.25
    val   = val.where(~mask1, -1.0)
    mask2 = (~mask1) & (term < 0.0)
    val   = val.where(~mask2, 1.0)
    mask3 = (~mask1) & (~mask2)
    val   = val.where(~mask3, -1.0 * (c - c1))
    return val.where(sample == 1)

# ─── Alpha 47 ───────────────────────────────────────────────────────────────
def load_alpha47(close_shifted, high_shifted, volume_shifted,
                 adv20_shifted, vwap_shifted, sample):
    inv_c     = close_shifted.rpow(-1)
    r_invc    = rank(inv_c)
    partA1    = r_invc * volume_shifted.div(adv20_shifted)

    diff_hc   = high_shifted - close_shifted
    r_diff_hc = rank(diff_hc)
    sma_h5    = ts_sum(high_shifted, 5) / 5
    partA2    = high_shifted * r_diff_hc.div(sma_h5)

    A         = partA1 * partA2
    r_vwd     = rank(vwap_shifted - delay(vwap_shifted, 5))

    val = A - r_vwd
    return val.where(sample == 1)

# ─── Alpha 49 ───────────────────────────────────────────────────────────────
def load_alpha49(close_shifted, sample):
    c    = close_shifted
    c10  = delay(c, 10)
    c20  = delay(c, 20)
    c1   = delay(c, 1)
    term = ((c20 - c10) / 10.0) - ((c10 - c) / 10.0)

    val   = pd.DataFrame(index=c.index, columns=c.columns, dtype=float)
    mask1 = term < -0.1
    val   = val.where(~mask1, 1.0)
    val   = val.where(mask1, -1.0 * (c - c1))
    return val.where(sample == 1)

# ─── Alpha 50 ───────────────────────────────────────────────────────────────
def load_alpha50(volume_shifted, vwap_shifted, sample):
    r_vol  = rank(volume_shifted)
    r_vwap = rank(vwap_shifted)
    corr5  = correlation(r_vol, r_vwap, 5)
    r_corr = rank(corr5)
    tmax5  = ts_max(r_corr, 5)
    val    = -tmax5
    return val.where(sample == 1)

# ─── Alpha 51 ───────────────────────────────────────────────────────────────
def load_alpha51(close_shifted, sample):
    c    = close_shifted
    c10  = delay(c, 10)
    c20  = delay(c, 20)
    c1   = delay(c, 1)
    term = ((c20 - c10) / 10.0) - ((c10 - c) / 10.0)

    val  = pd.DataFrame(index=c.index, columns=c.columns, dtype=float)
    mask = term < -0.05
    val  = val.where(~mask, 1.0)
    val  = val.where(mask, -1.0 * (c - c1))
    return val.where(sample == 1)

# ─── Alpha 52 ───────────────────────────────────────────────────────────────
def load_alpha52(low_shifted, logret_shifted, volume_shifted, sample):
    tmin5      = ts_min(low_shifted, 5)
    tmin5_d5   = delay(tmin5, 5)
    partA      = -tmin5 + tmin5_d5

    sum240     = ts_sum(logret_shifted, 240)
    sum20      = ts_sum(logret_shifted, 20)
    ratio      = (sum240 - sum20).div(220.0)
    r_ratio    = rank(ratio)

    tr_vol5    = ts_rank(volume_shifted, 5)

    val        = partA * r_ratio * tr_vol5
    return val.where(sample == 1)

# ─── Alpha 55 ───────────────────────────────────────────────────────────────
def load_alpha55(close_shifted, high_shifted, low_shifted,
                 volume_shifted, sample):
    tmin12     = ts_min(low_shifted, 12)
    tmax12     = ts_max(high_shifted, 12)
    denom      = (tmax12 - tmin12).replace(0, np.nan)
    ratio      = (close_shifted - tmin12).div(denom)
    r_ratio    = rank(ratio)
    r_vol      = rank(volume_shifted)
    corr6      = correlation(r_ratio, r_vol, 6)
    val        = -corr6
    return val.where(sample == 1)

def load_alpha56(
    logret_shifted: pd.DataFrame,
    market_cap_shifted: pd.DataFrame,
    sample: pd.DataFrame
) -> pd.DataFrame:
    sum10  = ts_sum(logret_shifted, 10)
    sum2   = ts_sum(logret_shifted, 2)
    sum2_3 = ts_sum(sum2, 3)
    ratio  = sum10.div(sum2_3)
    part1  = rank(ratio)
    part2  = rank(logret_shifted.mul(market_cap_shifted))
    alpha56 = - part1.mul(part2)

    return alpha56.where(sample == 1)

# ─── Alpha 57 ───────────────────────────────────────────────────────────────
def load_alpha57(close_shifted, vwap_shifted, sample):
    def _argmax(arr: np.ndarray) -> float:
        return float(np.argmax(arr))
    argmax30   = close_shifted.rolling(window=30, min_periods=30).apply(_argmax, raw=True)
    r_argmax30 = rank(argmax30)
    dl2        = decay_linear(r_argmax30, 2)
    expr       = (close_shifted - vwap_shifted).div(dl2.replace(0, np.nan))
    val        = -expr
    return val.where(sample == 1)

# ─── Alpha 58 ───────────────────────────────────────────────────────────────
def load_alpha58(vwap_shifted, volume_shifted,
                 sector_aligned_shifted, sample):
    w_corr  = int(np.floor(3.92795))   # =3
    w_decay = int(np.floor(7.89291))   # =7
    w_rank  = int(np.floor(5.50322))   # =5

    vwap_ind = indneutralize(vwap_shifted, sector_aligned_shifted)
    corr3    = correlation(vwap_ind, volume_shifted, w_corr)
    decay7   = decay_linear(corr3, w_decay)
    rank5    = ts_rank(decay7, w_rank)
    val      = -rank5
    return val.where(sample == 1)

# ─── Alpha 59 ───────────────────────────────────────────────────────────────
def load_alpha59(vwap_shifted, volume_shifted,
                 industry_aligned_shifted, sample):
    w_corr  = int(np.floor(4.25197))    # =4
    w_decay = int(np.floor(16.2289))    # =16
    w_rank  = int(np.floor(8.19648))    # =8

    vwap_mix = vwap_shifted  # since 0.728317 + (1 - 0.728317) = 1
    vwap_ind = indneutralize(vwap_mix, industry_aligned_shifted)
    corr4    = correlation(vwap_ind, volume_shifted, w_corr)
    decay16  = decay_linear(corr4, w_decay)
    rank8    = ts_rank(decay16, w_rank)
    val      = -rank8
    return val.where(sample == 1)

# ─── Alpha 60 ───────────────────────────────────────────────────────────────
def load_alpha60(close_shifted, high_shifted, low_shifted,
                 volume_shifted, sample):
    num = (close_shifted - low_shifted) - (high_shifted - close_shifted)
    den = (high_shifted - low_shifted).replace(0, np.nan)
    X   = num.div(den)

    A_r = rank(X * volume_shifted)
    A_s = scale(A_r)

    B_r = rank(ts_argmax(close_shifted, 10))
    B_s = scale(B_r)

    alpha = - (2 * A_s - B_s)
    return alpha.where(sample == 1)

# ─── Alpha 61 ───────────────────────────────────────────────────────────────
def load_alpha61(vwap_shifted, adv180_shifted, sample):
    left   = rank(vwap_shifted - ts_min(vwap_shifted, 16.1219))
    right  = rank(correlation(vwap_shifted, adv180_shifted, 17.9282))
    alpha  = left.lt(right).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 62 ───────────────────────────────────────────────────────────────
def load_alpha62(vwap_shifted, adv20_shifted, open_shifted,
                 high_shifted, low_shifted, sample):
    sum_adv  = ts_sum(adv20_shifted, 22.4101)
    left     = rank(correlation(vwap_shifted, sum_adv, 9.91009))

    r_open   = rank(open_shifted)
    r_mid    = rank((high_shifted + low_shifted) * 0.5)
    r_high   = rank(high_shifted)
    cond     = (r_open + r_open).lt(r_mid + r_high)
    right    = rank(cond.astype(float))

    alpha    = - left.lt(right).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 63 ───────────────────────────────────────────────────────────────
def load_alpha63(vwap_shifted, adv180_shifted, close_shifted,
                 open_shifted, high_shifted, low_shifted,
                 industry_aligned_shifted, sample):
    ind_cl   = indneutralize(close_shifted, industry_aligned_shifted)
    d1       = delta(ind_cl, 2.25164)
    d1_dec   = decay_linear(d1, 8.22237)
    p1       = rank(d1_dec)

    mix      = vwap_shifted * 0.318108 + open_shifted * (1 - 0.318108)
    sum2     = ts_sum(adv180_shifted, 37.2467)
    c2       = correlation(mix, sum2, 13.557)
    d2_dec   = decay_linear(c2, 12.2883)
    p2       = rank(d2_dec)

    alpha    = - (p1 - p2)
    return alpha.where(sample == 1)

# ─── Alpha 64 ───────────────────────────────────────────────────────────────
def load_alpha64(vwap_shifted, adv120_shifted, close_shifted,
                 high_shifted, low_shifted, open_shifted, sample):
    mix1     = open_shifted * 0.178404 + low_shifted * (1 - 0.178404)
    sum1     = ts_sum(mix1, 12.7054)
    sum_adv  = ts_sum(adv120_shifted, 12.7054)
    left     = rank(correlation(sum1, sum_adv, 16.6208))

    mix2     = ((high_shifted + low_shifted) * 0.5) * 0.178404 + vwap_shifted * (1 - 0.178404)
    right    = rank(delta(mix2, 3.69741))

    alpha    = -left.lt(right).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 65 ───────────────────────────────────────────────────────────────
def load_alpha65(vwap_shifted, adv60_shifted, open_shifted, sample):
    sum_adv  = ts_sum(adv60_shifted, 8.6911)
    mix      = open_shifted * 0.00817205 + vwap_shifted * (1 - 0.00817205)
    left     = rank(correlation(mix, sum_adv, 6.40374))
    right    = rank(open_shifted - ts_min(open_shifted, 13.635))
    alpha    = -left.lt(right).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 66 ───────────────────────────────────────────────────────────────
def load_alpha66(vwap_shifted, high_shifted, low_shifted,
                 open_shifted, sample):
    d1       = delta(vwap_shifted, 3.51013)
    d1_dec   = decay_linear(d1, 7.23052)
    p1       = rank(d1_dec)

    mix2_num = low_shifted - vwap_shifted
    mix2_den = (open_shifted - (high_shifted + low_shifted) * 0.5).replace(0, np.nan)
    mix2     = mix2_num.div(mix2_den)
    d2_dec   = decay_linear(mix2, 11.4157)
    p2       = ts_rank(d2_dec, 6.72611)

    alpha    = - (p1 + p2)
    return alpha.where(sample == 1)

# ─── Alpha 67 ───────────────────────────────────────────────────────────────
def load_alpha67(vwap_shifted, adv20_shifted,
                 sector_aligned_shifted, subindustry_aligned_shifted,
                 high_shifted, sample):
    w_min    = 2.14593
    left     = rank(high_shifted - ts_min(high_shifted, w_min))

    vwap_n   = indneutralize(vwap_shifted, sector_aligned_shifted)
    adv_n    = indneutralize(adv20_shifted, subindustry_aligned_shifted)
    w_c      = 6.02936
    right    = rank(correlation(vwap_n, adv_n, w_c))

    alpha    = - (left ** right)
    return alpha.where(sample == 1)

# ─── Alpha 68 ───────────────────────────────────────────────────────────────
def load_alpha68(high_shifted, adv15_shifted, close_shifted,
                 low_shifted, sample):
    r_high   = rank(high_shifted)
    r_adv15  = rank(adv15_shifted)
    corr1    = correlation(r_high, r_adv15, 8.91644)
    left     = ts_rank(corr1, 13.9333)

    mix2     = close_shifted * 0.518371 + low_shifted * (1 - 0.518371)
    right    = rank(delta(mix2, 1.06157))

    alpha    = - left.lt(right).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 69 ───────────────────────────────────────────────────────────────
def load_alpha69(vwap_shifted, close_shifted, adv20_shifted,
                 industry_aligned_shifted, sample):
    indn     = indneutralize(vwap_shifted, industry_aligned_shifted)
    d1       = delta(indn, 2.72412)
    m1       = ts_max(d1, 4.79344)
    left     = rank(m1)

    mix2     = close_shifted * 0.490655 + vwap_shifted * (1 - 0.490655)
    corr2    = correlation(mix2, adv20_shifted, 4.92416)
    right    = ts_rank(corr2, 9.0615)

    alpha    = - (left ** right)
    return alpha.where(sample == 1)

# ─── Alpha 70 ───────────────────────────────────────────────────────────────
def load_alpha70(vwap_shifted, close_shifted, adv50_shifted,
                 industry_aligned_shifted, sample):
    r1 = rank(delta(vwap_shifted, 1.29456))

    indc = indneutralize(close_shifted, industry_aligned_shifted)
    c2   = correlation(indc, adv50_shifted, 17.8256)
    t2   = ts_rank(c2, 17.9171)

    alpha = -(r1 ** t2)
    return alpha.where(sample == 1)

# ─── Alpha 71 ───────────────────────────────────────────────────────────────
def load_alpha71(close_shifted, adv180_shifted, low_shifted,
                 open_shifted, vwap_shifted, sample):
    t_close  = ts_rank(close_shifted, 3.43976)
    t_adv180 = ts_rank(adv180_shifted, 12.0647)
    c1       = correlation(t_close, t_adv180, 18.0175)
    d1       = decay_linear(c1, 4.20501)
    term1    = ts_rank(d1, 15.6948)

    expr     = (low_shifted + open_shifted) - 2 * vwap_shifted
    r_expr   = rank(expr)
    sqr_expr = r_expr * r_expr
    d2       = decay_linear(sqr_expr, 16.4662)
    term2    = ts_rank(d2, 4.4388)

    alpha    = term1.combine(term2, np.maximum)
    return alpha.where(sample == 1)

# ─── Alpha 72 ───────────────────────────────────────────────────────────────
def load_alpha72(high_shifted, low_shifted, adv40_shifted,
                 vwap_shifted, volume_shifted, sample):
    mid      = (high_shifted + low_shifted) * 0.5
    c1       = correlation(mid, adv40_shifted, 8.93345)
    d1       = decay_linear(c1, 10.1519)
    num_rank = rank(d1)

    r_vwap   = ts_rank(vwap_shifted, 3.72469)
    r_vol    = ts_rank(volume_shifted, 18.5188)
    c2       = correlation(r_vwap, r_vol, 6.86671)
    d2       = decay_linear(c2, 2.95011)
    den_rank = rank(d2)

    alpha    = num_rank.div(den_rank.replace(0, np.nan))
    return alpha.where(sample == 1)

# ─── Alpha 73 ───────────────────────────────────────────────────────────────
def load_alpha73(vwap_shifted, high_shifted, low_shifted,
                 open_shifted, sample):
    d1      = delta(vwap_shifted, 4.72775)
    d1_dec  = decay_linear(d1, 2.91864)
    r1      = rank(d1_dec)

    mix     = open_shifted * 0.147155 + low_shifted * (1 - 0.147155)
    d_mix   = delta(mix, 2.03608)
    expr2   = d_mix.div(mix.replace(0, np.nan)) * -1.0
    d2_dec  = decay_linear(expr2, 3.33829)
    r2      = ts_rank(d2_dec, 16.7411)

    alpha   = - r1.combine(r2, np.maximum)
    return alpha.where(sample == 1)

# ─── Alpha 74 ───────────────────────────────────────────────────────────────
def load_alpha74(close_shifted, high_shifted, vwap_shifted,
                 volume_shifted, adv30_shifted, sample):
    sum_adv = ts_sum(adv30_shifted, 37.4843)
    left    = rank(correlation(close_shifted, sum_adv, 15.1365))

    mix2    = high_shifted * 0.0261661 + vwap_shifted * (1 - 0.0261661)
    right   = rank(correlation(rank(mix2), rank(volume_shifted), 11.4791))

    alpha   = - left.lt(right).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 75 ───────────────────────────────────────────────────────────────
def load_alpha75(vwap_shifted, volume_shifted, low_shifted,
                 adv50_shifted, sample):
    left  = rank(correlation(vwap_shifted, volume_shifted, 4.24304))
    right = rank(correlation(rank(low_shifted), rank(adv50_shifted), 12.4413))
    alpha = left.lt(right).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 76 ───────────────────────────────────────────────────────────────
def load_alpha76(vwap_shifted, volume_shifted, low_shifted,
                 adv50_shifted, adv81_shifted,
                 sector_aligned_shifted, sample):
    d1      = delta(vwap_shifted, 1.24383)
    d1_dec  = decay_linear(d1, 11.8259)
    left    = rank(d1_dec)

    ind_l   = indneutralize(low_shifted, sector_aligned_shifted)
    c2      = correlation(ind_l, adv81_shifted, 8.14941)
    t1      = ts_rank(c2, 19.569)
    d2_dec  = decay_linear(t1, 17.1543)
    right   = ts_rank(d2_dec, 19.383)

    both    = pd.DataFrame(
                  np.maximum(left.values, right.values),
                  index=left.index,
                  columns=left.columns
              )
    alpha   = -both
    return alpha.where(sample == 1)

# ─── Alpha 77 ───────────────────────────────────────────────────────────────
def load_alpha77(high_shifted, low_shifted, vwap_shifted,
                 adv40_shifted, sample):
    expr1   = ((high_shifted + low_shifted) * 0.5) - vwap_shifted
    r1      = rank(decay_linear(expr1, 20.0451))

    mid     = (high_shifted + low_shifted) * 0.5
    c2      = correlation(mid, adv40_shifted, 3.1614)
    r2      = rank(decay_linear(c2, 5.64125))

    alpha   = r1.combine(r2, np.minimum)
    return alpha.where(sample == 1)

# ─── Alpha 78 ───────────────────────────────────────────────────────────────
def load_alpha78(low_shifted, vwap_shifted, volume_shifted,
                 adv40_shifted, sample):
    mix1    = low_shifted * 0.352233 + vwap_shifted * (1 - 0.352233)
    s1      = ts_sum(mix1, 19.7428)
    s2      = ts_sum(adv40_shifted, 19.7428)
    c1      = correlation(s1, s2, 6.83313)
    r1      = rank(c1)

    r_vwap  = rank(vwap_shifted)
    r_vol   = rank(volume_shifted)
    c2      = correlation(r_vwap, r_vol, 5.77492)
    r2      = rank(c2)

    alpha   = r1 ** r2
    return alpha.where(sample == 1)

# ─── Alpha 79 ───────────────────────────────────────────────────────────────
def load_alpha79(close_shifted, open_shifted,
                 sector_aligned_shifted, vwap_shifted,
                 adv150_shifted, sample):
    mix     = close_shifted * 0.60733 + open_shifted * (1 - 0.60733)
    ind_mix = indneutralize(mix, sector_aligned_shifted)
    r1      = rank(delta(ind_mix, 1.23438))

    t1      = ts_rank(vwap_shifted, 3.60973)
    t2      = ts_rank(adv150_shifted, 9.18637)
    c2      = correlation(t1, t2, 14.6644)
    r2      = rank(c2)

    alpha   = r1.lt(r2).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 80 ───────────────────────────────────────────────────────────────
def load_alpha80(open_shifted, high_shifted, volume_shifted,
                 adv10_shifted, industry_aligned_shifted, sample):
    mix     = open_shifted * 0.868128 + high_shifted * (1 - 0.868128)
    ind_mix = indneutralize(mix, industry_aligned_shifted)
    s1      = sign(delta(ind_mix, 4.04545))
    r1      = rank(s1)

    c2      = correlation(high_shifted, adv10_shifted, 5.11456)
    t2      = ts_rank(c2, 5.53756)

    alpha   = - (r1 ** t2)
    return alpha.where(sample == 1)

# ─── Alpha 81 ───────────────────────────────────────────────────────────────
def load_alpha81(vwap_shifted, volume_shifted, adv10_shifted, sample):
    s_adv10  = ts_sum(adv10_shifted, 49.6054)
    c1       = correlation(vwap_shifted, s_adv10, 8.47743)
    r_c1     = rank(c1)
    p4       = r_c1.pow(4)
    r4       = rank(p4)
    prod1    = product(r4, 14.9655)
    logp     = np.log(prod1)
    r_logp   = rank(logp)

    r_vwap   = rank(vwap_shifted)
    r_vol    = rank(volume_shifted)
    c2       = correlation(r_vwap, r_vol, 5.07914)
    r2       = rank(c2)

    alpha    = - r_logp.lt(r2).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 82 ───────────────────────────────────────────────────────────────
def load_alpha82(open_shifted, volume_shifted,
                 sector_aligned_shifted, sample):
    d1       = delta(open_shifted, 1.46063)
    d1_dec   = decay_linear(d1, 14.8717)
    r1       = rank(d1_dec)

    ind_v    = indneutralize(volume_shifted, sector_aligned_shifted)
    c2       = correlation(ind_v, open_shifted, 17.4842)
    d2_dec   = decay_linear(c2, 6.92131)
    t2       = ts_rank(d2_dec, 13.4283)

    alpha    = - r1.combine(t2, np.minimum)
    return alpha.where(sample == 1)

# ─── Alpha 83 ───────────────────────────────────────────────────────────────
def load_alpha83(high_shifted, low_shifted, close_shifted,
                 vwap_shifted, volume_shifted, sample):
    mid_range= (high_shifted - low_shifted).div(ts_sum(close_shifted, 5).div(5))
    d1       = delay(mid_range, 2)
    r1       = rank(d1)

    r_vol    = rank(volume_shifted)
    r2       = rank(r_vol)

    numerator= r1 * r2
    denom    = mid_range.div(vwap_shifted - close_shifted.replace(0, np.nan))
    alpha    = numerator.div(denom.replace(0, np.nan))
    return alpha.where(sample == 1)

# ─── Alpha 84 ───────────────────────────────────────────────────────────────
def load_alpha84(vwap_shifted, close_shifted, sample):
    d1       = vwap_shifted - ts_max(vwap_shifted, 15.3217)
    t1       = ts_rank(d1, 20.7127)
    pwr      = delta(close_shifted, 4.96796)
    alpha    = signedpower(t1, pwr)
    return alpha.where(sample == 1)

# ─── Alpha 85 ───────────────────────────────────────────────────────────────
def load_alpha85(high_shifted, low_shifted, close_shifted,
                 adv30_shifted, volume_shifted, sample):
    mix1 = high_shifted * 0.876703 + close_shifted * (1 - 0.876703)
    c1   = correlation(mix1, adv30_shifted, 9.61331)
    r1   = rank(c1)

    mid = (high_shifted + low_shifted) * 0.5
    t_mid = ts_rank(mid, 3.70596)
    t_vol = ts_rank(volume_shifted, 10.1595)
    c2    = correlation(t_mid, t_vol, 7.11408)
    r2    = rank(c2)

    alpha = r1 ** r2
    return alpha.where(sample == 1)

# ─── Alpha 86 ───────────────────────────────────────────────────────────────
def load_alpha86(open_shifted, close_shifted, vwap_shifted,
                 adv20_shifted, sample):
    s_adv    = ts_sum(adv20_shifted, 14.7444)
    c1       = correlation(close_shifted, s_adv, 6.00049)
    t1       = ts_rank(c1, 20.4195)

    expr     = (open_shifted + close_shifted) - (vwap_shifted + open_shifted)
    r2       = rank(expr)

    alpha    = - t1.lt(r2).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 87 ───────────────────────────────────────────────────────────────
def load_alpha87(close_shifted, vwap_shifted, adv81_shifted,
                 industry_aligned_shifted, sample):
    mix1     = close_shifted * 0.369701 + vwap_shifted * (1 - 0.369701)
    d1       = delta(mix1, 1.91233)
    d1_dec   = decay_linear(d1, 2.65461)
    r1       = rank(d1_dec)

    ind_a    = indneutralize(adv81_shifted, industry_aligned_shifted)
    c2       = correlation(ind_a, close_shifted, 13.4132).abs()
    d2_dec   = decay_linear(c2, 4.89768)
    t2       = ts_rank(d2_dec, 14.4535)

    alpha    = - r1.combine(t2, np.maximum)
    return alpha.where(sample == 1)

# ─── Alpha 88 ───────────────────────────────────────────────────────────────
def load_alpha88(open_shifted, high_shifted, low_shifted, close_shifted,
                 adv60_shifted, sample):
    expr1    = rank(open_shifted) + rank(low_shifted) - (rank(high_shifted) + rank(close_shifted))
    d1       = decay_linear(expr1, 8.06882)
    r1       = rank(d1)

    t1       = ts_rank(close_shifted, 8.44728)
    t2       = ts_rank(adv60_shifted, 20.6966)
    c2       = correlation(t1, t2, 8.01266)
    d2_dec   = decay_linear(c2, 6.65053)
    r2       = ts_rank(d2_dec, 2.61957)

    alpha    = r1.combine(r2, np.minimum)
    return alpha.where(sample == 1)

# ─── Alpha 89 ───────────────────────────────────────────────────────────────
def load_alpha89(vwap_shifted, close_shifted, low_shifted,
                 adv10_shifted, industry_aligned_shifted, sample):
    low_mix  = low_shifted
    c1       = correlation(low_mix, adv10_shifted, 6.94279)
    d1_dec   = decay_linear(c1, 5.51607)
    left     = ts_rank(d1_dec, 3.79744)

    ind_v    = indneutralize(vwap_shifted, industry_aligned_shifted)
    d2       = delta(ind_v, 3.48158)
    d2_dec   = decay_linear(d2, 10.1466)
    right    = ts_rank(d2_dec, 15.3012)

    alpha    = left - right
    return alpha.where(sample == 1)

# ─── Alpha 90 ───────────────────────────────────────────────────────────────
def load_alpha90(close_shifted, low_shifted, adv40_shifted,
                 subindustry_aligned_shifted, sample):
    left_expr = close_shifted - ts_max(close_shifted, 4.66719)
    left_r    = rank(left_expr)

    ind_a     = indneutralize(adv40_shifted, subindustry_aligned_shifted)
    c2        = correlation(ind_a, low_shifted, 5.38375)
    right_t   = ts_rank(c2, 3.21856)

    alpha     = - (left_r ** right_t)
    return alpha.where(sample == 1)

# ─── Alpha 91 ───────────────────────────────────────────────────────────────
def load_alpha91(close_shifted, volume_shifted, vwap_shifted,
                 adv30_shifted, industry_aligned_shifted, sample):
    ind_c    = indneutralize(close_shifted, industry_aligned_shifted)
    c1       = correlation(ind_c, volume_shifted, 9.74928)
    d1       = decay_linear(c1, 16.398)
    d1_dec   = decay_linear(d1, 3.83219)
    part1    = ts_rank(d1_dec, 4.8667)

    c2       = correlation(vwap_shifted, adv30_shifted, 4.01303)
    d2_dec   = decay_linear(c2, 2.6809)
    part2    = rank(d2_dec)

    alpha    = - (part1 - part2)
    return alpha.where(sample == 1)

# ─── Alpha 92 ───────────────────────────────────────────────────────────────
def load_alpha92(high_shifted, low_shifted, close_shifted,
                 open_shifted, adv30_shifted, sample):
    cond_bool    = ((high_shifted + low_shifted) * 0.5 + close_shifted) < (low_shifted + open_shifted)
    d1            = decay_linear(cond_bool.astype(float), 14.7221)
    part1         = ts_rank(d1, 18.8683)

    c2_attrs         = correlation(rank(low_shifted), rank(adv30_shifted), 7.58555)
    d2_attrs         = decay_linear(c2_attrs, 6.94024)
    part2            = ts_rank(d2_attrs, 6.80584)

    alpha            = part1.combine(part2, np.minimum)
    return alpha.where(sample == 1)

# ─── Alpha 93 ───────────────────────────────────────────────────────────────
def load_alpha93(close_shifted, vwap_shifted, adv81_shifted,
                 industry_aligned_shifted, sample):
    ind_v   = indneutralize(vwap_shifted, industry_aligned_shifted)
    c1      = correlation(ind_v, adv81_shifted, 17.4193)
    d1      = decay_linear(c1, 19.848)
    left    = ts_rank(d1, 7.54455)

    mix     = close_shifted * 0.524434 + vwap_shifted * (1 - 0.524434)
    d2      = delta(mix, 2.77377)
    d2_dec  = decay_linear(d2, 16.2664)
    right   = rank(d2_dec)

    alpha   = left.div(right.replace(0, np.nan))
    return alpha.where(sample == 1)

# ─── Alpha 94 ───────────────────────────────────────────────────────────────
def load_alpha94(vwap_shifted, adv60_shifted, sample):
    left_expr = vwap_shifted - ts_min(vwap_shifted, 11.5783)
    left_r    = rank(left_expr)

    t_vwap     = ts_rank(vwap_shifted, 19.6462)
    t_adv60    = ts_rank(adv60_shifted, 4.02992)
    c2         = correlation(t_vwap, t_adv60, 18.0926)
    right_t    = ts_rank(c2, 2.70756)

    alpha      = - (left_r ** right_t)
    return alpha.where(sample == 1)

# ─── Alpha 95 ───────────────────────────────────────────────────────────────
def load_alpha95(open_shifted, high_shifted, low_shifted,
                 adv40_shifted, sample):
    left_expr = open_shifted - ts_min(open_shifted, 12.4105)
    left_r    = rank(left_expr)

    s1        = ts_sum((high_shifted + low_shifted) * 0.5, 19.1351)
    s2        = ts_sum(adv40_shifted, 19.1351)
    c2        = correlation(s1, s2, 12.8742)
    r_c2      = rank(c2)
    p5        = r_c2.pow(5)
    right_t   = ts_rank(p5, 11.7584)

    alpha     = left_r.lt(right_t).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 96 ───────────────────────────────────────────────────────────────
def load_alpha96(close_shifted, low_shifted, high_shifted,
                 volume_shifted, adv60_shifted, vwap_shifted, sample):
    c1     = correlation(rank(vwap_shifted), rank(volume_shifted), 3.83878)
    d1     = decay_linear(c1, 4.16783)
    part1  = ts_rank(d1, 8.38151)

    t1     = ts_rank(close_shifted, 7.45404)
    t2     = ts_rank(adv60_shifted, 4.13242)
    c2     = correlation(t1, t2, 3.65459)
    argpos = ts_argmax(c2, 12.6556)
    d2     = decay_linear(argpos, 14.0365)
    part2  = ts_rank(d2, 13.4143)

    alpha  = - part1.combine(part2, np.maximum)
    return alpha.where(sample == 1)

# ─── Alpha 97 ───────────────────────────────────────────────────────────────
def load_alpha97(low_shifted, vwap_shifted, adv60_shifted,
                 industry_aligned_shifted, sample):
    mix1     = low_shifted * 0.721001 + vwap_shifted * (1 - 0.721001)
    ind_m1   = indneutralize(mix1, industry_aligned_shifted)
    d1       = delta(ind_m1, 3.3705)
    d1_dec   = decay_linear(d1, 20.4523)
    p1       = rank(d1_dec)

    t_low    = ts_rank(low_shifted, 7.87871)
    t_adv60  = ts_rank(adv60_shifted, 17.255)
    c2       = correlation(t_low, t_adv60, 4.97547)
    r_c2     = ts_rank(c2, 18.5925)
    d2       = decay_linear(r_c2, 15.7152)
    p2       = ts_rank(d2, 6.71659)

    alpha    = - (p1 - p2)
    return alpha.where(sample == 1)

# ─── Alpha 98 ───────────────────────────────────────────────────────────────
def load_alpha98(open_shifted, vwap_shifted, adv5_shifted,
                 adv15_shifted, sample):
    s1   = ts_sum(adv5_shifted, 26.4719)
    c1   = correlation(vwap_shifted, s1, 4.58418)
    d1   = decay_linear(c1, 7.18088)
    p1   = rank(d1)

    r_o    = rank(open_shifted)
    r_a15  = rank(adv15_shifted)
    c2     = correlation(r_o, r_a15, 20.8187)
    amin   = ts_argmin(c2, 8.62571)
    t3     = ts_rank(amin, 6.95668)
    d2     = decay_linear(t3, 8.07206)
    p2     = rank(d2)

    alpha  = p1 - p2
    return alpha.where(sample == 1)

# ─── Alpha 99 ───────────────────────────────────────────────────────────────
def load_alpha99(high_shifted, low_shifted, volume_shifted,
                 adv60_shifted, sample):
    s1       = ts_sum((high_shifted + low_shifted) * 0.5, 19.8975)
    s2       = ts_sum(adv60_shifted, 19.8975)
    c1       = correlation(s1, s2, 8.8136)
    left_r   = rank(c1)

    c2       = correlation(low_shifted, volume_shifted, 6.28259)
    right_r  = rank(c2)

    alpha    = - left_r.lt(right_r).astype(float)
    return alpha.where(sample == 1)

# ─── Alpha 100 ──────────────────────────────────────────────────────────────
def load_alpha100(close_shifted, low_shifted, high_shifted,
                  volume_shifted, adv20_shifted,
                  subindustry_aligned_shifted, sample):
    ratio = ((close_shifted - low_shifted) - (high_shifted - close_shifted)).div(
                (high_shifted - low_shifted).replace(0, np.nan)
            )
    r1    = rank(ratio * volume_shifted)
    in1   = indneutralize(r1, subindustry_aligned_shifted)
    s_in1 = scale(in1)

    c2    = correlation(close_shifted, rank(adv20_shifted), 5)
    argm  = ts_argmin(close_shifted, 30)
    r_arg = rank(argm)
    diff2 = c2 - r_arg
    in2   = indneutralize(diff2, subindustry_aligned_shifted)
    s_in2 = scale(in2)
    partA = 1.5 * s_in1 - s_in2

    ratio2 = volume_shifted.div(adv20_shifted.replace(0, np.nan))
    alpha  = - (partA * ratio2)
    return alpha.where(sample == 1)

# ─── Alpha 101 ──────────────────────────────────────────────────────────────
def load_alpha101(open_shifted, high_shifted, low_shifted,
                  close_shifted, sample):
    alpha = (close_shifted - open_shifted).div((high_shifted - low_shifted) + 0.001)
    return alpha.where(sample == 1)



'''Delay-0 Alphas'''

def load_alpha42_d0(amount: pd.DataFrame,
                    volume: pd.DataFrame,
                    close: pd.DataFrame,
                    sample: pd.DataFrame
                   ) -> pd.DataFrame:
    vwap = amount.div(volume).replace([np.inf, -np.inf], np.nan)
    raw  = rank(vwap - close) / rank(vwap + close)
    return raw.where(sample == 1)


def load_alpha48_d0(close: pd.DataFrame,
                    subindustry_aligned: pd.DataFrame,
                    sample: pd.DataFrame
                   ) -> pd.DataFrame:
    # 1）计算一阶差分和滞后差分
    delta_c      = delta(close, 1)
    delta_c_lag1 = delta(delay(close, 1), 1)
    # 2）250 日相关
    corr_250     = correlation(delta_c, delta_c_lag1, 250)

    # 3）构造 top_raw，并用新的 subindustry_aligned 做中性化
    top_raw      = corr_250.mul(delta_c).div(close)
    top_adj      = indneutralize(top_raw, subindustry_aligned)

    # 4）分母部分不变
    denom        = ts_sum((delta_c.div(delay(close, 1)))**2, 250)
    raw          = top_adj.div(denom.replace(0, np.nan))

    return raw.where(sample == 1)


def load_alpha53_d0(close: pd.DataFrame,
                    high: pd.DataFrame,
                    low: pd.DataFrame,
                    sample: pd.DataFrame
                   ) -> pd.DataFrame:
    x    = ((close - low) - (high - close)).div(close - low + 1e-9)
    raw  = -delta(x, 9)
    return raw.where(sample == 1)


def load_alpha54_d0(open_price: pd.DataFrame,
                    high:        pd.DataFrame,
                    low:         pd.DataFrame,
                    close:       pd.DataFrame,
                    sample:      pd.DataFrame
                   ) -> pd.DataFrame:
    num  = (low - close).mul(signedpower(open_price, 5))
    den  = (low - high).mul(signedpower(close,      5))
    raw  = -num.div(den.replace(0, np.nan))
    return raw.where(sample == 1)



''' IC Info'''

def compute_ic_series(
        alpha_df: pd.DataFrame,
        logret_df: pd.DataFrame,
        sample_df: pd.DataFrame,
        method: str = "spearman"
) -> pd.Series:
    """
    Same docstring as before …
    """
    # 1) 对齐列（股票代码）
    alpha_df, logret_df = alpha_df.align(logret_df, join='inner', axis=1)
    alpha_df, sample_df = alpha_df.align(sample_df, join='inner', axis=1)

    # 2) r_{t+1}
    ret_next = logret_df.shift(-1)

    # 3) 日期交集 + 排序
    dates = alpha_df.index.intersection(ret_next.index).sort_values()

    ic_vals = []
    for dt in dates:
        mask = (sample_df.loc[dt] == 1)
        a = alpha_df.loc[dt, mask]
        r = ret_next.loc[dt, mask]

        valid = a.notna() & r.notna()
        if valid.sum() < 2:
            ic_vals.append(np.nan)
            continue

        if method.lower() == "spearman":
            ic = a[valid].corr(r[valid], method="spearman")
        elif method.lower() == "pearson":
            ic = a[valid].corr(r[valid], method="pearson")
        else:
            raise ValueError("method must be 'spearman' or 'pearson'")
        ic_vals.append(ic)

    return pd.Series(ic_vals, index=dates,
                     name="rank_ic" if method.lower()=="spearman" else "ic")

def compute_ic_all(
        alphas: dict[str, pd.DataFrame],
        logret_df: pd.DataFrame,
        sample_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    ic_mat     = {}
    rank_ic_mat = {}

    for nm, a_df in alphas.items():
        ic_mat[nm]      = compute_ic_series(a_df, logret_df, sample_df, "pearson")
        rank_ic_mat[nm] = compute_ic_series(a_df, logret_df, sample_df, "spearman")

    ic_df      = pd.DataFrame(ic_mat)
    rank_ic_df = pd.DataFrame(rank_ic_mat)

    summary = []
    for nm in alphas.keys():
        s_ic  = ic_df[nm].dropna()
        s_ric = rank_ic_df[nm].dropna()

        mean_ic,  std_ic  = s_ic.mean(),  s_ic.std(ddof=1)
        mean_ric, std_ric = s_ric.mean(), s_ric.std(ddof=1)

        summary.append({
            "alpha": nm,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ic_ir": np.nan if (std_ic == 0 or np.isnan(std_ic)) else mean_ic / std_ic,
            "mean_rank_ic": mean_ric,
            "std_rank_ic": std_ric,
            "rank_ic_ir": np.nan if (std_ric == 0 or np.isnan(std_ric)) else mean_ric / std_ric,
        })

    summary_df = (pd.DataFrame(summary)
                    .set_index("alpha")
                    .sort_values("mean_rank_ic", ascending=False))
    return ic_df, rank_ic_df, summary_df





'''Hypothesis Testing'''

def hac_se_vectorized(df: pd.DataFrame, maxlags: int = 5) -> pd.Series:
    X = df.to_numpy()
    T, K = X.shape
    n_obs = np.sum(~np.isnan(X), axis=0).astype(float)  # 有效样本数 (K,)

    L = min(maxlags, int(np.nanmax(n_obs))-1)
    mu = np.nanmean(X, axis=0)                         # (K,)
    E  = X - mu                                         # (T, K)

    gamma0 = np.nanmean(E*E, axis=0)

    sum_term = np.zeros(K)
    for l in range(1, L+1):
        cov_l = np.nanmean(E[l:]*E[:-l], axis=0)
        weight = 1 - l/(L+1)
        n_eff  = np.maximum(n_obs - l, 1)               # 防 0
        sum_term += 2*weight*cov_l*n_obs/n_eff          # Small-sample 修正

    var_mean = (gamma0 + sum_term) / n_obs
    var_mean = np.clip(var_mean, 0, None)
    return pd.Series(np.sqrt(var_mean), index=df.columns)

def alpha_significance(df: pd.DataFrame, alpha_lv=[0.05],
                       null_h_pos=0.0, null_h_neg=0.0,
                       ci_level=0.95, hac_lags=5):
    n_obs  = df.count()
    means  = df.mean()
    ses    = hac_se_vectorized(df, hac_lags)

    sign   = np.where(means >= 0, 1, -1)
    nulls  = np.where(sign > 0, null_h_pos, null_h_neg)
    dfree  = n_obs - 1
    t_stat = (means - nulls) / ses

    # 单侧 P-value
    right_tail = (sign > 0)
    p_val = np.where(right_tail, 1 - t.cdf(t_stat, dfree),
                                 t.cdf(t_stat, dfree))

    # 置信区间
    tcrit = t.ppf(ci_level, dfree)
    ci_lower = np.where(right_tail, means - tcrit*ses, np.nan)
    ci_upper = np.where(~right_tail, means + tcrit*ses, np.nan)

    out = pd.DataFrame({
        "n": n_obs, "mean": means, "se": ses,
        "t_stat": t_stat, "p_value": p_val,
        "CI_lower": ci_lower, "CI_upper": ci_upper,
    })

    for lev in alpha_lv:
        out[f"reject_by_p_{lev}"] = p_val < lev

    return out.sort_values("t_stat", ascending=False)


def filter_df(df):
    same_direction = np.sign(df["mean_ic"]) == np.sign(df["mean_ric"])

    pos_mask = (
        (df["mean_ic"] >= 0) &
        (df["mean_ric"] >= 0) &
        (df["reject_by_p_0.05_ic"]) &
        (df["reject_by_p_0.05_ric"]) &
        (df["CI_lower_ic"]  > 0) &
        (df["CI_lower_ric"] > 0) &
        (df["mean_ic"]  > ic_thresh_pos) &
        (df["mean_ric"] > ric_thresh_pos)
    )

    neg_mask = (
        (df["mean_ic"] < 0) &
        (df["mean_ric"] < 0) &
        (df["reject_by_p_0.05_ic"]) &
        (df["reject_by_p_0.05_ric"]) &
        (df["CI_upper_ic"] < 0) &
        (df["CI_upper_ric"] < 0) &
        (df["mean_ic"]  < ic_thresh_neg) &
        (df["mean_ric"] < ric_thresh_neg)
    )

    return df[same_direction & (pos_mask | neg_mask)]


# ---------- 6. 原始方向统计 ----------
def attach_orig_stats(row):
    src = (orig_mean_ic_d1, orig_mean_ric_d1, 'd1') if row['delay']=='d1' \
          else (orig_mean_ic_d0, orig_mean_ric_d0, 'd0')
    om_ic, om_ric = src[0][row['alpha']], src[1][row['alpha']]
    return pd.Series({
        'orig_mean_ic': om_ic,
        'abs_orig_mean_ic': np.abs(om_ic),
        'orig_mean_ric': om_ric,
        'abs_orig_mean_ric': np.abs(om_ric),
        'orig_sign_ic': np.sign(om_ic),
        'orig_sign_ric': np.sign(om_ric)
    })


def plot_alpha_corr_by_list(
    alpha_list: list[str],
    alpha_dict: dict[str, pd.DataFrame],
    corr_method: str = "pearson"
) -> None:
    """
    直接用 alpha_list（列表）画相关热力图。
    """
    # 1. 用 alpha_list 作为 names
    names = alpha_list

    # 2–5 同前
    series_dict = {}
    for name in names:
        df = alpha_dict[name]
        s = df.stack(dropna=True)
        s.name = name
        series_dict[name] = s

    combined = pd.DataFrame(series_dict)
    corr = combined.corr(method=corr_method)

    plt.figure(figsize=(max(6, len(names)), max(6, len(names))))
    sns.heatmap(corr, annot=True, fmt=".2f", square=True,
                cbar_kws={"shrink": .8}, linewidths=0.5)
    plt.title(f"Alpha Correlation ({corr_method})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def get_alpha_corr_matrix_by_list(
    alpha_list: list[str],
    alpha_dict: dict[str, pd.DataFrame],
    corr_method: str = "pearson"
) -> pd.DataFrame:
    """
    直接用 alpha_list（列表）返回相关系数矩阵。
    """
    names = alpha_list
    series_dict = {}
    for name in names:
        df = alpha_dict[name]
        s = df.stack(dropna=True)
        s.name = name
        series_dict[name] = s

    combined = pd.DataFrame(series_dict)
    return combined.corr(method=corr_method)



def fama_macbeth_with_p(final_alphas: pd.DataFrame,
                        alpha_dict: dict[str, pd.DataFrame],
                        logret: pd.DataFrame,
                        shift: int = 1,
                        add_const: bool = True,
                        ridge: float = 0.0,
                        acf_lags: int = 5,
                        alpha: float = 0.05) -> pd.DataFrame:
    """
    Fama–MacBeth 两步 + ACF/PACF 自相关检验自动选用普通或 HAC 标准误，
    并打印每个系数的检验方法决策报告（不绘图）。

    新增参数：
      - alpha: float, 双尾检验显著性水平（例如 0.10 对应 90% 置信区间，默认 0.05=95%）
    """
    names = final_alphas["alpha"].tolist()

    # 1）构造次期收益
    ret_next = logret.shift(-shift)
    if shift > 0:
        ret_next = ret_next.iloc[:-shift]

    # 2）取公共日期
    dates = ret_next.index
    for n in names:
        dates = dates.intersection(alpha_dict[n].index)

    # 3）逐期横截面回归，收集 β_t
    betas_ts = []
    for date in dates:
        X_df = pd.DataFrame({n: alpha_dict[n].loc[date] for n in names})
        y    = ret_next.loc[date].rename("y")
        df   = pd.concat([y, X_df], axis=1).dropna()
        N_obs = df.shape[0]
        if N_obs == 0:
            continue

        # 剔除零方差因子
        fac_std = df[names].std(axis=0)
        valid   = fac_std[fac_std>0].index.tolist()
        p_req   = len(valid) + (1 if add_const else 0)
        if N_obs < p_req:
            continue

        X_part = df[valid].values
        if add_const:
            X_vals = np.column_stack([np.ones(N_obs), X_part])
            cols   = ["const"] + valid
        else:
            X_vals = X_part
            cols   = valid

        y_vals = df["y"].values
        try:
            if ridge > 0:
                XtX    = X_vals.T @ X_vals
                I      = np.eye(XtX.shape[0])
                beta_t = np.linalg.solve(XtX + ridge*I, X_vals.T @ y_vals)
            else:
                beta_t, *_ = np.linalg.lstsq(X_vals, y_vals, rcond=None)
        except LinAlgError:
            continue

        full_idx   = (['const'] if add_const else []) + names
        beta_full  = pd.Series(index=full_idx, dtype=float)
        beta_full.loc[cols] = beta_t
        betas_ts.append(beta_full)

    betas_df = pd.DataFrame(betas_ts)
    if betas_df.empty:
        raise ValueError("No valid cross‐sections for regression.")

    # 4）平均系数 & 期数
    T_k       = betas_df.count()
    beta_mean = betas_df.mean(skipna=True)

    # 5）准备两种标准误
    se_iid = betas_df.std(ddof=1) / np.sqrt(T_k)
    se_hac = hac_se_vectorized(betas_df, maxlags=acf_lags)

    # 6）因子级 ACF/PACF 检验 → 选择 se
    se_final   = pd.Series(index=beta_mean.index, dtype=float)
    method_map = {}
    for j in beta_mean.index:
        series = betas_df[j].dropna()
        Tj     = len(series)
        bound  = 1.96 / np.sqrt(Tj)

        acfs  = acf(series, nlags=acf_lags, fft=False)
        pacfs = pacf(series, nlags=acf_lags)
        if np.any(np.abs(acfs[1:]) > bound) or np.any(np.abs(pacfs[1:]) > bound):
            se_final[j]   = se_hac[j]
            method_map[j] = "HAC"
        else:
            se_final[j]   = se_iid[j]
            method_map[j] = "IID"

    # 7）t 统计量与双尾 p-value
    t_stat       = beta_mean / se_final
    df_res       = pd.DataFrame({
        "beta":   beta_mean,
        "se":     se_final,
        "t_stat": t_stat,
        "T_k":    T_k,
    })
    df_res["df"]      = df_res["T_k"] - 1
    df_res["p_value"] = 2 * (1 - t.cdf(np.abs(df_res["t_stat"]), df_res["df"]))

    # 8）显著性判定，列名依据置信度自动命名
    ci = int((1 - alpha) * 100)
    sig_col = f"sig_{ci}%"
    df_res[sig_col]    = df_res["p_value"] < alpha

    # 9）打印方法报告与显著性水平
    print(f"—— 自相关检验方法选择 ——")
    print(pd.Series(method_map).to_frame("method"))
    print(f"—— 双尾显著性水平 alpha = {alpha:.2%} (置信度 {ci}%) ——")

    return df_res






        
'''Key Operators'''

open_price     = load_open_price()
high           = load_high_price()
close          = load_close_price()
low            = load_low_price()
volume         = load_volume()
amount         = load_amount()
turnover       = load_turnover()
market_cap     = load_market_cap()
market_cap_float = load_market_cap_float()
zz1000         = load_zz1000_index()




div      = load_dividend(close_price=close)
bonus    = load_bonus(close_price=close)
split    = load_split(close_price=close)
logret   = load_logret(close, div, bonus, split)
amt_flag = load_amt_flag(amount)
list_flag= load_list_flag(close)
sample   = load_sample_space(amt_flag, list_flag)



# Raw data (shift once globally)
open_shifted        = open_price.shift(1)
high_shifted        = high.shift(1)
close_shifted       = close.shift(1)
low_shifted         = low.shift(1)
volume_shifted      = volume.shift(1)
amount_shifted      = amount.shift(1)
turnover_shifted    = turnover.shift(1)
market_cap_shifted  = market_cap.shift(1)

# Derived features (also shift once)
div_shifted      = div.shift(1)
bonus_shifted    = bonus.shift(1)
split_shifted    = split.shift(1)
logret_shifted   = logret.shift(1)


stock_sector_df = load_stock_sector_table(CONN)
industry_code_df = build_industry_code_df(stock_sector_df,start_date=None,end_date=None)
mapping_file_path = r"C:\Users\19874\OneDrive\桌面\九坤投资实习\申万分类\SwClassCode_2021.xls"
mapping_df = load_mapping_df(mapping_file_path,code_col="行业代码", level1_col="一级行业名称", level2_col="二级行业名称", level3_col="三级行业名称")

sector_df, industry_df, subindustry_df = build_sector_and_industry_dfs(industry_code_df,mapping_df)

sector_aligned         = align_factor_df_to_close(sector_df, close, default_value="UNK")
subindustry_aligned    = align_factor_df_to_close(subindustry_df, close, default_value="UNK")
industry_aligned       = align_factor_df_to_close(industry_df, close, default_value="UNK")

sector_aligned_shifted      = sector_aligned.shift(1)
subindustry_aligned_shifted = subindustry_aligned.shift(1)
industry_aligned_shifted    = industry_aligned.shift(1)

daily_counts = sample.sum(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, daily_counts.values, linewidth=2)
plt.title("Number of Stocks Satisfying Sample Space")
plt.xlabel("Date")
plt.ylabel("Number of Stocks")
plt.grid(True)
plt.tight_layout()
plt.show()


sector_aligned.loc['2023-01-03':'2025-06-13',:]


adv5_shifted   = volume.rolling(window=5,   min_periods=5).mean().shift(1)
adv10_shifted  = volume.rolling(window=10,  min_periods=10).mean().shift(1)
adv15_shifted  = volume.rolling(window=15,  min_periods=15).mean().shift(1)
adv20_shifted  = volume.rolling(window=20,  min_periods=20).mean().shift(1)
adv30_shifted  = volume.rolling(window=30,  min_periods=30).mean().shift(1)
adv40_shifted  = volume.rolling(window=40,  min_periods=40).mean().shift(1)
adv50_shifted  = volume.rolling(window=50,  min_periods=50).mean().shift(1)
adv60_shifted  = volume.rolling(window=60,  min_periods=60).mean().shift(1)
adv81_shifted  = volume.rolling(window=81,  min_periods=81).mean().shift(1)
adv120_shifted = volume.rolling(window=120, min_periods=120).mean().shift(1)
adv150_shifted = volume.rolling(window=150, min_periods=150).mean().shift(1)
adv180_shifted = volume.rolling(window=180, min_periods=180).mean().shift(1)

vwap_shifted = amount.div(volume).replace([np.inf, -np.inf], np.nan).shift(1)



pd.reset_option("all")




# ── Delay-1 Alphas ─────────────────────────────────────────────────────────
alpha_1   = load_alpha1( sl(close_shifted), sl(logret_shifted),      sl(sample) )
alpha_2   = load_alpha2( sl(close_shifted), sl(open_shifted), sl(volume_shifted), sl(sample) )
alpha_3   = load_alpha3( sl(open_shifted),  sl(volume_shifted),        sl(sample) )
alpha_4   = load_alpha4( sl(low_shifted),            sl(sample) )
alpha_5   = load_alpha5( sl(open_shifted), sl(close_shifted), sl(vwap_shifted),        sl(sample) )
alpha_6   = load_alpha6( sl(open_shifted), sl(volume_shifted),        sl(sample) )
alpha_7   = load_alpha7( sl(close_shifted), sl(adv20_shifted), sl(volume_shifted), sl(sample) )
alpha_8   = load_alpha8( sl(open_shifted),  sl(logret_shifted),        sl(sample) )
alpha_9   = load_alpha9( sl(close_shifted),            sl(sample) )
alpha_10  = load_alpha10( sl(close_shifted),           sl(sample) )
alpha_11  = load_alpha11(sl(close_shifted), sl(vwap_shifted), sl(volume_shifted), sl(sample))
alpha_12  = load_alpha12(sl(close_shifted), sl(volume_shifted),        sl(sample))
alpha_13  = load_alpha13(sl(close_shifted), sl(volume_shifted),        sl(sample))
alpha_14  = load_alpha14(sl(open_shifted), sl(volume_shifted), sl(logret_shifted), sl(sample))
alpha_15  = load_alpha15(sl(high_shifted), sl(volume_shifted),        sl(sample))
alpha_16  = load_alpha16(sl(high_shifted), sl(volume_shifted),        sl(sample))
alpha_17  = load_alpha17(sl(close_shifted), sl(volume_shifted), sl(adv20_shifted), sl(sample))
alpha_18  = load_alpha18(sl(close_shifted), sl(open_shifted),         sl(sample))
alpha_19  = load_alpha19(sl(close_shifted), sl(logret_shifted),       sl(sample))
alpha_20  = load_alpha20(sl(open_shifted), sl(high_shifted), sl(close_shifted), sl(low_shifted), sl(sample))
alpha_21  = load_alpha21(sl(close_shifted), sl(volume_shifted), sl(adv20_shifted), sl(sample))
alpha_22  = load_alpha22(sl(high_shifted), sl(volume_shifted), sl(close_shifted), sl(sample))
alpha_23  = load_alpha23(sl(high_shifted),           sl(sample))
alpha_24  = load_alpha24(sl(close_shifted),          sl(sample))
alpha_25  = load_alpha25(sl(logret_shifted), sl(adv20_shifted), sl(vwap_shifted), sl(high_shifted), sl(close_shifted), sl(sample))
alpha_26  = load_alpha26(sl(volume_shifted), sl(high_shifted),       sl(sample))
alpha_27  = load_alpha27(sl(volume_shifted), sl(amount_shifted),     sl(sample))
alpha_28  = load_alpha28(sl(adv20_shifted), sl(low_shifted), sl(high_shifted), sl(close_shifted), sl(sample))
alpha_29  = load_alpha29(sl(close_shifted), sl(logret_shifted),       sl(sample))
alpha_30  = load_alpha30(sl(close_shifted), sl(volume_shifted),      sl(sample))
alpha_31  = load_alpha31(sl(close_shifted), sl(adv20_shifted), sl(low_shifted), sl(sample))
alpha_32  = load_alpha32(sl(close_shifted), sl(vwap_shifted),        sl(sample))
alpha_33  = load_alpha33(sl(open_shifted), sl(close_shifted),        sl(sample))
alpha_34  = load_alpha34(sl(close_shifted), sl(logret_shifted),      sl(sample))
alpha_35  = load_alpha35(sl(close_shifted), sl(high_shifted), sl(low_shifted), sl(volume_shifted), sl(logret_shifted), sl(sample))
alpha_36  = load_alpha36(sl(close_shifted), sl(open_shifted), sl(volume_shifted), sl(vwap_shifted), sl(adv20_shifted), sl(logret_shifted), sl(sample))
alpha_37  = load_alpha37(sl(open_shifted), sl(close_shifted),        sl(sample))
alpha_38  = load_alpha38(sl(close_shifted), sl(open_shifted),        sl(sample))
alpha_39  = load_alpha39(sl(close_shifted), sl(volume_shifted), sl(adv20_shifted), sl(logret_shifted), sl(sample))
alpha_40  = load_alpha40(sl(high_shifted), sl(volume_shifted),       sl(sample))
alpha_41  = load_alpha41(sl(high_shifted), sl(low_shifted), sl(vwap_shifted), sl(sample))
alpha_43  = load_alpha43(sl(close_shifted), sl(volume_shifted), sl(adv20_shifted), sl(sample))
alpha_44  = load_alpha44(sl(high_shifted), sl(volume_shifted),       sl(sample))
alpha_45  = load_alpha45(sl(close_shifted), sl(volume_shifted),      sl(sample))
alpha_46  = load_alpha46(sl(close_shifted),         sl(sample))
alpha_47  = load_alpha47(sl(close_shifted), sl(high_shifted), sl(volume_shifted), sl(adv20_shifted), sl(vwap_shifted), sl(sample))
alpha_49  = load_alpha49(sl(close_shifted),         sl(sample))
alpha_50  = load_alpha50(sl(volume_shifted), sl(vwap_shifted),       sl(sample))
alpha_51  = load_alpha51(sl(close_shifted),         sl(sample))
alpha_52  = load_alpha52(sl(low_shifted), sl(logret_shifted), sl(volume_shifted), sl(sample))
alpha_55  = load_alpha55(sl(close_shifted), sl(high_shifted), sl(low_shifted), sl(volume_shifted), sl(sample))
alpha_56  = load_alpha56(sl(logret_shifted), sl(market_cap_shifted), sl(sample))
alpha_57  = load_alpha57(sl(close_shifted), sl(vwap_shifted),        sl(sample))
alpha_58  = load_alpha58(sl(vwap_shifted), sl(volume_shifted), sl(sector_aligned_shifted), sl(sample))
alpha_59  = load_alpha59(sl(vwap_shifted), sl(volume_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_60  = load_alpha60(sl(close_shifted), sl(high_shifted), sl(low_shifted), sl(volume_shifted), sl(sample))
alpha_61  = load_alpha61(sl(vwap_shifted), sl(adv180_shifted),      sl(sample))
alpha_62  = load_alpha62(sl(vwap_shifted), sl(adv20_shifted), sl(open_shifted), sl(high_shifted), sl(low_shifted), sl(sample))
alpha_63  = load_alpha63(sl(vwap_shifted), sl(adv180_shifted), sl(close_shifted), sl(open_shifted), sl(high_shifted), sl(low_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_64  = load_alpha64(sl(vwap_shifted), sl(adv120_shifted), sl(close_shifted), sl(high_shifted), sl(low_shifted), sl(open_shifted), sl(sample))
alpha_65  = load_alpha65(sl(vwap_shifted), sl(adv60_shifted), sl(open_shifted), sl(sample))
alpha_66  = load_alpha66(sl(vwap_shifted), sl(high_shifted), sl(low_shifted), sl(open_shifted), sl(sample))
alpha_67  = load_alpha67(sl(vwap_shifted), sl(adv20_shifted), sl(sector_aligned_shifted), sl(subindustry_aligned_shifted), sl(high_shifted), sl(sample))
alpha_68  = load_alpha68(sl(high_shifted), sl(adv15_shifted), sl(close_shifted), sl(low_shifted), sl(sample))
alpha_69  = load_alpha69(sl(vwap_shifted), sl(close_shifted), sl(adv20_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_70  = load_alpha70(sl(vwap_shifted), sl(close_shifted), sl(adv50_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_71  = load_alpha71(sl(close_shifted), sl(adv180_shifted), sl(low_shifted), sl(open_shifted), sl(vwap_shifted), sl(sample))
alpha_72  = load_alpha72(sl(high_shifted), sl(low_shifted), sl(adv40_shifted), sl(vwap_shifted), sl(volume_shifted), sl(sample))
alpha_73  = load_alpha73(sl(vwap_shifted), sl(high_shifted), sl(low_shifted), sl(open_shifted), sl(sample))
alpha_74  = load_alpha74(sl(close_shifted), sl(high_shifted), sl(vwap_shifted), sl(volume_shifted), sl(adv30_shifted), sl(sample))
alpha_75  = load_alpha75(sl(vwap_shifted), sl(volume_shifted), sl(low_shifted), sl(adv50_shifted), sl(sample))
alpha_76  = load_alpha76(sl(vwap_shifted), sl(volume_shifted), sl(low_shifted), sl(adv50_shifted), sl(adv81_shifted), sl(sector_aligned_shifted), sl(sample))
alpha_77  = load_alpha77(sl(high_shifted), sl(low_shifted), sl(vwap_shifted), sl(adv40_shifted), sl(sample))
alpha_78  = load_alpha78(sl(low_shifted), sl(vwap_shifted), sl(volume_shifted), sl(adv40_shifted), sl(sample))
alpha_79  = load_alpha79(sl(close_shifted), sl(open_shifted), sl(sector_aligned_shifted), sl(vwap_shifted), sl(adv150_shifted), sl(sample))
alpha_80  = load_alpha80(sl(open_shifted), sl(high_shifted), sl(volume_shifted), sl(adv10_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_81  = load_alpha81(sl(vwap_shifted), sl(volume_shifted), sl(adv10_shifted), sl(sample))
alpha_82  = load_alpha82(sl(open_shifted), sl(volume_shifted), sl(sector_aligned_shifted), sl(sample))
alpha_83  = load_alpha83(sl(high_shifted), sl(low_shifted), sl(close_shifted), sl(vwap_shifted), sl(volume_shifted), sl(sample))
alpha_84  = load_alpha84(sl(vwap_shifted), sl(close_shifted), sl(sample))
alpha_85  = load_alpha85(sl(high_shifted), sl(low_shifted), sl(close_shifted), sl(adv30_shifted), sl(volume_shifted), sl(sample))
alpha_86  = load_alpha86(sl(open_shifted), sl(close_shifted), sl(vwap_shifted), sl(adv20_shifted), sl(sample))
alpha_87  = load_alpha87(sl(close_shifted), sl(vwap_shifted), sl(adv81_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_88  = load_alpha88(sl(open_shifted), sl(high_shifted), sl(low_shifted), sl(close_shifted), sl(adv60_shifted), sl(sample))
alpha_89  = load_alpha89(sl(vwap_shifted), sl(close_shifted), sl(low_shifted), sl(adv10_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_90  = load_alpha90(sl(close_shifted), sl(low_shifted), sl(adv40_shifted), sl(subindustry_aligned_shifted), sl(sample))
alpha_91  = load_alpha91(sl(close_shifted), sl(volume_shifted), sl(vwap_shifted), sl(adv30_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_92  = load_alpha92(sl(high_shifted), sl(low_shifted), sl(close_shifted), sl(open_shifted), sl(adv30_shifted), sl(sample))
alpha_93  = load_alpha93(sl(close_shifted), sl(vwap_shifted), sl(adv81_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_94  = load_alpha94(sl(vwap_shifted), sl(adv60_shifted), sl(sample))
alpha_95  = load_alpha95(sl(open_shifted), sl(high_shifted), sl(low_shifted), sl(adv40_shifted), sl(sample))
alpha_96  = load_alpha96(sl(close_shifted), sl(low_shifted), sl(high_shifted), sl(volume_shifted), sl(adv60_shifted), sl(vwap_shifted), sl(sample))
alpha_97  = load_alpha97(sl(low_shifted), sl(vwap_shifted), sl(adv60_shifted), sl(industry_aligned_shifted), sl(sample))
alpha_98  = load_alpha98(sl(open_shifted), sl(vwap_shifted), sl(adv5_shifted), sl(adv15_shifted), sl(sample))
alpha_99  = load_alpha99(sl(high_shifted), sl(low_shifted), sl(volume_shifted), sl(adv60_shifted), sl(sample))
alpha_100 = load_alpha100(sl(close_shifted), sl(low_shifted), sl(high_shifted), sl(volume_shifted), sl(adv20_shifted), sl(subindustry_aligned_shifted), sl(sample))
alpha_101 = load_alpha101(sl(open_shifted), sl(high_shifted), sl(low_shifted), sl(close_shifted), sl(sample))

# ── Delay-0 Alphas ─────────────────────────────────────────────────────────
alpha_42 = load_alpha42_d0( sl(amount), sl(volume), sl(close), sl(sample) )
alpha_48 = load_alpha48_d0( sl(close), sl(subindustry_aligned), sl(sample) )
alpha_53 = load_alpha53_d0( sl(close), sl(high), sl(low), sl(sample) )
alpha_54 = load_alpha54_d0( sl(open_price), sl(high), sl(low), sl(close), sl(sample) )


ic_matrix, rank_ic_matrix, ic_summary = compute_ic_all(
    alphas = {
        "alpha_1":   alpha_1,
        "alpha_2":   alpha_2,
        "alpha_3":   alpha_3,
        "alpha_4":   alpha_4,
        "alpha_5":   alpha_5,
        "alpha_6":   alpha_6,
        "alpha_7":   alpha_7,
        "alpha_8":   alpha_8,
        "alpha_9":   alpha_9,
        "alpha_10":  alpha_10,
        "alpha_11":  alpha_11,
        "alpha_12":  alpha_12,
        "alpha_13":  alpha_13,
        "alpha_14":  alpha_14,
        "alpha_15":  alpha_15,
        "alpha_16":  alpha_16,
        "alpha_17":  alpha_17,
        "alpha_18":  alpha_18,
        "alpha_19":  alpha_19,
        "alpha_20":  alpha_20,
        "alpha_21":  alpha_21,
        "alpha_22":  alpha_22,
        "alpha_23":  alpha_23,
        "alpha_24":  alpha_24,
        "alpha_25":  alpha_25,
        "alpha_26":  alpha_26,
        "alpha_27":  alpha_27,
        "alpha_28":  alpha_28,
        "alpha_29":  alpha_29,
        "alpha_30":  alpha_30,
        "alpha_31":  alpha_31,
        "alpha_32":  alpha_32,
        "alpha_33":  alpha_33,
        "alpha_34":  alpha_34,
        "alpha_35":  alpha_35,
        "alpha_36":  alpha_36,
        "alpha_37":  alpha_37,
        "alpha_38":  alpha_38,
        "alpha_39":  alpha_39,
        "alpha_40":  alpha_40,
        "alpha_41":  alpha_41,
        "alpha_43":  alpha_43,
        "alpha_44":  alpha_44,
        "alpha_45":  alpha_45,
        "alpha_46":  alpha_46,
        "alpha_47":  alpha_47,
        "alpha_49":  alpha_49,
        "alpha_50":  alpha_50,
        "alpha_51":  alpha_51,
        "alpha_52":  alpha_52,
        "alpha_55":  alpha_55,
        "alpha_57":  alpha_57,
        "alpha_58":  alpha_58,
        "alpha_59":  alpha_59,
        "alpha_60":  alpha_60,
        "alpha_61":  alpha_61,
        "alpha_62":  alpha_62,
        "alpha_63":  alpha_63,
        "alpha_64":  alpha_64,
        "alpha_65":  alpha_65,
        "alpha_66":  alpha_66,
        "alpha_67":  alpha_67,
        "alpha_68":  alpha_68,
        "alpha_69":  alpha_69,
        "alpha_70":  alpha_70,
        "alpha_71":  alpha_71,
        "alpha_72":  alpha_72,
        "alpha_73":  alpha_73,
        "alpha_74":  alpha_74,
        "alpha_75":  alpha_75,
        "alpha_76":  alpha_76,
        "alpha_77":  alpha_77,
        "alpha_78":  alpha_78,
        "alpha_79":  alpha_79,
        "alpha_80":  alpha_80,
        "alpha_81":  alpha_81,
        "alpha_82":  alpha_82,
        "alpha_83":  alpha_83,
        "alpha_84":  alpha_84,
        "alpha_85":  alpha_85,
        "alpha_86":  alpha_86,
        "alpha_87":  alpha_87,
        "alpha_88":  alpha_88,
        "alpha_89":  alpha_89,
        "alpha_90":  alpha_90,
        "alpha_91":  alpha_91,
        "alpha_92":  alpha_92,
        "alpha_93":  alpha_93,
        "alpha_94":  alpha_94,
        "alpha_95":  alpha_95,
        "alpha_96":  alpha_96,
        "alpha_97":  alpha_97,
        "alpha_98":  alpha_98,
        "alpha_99":  alpha_99,
        "alpha_100": alpha_100,
        "alpha_101": alpha_101,
    },
    logret_df = logret,
    sample_df = sample,
)

# Delay-0 alphas: same pattern
ic_matrix_d0, rank_ic_matrix_d0, ic_summary_d0 = compute_ic_all(
    alphas = {
        "alpha_42": alpha_42,
        "alpha_48": alpha_48,
        "alpha_53": alpha_53,
        "alpha_54": alpha_54,
    },
    logret_df = logret,
    sample_df = sample,
)




''' 假设检验过程（已自动返回单侧 CI 下界，无需额外提取）'''


# ——— 0. 计算原始 mean_ic, mean_ric ———
orig_mean_ic_d1    = ic_matrix.mean(axis=0)
orig_mean_ric_d1   = rank_ic_matrix.mean(axis=0)
orig_mean_ic_d0    = ic_matrix_d0.mean(axis=0)
orig_mean_ric_d0   = rank_ic_matrix_d0.mean(axis=0)



# ========= 可调参数 =========
alpha_levels     = [0.05]
ci_level         = 0.95       # 单侧置信水平

# 正向检验  H0: μ ≤ pos_null_mean
pos_null_mean    = 0.0
# 反向检验  H0: μ ≥ neg_null_mean
neg_null_mean    = 0.0

# 经济显著性阈值
ic_thresh_pos   = 0.001   # IC   正向阈值
ric_thresh_pos  = 0.001   # Rank 正向阈值
ic_thresh_neg   = -0.001  # IC   反向阈值
ric_thresh_neg  = -0.001  # Rank 反向阈值



# ---------- 2. 直接对原始矩阵做检验 ----------
ic_hptest         = alpha_significance(ic_matrix,       alpha_levels, pos_null_mean, neg_null_mean, ci_level)
rank_ic_hptest    = alpha_significance(rank_ic_matrix,  alpha_levels, pos_null_mean, neg_null_mean, ci_level)
ic_hptest_d0      = alpha_significance(ic_matrix_d0,    alpha_levels, pos_null_mean, neg_null_mean, ci_level)
rank_ic_hptest_d0 = alpha_significance(rank_ic_matrix_d0,alpha_levels,pos_null_mean, neg_null_mean, ci_level)

# ---------- 3. 合并 ----------
d1 = ic_hptest.merge(rank_ic_hptest, left_index=True, right_index=True, suffixes=('_ic','_ric'))
d0 = ic_hptest_d0.merge(rank_ic_hptest_d0, left_index=True, right_index=True, suffixes=('_ic','_ric'))


d1_filtered = filter_df(d1).assign(delay='d1')
d0_filtered = filter_df(d0).assign(delay='d0')

# ---------- 5. 合并 ----------
final = (pd.concat([d1_filtered, d0_filtered])
           .reset_index().rename(columns={'index':'alpha'}))
final_alphas = pd.concat(
    [final[['alpha','delay']], final.apply(attach_orig_stats, axis=1), final.drop(columns=['alpha','delay'])],
    axis=1
)

print(final)






# ---------- 7. 输出 ----------
pd.set_option('display.max_columns', None, 'display.width', 150, 'display.max_colwidth', None)
print(final_alphas.sort_values('abs_orig_mean_ic', ascending=False))
print(final_alphas)


# 1. 构造一个 name → DataFrame 的映射
names = final_alphas["alpha"].tolist()
alpha_dict = {name: globals()[name] for name in names}

#设立显著性 0.05， 或者其它
fm_alpha = 0.10
fama_macbeth_results = fama_macbeth_with_p(final_alphas, alpha_dict, logret,
                              shift=1, add_const=True, ridge=0.0, acf_lags=5, alpha=fm_alpha)

print(fama_macbeth_results.sort_values('t_stat', ascending = False))


# 如果你只想看显著的因子：

fm_ci = int((1 - fm_alpha)*100)
sig_col = f"sig_{fm_ci}%"
sig_factors = fama_macbeth_results.index[fama_macbeth_results[sig_col]].tolist()
D0_ALPHAS = {"alpha_42", "alpha_48", "alpha_53", "alpha_54"}

alpha_ic = final_alphas.set_index('alpha')['orig_mean_ic']
sig_factors = [a for a in sig_factors if a not in D0_ALPHAS and abs(alpha_ic.loc[a]) >= 0.005]

print(f"筛选后 {sig_col} 双尾显著且 |IC|>=0.005 的 Alphas:", sig_factors)
print(alpha_ic.loc[sig_factors])

# 画 Pearson 相关的热力图
plot_alpha_corr_by_list(sig_factors, alpha_dict, corr_method="pearson")
corr_matrix = get_alpha_corr_matrix_by_list(sig_factors, alpha_dict, corr_method="pearson")
print(corr_matrix)











'''Back Testing'''




'''EGARCH & HMM Model'''


start_test  = (pd.to_datetime(end_train) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
end_test    = "2025-06-13"
# 裁剪 log-return
upper, lower   = np.log(2), np.log(0.5)
logret_clipped = logret.where((logret > lower) & (logret < upper))



EPS = 1e-8

# 1) 拟合 EGARCH 参数部分（不变）
# =============================================================================
# 1) EGARCH parameter fitting (unchanged)
def _fit_one(col, series, p, o, q, dist):
    if series.size < (p + o + q + 10):
        return col, None

    # 构造模型，关闭内部自动缩放
    am = arch_model(
        series,
        mean="Zero",
        vol="EGARCH",
        p=p, o=o, q=q,
        dist=dist,
        rescale=False
    )

    # 优先用默认 L-BFGS 进行快速拟合
    try:
        res = am.fit(
            disp="off",
            show_warning=False,
            tol=1e-4,            # 较宽松的收敛公差，加快速度
            options={"maxiter": 200}  # 最多迭代 200 步
        )
    except Exception:
        # 若默认方法失败，则改用 Nelder–Mead 保底
        res = am.fit(
            disp="off",
            show_warning=False,
            solver="nm",         # Nelder–Mead，无导数更稳
            tol=1e-6,            # 更严格的收敛
            options={"maxiter": 500}
        )

    # 返回参数字典
    return col, {k: float(v) for k, v in res.params.items()}

def fit_egarch_params_parallel(logret_df, p, o, q, dist, n_jobs):
    tasks = [(c, logret_df[c].dropna().to_numpy(), p, o, q, dist)
             for c in logret_df.columns]
    out = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one)(*t) for t in tasks
    )
    return {c: d for c, d in out if d is not None}

def filter_extreme(params, alpha_max=1.5, gamma_max=1.5, beta_max=0.90):
    return {
        s: p for s, p in params.items()
        if abs(p["alpha[1]"]) <= alpha_max
        and abs(p["gamma[1]"]) <= gamma_max
        and p["beta[1]"] < beta_max
    }

# 2) ω recalibration & vectorization (unchanged)
def _const_for_dist(nu: float | None) -> float:
    if nu is None or nu <= 2.0 or nu > 1e5:
        return math.sqrt(2.0 / math.pi)
    log_const = (
        0.5 * math.log(nu)
        + math.lgamma((nu - 1.0) / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma(nu / 2.0)
    )
    return math.exp(log_const)

def _params_to_vec(params, symbols, beta_max=0.9):
    K = len(symbols)
    omega_arr  = np.empty(K, dtype=np.float64)
    alpha_arr  = np.empty(K, dtype=np.float64)
    beta_arr   = np.empty(K, dtype=np.float64)
    gamma_arr  = np.empty(K, dtype=np.float64)
    const_arr  = np.empty(K, dtype=np.float64)

    for i, s in enumerate(symbols):
        p = params[s]
        ω0, β0 = p["omega"], p["beta[1]"]
        if β0 > beta_max:
            β1 = beta_max
            ω1 = ω0 * (1 - β1) / (1 - β0)
            omega_arr[i], beta_arr[i] = ω1, β1
        else:
            omega_arr[i], beta_arr[i] = ω0, β0
        alpha_arr[i] = p["alpha[1]"]
        gamma_arr[i] = p["gamma[1]"]
        const_arr[i] = _const_for_dist(p.get("nu", None))

    return omega_arr, alpha_arr, beta_arr, gamma_arr, const_arr

# 3) Revised Numba forward kernel
@njit
def _egarch_forward_numba(
    ret_mat, reest_idx,
    omega_arr, alpha_arr, beta_arr, gamma_arr, const_arr,
    ln_low, ln_high
):
    N, K = ret_mat.shape
    mv = np.empty(N, dtype=np.float64)
    mv[:] = np.nan  # Modified: initialize all to NaN

    prev_s2 = np.empty(K, dtype=np.float64)
    prev_e  = np.zeros(K, dtype=np.float64)

    # 1) warm-start unconditional variance
    for k in range(K):
        unc = omega_arr[0, k] / (1.0 - beta_arr[0, k])
        if unc < ln_low:    unc = ln_low
        elif unc > ln_high: unc = ln_high
        prev_s2[k] = math.exp(unc)

    # 2) segmented recursion
    S = omega_arr.shape[0]
    for seg in range(S):
        start, end = reest_idx[seg], reest_idx[seg + 1]
        ω, α, β, γ, c = (
            omega_arr[seg], alpha_arr[seg],
            beta_arr[seg], gamma_arr[seg],
            const_arr[seg]
        )
        for t in range(start, end):
            for k in range(K):
                ret = ret_mat[t, k]
                if math.isnan(ret):  # Modified: skip NaN and clear residual
                    prev_e[k] = 0.0
                    continue

                sqrt_s2 = math.sqrt(prev_s2[k])
                z = prev_e[k] / sqrt_s2

                ln_s2 = (
                    ω[k]
                    + β[k] * math.log(prev_s2[k])
                    + α[k] * (abs(prev_e[k]) / sqrt_s2 - c[k])
                    + γ[k] * z
                )
                if ln_s2 < ln_low:    ln_s2 = ln_low
                elif ln_s2 > ln_high: ln_s2 = ln_high

                s2 = math.exp(ln_s2)
                prev_s2[k], prev_e[k] = s2, ret

            # compute market volatility
            s_sum, cnt = 0.0, 0
            for k in range(K):
                if not math.isnan(prev_s2[k]):
                    s_sum += math.sqrt(prev_s2[k])
                    cnt   += 1
            mv[t] = s_sum / cnt if cnt > 0 else np.nan

    return mv

# 4) Revised rolling EGARCH function
def rolling_egarch_vol(
    logret: pd.DataFrame,
    p=1, o=1, q=1, dist="t",
    window=252, reest_freq=21,
    n_jobs=-1
) -> pd.Series:
    # Modified: keep NaN in clipped returns
    logret_clipped = logret.where((logret > np.log(0.5)) & (logret < np.log(2)))

    dates = logret_clipped.index
    N     = len(dates)

    # initial fit using days 0…window-1
    params0 = filter_extreme(
        fit_egarch_params_parallel(
            logret_clipped.iloc[:window], p, o, q, dist, n_jobs
        )
    )
    keep = list(params0.keys())

    # Modified: start first segment at t=0
    reest_pts = [0] + list(range(window, N, reest_freq))
    reest_pts.append(N)

    S, K = len(reest_pts), len(keep)
    omega_arr = np.empty((S,K), dtype=np.float64)
    alpha_arr = np.empty((S,K), dtype=np.float64)
    beta_arr  = np.empty((S,K), dtype=np.float64)
    gamma_arr = np.empty((S,K), dtype=np.float64)
    const_arr = np.empty((S,K), dtype=np.float64)

    ω, α, β, γ, c = _params_to_vec(params0, keep)
    omega_arr[0], alpha_arr[0], beta_arr[0], gamma_arr[0], const_arr[0] = ω, α, β, γ, c
    prev_params = params0.copy()

    for i in range(1, S):
        t0 = reest_pts[i-1]
        seg_data = logret_clipped.iloc[t0-window:t0][keep]
        params_k = filter_extreme(
            fit_egarch_params_parallel(seg_data, p, o, q, dist, n_jobs)
        )
        for s in keep:
            if s in params_k:
                prev_params[s] = params_k[s]
        ω, α, β, γ, c = _params_to_vec(prev_params, keep)
        omega_arr[i], alpha_arr[i], beta_arr[i], gamma_arr[i], const_arr[i] = ω, α, β, γ, c

    ret_mat = logret_clipped[keep].to_numpy(dtype=np.float64)
    mv_arr = _egarch_forward_numba(
        ret_mat,
        np.array(reest_pts, dtype=np.int64),
        omega_arr, alpha_arr, beta_arr, gamma_arr, const_arr,
        ln_low=-14, ln_high=-2
    )

    # ensure non-negative
    return pd.Series(mv_arr, index=dates).clip(lower=0.0)

# 5) HMM prediction (unchanged)
def _one_step_pred_prob(model, X):
    gamma = model.predict_proba(X)
    return (gamma[-1] @ model.transmat_) / gamma[-1].sum()

def rolling_hmm_probabilities(
    mv, window_days=30,
    n_components=3, n_iter=300,
    tol=1e-2, random_state=42,
    n_jobs=-1
):
    dates = mv.index.to_list()
    def fit_pred(i):
        X = mv.iloc[max(0, i-window_days):i].dropna().values.reshape(-1,1)
        if len(X) < n_components:
            return np.full(n_components, 1.0/n_components)
        try:
            m = GaussianHMM(n_components, "diag", n_iter=n_iter, tol=tol, random_state=random_state)
            perc = np.linspace(0, 100, n_components+1)[1:-1]
            m.means_init = np.percentile(X, perc).reshape(-1,1)
            m.fit(X)
            probs = _one_step_pred_prob(m, X)
            order = np.argsort(m.means_.flatten())
            return probs[order]
        except:
            return np.full(n_components, 1.0/n_components)

    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fit_pred)(i) for i in range(len(dates)))
    cols = [f"prob_{l}" for l in ["low","mid","high"][:n_components]]
    return pd.DataFrame(np.vstack(out), index=dates, columns=cols)

# ===== Invocation, printing, plotting (unchanged) =====

upper, lower = np.log(1.5), np.log(0.6)
logret_clipped = logret.where((logret > lower) & (logret < upper))

mv_all = rolling_egarch_vol(
    logret        = logret_clipped,
    p=1, o=1, q=1, dist="t",
    window        = 120,
    reest_freq    = 10,
    n_jobs        = -1
)

prob_matrix = rolling_hmm_probabilities(
    mv             = mv_all,
    window_days    = 60,
    n_components   = 3,
    n_iter         = 300,
    tol            = 1e-3,
    random_state   = 42,
    n_jobs         = -1
)

print("HMM 概率矩阵统计：")
print(prob_matrix.describe())
print("\n尾部几行：")
print(prob_matrix.tail())



plt.figure(figsize=(10,4))
plt.plot(prob_matrix.index, prob_matrix['prob_high'], label='prob_high')
plt.xlabel('Date'); plt.ylabel('Probability'); plt.legend()
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(mv_all.index, mv_all, label='mv_all')
plt.xlabel('Date'); plt.ylabel('Market Volatility')
plt.title('Rolloing EGARCH MV')
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

print(mv_all.describe())

print(prob_matrix)
print(mv_all)


counts = {
    'low' : (prob_matrix['prob_low']  > 0.5).sum(),
    'mid' : (prob_matrix['prob_mid']  > 0.5).sum(),
    'high': (prob_matrix['prob_high'] > 0.5).sum()
}
print(counts)








ic_summary.sort_values('ic_ir', ascending = False)










''' Back Testing Implementation'''




"""
Dual-Leg Softmax Back-tester
包含所有依赖函数：build_factor_dict、权重计算、_bt_loop、画图打印
数学模型为 HMM 三状态资金拆分 + 波动率分层 + Softmax 双腿选股，
交易/NAV/资金逻辑完全沿用原版 _bt_loop。
"""


# ============== 全局可调参数 ==============
PARAMS = dict(
    w_L=0.20, w_M=0.30, w_H=0.50,
    alpha_z=1.0, z_up=1.8, z_dn=-0.1,
    max_hold=10,
    alpha_vol=0.30,
    gamma_HF=5.0, gamma_LF=2.0,
    top_pct_HF=0.30, top_pct_LF=0.50,
    min_lf_pick=5,
    stop_loss_pct=0.07, take_profit_pct=0.15,
    portfolio_dd_pct=0.10,
    cooldown_days=1,
    delay_days=0
)






'''Note: alpha_vol is locked in the calculation
run compute_vol_proxy function and vol_proxy_sub before backtesting'''






# ============== 交易常数 ==============
INIT_CASH = 1e8
LOT       = 100
C_COMM    = 0.0003
C_IMP     = 0.0002
C_STAMP   = 0.0005
UP_20     = {"688","689","300","301"}  # 20% 涨跌停板标的

# ------------------------------------------------------------------------
# 1. 因子构建：build_factor_dict
# ------------------------------------------------------------------------
def build_factor_dict(alpha_names: list[str],
                      start: str, end: str,
                      dates: pd.Index | None = None) -> dict:
    """
    遍历 alpha_names，在 [start:end] 区间调用 load_<alpha> 或 load_<alpha>_d0；
    若提供 dates，则 reindex→ffill→bfill 保证无全 NaN 行。
    """
    out: dict = {}
    for name in alpha_names:
        base = name.replace("_", "")
        func_name = f"load_{base}_d0" if name in D0_ALPHAS else f"load_{base}"
        if func_name not in globals():
            raise NameError(f"未找到因子计算函数 {func_name}")
        func = globals()[func_name]

        sig = inspect.signature(func)
        args = []
        for p in sig.parameters.values():
            if p.name not in globals():
                raise NameError(f"{func_name} 缺少全局变量 {p.name}")
            val = globals()[p.name]
            args.append(val.loc[start:end] if isinstance(val, pd.DataFrame) else val)

        fac = func(*args).loc[start:end]
        if dates is not None:
            fac = fac.reindex(dates).ffill().bfill()
        out[name] = fac
    return out

# ------------------------------------------------------------------------
# 2. 波动代理 & 工具函数
# ------------------------------------------------------------------------
def _uplim_rate(code: str) -> float:
    bare = code[2:] if code.startswith(("sh", "sz")) else code
    return 0.20 if any(bare.startswith(p) for p in UP_20) else 0.10

def compute_vol_proxy(logret_df: pd.DataFrame) -> pd.DataFrame:
    """
    EWMA 波动代理：vp[t] = α * |r_t| + (1-α) * vp[t-1]
    """
    ret = logret_df.to_numpy(dtype=float)
    T, N = ret.shape
    vp = np.empty_like(ret)
    alpha = PARAMS['alpha_vol']
    vp[0] = np.nan_to_num(np.abs(ret[0]), nan=0.0)
    for t in range(1, T):
        abs_r = np.nan_to_num(np.abs(ret[t]), nan=vp[t-1])
        vp[t] = alpha * abs_r + (1 - alpha) * vp[t-1]
    return (pd.DataFrame(vp, index=logret_df.index, columns=logret_df.columns)
            .ffill().fillna(0.0))

def _softmax_vec(v: np.ndarray, g: float) -> np.ndarray:
    v = np.nan_to_num(v, nan=0.0)
    ex = np.exp(g * (v - v.max()))
    return ex / ex.sum() if ex.sum() > 0 else np.full_like(ex, 1 / len(ex))

# ------------------------------------------------------------------------
# 3. 双腿权重：build_dualleg_weights
# ------------------------------------------------------------------------
def build_dualleg_weights(
    dates: pd.Index, stocks: pd.Index,
    S_pos: np.ndarray, price_ok: np.ndarray,
    vol_proxy: pd.DataFrame, prob_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输出：W_pos, leg_flag, exit_flag
      leg_flag: 0=HF, 1=LF
      exit_flag: LF z<阈值 强制出场
    """
    T, N = S_pos.shape
    total = PARAMS['w_L'] + PARAMS['w_M'] + PARAMS['w_H']
    w_Ln, w_Mn, w_Hn = (PARAMS['w_L']/total,
                        PARAMS['w_M']/total,
                        PARAMS['w_H']/total)

    W_pos    = np.zeros_like(S_pos)
    leg_flag = np.zeros_like(S_pos, dtype=np.uint8)
    exit_flag= np.zeros_like(S_pos, dtype=np.uint8)

    for t in range(T):
        pL, pM, pH = prob_df.iloc[t][['prob_low','prob_mid','prob_high']]
        hf_share   = np.clip(pL*w_Ln + pM*w_Mn + pH*w_Hn, 0, 1)
        vp_row     = vol_proxy.iloc[t].to_numpy()

        med_v  = np.nanmedian(vp_row[price_ok[t] == 1])
        U_low  = [i for i in range(N) if vp_row[i] <= med_v and price_ok[t, i] == 1]
        U_high = [i for i in range(N) if vp_row[i] >  med_v and price_ok[t, i] == 1]

        # — HF 腿 —
        nh = max(1, int(len(U_low) * PARAMS['top_pct_HF']))
        hf_cand = sorted(U_low, key=lambda i: -S_pos[t, i])[:nh]
        if hf_cand:
            w_hf = _softmax_vec(S_pos[t, hf_cand], PARAMS['gamma_HF'])
            for k, i in enumerate(hf_cand):
                W_pos[t, i]    += hf_share * w_hf[k]
                leg_flag[t, i] = 0

        # — LF 腿 —
        nl = max(PARAMS['min_lf_pick'], int(len(U_high) * PARAMS['top_pct_LF']))
        lf_cand = sorted(U_high, key=lambda i: -S_pos[t, i])[:nl]
        if lf_cand:
            Zr = PARAMS['alpha_z'] * S_pos[t, lf_cand]
            pick_up = [j for j, z in enumerate(Zr) if z >  PARAMS['z_up']]
            pick_dn = [j for j, z in enumerate(Zr) if z <  PARAMS['z_dn']]

            if pick_up:
                w_lf = _softmax_vec(Zr[pick_up], PARAMS['gamma_LF'])
                for k, j in enumerate(pick_up):
                    i = lf_cand[j]
                    W_pos[t, i]    += (1 - hf_share) * w_lf[k]
                    leg_flag[t, i] = 1

            for j in pick_dn:
                i = lf_cand[j]
                exit_flag[t, i] = 1
                leg_flag[t, i]  = 1

    return W_pos, leg_flag, exit_flag

# ------------------------------------------------------------------------
# 4. 回测核心：_bt_loop（含 HF 平仓 & 涨跌停撮合）
# ------------------------------------------------------------------------
@nb.njit(cache=True, fastmath=True)
def _bt_loop(
    price: np.ndarray,
    W:     np.ndarray,
    price_ok: np.ndarray,
    sample_day: np.ndarray,
    lim_up_mat: np.ndarray,
    lim_dn_mat: np.ndarray,
    change_mat: np.ndarray,
    resume_mat: np.ndarray,
    leg_flag_mat: np.ndarray,
    exit_flag_mat: np.ndarray,
    max_hold: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    portfolio_dd_pct: float,
    cooldown_days: int,
    delay_days: int
):
    T, N   = price.shape
    c_fee  = C_COMM + C_IMP

    nav             = np.empty(T, dtype=np.float64)
    exec_arr        = np.zeros(T-1, dtype=np.float64)
    util_arr        = np.zeros(T-1, dtype=np.float64)
    sell_flow       = np.zeros(T-1, dtype=np.float64)
    buy_flow        = np.zeros(T-1, dtype=np.float64)
    sell_count      = np.zeros(T-1, dtype=np.int64)
    buy_count       = np.zeros(T-1, dtype=np.int64)
    fail_sell_count = np.zeros(T-1, dtype=np.int64)
    fail_buy_count  = np.zeros(T-1, dtype=np.int64)
    cum_pnl         = np.zeros(N, dtype=np.float64)
    theo_lot        = np.zeros((T-1, N), dtype=np.int64)
    exec_lot        = np.zeros((T-1, N), dtype=np.int64)
    sell_lot        = np.zeros((T-1, N), dtype=np.int64)

    cash            = INIT_CASH
    pos             = np.zeros(N, dtype=np.float64)
    entry_day       = -np.ones(N, dtype=np.int64)
    entry_price     = np.zeros(N, dtype=np.float64)
    next_trade_day  = np.zeros(N, dtype=np.int64)
    NAV_peak        = INIT_CASH
    drawdown_active = False
    block_until     = -1
    pending_exit    = np.zeros(N, dtype=np.uint8)

    entry_leg       = -np.ones(N, dtype=np.int8)  # 记录下单时腿类型

    for t in range(T-1):
        # 1) 组合回撤风控
        prev_nav = cash + np.dot(pos, price[t-1]) if t>0 else INIT_CASH
        if prev_nav > NAV_peak:
            NAV_peak = prev_nav
        if (not drawdown_active) and prev_nav < NAV_peak*(1-portfolio_dd_pct):
            for i in range(N):
                if pos[i] > 0:
                    pending_exit[i] = 1
            block_until     = t + cooldown_days
            drawdown_active = True
            NAV_peak        = prev_nav
        if drawdown_active and prev_nav >= NAV_peak:
            drawdown_active = False

        # 2) 个股层面平仓信号
        if t > 0:
            for i in range(N):
                if pos[i] > 0:
                    # HF T+1 必卖（基于 entry_leg）
                    if entry_leg[i] == 0 and (t - entry_day[i] >= 1):
                        pending_exit[i] = 1
                    # LF 超持仓天数卖
                    if entry_leg[i] == 1 and (t - entry_day[i] >= max_hold):
                        pending_exit[i] = 1
                    # LF z<阈值卖
                    if exit_flag_mat[t, i] == 1:
                        pending_exit[i] = 1
                    # 止盈/止损
                    ret_i = (price[t-1, i] - entry_price[i]) / entry_price[i]
                    if ret_i <= -stop_loss_pct or ret_i >= take_profit_pct:
                        pending_exit[i] = 1

        # 3) 卖出撮合：考虑跌停 & 交易日限制
        proceeds, sc, fs = 0.0, 0, 0
        for i in range(N):
            if pos[i] > 0 and pending_exit[i]:
                if change_mat[t, i] <= lim_dn_mat[t, i] or price_ok[t, i] == 0 or t < next_trade_day[i]:
                    fs += 1
                else:
                    lots = int(pos[i]/LOT)
                    if lots > 0:
                        val = lots*LOT*price[t, i]*(1-c_fee-C_STAMP)
                        proceeds    += val
                        cum_pnl[i]  += val
                        pos[i]      = 0.0
                        entry_day[i]   = -1
                        entry_price[i] = 0.0
                        pending_exit[i]= 0
                        sc         += 1
                        next_trade_day[i] = t + delay_days + 1
                        sell_lot[t, i]    = lots
        sell_flow[t], sell_count[t], fail_sell_count[t] = proceeds, sc, fs
        cash += proceeds

        # 4) 冷却期内跳过买入
        if t <= block_until:
            nav[t] = cash + np.dot(pos, price[t])
            continue

        # 5) 买入撮合：考虑涨停 & 延迟天数
        cash_left, bc, fb = cash, 0, 0
        samp = sample_day[t]
        if np.all(samp == 0):
            samp[:] = 1
        order_i = np.empty(N, dtype=np.int64)
        order_w = np.empty(N, dtype=np.float64)
        cnt = 0
        for i in range(N):
            if W[t, i] > 0 and samp[i]==1 and price_ok[t, i]==1 and pending_exit[i]==0:
                order_i[cnt] = i
                order_w[cnt] = W[t, i]
                cnt += 1

        # 权重降序排序
        for a in range(cnt):
            mx = a
            for b in range(a+1, cnt):
                if order_w[b] > order_w[mx]:
                    mx = b
            order_i[a], order_i[mx] = order_i[mx], order_i[a]
            order_w[a], order_w[mx] = order_w[mx], order_w[a]

        for k in range(cnt):
            i = order_i[k]
            if t < next_trade_day[i]:
                fb += 1
                continue
            if change_mat[t, i] >= lim_up_mat[t, i]:
                fb += 1
                continue

            p   = price[t, i]
            tgt = cash_left * order_w[k]
            theo = math.floor(tgt / (p * LOT))
            theo_lot[t, i] = theo
            lots = theo
            if lots <= 0 or cash_left < p * LOT:
                fb += 1
                continue
            cost = lots * LOT * p * (1 + c_fee)
            if cost > cash_left:
                lots = math.floor(cash_left / (p * LOT))
                theo_lot[t, i] = lots
                cost = lots * LOT * p * (1 + c_fee)
                if lots <= 0:
                    fb += 1
                    continue

            cash_left      -= cost
            pos[i]         += lots * LOT
            entry_day[i]    = t
            entry_price[i]  = p
            entry_leg[i]    = leg_flag_mat[t, i]  # 记录买入时的腿类型
            exec_lot[t, i]  = lots
            bc             += 1

        buy_flow[t], buy_count[t], fail_buy_count[t] = -(cash - cash_left), bc, fb
        cash = cash_left

        # 6) 记录 exec/util/NAV
        tv = np.sum(W[t, :])
        exec_arr[t] = 0.0 if tv == 0 else (INIT_CASH - cash_left) / (tv * INIT_CASH)
        util_arr[t] = (INIT_CASH - cash_left) / INIT_CASH
        nav[t]      = cash + np.dot(pos, price[t])

    nav[T-1] = cash + np.dot(pos, price[T-1])

    return (
        nav, exec_arr, util_arr,
        sell_flow, buy_flow, sell_count, buy_count,
        fail_sell_count, fail_buy_count,
        cum_pnl, theo_lot, exec_lot, sell_lot
    )

# ------------------------------------------------------------------------
# 5. 回测接口：run_backtest_dual_leg_vec（完整打印 & 绘图）
# ------------------------------------------------------------------------
def run_backtest_dual_leg_vec(
    close_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    factor_dict: dict,
    w_dict: dict,
    prob_df: pd.DataFrame,
    vol_proxy_df: pd.DataFrame,
    start: str,
    end: str,
    index_series: pd.Series|None=None
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    # 1) 数据准备
    seg_c = close_df.loc[start:end].ffill().bfill()
    seg_s = sample_df.loc[start:end]
    dates, stocks = seg_c.index, seg_c.columns
    T, N = len(dates), len(stocks)
    price    = seg_c.to_numpy(dtype=np.float64)
    price_ok = (~np.isnan(price)).astype(np.uint8)

    # 2) 涨跌停 / 涨跌幅 / 复牌
    uplim = np.array([_uplim_rate(c) for c in stocks], dtype=np.float64)
    lim_up = np.zeros((T, N)); lim_dn = np.zeros((T, N))
    for t in range(1, T):
        prev = price[t-1]
        lim_up[t] = np.where(np.isnan(prev), lim_up[t-1],
                             np.round(prev*(1+uplim)+1e-8,2)/prev - 1)
        lim_dn[t] = np.where(np.isnan(prev), lim_dn[t-1],
                             np.round(prev*(1-uplim)+1e-8,2)/prev - 1)
    change = np.zeros((T, N)); change[1:] = price[1:]/price[:-1] - 1
    change = np.clip(change, lim_dn, lim_up)
    resume = np.zeros((T, N), dtype=np.uint8)
    resume[1:] = ((price_ok[1:] == 1) & (price_ok[:-1] == 0)).astype(np.uint8)

    # 3) 因子打分 → S_pos
    ic_map = final_alphas.set_index('alpha')['orig_mean_ic'].to_dict()
    Zs = []
    for name, wt in w_dict.items():
        arr = factor_dict[name].loc[dates, stocks].to_numpy(dtype=np.float64)
        m   = np.nanmean(arr, axis=1, keepdims=True)
        sd  = np.nanstd(arr, axis=1, keepdims=True)
        z   = np.nan_to_num((arr - m) / np.where(sd == 0, 1, sd))
        if ic_map.get(name, 0) < 0:
            z = -z
        Zs.append(wt * z)
    S_pos = np.sum(Zs, axis=0)

    # 4) 构建权重矩阵 & 标记
    W, leg_flag, exit_flag = build_dualleg_weights(
        dates, stocks, S_pos, price_ok,
        vol_proxy_df.loc[dates],
        prob_df.loc[dates]
    )
    sample_day = (seg_s.to_numpy() == 1).astype(np.uint8)

    # 5) 主回测
    (nav, ex_arr, ut_arr,
     sell_flow, buy_flow, sell_cnt, buy_cnt,
     fail_sell_cnt, fail_buy_cnt,
     cum_pnl, theo_lot, exec_lot, sell_lot) = _bt_loop(
        price, W, price_ok, sample_day,
        lim_up, lim_dn, change, resume,
        leg_flag, exit_flag,
        PARAMS['max_hold'],
        PARAMS['stop_loss_pct'],
        PARAMS['take_profit_pct'],
        PARAMS['portfolio_dd_pct'],
        PARAMS['cooldown_days'],
        PARAMS['delay_days']
    )

    # 6) 交易前计划表 plan_df
    idx_dates = dates[:-1].repeat(N)
    idx_tcks  = np.tile(stocks, T-1)
    plan_df = pd.DataFrame({
        "date": idx_dates,
        "ticker": idx_tcks,
        "in_top": (W[:-1].reshape(-1) > 0),
        "greedy_selected": (theo_lot.reshape(-1) > 0),
        "theoretical_lots": theo_lot.reshape(-1)
    }).set_index(["date","ticker"]).sort_index()

    # 7) 交易执行表 exec_df
    sel = (exec_lot.reshape(-1) > 0)
    exec_df = pd.DataFrame({
        "date": idx_dates[sel],
        "ticker": idx_tcks[sel],
        "theoretical_lots": theo_lot.reshape(-1)[sel],
        "executed_lots": exec_lot.reshape(-1)[sel]
    })
    exec_df["ratio"] = exec_df["executed_lots"] / exec_df["theoretical_lots"].replace(0, np.nan)
    exec_df = exec_df.set_index(["date","ticker"]).sort_index()

    # 8) 交易日志 & 按净现金流排序 Top-10
    records = []
    for t in range(T-1):
        dt = dates[t]
        for i in range(N):
            bl = exec_lot[t,i]
            if bl > 0:
                pr = price[t,i]
                cf = -bl * LOT * pr * (1 + C_COMM + C_IMP)
                records.append([stocks[i], dt, "BUY", bl, pr, cf])
            sl = sell_lot[t,i]
            if sl > 0:
                pr = price[t,i]
                cf =  sl * LOT * pr * (1 - (C_COMM + C_IMP) - C_STAMP)
                records.append([stocks[i], dt, "SELL", sl, pr, cf])
    trades_df = pd.DataFrame(records,
                             columns=["ticker","date","action","lots","price","cash_flow"])
    trades_df = trades_df.set_index(["ticker","date"]).sort_index()

    net_pnl = trades_df.groupby(level="ticker")["cash_flow"].sum()
    top10   = net_pnl.sort_values(ascending=False).head(10)
    trade_top10_df = trades_df.loc[top10.index]

    # 暴露 DataFrame 供后续查询
    globals().update({
        "plan_df": plan_df,
        "exec_df": exec_df,
        "trade_top10_df": trade_top10_df
    })

    # 9) 结果拼装 & 打印 & 画图
    nav_s  = pd.Series(nav,  index=dates,       name="NAV")
    exec_s = pd.Series(ex_arr, index=dates[:-1], name="Exec Rate")
    util_s = pd.Series(ut_arr, index=dates[:-1], name="Util Rate")
    strat_ret = nav_s.pct_change().fillna(0.0)

    print("== 前5日净值 ==\n", nav_s.head(), "\n")
    print(f"Exec = {exec_s.mean():.2%}, Util = {util_s.mean():.2%}\n")

    # — 日度候选/成交/失败 可视化 —
    plt.figure(figsize=(12,3))
    plt.plot(dates[:-1], (W[:-1]>0).sum(axis=1), label="Weighted candidates")
    plt.plot(dates[:-1], buy_cnt,       label="Buy count")
    plt.plot(dates[:-1], fail_buy_cnt,  label="Failed Buys")
    plt.plot(dates[:-1], fail_sell_cnt, label="Failed Sells")
    plt.title("Daily Candidates & Trades"); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    # — NAV 曲线 —
    plt.figure(figsize=(12,4))
    nav_s.plot(title="NAV"); plt.grid(); plt.tight_layout(); plt.show()

    # — Exec / Util —
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6), sharex=True)
    exec_s.plot(ax=ax1, title="Exec Rate"); ax1.grid(True)
    util_s.plot(ax=ax2, title="Util Rate"); ax2.grid(True)
    plt.tight_layout(); plt.show()

    # — Greedy 候选数 & 理论/实际 lots 比例 —
    plt.figure(figsize=(12,4))
    plt.plot(dates[:-1], (theo_lot>0).sum(axis=1), label="Greedy Candidates")
    plt.title("Daily Candidate Counts"); plt.ylabel("Count"); plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

    theory_tot = theo_lot.sum(axis=1)
    exec_tot   = exec_lot.sum(axis=1)
    ratio_day  = np.where(theory_tot>0, exec_tot/theory_tot, np.nan)
    plt.figure(figsize=(12,3))
    plt.plot(dates[:-1], ratio_day)
    plt.title("Executed/Theoretical Lots Ratio"); plt.ylabel("Ratio"); plt.grid(); plt.tight_layout(); plt.show()

    # — 日度 & 累计收益 vs 指数 —
    plt.figure(figsize=(12,4))
    plt.plot(dates, strat_ret*100, label="Strategy Daily Return")
    if index_series is not None:
        idx_ser   = index_series.loc[dates].ffill()
        idx_ret   = idx_ser.pct_change().fillna(0.0)
        net_ret   = strat_ret - idx_ret
        plt.plot(dates, idx_ret*100, label="Index Daily Return")
        plt.plot(dates, net_ret*100, label="Hedged Daily Return")
    plt.ylabel("Daily Return (%)"); plt.title("Daily Returns"); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(dates, (1+strat_ret).cumprod()-1, label="Strategy Cum Return")
    if index_series is not None:
        plt.plot(dates, (1+idx_ret).cumprod()-1,  label="Index Cum Return")
        plt.plot(dates, (1+net_ret).cumprod()-1,  label="Hedged Cum Return")
    plt.ylabel("Cumulative Return (%)"); plt.title("Cumulative Returns"); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    # — 现金流 & 交易计数 —
    sf_s = pd.Series(sell_flow, index=dates[:-1], name="Sell Cash Net")
    bf_s = pd.Series(buy_flow,  index=dates[:-1], name="Buy Cost Net")
    fig, (ax3, ax4) = plt.subplots(2,1,figsize=(10,6), sharex=True)
    sf_s.plot(ax=ax3); ax3.axhline(0,linewidth=0.5); ax3.set_title("Sell Proceeds Net"); ax3.grid(True)
    bf_s.plot(ax=ax4); ax4.axhline(0,linewidth=0.5); ax4.set_title("Buy Cost Net");    ax4.grid(True)
    plt.tight_layout(); plt.show()

    sc_s = pd.Series(sell_cnt, index=dates[:-1], name="Sell Count")
    bc_s = pd.Series(buy_cnt,  index=dates[:-1], name="Buy Count")
    plt.figure(figsize=(12,4))
    sc_s.plot(label="Stocks Sold"); bc_s.plot(label="Stocks Bought")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend(); plt.title("Daily Transaction Counts"); plt.grid(); plt.tight_layout(); plt.show()

    # — Top 10 净收益条形图 —
    plt.figure(figsize=(10,4))
    top10.plot.bar()
    plt.title("Top 10 Stocks by Net Realized PnL")
    plt.ylabel("Net PnL (CNY)"); plt.xlabel("Ticker")
    plt.grid(axis="y"); plt.tight_layout(); plt.show()

    # — 回测报告 & 对比指标 —
    start_nv, end_nv = nav_s.iloc[0], nav_s.iloc[-1]
    total_rt         = end_nv / start_nv - 1
    days             = (dates[-1] - dates[0]).days; years = days / 365.0
    ann_ret          = (end_nv / start_nv)**(1/years) - 1
    ann_vol          = strat_ret.std() * np.sqrt(len(strat_ret) / years)
    sharpe           = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd               = nav_s / nav_s.cummax() - 1

    print("=== 回测报告 ===")
    print(f"Start NAV           : {start_nv:.2f}")
    print(f"End NAV             : {end_nv:.2f}")
    print(f"Total Return        : {total_rt:.2%}")
    print(f"Annualized Return   : {ann_ret:.2%}")
    print(f"Annualized Vol      : {ann_vol:.2%}")
    print(f"Sharpe Ratio        : {sharpe:.2f}")
    print(f"Max Drawdown        : {dd.min():.2%}\n")

    if index_series is not None:
        idx_ser           = index_series.loc[dates].ffill()
        idx_total_rt      = idx_ser.iloc[-1]/idx_ser.iloc[0] - 1
        hedged_total_rt   = net_ret.cumsum().iloc[-1]   # or (1+net_ret).cumprod()[-1]-1
        hedged_ann_ret    = (1+net_ret).cumprod().iloc[-1]**(1/years) - 1
        hedged_ann_vol    = net_ret.std() * np.sqrt(len(net_ret)/years)
        hedged_sharpe     = hedged_ann_ret / hedged_ann_vol if hedged_ann_vol>0 else np.nan
        hedged_dd         = net_ret.cumsum().cummax() - net_ret.cumsum()

        print("=== 对比指标 ===")
        print(f"Index Total Return    : {idx_total_rt:.2%}")
        print(f"Strategy Total Return : {total_rt:.2%}")
        print(f"Hedged Total Return   : {hedged_total_rt:.2%}")
        print(f"Average Daily Excess  : {net_ret.mean():.4%}\n")

        print("=== Hedged 年化指标 ===")
        print(f"Hedged Annualized Return : {hedged_ann_ret:.2%}")
        print(f"Hedged Annualized Vol    : {hedged_ann_vol:.2%}")
        print(f"Hedged Sharpe Ratio      : {hedged_sharpe:.2f}")
        print(f"Hedged Max Drawdown      : {hedged_dd.min():.2%}\n")

    return nav_s, exec_s, ut_arr



# ------------------------------------------------------------------------
# 6. 示例调用（完整一条龙）
# ------------------------------------------------------------------------

factor_dict   = build_factor_dict(
    sig_factors, start_test, end_test,
    dates=close.loc[start_test:end_test].index
)
w_dict        = {name: 1.0 for name in sig_factors}
prob_sub      = prob_matrix.loc[start_test:end_test]






vol_proxy_sub = compute_vol_proxy(logret.loc[start_test:end_test])
nav_s, exec_s, util_s = run_backtest_dual_leg_vec(
    close_df     = close,
    sample_df    = sample,
    factor_dict  = factor_dict,
    w_dict       = w_dict,
    prob_df      = prob_sub,
    vol_proxy_df = vol_proxy_sub,
    start        = start_test,
    end          = end_test,
    index_series = zz1000.squeeze()
)





# 现在可以：
plan_df.loc["2025-06-03"]
exec_df.loc["2024-02-05"]  #检查执行日股票的买卖数量

check_symbol = 'sh688578'
cf = trade_top10_df.loc[check_symbol]  # 获取该股票的所有交易记录
print(cf)
net_cf = cf['cash_flow'].sum()        # 计算净现金流
print(f"Net cashflow of stock {check_symbol}: {net_cf:.2f} CNY")










'''Barra CNE5 Model Factor Analysis'''


ret = (close / close.shift(1)) - 1
print(ret)



# 行业因子市值加权正交化

# ===================== 参数区 =====================
PATH_INDUSTRY_CSV = Path(r"C:/Users/19874/OneDrive/桌面/九坤投资实习/Barra Model/GICS_Industry.csv")

# 策略选项：未匹配股票怎么处理？
USE_UNK_INDUSTRY   = True      # True -> 未匹配股票归为 'UNK'
DROP_UNMATCHED     = False     # True -> 直接剔除未匹配股票
COVERAGE_THRESHOLD = 0.95      # 覆盖率阈值（仅在不开启 UNK/剔除时强制报错）
UNK_NAME           = 'UNK'

# ===================== 工具函数 =====================
def csv_to_close_ticker(csv_ticker: str) -> str:
    """600000.SS → sh600000 ; 000001.SZ → sz000001"""
    code, exch = csv_ticker.strip().split('.')
    return ('sh' if exch.upper() == 'SS' else 'sz') + code

def close_to_csv_ticker(close_ticker: str) -> str:
    """sh600000 → 600000.SS ; sz000001 → 000001.SZ"""
    prefix, code = close_ticker[:2], close_ticker[2:]
    return f"{code}.SS" if prefix == 'sh' else f"{code}.SZ"

# ===================== 1. 读取 & 清洗映射 =====================
map_df = pd.read_csv(PATH_INDUSTRY_CSV, dtype=str).rename(columns={
    'ticker'           : 'ticker_csv',
    'CNE5 Factor Name' : 'CNE5_Name'
})

required_cols = ['ticker_csv','CSV_Sector','CSV_Industry','CNE5_Name','GICS Sector']
missing_cols  = [c for c in required_cols if c not in map_df.columns]
if missing_cols:
    raise KeyError(f"GICS_Industry.csv 缺少必要列: {missing_cols}")

# 去空格
for c in ['ticker_csv','CSV_Sector','CSV_Industry','CNE5_Name','GICS Sector']:
    map_df[c] = map_df[c].str.strip()

# 转换为 close 风格代码，确保与矩阵统一
map_df['close_ticker'] = map_df['ticker_csv'].apply(csv_to_close_ticker)

# 同一只股票可能多行（不同分类来源），保留第一条或检查冲突
dup = map_df.duplicated(subset='close_ticker', keep=False)
if dup.any():
    # 检查是否真的有多个不同行业
    grp = map_df[dup].groupby('close_ticker')['CNE5_Name'].nunique()
    conflicts = grp[grp > 1]
    if len(conflicts) > 0:
        raise ValueError(f"同一股票在 CSV 中出现多个不同行业，请清洗：\n{conflicts}")
    # 若只是重复行但行业一样，直接去重
map_df = map_df.drop_duplicates(subset='close_ticker', keep='first')

# ===================== 2. 构建行业向量 =====================
# 以 close.columns 为股票池
univ_raw = [c for c in close.columns if c.startswith(('sh','sz'))]
ticker2ind = map_df.set_index('close_ticker')['CNE5_Name']
industry_vector = ticker2ind.reindex(univ_raw)
industry_vector.name = 'industry'

# ===================== 3. 覆盖率诊断 =====================
covered = industry_vector.notna().sum()
total   = len(industry_vector)
coverage = covered / total if total else 1.0
print(f"[INFO] 行业已匹配: {covered}/{total} ({coverage:.2%})")

unmatched = industry_vector[industry_vector.isna()]
if len(unmatched) > 0:
    print("WARN: 未匹配行业股票数 =", len(unmatched))
    print("示例：")
    print(unmatched.head(15))

if (not USE_UNK_INDUSTRY) and (not DROP_UNMATCHED) and coverage < COVERAGE_THRESHOLD:
    raise ValueError(f"行业覆盖率仅 {coverage:.2%}。请补 CSV 或启用 UNK / 剔除策略。")

# 处理未匹配
if USE_UNK_INDUSTRY:
    industry_vector = industry_vector.fillna(UNK_NAME)

if DROP_UNMATCHED:
    keep_mask = industry_vector.notna()
    dropped_n = (~keep_mask).sum()
    if dropped_n > 0:
        print(f"[INFO] 剔除未匹配股票 {dropped_n} 个")
    industry_vector = industry_vector[keep_mask]

# ===================== 4. 从 CSV 自动提取行业全集 =====================
all_inds_in_csv = sorted(industry_vector.unique())
print(f"[INFO] CSV 中实际行业数量: {len(all_inds_in_csv)}")

# 若 UNK 在里面，保持其在列末尾
if USE_UNK_INDUSTRY and UNK_NAME in all_inds_in_csv:
    all_inds = [ind for ind in all_inds_in_csv if ind != UNK_NAME] + [UNK_NAME]
else:
    all_inds = all_inds_in_csv

# ===================== 5. 构建 One-Hot 行业矩阵 =====================
industry_matrix = pd.get_dummies(industry_vector).astype(int)
industry_matrix = industry_matrix.reindex(columns=all_inds, fill_value=0)

# 每行应当恰好为 1（如果有 UNK 列，允许该列为 1）
check_cols = [c for c in all_inds if c != UNK_NAME] if USE_UNK_INDUSTRY else all_inds
row_sum = industry_matrix[check_cols].sum(axis=1)
bad_rows = row_sum[row_sum != 1]
if len(bad_rows) > 0:
    print("WARN: 以下股票行业指示不为单一 1：")
    print(bad_rows)

# ===================== 6. 国家矩阵（股票 × 1） =====================
country_matrix = pd.DataFrame(1, index=industry_matrix.index, columns=['Country_CN'])

# ===================== 7. 日度行业市值聚合 =====================
def agg_by_industry(df_mcap: pd.DataFrame, ind_mat: pd.DataFrame) -> pd.DataFrame:
    """
    df_mcap: (T × N_stock)
    ind_mat: (N_stock × J)
    return : (T × J)  = df_mcap @ ind_mat
    """
    df_mcap = df_mcap.loc[:, ind_mat.index].fillna(0.0)
    out = df_mcap.to_numpy(float) @ ind_mat.to_numpy(float)
    return pd.DataFrame(out, index=df_mcap.index, columns=ind_mat.columns)

# 如果 DROP_UNMATCHED，market_cap 也要裁剪；如果 UNK，仅需保持列一致
cols_keep = industry_matrix.index.tolist()
market_cap_aligned       = market_cap.reindex(columns=cols_keep)
market_cap_float_aligned = market_cap_float.reindex(columns=cols_keep)

# 对齐日期
common_dates = market_cap_aligned.index.intersection(market_cap_float_aligned.index)
market_cap_aligned       = market_cap_aligned.loc[common_dates]
market_cap_float_aligned = market_cap_float_aligned.loc[common_dates]



M_total = agg_by_industry(market_cap_aligned, industry_matrix)
M_float = agg_by_industry(market_cap_float_aligned, industry_matrix)

# ===================== 8. 输出检查 =====================
print("country_matrix :", country_matrix)
print("industry_vector:", industry_vector)
print("industry_matrix:", industry_matrix)
print("M_total        :", M_total)
print("M_float        :", M_float)



# 统一只看 A股代码（假设都是 sh/sz 开头）
cols_close = [c for c in close.columns if c.startswith(('sh','sz'))]

miss_in_total = set(cols_close) - set(market_cap.columns)
miss_in_float = set(cols_close) - set(market_cap_float.columns)

print("缺在 total_mv 的股票数:", len(miss_in_total))
print(sorted(list(miss_in_total))[:20])
print("缺在 circulating_mv 的股票数:", len(miss_in_float))
print(sorted(list(miss_in_float))[:20])







'''行业因子矩阵市值加权正交化处理'''


# ---------------- 1. 预处理 ----------------
def prepare_industry_for_orth(industry_matrix: pd.DataFrame,
                              country_matrix:  pd.DataFrame,
                              mv_float:        pd.DataFrame,
                              drop_missing:    bool = True,
                              drop_unk:        bool = True,
                              unk_name:        str = "UNK"):
    """
    预处理所有静态对象，只做一次；返回一个字典 ctx，后面 get_D_tilde_for_day 用。
    """
    # 对齐股票
    if drop_missing:
        stocks = mv_float.columns.intersection(industry_matrix.index)
    else:
        stocks = industry_matrix.index.union(mv_float.columns)

    D = industry_matrix.loc[stocks]
    C = country_matrix.loc[stocks]

    # 去掉 UNK 列
    if drop_unk and unk_name in D.columns:
        D = D.drop(columns=unk_name)

    # 保存 numpy，加速广播
    D_val = D.values.astype(np.float64, copy=False)
    # 常量向量 1_N
    ones_N = np.ones(len(stocks), dtype=np.float64)

    ctx = {
        "stocks":  stocks,           # 股票顺序
        "industries": D.columns,     # 行业列名
        "D_val":   D_val,            # (N×K)
        "C":       C,                # 国家矩阵(实际上只有1列，这里保留DataFrame形式)
        "mv_float": mv_float.loc[:, stocks],  # 重新对齐后的权重矩阵
        "ones_N":  ones_N,
    }
    return ctx


# ---------------- 2. 单日正交化 ----------------
def get_D_tilde_for_day(date, ctx):
    """
    给定日期，返回：
      X_t:  (N×(1+K)) 设计矩阵  [Country, D_tilde]
      w_t:  (N,)      权重（归一化后）
    若当日权重全0则返回 None, None
    """
    mv = ctx["mv_float"]
    if date not in mv.index:
        raise KeyError(f"{date} 不在 mv_float 的索引中")

    w = mv.loc[date].fillna(0.0).values.astype(np.float64)
    tot = w.sum()
    if tot <= 0:
        return None, None

    w_norm = w / tot                      # (N,)
    D_val  = ctx["D_val"]                 # (N×K)
    # s_j = w' * D
    s = w_norm @ D_val                    # (K,)
    # D_tilde = D - 1 * s
    D_tilde = D_val - s                   # 广播 (N×K) - (K,)
    # 约束检查
    if not np.allclose(w_norm @ D_tilde, 0, atol=1e-10):
        raise AssertionError("w' * D_tilde != 0")

    # 组合设计矩阵：国家列 + 正交化行业列
    country_col = ctx["ones_N"][:, None]  # (N×1)
    X_t = np.concatenate([country_col, D_tilde], axis=1)  # (N×(1+K))

    # 转回 DataFrame 方便你看
    cols = ["Country_CN"] + list(ctx["industries"])
    X_t_df = pd.DataFrame(X_t, index=ctx["stocks"], columns=cols)

    return X_t_df, pd.Series(w_norm, index=ctx["stocks"], name="weight")


# ---------------- 3. 使用示例 ----------------
# 假设你已有：
#   industry_matrix (N×K)
#   country_matrix  (N×1)
#   market_cap_float(T×N)
#   ret             (T×N) 横截面收益

# 1) 初始化
ctx = prepare_industry_for_orth(
    industry_matrix,     # 股票×行业 one-hot
    country_matrix,      # 股票×1，全1
    market_cap_float,    # 日期×股票，流通市值
    drop_missing=True,   # 只保留有流通市值的股票
    drop_unk=True,       # 去掉 UNK 行业列
    unk_name="UNK"
)

# 2) 某一天直接拿 X_t
day = market_cap_float.index[102]
coun_ind_otg_t, w_t = get_D_tilde_for_day(day, ctx)


coun_ind_otg_t
w_t

# 3) 约束检查：如果这个 max 值 ~ 1e-12 级别，就说明正交化成功。
check = (w_t.values @ coun_ind_otg_t.iloc[:, 1:].values)
print("max|w'D_tilde| =", np.abs(check).max())





'''风格因子-基本面-矩阵处理'''











