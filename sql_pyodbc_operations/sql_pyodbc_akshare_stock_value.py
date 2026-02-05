import akshare as ak
import pyodbc
import pandas as pd
import time
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

conn_str        = 'DSN,UID,PWD'
valuation_table = "stock_valuation"
batch_size      = 1000   
sleep_interval  = 0.1   

sse_prefixes  = {"600", "601", "603", "605", "688", "689"}
szse_prefixes = {"000", "001", "002", "003", "300", "301"}

def add_prefix(code: str) -> str:
    """
    给6位代码加上交易所前缀，返回完整symbol（如 "sh600000" 或 "sz000001"）。
    如果不属于A股前缀范围，则返回 None。
    """
    for p in sse_prefixes:
        if code.startswith(p):
            return "sh" + code
    for p in szse_prefixes:
        if code.startswith(p):
            return "sz" + code
    return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_all_codes() -> pd.DataFrame:
    """
    拉取当前所有A股的spot数据，用于获取股票列表。
    这里只需要 symbol_code 列。
    """
    df = ak.stock_zh_a_spot_em()
    return df[['代码']].rename(columns={'代码': 'symbol_code'})

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_valuation_for_symbol(code: str) -> pd.DataFrame:
    """
    调用 ak.stock_a_indicator_lg 获取单只股票的估值数据（历史所有期）。
    返回 DataFrame，列名包含：
    ['trade_date', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_mv']
    """
    return ak.stock_a_indicator_lg(symbol=code)

def main():
    # —— 1. 获取所有A股代码 —— 
    try:
        codes_df = fetch_all_codes()
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        return

    # 提取去重后的 symbol_code 列，并筛选能加前缀的代码
    codes_df['symbol'] = codes_df['symbol_code'].astype(str).map(add_prefix)
    codes_df = codes_df[codes_df['symbol'].notna()]
    all_codes = codes_df['symbol_code'].astype(str).unique().tolist()
    print(f"ℹ️ 共获取到 {len(all_codes)} 支A股代码，将逐一拉取估值数据。")

    # —— 2. 连接数据库并创建表 stock_valuation —— 
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.fast_executemany = True

        # 如果表不存在，则创建新表
        ddl = f"""
IF OBJECT_ID(N'dbo.{valuation_table}', 'U') IS NULL
BEGIN
  CREATE TABLE dbo.{valuation_table} (
    symbol       VARCHAR(10)    NOT NULL,
    trade_date   DATE           NOT NULL,
    pe           DECIMAL(18,6)  NULL,
    pe_ttm       DECIMAL(18,6)  NULL,
    pb           DECIMAL(18,6)  NULL,
    ps           DECIMAL(18,6)  NULL,
    ps_ttm       DECIMAL(18,6)  NULL,
    dv_ratio     DECIMAL(18,6)  NULL,
    dv_ttm       DECIMAL(18,6)  NULL,
    total_mv     DECIMAL(20,2)  NULL,
    update_time  DATETIME       NOT NULL,
    CONSTRAINT PK_{valuation_table} PRIMARY KEY (symbol, trade_date)
  );
END
"""
        cursor.execute(ddl)
        conn.commit()

        total_inserted = 0

        # —— 3. 遍历每个股票代码，拉取并写入估值数据 —— 
        for idx, code in enumerate(all_codes, start=1):
            prefixed_symbol = add_prefix(code)
            if prefixed_symbol is None:
                continue  # 过滤掉非A股

            try:
                df_val = fetch_valuation_for_symbol(code)
            except Exception as e:
                print(f"⚠️ [{code}] 拉取估值数据失败: {e}，跳过该股票。")
                continue

            if df_val.empty:
                print(f"⚠️ [{code}] 拉取到空 DataFrame，跳过。")
                continue

            # 重命名、筛选字段
            df_val = df_val[[
                'trade_date', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                'dv_ratio', 'dv_ttm', 'total_mv'
            ]].copy()

            # 添加 symbol 列
            df_val['symbol'] = prefixed_symbol

            # 数据类型清洗
            df_val['trade_date'] = pd.to_datetime(df_val['trade_date'], errors='coerce').dt.date
            for col in ['pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm']:
                df_val[col] = pd.to_numeric(df_val[col], errors='coerce').round(6)
            df_val['total_mv'] = pd.to_numeric(df_val['total_mv'], errors='coerce').round(2)

            # 丢弃 trade_date 为空的行
            df_val.dropna(subset=['trade_date'], inplace=True)

            # 添加 update_time 字段
            now_ts = datetime.now()
            df_val['update_time'] = now_ts

            # 准备插入 DataFrame
            df_insert = df_val[[
                'symbol', 'trade_date', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                'dv_ratio', 'dv_ttm', 'total_mv', 'update_time'
            ]]

            # 按 batch_size 分批插入
            inserted_this_stock = 0
            for i in range(0, len(df_insert), batch_size):
                batch = df_insert.iloc[i:i + batch_size]
                records = [
                    (
                        row.symbol,
                        row.trade_date,
                        row.pe,
                        row.pe_ttm,
                        row.pb,
                        row.ps,
                        row.ps_ttm,
                        row.dv_ratio,
                        row.dv_ttm,
                        row.total_mv,
                        row.update_time
                    )
                    for row in batch.itertuples(index=False)
                ]

                insert_sql = f"""
INSERT INTO dbo.{valuation_table}
  (symbol, trade_date, pe, pe_ttm, pb, ps, ps_ttm,
   dv_ratio, dv_ttm, total_mv, update_time)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""
                try:
                    cursor.executemany(insert_sql, records)
                    conn.commit()
                    count_inserted = len(records)
                    inserted_this_stock += count_inserted
                except Exception as e:
                    print(f"❌ [{prefixed_symbol}] 插入批次失败: {e}")
                    conn.rollback()
                    break  # 如果某批失败，则跳出本股循环

            if inserted_this_stock > 0:
                total_inserted += inserted_this_stock
                print(f"✅ [{idx}/{len(all_codes)}] {prefixed_symbol} 共插入 {inserted_this_stock} 条估值记录。")

            # 暂停一段时间，防止接口或数据库压力过大
            time.sleep(sleep_interval)

        print(f"\n🎉 全部完成，共插入 {total_inserted} 条估值数据到 [{valuation_table}]。")

if __name__ == "__main__":
    main()

