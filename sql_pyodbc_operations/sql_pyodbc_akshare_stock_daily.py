import akshare as ak
import pyodbc
import pandas as pd
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from json import JSONDecodeError


conn_str   = 'DSN,UID,PWD'  
table_name = "stock_a_daily"                  
start_date = "2010-01-01"                     

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_daily(symbol: str) -> pd.DataFrame:
    return ak.stock_zh_a_daily(symbol=symbol, start_date=start_date)

def main():
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.fast_executemany = True
            print("✔️ 数据库连接成功")

            ddl = f"""
IF OBJECT_ID(N'dbo.{table_name}', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.{table_name} (
        symbol      VARCHAR(10)   NOT NULL,
        trade_date  DATE          NOT NULL,
        [open]      DECIMAL(9,4)  NULL,
        high        DECIMAL(9,4)  NULL,
        low         DECIMAL(9,4)  NULL,
        [close]     DECIMAL(9,4)  NULL,
        volume      DECIMAL(15,2) NULL,
        amount      DECIMAL(20,2) NULL,
        turnover    DECIMAL(9,6)  NULL,
        update_time DATETIME      NOT NULL,
        CONSTRAINT PK_{table_name} PRIMARY KEY (symbol, trade_date)
    );
END
"""
            cursor.execute(ddl)
            conn.commit()
            print(f"✔️ 表 [{table_name}] 创建/检查成功")
            
            try:
                stock_df = ak.stock_info_a_code_name()
            except JSONDecodeError:
                time.sleep(0.5)
                stock_df = ak.stock_info_a_code_name()
            codes = stock_df['code'].astype(str).tolist()

            sse_prefixes  = {"600","601","603","605","688","689"}
            szse_prefixes = {"000","001","002","003","300","301"}

            symbols = []
            for c in codes:
                if any(c.startswith(p) for p in sse_prefixes):
                    symbols.append(f"sh{c}")
                elif any(c.startswith(p) for p in szse_prefixes):
                    symbols.append(f"sz{c}")

            print(f"✔️ 共获取 {len(symbols)} 只沪深交易所 A 股代码")

            failed = []
            for symbol in symbols:
                try:
                    df = fetch_daily(symbol)
                    if df.empty:
                        print(f"⚠️ {symbol} 无数据，跳过")
                        continue

                    df.reset_index(inplace=True)                       
                    df.rename(columns={'date':'trade_date'}, inplace=True)
                    df['symbol']      = symbol
                    df['update_time'] = datetime.now()
                    df['trade_date']  = pd.to_datetime(df['trade_date']).dt.date

                    df = df[[
                        'symbol','trade_date','open','high','low','close',
                        'volume','amount','turnover','update_time'
                    ]]

                    records = [tuple(row) for row in df.values]
                    insert_sql = f"""
INSERT INTO dbo.{table_name}
    (symbol, trade_date, [open], high, low, [close],
     volume, amount, turnover, update_time)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
                    cursor.executemany(insert_sql, records)
                    conn.commit()
                    print(f"✅ {symbol} 插入 {len(df)} 条")
                    time.sleep(0.1)  

                except Exception as e:
                    print(f"❌ {symbol} 处理失败: {e}")
                    conn.rollback()
                    failed.append(symbol)

            if failed:
                print("以下股票多次重试后仍失败，需要后续手动补跑：")
                print(failed)

    except pyodbc.Error as err:
        print(f"❌ 数据库操作失败: {err}")

    print("▶️ 脚本执行完毕")

if __name__ == "__main__":
    main()


