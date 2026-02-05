import akshare as ak
import pyodbc
import pandas as pd
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from json import JSONDecodeError


conn_str   = 'DSN,UID,PWD'
table_name = "stock_a_share_cap"           


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_share_cap(code: str) -> pd.DataFrame:
    return ak.stock_value_em(symbol=code)

def main():
    try:
        # 1. 连接数据库
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.fast_executemany = True
            print("✔️ 数据库连接成功")

            # 2. 建表（若不存在），并确保列精度足够大
            ddl = f"""
IF OBJECT_ID(N'dbo.{table_name}', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.{table_name} (
        symbol          VARCHAR(10)    NOT NULL,
        data_date       DATE           NOT NULL,
        [close]         DECIMAL(15,4)  NULL,
        change_pct      DECIMAL(15,6)  NULL,
        total_mv        DECIMAL(20,2)  NULL,
        circulating_mv  DECIMAL(20,2)  NULL,
        total_share     DECIMAL(38,0)  NULL,
        float_share     DECIMAL(38,0)  NULL,
        pe_ttm          DECIMAL(15,4)  NULL,
        pe_static       DECIMAL(15,4)  NULL,
        pb              DECIMAL(15,4)  NULL,
        peg             DECIMAL(15,4)  NULL,
        pcf             DECIMAL(15,4)  NULL,
        ps              DECIMAL(15,4)  NULL,
        update_time     DATETIME       NOT NULL,
        CONSTRAINT PK_{table_name} PRIMARY KEY(symbol, data_date)
    );
END
"""
            cursor.execute(ddl)
            conn.commit()
            print(f"✔️ 表 [{table_name}] 创建/检查成功")

            # 3. 若已存在，再确保股本列最大精度
            for col in ('total_share', 'float_share'):
                try:
                    cursor.execute(f"""
ALTER TABLE dbo.{table_name}
ALTER COLUMN {col} DECIMAL(38,0) NULL;
""")
                    conn.commit()
                except:
                    pass

            # 4. 获取所有 A 股代码
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
            print(f"✔️ 共获取 {len(symbols)} 只沪深 A 股")

            # 5. 抓取并入库
            failed = []
            for symbol in symbols:
                code = symbol[2:]
                try:
                    df = fetch_share_cap(code)
                    if df is None or df.empty:
                        print(f"⚠️ {symbol} 无数据，跳过")
                        continue

                    # —— 重命名列 —— 
                    df = df.rename(columns={
                        '数据日期':   'data_date',
                        '当日收盘价': 'close',
                        '当日涨跌幅': 'change_pct',
                        '总市值':     'total_mv',
                        '流通市值':   'circulating_mv',
                        '总股本':     'total_share',
                        '流通股本':   'float_share',
                        'PE(TTM)':   'pe_ttm',
                        'PE(静)':    'pe_static',
                        '市净率':     'pb',
                        'PEG值':     'peg',
                        '市现率':     'pcf',
                        '市销率':     'ps',
                    })

                    # —— 强制数值 & round —— 
                    numeric_cols = [
                        'close','change_pct','total_mv','circulating_mv',
                        'total_share','float_share','pe_ttm','pe_static',
                        'pb','peg','pcf','ps'
                    ]
                    # 非法字符变 NaN
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # 四舍五入到表定义小数位
                    df['close']          = df['close'].round(4)
                    df['change_pct']     = df['change_pct'].round(6)
                    df['total_mv']       = df['total_mv'].round(2)
                    df['circulating_mv'] = df['circulating_mv'].round(2)
                    df['pe_ttm']         = df['pe_ttm'].round(4)
                    df['pe_static']      = df['pe_static'].round(4)
                    df['pb']             = df['pb'].round(4)
                    df['peg']            = df['peg'].round(4)
                    df['pcf']            = df['pcf'].round(4)
                    df['ps']             = df['ps'].round(4)

                    # 股本列转整数
                    df['total_share']    = df['total_share'].round().astype('Int64')
                    df['float_share']    = df['float_share'].round().astype('Int64')

                    # NaN → None
                    df = df.where(pd.notnull(df), None)

                    df['data_date']      = pd.to_datetime(df['data_date']).dt.date
                    df['symbol']         = symbol
                    df['update_time']    = datetime.now()

                    cols = [
                        'symbol','data_date','close','change_pct',
                        'total_mv','circulating_mv','total_share','float_share',
                        'pe_ttm','pe_static','pb','peg','pcf','ps','update_time'
                    ]
                    df = df[cols]

                    # —— 插入数据库 —— 
                    col_list = ', '.join('[close]' if c=='close' else c for c in cols)
                    placeholders = ', '.join('?' for _ in cols)
                    insert_sql = f"INSERT dbo.{table_name} ({col_list}) VALUES ({placeholders})"

                    records = [tuple(row) for row in df.values]
                    cursor.executemany(insert_sql, records)
                    conn.commit()
                    print(f"✅ {symbol} 插入 {len(df)} 条")
                    time.sleep(0.1)

                except Exception as e:
                    print(f"❌ {symbol} 失败: {e}")
                    conn.rollback()
                    failed.append(symbol)

            if failed:
                print("以下股票重试后仍失败：", failed)

    except pyodbc.Error as err:
        print(f"❌ 数据库操作出错: {err}")

    print("▶️ 脚本执行完毕")

if __name__ == "__main__":
    main()


