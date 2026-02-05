import akshare as ak
import pyodbc
import pandas as pd
from datetime import datetime


DSN = 'DSN,UID,PWD'

def create_table_if_not_exists():
    """create new table"""
    conn = pyodbc.connect(DSN)
    cursor = conn.cursor()
    create_sql = """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'stock_dividend_new')
    CREATE TABLE stock_dividend_new (
        symbol                  VARCHAR(10)      NOT NULL,
        ex_dividend_date        DATE             NOT NULL,
        name                    VARCHAR(100)     NULL,
        total_bonus_split       DECIMAL(9,4)     NULL,
        bonus_share             DECIMAL(9,4)     NULL,
        split_share             DECIMAL(9,4)     NULL,
        cash_dividend           DECIMAL(9,4)     NULL,
        dividend_yield          DECIMAL(9,4)     NULL,
        eps                     DECIMAL(9,4)     NULL,
        bps                     DECIMAL(9,4)     NULL,
        capital_reserve         DECIMAL(9,4)     NULL,
        undistributed_profit    DECIMAL(9,4)     NULL,
        net_profit_growth       DECIMAL(9,4)     NULL,
        total_shares            BIGINT           NULL,
        proposal_date           DATE             NULL,
        record_date             DATE             NULL,
        progress                VARCHAR(50)      NULL,
        latest_announcement     DATE             NULL,
        update_time             DATETIME         NOT NULL,
        CONSTRAINT PK_stock_dividend PRIMARY KEY (symbol, ex_dividend_date)
    )
    """
    cursor.execute(create_sql)
    conn.commit()
    cursor.close()
    conn.close()

def process_and_insert_data():
    """insert 2009-2025 data"""
    # 生成所有需要查询的日期（每年0630和1231）
    query_dates = [f"{year}{month}" for year in range(2009, 2026) for month in ['0630', '1231']]
    
    # 连接数据库
    conn = pyodbc.connect(DSN)
    cursor = conn.cursor()
    
    total_dates = len(query_dates)
    success_count = 0
    fail_count = 0

    for idx, date_str in enumerate(query_dates):
        try:
            print(f"[{datetime.now()}] 处理第 {idx+1}/{total_dates} 个日期：{date_str}")
            
            # 获取AkShare数据
            df = ak.stock_fhps_em(date=date_str)
            if df is None or df.empty:
                print(f"  无有效数据，跳过")
                continue
            
            # 数据清洗（关键步骤：转换为每股数据）
            df_clean = df.rename(columns={
                '代码': 'symbol',
                '名称': 'name',
                '送转股份-送转总比例': 'total_bonus_split',
                '送转股份-送转比例': 'bonus_share',
                '送转股份-转股比例': 'split_share',
                '现金分红-现金分红比例': 'cash_dividend',
                '现金分红-股息率': 'dividend_yield',
                '每股收益': 'eps',
                '每股净资产': 'bps',
                '每股公积金': 'capital_reserve',
                '每股未分配利润': 'undistributed_profit',
                '净利润同比增长': 'net_profit_growth',
                '总股本': 'total_shares',
                '预案公告日': 'proposal_date',
                '股权登记日': 'record_date',
                '除权除息日': 'ex_dividend_date',
                '方案进度': 'progress',
                '最新公告日期': 'latest_announcement'
            })

            # 关键处理：将每10股数据转换为每股（除以10）
            ratio_cols = ['total_bonus_split', 'bonus_share', 'split_share', 'cash_dividend']
            df_clean[ratio_cols] = df_clean[ratio_cols].apply(pd.to_numeric, errors='coerce') / 10

            # 日期字段转换（字符串转DATE）
            date_cols = ['proposal_date', 'record_date', 'ex_dividend_date', 'latest_announcement']
            for col in date_cols:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce').dt.date

            # 添加更新时间
            df_clean['update_time'] = datetime.now()

            # 插入数据库（使用MERGE避免重复）
            merge_sql = """
            MERGE INTO stock_dividend_new AS t
            USING (VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?))  
                AS s (symbol, ex_dividend_date, name, total_bonus_split, bonus_share, split_share, 
                      cash_dividend, dividend_yield, eps, bps, capital_reserve, undistributed_profit, 
                      net_profit_growth, total_shares, proposal_date, record_date, progress, latest_announcement, update_time)
            ON t.symbol = s.symbol AND t.ex_dividend_date = s.ex_dividend_date
            WHEN MATCHED THEN
                UPDATE SET 
                    name = s.name,
                    total_bonus_split = s.total_bonus_split,
                    bonus_share = s.bonus_share,
                    split_share = s.split_share,
                    cash_dividend = s.cash_dividend,
                    dividend_yield = s.dividend_yield,
                    eps = s.eps,
                    bps = s.bps,
                    capital_reserve = s.capital_reserve,
                    undistributed_profit = s.undistributed_profit,
                    net_profit_growth = s.net_profit_growth,
                    total_shares = s.total_shares,
                    proposal_date = s.proposal_date,
                    record_date = s.record_date,
                    progress = s.progress,
                    latest_announcement = s.latest_announcement,
                    update_time = s.update_time
            WHEN NOT MATCHED THEN
                INSERT (
                    symbol, ex_dividend_date, name, total_bonus_split, bonus_share, split_share, 
                    cash_dividend, dividend_yield, eps, bps, capital_reserve, undistributed_profit, 
                    net_profit_growth, total_shares, proposal_date, record_date, progress, latest_announcement, update_time
                )
                VALUES (
                    s.symbol, s.ex_dividend_date, s.name, s.total_bonus_split, s.bonus_share, s.split_share, 
                    s.cash_dividend, s.dividend_yield, s.eps, s.bps, s.capital_reserve, s.undistributed_profit, 
                    s.net_profit_growth, s.total_shares, s.proposal_date, s.record_date, s.progress, s.latest_announcement, s.update_time
                );
            """

            # 逐行插入（处理大数据量建议用批量插入，这里为清晰用逐行）
            row_count = 0
            for _, row in df_clean.iterrows():
                try:
                    values = (
                        row['symbol'],
                        row['ex_dividend_date'],
                        row['name'] if pd.notna(row['name']) else None,
                        row['total_bonus_split'] if pd.notna(row['total_bonus_split']) else None,
                        row['bonus_share'] if pd.notna(row['bonus_share']) else None,
                        row['split_share'] if pd.notna(row['split_share']) else None,
                        row['cash_dividend'] if pd.notna(row['cash_dividend']) else None,
                        row['dividend_yield'] if pd.notna(row['dividend_yield']) else None,
                        row['eps'] if pd.notna(row['eps']) else None,
                        row['bps'] if pd.notna(row['bps']) else None,
                        row['capital_reserve'] if pd.notna(row['capital_reserve']) else None,
                        row['undistributed_profit'] if pd.notna(row['undistributed_profit']) else None,
                        row['net_profit_growth'] if pd.notna(row['net_profit_growth']) else None,
                        row['total_shares'] if pd.notna(row['total_shares']) else None,
                        row['proposal_date'] if pd.notna(row['proposal_date']) else None,
                        row['record_date'] if pd.notna(row['record_date']) else None,
                        row['progress'] if pd.notna(row['progress']) else None,
                        row['latest_announcement'] if pd.notna(row['latest_announcement']) else None,
                        row['update_time']
                    )
                    cursor.execute(merge_sql, values)
                    row_count += 1
                except Exception as e:
                    print(f"  行插入失败：{e}")
                    fail_count += 1

            conn.commit()
            success_count += row_count
            print(f"  成功插入 {row_count} 行，累计成功 {success_count} 行，失败 {fail_count} 行")

        except Exception as e:
            print(f"  日期 {date_str} 处理失败：{e}")
            fail_count += 1
            continue

    cursor.close()
    conn.close()
    print(f"[{datetime.now()}] 全部处理完成！总成功：{success_count}，总失败：{fail_count}")

if __name__ == "__main__":
    create_table_if_not_exists()
    process_and_insert_data()
