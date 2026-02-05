import akshare as ak
import pyodbc
import pandas as pd
from datetime import datetime

# Step 1: Fetch PB data for all 4 indices
index_map = {
    "上证": "SH",
    "深证": "SZ",
    "创业板": "CYB",
    "科创版": "STAR"
}

df_list = []

for cn_name, en_code in index_map.items():
    df = ak.stock_market_pb_lg(symbol=cn_name)
    df.columns = [
        'trade_date',         
        'index_value',        
        'pb',                 
        'pb_weighted',        
        'pb_median'           
    ]
    df['market'] = en_code
    df_list.append(df)

# Step 2: Combine all DataFrames
df_all = pd.concat(df_list, ignore_index=True)
df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])

# Step 3: Connect to SQL Server
conn = pyodbc.connect('DSN,UID,PWD')
cursor = conn.cursor()

# Step 4: Create the market_index table
create_table_sql = """
IF OBJECT_ID('market_index', 'U') IS NOT NULL
    DROP TABLE market_index;

CREATE TABLE market_index (
    trade_date     DATE           NOT NULL,
    market         VARCHAR(10)    NOT NULL,
    index_value    DECIMAL(10, 2) NULL,
    pb             DECIMAL(10, 4) NULL,
    pb_weighted    DECIMAL(10, 4) NULL,
    pb_median      DECIMAL(10, 4) NULL,
    update_time    DATETIME       NOT NULL DEFAULT GETDATE(),
    PRIMARY KEY (trade_date, market)
)
"""
cursor.execute(create_table_sql)
conn.commit()

# Step 5: Prepare and insert data
insert_sql = """
INSERT INTO market_index (trade_date, market, index_value, pb, pb_weighted, pb_median, update_time)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

now = datetime.now()
data = [
    (
        row.trade_date.date(),
        row.market,
        row.index_value,
        row.pb,
        row.pb_weighted,
        row.pb_median,
        now
    )
    for _, row in df_all.iterrows()
]

cursor.executemany(insert_sql, data)
conn.commit()

print(f"✅ Inserted {len(data)} rows into [market_index].")

# Step 6: Cleanup
cursor.close()
conn.close()

