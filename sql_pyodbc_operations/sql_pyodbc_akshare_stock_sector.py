import pandas as pd
import pyodbc
from datetime import datetime

CONN = pyodbc.connect("DSN,UID,PWD")

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
) -> (pd.DataFrame, pd.DataFrame):
    level1_map = mapping_df['level1']
    level3_map = mapping_df['level3']

    def map_to_level1(code: str) -> str:
        if code == 'UNK' or pd.isna(code):
            return 'UNK'
        return level1_map.get(code, 'UNK')

    def map_to_level3(code: str) -> str:
        if code == 'UNK' or pd.isna(code):
            return 'UNK'
        return level3_map.get(code, 'UNK')

    sector_df = industry_code_df.applymap(map_to_level1)
    industry_df = industry_code_df.applymap(map_to_level3)
    return sector_df, industry_df


stock_sector_df = load_stock_sector_table(CONN)
industry_code_df = build_industry_code_df(
stock_sector_df,
start_date=None,
end_date=None
)
mapping_file_path = r"C:\Users\19874\OneDrive\桌面\九坤投资实习\申万分类\SwClassCode_2021.xls"
mapping_df = load_mapping_df(
mapping_file_path,
code_col="行业代码",
level1_col="一级行业名称",
level2_col="二级行业名称",
level3_col="三级行业名称"
)
sector_df, industry_df = build_sector_and_industry_dfs(industry_code_df, mapping_df)
print(sector_df)
print(industry_df)

