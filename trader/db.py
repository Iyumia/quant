# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
from collections.abc import Iterable
from sqlalchemy import create_engine, text



HOST = 'localhost'
PORT = 3306
USER = 'root'
PASSWORD = '419710'
DATABASE = 'quant'

# salalchemy
engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
conn = engine.connect()

# pymysql
connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
cursor = connection.cursor()


def get_data(sql):
    """使用SQL查询, 返回DataFrame表格"""
    data = pd.io.sql.read_sql(text(sql), con=conn)
    return data

def df_2_table(df, tablename, index=False, create=False):
    """将DataFrame 插入数据库表"""
    if create:
        df.to_sql(tablename, con=engine, index=index)
    else:
        df.to_sql(tablename, con=engine, if_exists='append', index=index)

def excute(sql):
    """单独执行SQL 语句, 主要用于个别数据的插入和更新等"""
    try:
        cursor.execute(sql)
        connection.commit()
    except Exception as e:
        raise ValueError(f'SQL Error:{sql}')

def get_data_magic(sql, database=DATABASE):
    """在jupyter-magic中调用"""
    engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{database}")
    conn = engine.connect()
    data = pd.io.sql.read_sql(text(sql), con=conn)
    return data


def get_price(security, start:str, end:str,fields:list=None):
    """
    获取股票行情数据
    """
    if isinstance(security, Iterable) and not isinstance(security, str):
        security = tuple(security)
        sql = f"""
        SELECT * 
        FROM quant.daily_candle
        WHERE `security` IN {security} 
        AND `date` BETWEEN '{start}' AND '{end}'
        ORDER BY `security`,`date`;
        """
    elif isinstance(security, str):
        sql = f"""
        SELECT * 
        FROM quant.daily_candle
        WHERE `security`='{security}' 
        AND `date` BETWEEN '{start}' AND '{end}'
        ORDER BY `date`;
        """
    else:
        raise TypeError('security must be str or list or Series!')

    data = get_data(sql)
    if fields is None:
        return data 
    else:
        return data.get(fields)
    
