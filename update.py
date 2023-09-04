import time 
import talib
import smtplib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import wraps
from email.mime.text import MIMEText
from datetime import datetime, timedelta

from quant.utils import qcut
from quant.database.mysql import get_data
from quant.log import logger 
from quant.database.dolphin import selectBySQL


warnings.filterwarnings('ignore')


class LockDataError(Exception):
    """重写缺失最新股票数据的错误异常"""
    pass 
    

#******************* 公共函数 **************************
def intdate_to_date(intdate):
    """int日期类型转datetime"""
    strdate = str(intdate)
    date = datetime.strptime(strdate, '%Y%m%d').date()
    return date

def get_trade_date(days=0):
    """获取交易日历"""
    if days<=0:
        sql = f"select * from `trade_day`"
    else:
        fromdate = get_from_date(days=days).strftime('%Y%m%d')
        intfromdate = int(fromdate)
        sql = f"select * from `trade_day` where `date`>={intfromdate}"
    date_df = get_data(sql)
    return date_df

def get_now_date(test=False):
    """ 获取当前日期，以int形式返回 """
    now = datetime.now()
    if now.hour<15 or test:
        # 还未休市, now=前一日
        now = (now-timedelta(days=1)).date()
    else:
        # 已休市, now=当日
        now = now.date()
    return now

def get_from_date(days, test=False):
    """ 获取指定日期days日以前的日期 """
    now = get_now_date(test)
    fromdate = now-timedelta(days=days)
    return fromdate

def get_last_trade_date(date):
    """ 获取最近的一个交易日期 """
    date_array = get_trade_date().date.values
    last_trade_date = date_array[np.where(date_array<=date)][-1]
    return last_trade_date

def get_last_2_trade_date(date):
    """ 获取最近的一个交易日期 """
    date_array = get_trade_date().date.values
    last_2_trade_date = date_array[np.where(date_array<=date)][-2]
    return last_2_trade_date

def get_forward_trade_date(test=False):
    now_date = int(get_now_date(test=test).strftime('%Y%m%d'))
    date_array = get_trade_date().date.values
    forward_trade_date = date_array[np.where(date_array>now_date)][0]
    return forward_trade_date

def datascan(function):
    """对从Mysql数据库查询的数据进行初步判断"""
    now = int(get_now_date().strftime('%Y%m%d'))
    last_trade_date = get_last_trade_date(now)
    @wraps(function)
    def warp(*args, **kwargs):
        data = function(*args, **kwargs)
        data = data.drop_duplicates(subset=['date','code'])
        new = data.date.max()
        if new<last_trade_date:
            logger.error(f'{function.__name__}缺少股票最新数据, 最新数据截止到:{new}, 需要{last_trade_date}')
            raise LockDataError(f'{function.__name__}缺少股票最新数据, 最新数据截止到:{new}, 需要{last_trade_date}')
        else:
            return data
    return warp

# ****************** 加载最新基础数据 *********************
@datascan
def get_stock_st_data(days:int, test=False):
    """ 获取股票st数据, 数据来源: securities_extra"""
    fromdate = get_from_date(days=days, test=test).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select `date`,`code` from `securities_extras` where `date`>{intfromdate}"
    data = get_data(sql)
    data['is_st'] = True
    return data

@datascan
def get_valuation_data(days:int, fields=[], test=False):
    """ 获取股票估值等数据, 数据来源: valuation_data"""
    fromdate = get_from_date(days=days, test=test).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select * from `valuation_data` where `date`>{intfromdate} order by `code`,`date`"
    data = get_data(sql)
    if fields:
        data = data.get(fields)
    return data

@datascan
def get_stock_day_k_line_data(days=250, fields=[], test=False):
    """ 获取股票后复权数据, 数据来源: stock_day_k_line_data """
    fromdate = get_from_date(days=days, test=test).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select * from `stock_day_k_line_data` where `date`>{intfromdate} and `close`>0 order by `code`,`date`;"
    data = get_data(sql)
    if fields:
        data = data.get(fields)
    return data

@datascan
def get_stock_day_k_line_data_no_fq(days, fields=[], test=False):
    """ 获取股票后复权数据, 数据来源: stock_day_k_line_data_no_fq """
    fromdate = get_from_date(days=days, test=test).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select * from `stock_day_k_line_data_no_fq` where `date`>{intfromdate} and `close`>0 order by `code`,`date`;"
    data = get_data(sql)
    if fields:
        data = data.get(fields)
    return data

def get_securities_stock():
    """获取股票上市日期等基本数据, 数据来源: securities_stock """
    sql = f"select `code`,`name`,date(`start_date`) as `start_date` from `securities_stock`;"
    data = get_data(sql).drop_duplicates(subset=['code'])
    return data

@datascan
def get_money_flow(days, fields=[], test=False):
    """ 获取现金流等数据, 数据来源:money_flow """
    fromdate = get_from_date(days=days,test=test).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select * from `money_flow` where `date`>{intfromdate};"
    data= get_data(sql)
    if fields:
        data = data.get(fields)
    return data

def get_locked_shares(days=10, test=False):
    """ 获取股票解禁数据, 数据来源:locked_shares """
    fromdate = get_from_date(days, test=test).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select `date`,`code`,`rate1` as `rate` from `locked_shares` where `date`>{intfromdate} order by `code`,`date`;"
    data = get_data(sql).drop_duplicates(subset=['code','date'])
    data = data[~data.rate.str.isalpha()]
    data['rate'] = data['rate'].astype(float)
    data['release_date'] = data['date'].apply(intdate_to_date)
    return data

@datascan
def get_index_stock_data(index, days:int, test=False):
    """获取指数成分股, 数据来源:index_stock_data"""
    fromdate = get_from_date(days=days, test=test).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select `index`,`date`,`code` from `index_stock_data` where `index`='{index}' and `date`>{intfromdate};"
    data= get_data(sql)
    return data

@datascan
def get_index_day_k_line_data(index, days:int, fields=[], test=False):
    """获取指数行情数据, 数据来源:index_day_k_line_data"""
    fromdate = get_from_date(days=days, test=test).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select * from `index_day_k_line_data` where `code`='{index}' and `date`>{intfromdate} and `close`>0 order by `date`;"
    data = get_data(sql)
    if fields:
        data = data.get(fields)
    return data

@datascan
def get_sw1_daily_price(days:int, fields=[], test=False):
    """获取所有申万一级行业的行情数据, 数据来源:SW1_daily_price"""
    fromdate = get_from_date(days=days, test=False).strftime('%Y%m%d')
    intfromdate = int(fromdate)
    sql = f"select * from `SW1_daily_price` where `date`>{intfromdate} and `close`>0 order by `date`;"
    data = get_data(sql)
    if fields:
        data = data.get(fields)
    return data


# ************* 加载股票是否异常数据 *******************
def get_stock_is_st_data(days:int, test=False):
    """ 获取股票是否为st数据 """
    stock_candle_data = get_stock_day_k_line_data(days, fields=['date','code'], test=test)
    stock_st_data = get_stock_st_data(days,test=test)
    stock_is_st_data = pd.merge(stock_candle_data, stock_st_data, on=['date','code'], how='left')
    stock_is_st_data['is_st'] = stock_is_st_data['is_st'].fillna(False)
    return stock_is_st_data

def get_stock_is_new_data(days:int, ssts=30, test=False):
    """获取股票是否为新股数据"""
    # 获取行情数据
    stock_candle_data = get_stock_day_k_line_data(days, fields=['date','code'], test=test)
    stock_candle_data['now'] = stock_candle_data['date'].apply(intdate_to_date)
    # 获取上市日期数据
    stock_list_data = get_securities_stock()
    stock_is_new_data = pd.merge(stock_candle_data, stock_list_data, on=['code'], how='left')
    stock_is_new_data['is_new'] = (stock_is_new_data['now'] - stock_is_new_data['start_date']).apply(lambda x: True if x.days<ssts else False)
    stock_is_new_data = stock_is_new_data.get(['date','code','is_new'])
    return stock_is_new_data
    
def get_stock_is_release_data(days:int, release_day:int=180, ratio:float=0.05, test=False):
    """获取股票是否为即将解禁的数据""" 
    # 股票数据
    stock_candle_data = get_stock_day_k_line_data(days=days, fields=['date','code'], test=test)
    stock_candle_data['now'] = stock_candle_data['date'].apply(intdate_to_date)
    # 解禁数据
    locked_shares = get_locked_shares(days=days, test=test)
    # 合并后向前填充
    merge = pd.merge(stock_candle_data, locked_shares, on=['date','code'], how='outer')
    merge[['rate','release_date']] = merge.groupby(['code'])[['rate','release_date']].fillna(method='bfill')
    # 排除未来数据
    merge = merge[~merge['now'].isna()]
    # 计算下次解禁到现在的间隔日期
    merge['day'] = (merge['release_date'] - merge['now']).apply(lambda x: x.days)
    # 根据解禁比率和距今日期判断是否为解禁股票
    merge.loc[(merge['day']<release_day) & (merge['rate']>ratio), 'is_release'] = True
    merge['is_release'] = merge['is_release'].fillna(False)
    stock_is_release_data = merge.get(['date','code','is_release'])
    return stock_is_release_data 

def get_stock_is_recover_data(days:int, paused_day:int=10, recover_day:int=10, test=False):
    """计算股票是否为停牌超过10个交易日,复牌不满5个交易日的数据"""
    def cal_is_recover(group):
        group['is_recover0'] = group['paused'].rolling(paused_day).sum().apply(lambda x: 1 if x==paused_day else 0).rolling(recover_day).sum().apply(lambda x: True if x>0 else False)
        group.loc[(group['paused']==0) & (group['is_recover0']),'is_recover'] = True 
        group['is_recover'] = group['is_recover'].fillna(False)
        return group
    stock_candle_data = get_stock_day_k_line_data(days=days, fields=['date','code','paused'], test=test)
    stock_candle_data = stock_candle_data.groupby('code').apply(cal_is_recover)
    stock_is_recover_data = stock_candle_data.get(['date','code','is_recover'])
    return stock_is_recover_data

def get_stock_is_limit_data(days:int, test=False):
    # 计算股票是否涨停、跌停、一字涨停、一字跌停
    stock_candle_data = get_stock_day_k_line_data(days=days, fields=['date','code','open','high','low','close','pre_close'],test=False)
    def cal_is_limit(group):
        # 最大10%涨跌幅的股票涨停
        group.loc[(group['close']>group['pre_close']*1.095) & (~(group['code'].str.startswith('30') | group['code'].str.startswith('68'))), 'is_zt'] = True
        # 最大10%涨跌幅的股票跌停
        group.loc[(group['close']<group['pre_close']*0.905) & (~(group['code'].str.startswith('30') | group['code'].str.startswith('68'))), 'is_dt'] = True

        # 最大20%涨跌幅的股票涨停
        group.loc[(group['close']>group['pre_close']*1.195) & (group['code'].str.startswith('30') | group['code'].str.startswith('68')), 'is_zt'] = True
        # 最大20%涨跌幅的股票跌停
        group.loc[(group['close']<group['pre_close']*0.805) & (group['code'].str.startswith('30') | group['code'].str.startswith('68')), 'is_dt'] = True

        # 最大10%涨跌幅的股票一字涨停
        group.loc[(group['close']>group['pre_close']*1.095) & (group['high']==group['low']) & (~(group['code'].str.startswith('30') | group['code'].str.startswith('68'))), 'is_yzzt'] = True
        # 最大10%涨跌幅的股票一字跌停
        group.loc[(group['close']<group['pre_close']*0.905) & (group['high']==group['low']) & (~(group['code'].str.startswith('30') | group['code'].str.startswith('68'))), 'is_yzdt'] = True

        # 最大20%涨跌幅的股票一字涨停
        group.loc[(group['close']>group['pre_close']*1.195) & (group['high']==group['low']) & (group['code'].str.startswith('30') | group['code'].str.startswith('68')), 'is_yzzt'] = True
        # 最大20%涨跌幅的股票一字跌停
        group.loc[(group['close']<group['pre_close']*0.805) & (group['high']==group['low']) & (group['code'].str.startswith('30') | group['code'].str.startswith('68')), 'is_yzdt'] = True
        # 没有记录的默认为False
        group[['is_zt','is_dt','is_yzzt','is_yzdt']] = group[['is_zt','is_dt','is_yzzt','is_yzdt']].fillna(False)
        group = group.get(['date','code','is_zt','is_dt','is_yzzt','is_yzdt'])
        return group
    stock_is_limit_data = stock_candle_data.groupby('code').apply(cal_is_limit)
    return stock_is_limit_data
    

#  ************* 计算最新的因子数据 ***********************
def get_liquidity_data(days:int=400,test=False):
    """
    计算liquidity最新数据
    定义：liquidity = 0.35•share_turnover_monthly + 0.35•average_share_turnover_quarterly + 0.3•average_share_turnover_annual
    share_turnover_monthly:  过去21日的股票换手率之和的对数
    average_share_turnover_quarterly: 过去3个月的平均turnover_ratio，并取对数
    average_share_turnover_annual:过去12个月的平均turnover_ratio，并取对数
    """
    stock_turnover_ratio_data = get_valuation_data(days, fields=['date','code','turnover_ratio'], test=test)
    def cal_liquidity(group):
        group['share_turnover_monthly'] = group['turnover_ratio'].rolling(21).sum()
        group['share_turnover_monthly'] = np.log(group['share_turnover_monthly'])
        group['average_share_turnover_quarterly'] = group['turnover_ratio'].rolling(63).mean()
        group['average_share_turnover_quarterly'] = np.log(group['average_share_turnover_quarterly'])
        group['average_share_turnover_annual'] = group['turnover_ratio'].rolling(250).mean()
        group['average_share_turnover_annual'] = np.log(group['average_share_turnover_annual'])
        group['liquidity'] = 0.35*group['share_turnover_monthly'] + 0.35*group['average_share_turnover_quarterly'] + 0.3*group['average_share_turnover_annual']
        group = group.get(['date','code','liquidity'])
        return group
    liquidity_data = stock_turnover_ratio_data.groupby('code').apply(cal_liquidity)
    return liquidity_data
    
def get_natural_log_of_market_cap_data(days:int=20, test=False):
    """
    计算natural_log_of_market_cap最新数据
    对数市值 natural_log_of_market_cap: 公司的总市值的自然对数。
    """
    stock_circulating_cap_data = get_valuation_data(days, fields=['date','code','circulating_cap'], test=test)
    stock_circulating_cap_data['natural_log_of_market_cap'] = np.log(stock_circulating_cap_data['circulating_cap'])
    natural_log_of_market_cap_data = stock_circulating_cap_data.get(['date','code','natural_log_of_market_cap'])
    return natural_log_of_market_cap_data

def get_size_data(days:int=20, test=False):
    """ 
    计算size 最新数据
    size = natural_log_of_market_cap
    """
    size_data = get_natural_log_of_market_cap_data(days, test=test)
    size_data.rename(columns={'natural_log_of_market_cap':'size'}, inplace=True)
    return size_data

def get_TVSTD20_data(days:int=50, test=False):
    """
    计算TVSTD20 的最新数据
    TVSTD20: 20日成交额的标准差
    """
    def cal_TVSTD20(group):
        group['TVSTD20'] = group['money'].rolling(20, min_periods=5).std()
        return group
    stock_candle_data = get_stock_day_k_line_data(days, fields=['date','code','money'], test=test)
    stock_candle_data = stock_candle_data.groupby('code').apply(cal_TVSTD20)
    TVSTD20_data = stock_candle_data.get(['date','code','TVSTD20'])
    return TVSTD20_data

def get_fac_zdjezb_data(days:int=5, test=False):
    """
    计算fac_zdjezb 最新数据
    fac_zdjezb = zdje / money
    """
    zdjezb_data = get_money_flow(days, fields=['date','code','net_pct_m'], test=test)
    zdjezb_data.rename(columns={'net_pct_m':'fac_zdjezb'}, inplace=True)
    return zdjezb_data

def get_money_flow_20_data(days:int=50, test=False):
    """
    计算 money_flow_20
    money_flow_20 = mean((close+high+low)/3 * volume, 20)
    """
    stock_candle_data = get_stock_day_k_line_data(days, fields=['date','code','close','high','low','volume'], test=test)
    def cal_money_flow_20(group):
        group['price'] = (group['close']+group['high']+group['low']) / 3
        group['money_flow'] = group['price'] * group['volume']
        group['money_flow_20'] = group['money_flow'].rolling(20, min_periods=5).mean()
        return group.get(['date','code','money_flow_20'])
    money_flow_20_data = stock_candle_data.groupby('code').apply(cal_money_flow_20)
    return money_flow_20_data

def get_TVMA20_data(days:int=50, test=False):
    """
    计算 TVMA20
    TVMA20: 20日成交金额的移动平均值
    """
    def cal_TVMA20(group):
        group['TVMA20'] = group['money'].rolling(20, min_periods=5).mean()
        return group
    stock_candle_data = get_stock_day_k_line_data(days, fields=['date','code','money'], test=test)
    stock_candle_data = stock_candle_data.groupby('code').apply(cal_TVMA20)
    TVMA20_data = stock_candle_data.get(['date','code','TVMA20'])
    return TVMA20_data

def get_circulating_market_cap_data(days:int=5, test=False):
    """
    计算 circulating_market_cap
    circulating_market_cap: 股票流通市值
    """
    circulating_market_cap_data = get_valuation_data(days, fields=['date','code','circulating_market_cap'], test=test)
    return circulating_market_cap_data

def get_TVSTD6_data(days:int=30, test=False):
    """
    计算 TVSTD6
    TVSTD6: 6日成交金额的标准差
    """
    def cal_TVSTD6(group):
        group['TVSTD6'] = group['money'].rolling(6).std()
        return group
    stock_candle_data = get_stock_day_k_line_data(days, fields=['date','code','money'], test=test)
    stock_candle_data = stock_candle_data.groupby('code').apply(cal_TVSTD6)
    TVSTD6_data = stock_candle_data.get(['date','code','TVSTD6'])
    return TVSTD6_data

def get_fac_adtm_23_8_data(days:int=50, test=False):
    """
    计算 fac_adtm_23_8
    fac_adtm_23_8: 
    """
    def cal_fac_adtm_23_8(group):
        """计算 fac_adtm_23_8""" 
        group['dtm'] = list(map(lambda o,h,preo: 0 if o<=preo else max(h-o, o-preo), group['open'],group['high'],group['open'].shift(1)))
        group['dbm'] = list(map(lambda o, l, preo: 0 if o>=preo else max(o-l, o-preo), group['open'], group['low'], group['open'].shift(1)))
        group['stm'] = group['dtm'].rolling(23).sum()
        group['sbm'] = group['dbm'].rolling(23).sum()
        def cal_adtm(stm, sbm):
            if str(stm)=='nan' or str(sbm)=='nan':
                return np.NaN
            else:
                if stm>sbm:
                    return (stm-sbm) /stm
                elif stm==sbm:
                    return 0
                else:
                    return (stm-sbm) /sbm
        group['adtm'] = list(map(lambda stm, sbm: cal_adtm(stm, sbm), group['stm'],group['sbm']))
        if len(group.dropna())>=8:
            group['fac_adtm_23_8'] = talib.MA(group['adtm'], 8)
        else:
            group['fac_adtm_23_8'] = np.NaN
        fac = group.get(['date','code','fac_adtm_23_8'])
        return fac  
    stock_candle_data = get_stock_day_k_line_data(days, test=test)
    fac_adtm_23_8_data = stock_candle_data.groupby('code').apply(cal_fac_adtm_23_8)
    return fac_adtm_23_8_data


# ********************  每日选股  *********************
def get_factor_label(factor):
    """判断因子对收益率的影响是正还是负"""
    sql = f"select * from pt where name='{factor}' and period=1"
    fac_net_value = selectBySQL('dfs://FactorNetValueLib','FactorNetValueTable', sql)
    q1 = fac_net_value.query("`quantile`==1")['stock_net_value'].dropna().iloc[-1]
    q10 = fac_net_value.query("`quantile`==10")['stock_net_value'].dropna().iloc[-1]
    label = 10 if q10>q1 else 1
    return label

def get_factor_data(factor:str, days:int=10, test=False):
    """获取最新因子数据"""
    label = get_factor_label(factor)
    func = f"get_{factor}_data({days}, test={test})"
    factor_data = eval(func)
    if label==1:
        factor_data[factor] = -factor_data[factor]
    return factor_data

def get_factor_quantile(factor:str, days:int=10, quantiles=10, test=False):
    """ 将因子数据进行分组 """
    factor_data = get_factor_data(factor, days, test=test)
    factor_data.dropna(inplace=True)
    factor_data = factor_data \
        .groupby('date', group_keys=False) \
        .apply(lambda group:qcut(group, factor, quantiles, ascend=True)) \
        .rename(columns={'layer':f'{factor}_quantile'})
    return factor_data

def factor_destandard_denegative(group, factor):
    """ 因子在截面上：1. 先标准化， 2.进行非负处理 """
    group[factor] = (group[factor]-group[factor].mean()) / group[factor].std()
    group[factor] = group[factor] - group[factor].min()
    return group

def get_mfactor_data(factor_data1, factor_data2, test=False):
    """计算两个已经标准化、非负处理因子的乘积作为新的因子"""
    name1 = factor_data1.columns[-1]
    name2 = factor_data2.columns[-1]
    mfactor_data = pd.merge(left=factor_data1, right=factor_data2, on=['date','code'], how='inner')
    mfactor_data['mfactor'] = mfactor_data[name1] * mfactor_data[name2]
    mfactor_data = mfactor_data.get(['date','code','mfactor'])
    # 验证mfactor数据完整性
    new = mfactor_data.date.max()
    mfactor_data.dropna(inplace=True)
    now = int(get_now_date(test=test).strftime('%Y%m%d'))
    last_trade_date = get_last_trade_date(now)
    if len(mfactor_data)==0 or (len(mfactor_data)>0 and new<last_trade_date):
        raise LockDataError(f'mfactor缺少股票最新数据, 最新数据截止到:{new}, 需要到{last_trade_date}')
    else:
        return mfactor_data

def get_stock_info_data(days:int=100, test=False):
    """获取股票是否异常的最新数据"""
    is_st = get_stock_is_st_data(days,test=test).set_index(['date','code'])
    is_new = get_stock_is_new_data(days,test=test).set_index(['date','code'])
    is_release = get_stock_is_release_data(days,test=test).set_index(['date','code'])
    is_recover = get_stock_is_recover_data(days,test=test).set_index(['date','code'])
    is_limit = get_stock_is_limit_data(days,test=test).set_index(['date','code'])
    stock_info = pd.concat([is_st, is_new, is_release,is_recover, is_limit], axis=1)
    stock_info.reset_index(inplace=True)
    return stock_info

def get_clean_stock(stock_info):
    """剔除st, 解禁股票,新股，短期复牌，"""
    q = "`is_st`==False and `is_new`==False and `is_release`==False and `is_recover`==False"
    clean_stock = stock_info.query(q).get(['date','code'])
    return clean_stock

@datascan
def get_stock_daily_weight(factor1, factor2, stock_info, days=400, topN=10, test=False):
    """根据乘数因子每日筛选topN的股票及权重"""
    # 获取因子数据
    factor_data1 = get_factor_data(factor1, 90, test=test)
    factor_data2 = get_factor_data(factor2, 90, test=test)
    # 标准化，再非负处理
    destandard_denegative_factor_data1 = factor_data1.groupby('date', group_keys=False).apply(lambda group: factor_destandard_denegative(group, factor1))
    destandard_denegative_factor_data2 = factor_data2.groupby('date', group_keys=False).apply(lambda group: factor_destandard_denegative(group, factor2))
    # 两个处理过的因子相乘
    mfactor_data = get_mfactor_data(destandard_denegative_factor_data1, destandard_denegative_factor_data2, test=test)
    # 获取非异常股票
    clean_stock = get_clean_stock(stock_info)
    # 获取非异常因子数据
    clean_mfactor_data = pd.merge(mfactor_data, clean_stock, on=['date','code'], how='inner')
    # 剔除科创板股票
    clean_mfactor_data = clean_mfactor_data[~clean_mfactor_data['code'].str.startswith('68')]
    # 筛选每日选股结果
    stock_daily = clean_mfactor_data.sort_values(['date','mfactor'], ascending=[True, False]).groupby('date', group_keys=False).apply(lambda group: group.head(topN))
    # 计算股票每日权重
    weight = stock_daily.groupby('date', group_keys=False).count().reset_index().get(['date','code']).rename(columns={'code':'weight'})
    weight['weight'] = 1/weight['weight']
    # 股票每日权重
    stock_daily_weight = pd.merge(stock_daily, weight, on=['date'], how='inner')
    del stock_daily_weight['mfactor']
    return stock_daily_weight
    
def select_topN_run_daily(factor1, factor2, days=400, topN=10, test=False):
    """选取topN每日收盘运行 """
    stock_info = get_stock_info_data(60, test=test)
    stock_daily_weight = get_stock_daily_weight(factor1, factor2, stock_info, days, topN, test=test)
    # 只保留最新的选股结果
    now = int(get_now_date(test).strftime('%Y%m%d'))
    last_trade_date = get_last_trade_date(now)
    choose_today_stock = stock_daily_weight[stock_daily_weight['date']==last_trade_date]
    code_name = get_securities_stock().get(['code','name'])
    choose_today_stock = pd.merge(choose_today_stock, code_name, on=['code'], how='left').get(['date','code','name','weight'])
    return choose_today_stock

def select_quantile_run_daily(factor1, factor2, days=60, quantiles=10, test=False):
    """选取两个因子的top quantile"""
    # 股票异常数据
    stock_info_data = get_stock_info_data(days, test=test)
    # 获取因子数据（负因子的值取负号）并分组
    quantile1 = get_factor_quantile(factor1, days, quantiles, test=test)
    quantile2 = get_factor_quantile(factor2, days, quantiles, test=test)
    # 选取两个因子的组标签都是最大的股票记录
    quantile_data = pd.merge(quantile1, quantile2, on=['date','code'], how='inner').query(f"`{factor1}_quantile`=={quantiles} and `{factor2}_quantile`=={quantiles}")
    # 干净股票
    clean_stock = get_clean_stock(stock_info_data)
    clean_quantile_data = pd.merge(quantile_data, clean_stock, on=['date','code'], how='inner')
    # 剔除科创板股票
    clean_quantile_data = clean_quantile_data[~clean_quantile_data['code'].str.startswith('68')]
    # 选股
    stock_daily = clean_quantile_data.get(['date','code'])
    weight = stock_daily.groupby('date', group_keys=False).count().reset_index().get(['date','code']).rename(columns={'code':'weight'})
    weight['weight'] = 1/weight['weight']
    weight['weight'] = weight['weight'].apply(lambda x: round(x,2))
    # 股票每日权重
    stock_daily_weight = pd.merge(stock_daily, weight, on=['date'], how='inner')
    # 只保留最新的选股结果
    now = int(get_now_date().strftime('%Y%m%d'))
    last_trade_date = get_last_trade_date(now)
    choose_today_stock = stock_daily_weight[stock_daily_weight['date']==last_trade_date]
    return choose_today_stock

def send_email(subject='', msg='', receiver='yumi419710@126.com'):
    """发送文字邮件"""
    message = MIMEText(msg,'plain','utf-8')   
    message['Subject'] = subject
    message['From'] = '1031364477@qq.com'
    message['To'] = receiver
    try:
        server=smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login('1031364477@qq.com', 'brkbgbsqnavzbeba')
        server.sendmail('1031364477@qq.com', receiver, message.as_string())
        server.quit()
        logger.info(f"send email to {receiver} successfully!")
    except smtplib.SMTPException as e:
        logger.error(f"Send email to {receiver} error! ERROR:{e}")

def selectTopN(topN=10, test=False):
    """每日收盘后运行的主函数, 选取股票因子相乘后的top N只股票"""
    factors = [('TVSTD20','natural_log_of_market_cap'), ('TVMA20','natural_log_of_market_cap'), ('TVSTD20','fac_zdjezb')]
    now = datetime.now().strftime('%Y-%m-%d')
    subject = f"{now} 选股结果"
    full_text = ""
    selectResult = []
    for (factor1, factor2) in factors:
        choose_today_stock = select_topN_run_daily(factor1, factor2, topN=topN, test=test)
        forward_trade_date = get_forward_trade_date(test=test)
        last_trade_date = choose_today_stock['date'].max()
        print('last_trade_date:', last_trade_date)
        choose_today_stock['date'] = forward_trade_date
        strategy = title = f"{factor1}&{factor2} top:{topN}"
        choose_today_stock.insert(0, 'strategy', strategy)
        selectResult.append(choose_today_stock)
        text = f"{title}\n日期             股票代码         股票名称        权重\n"
        for row in choose_today_stock.iterrows():
            text += f"{row[1]['date']}\t{row[1]['code']}\t{row[1]['name']}\t{row[1]['weight']}\n"
        full_text = full_text + text + '\n\n\n'  
    send_email(subject, full_text) 
    return selectResult
    
def selectQuantile(days=60, quantiles=10, test=False):
    """ 将两个因子分别处理成正因子后，进行分区，均取因子值最大的那一组的股票 """
    factors = [('TVSTD20','natural_log_of_market_cap'), ('TVMA20','natural_log_of_market_cap'), ('TVSTD20','fac_zdjezb')]
    now = datetime.now().strftime('%Y-%m-%d')
    subject = f"{now} 选股结果"
    full_text = ""
    for (factor1, factor2) in factors:
        choose_today_stock = select_quantile_run_daily(factor1, factor2, days=days, quantiles=quantiles, test=test)
        forward_trade_date = get_forward_trade_date(test=test)
        choose_today_stock['date'] = forward_trade_date
        title = f"{factor1}&{factor2} quantiles:{quantiles}"
        text = f"{title}\n日期               股票代码         权重\n"
        for row in choose_today_stock.iterrows():
            text += f"{row[1]['date']}\t{row[1]['code']}\t{row[1]['weight']}\n"
        full_text = full_text + text + '\n\n\n'  
    send_email(subject, full_text) 

def test_is_last_date():
    candle = get_stock_day_k_line_data(10)
    valuation = get_valuation_data(10, fields=['date','code','turnover_ratio'])
    money = get_money_flow(10, fields=['date','code','net_pct_main'])
    is_new = get_stock_is_new_data(10)
    is_st = get_stock_is_st_data(10)
    is_release = get_stock_is_release_data(10)
    is_recover = get_stock_is_recover_data(60)
    is_limit = get_stock_is_limit_data(10)
    
    print('candle date max:', candle.date.max())
    print('valuation date max:', valuation.date.max())
    print('money date max:', money.date.max())
    print('is_new date max:', is_new.date.max())
    print('is_st date max:', is_st.date.max())
    print('is_release date max:', is_release.date.max())
    print('is_recover date max:', is_recover.date.max())
    print('is_limit date max:', is_limit.date.max())


