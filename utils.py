import empyrical
import pandas as pd
import numpy as np
from gm.api import *
from pandas import DataFrame
from itertools import accumulate
from datetime import datetime, timedelta
from scipy.stats.mstats import winsorize
from collections import defaultdict
from pandas import Series


from .decorator import timeit
from .secret import config
from .database.mysql import get_data
from .chart import eTable, eLine, ePage

set_token(config['eastmoney']['token'])



class DataLoader:
    """ 构造LSTM模型对时间序列进行预测 """
    def __init__(self, data:DataFrame, window:int, ratio:float, variables:list, targets:list):
        """ 
        初始化  
        data: index 为datetim64[ns]
        """
        self.data = data.values
        self.window = window
        self.ratio = ratio
        self.variables = variables
        self.targets = targets
        self.height, self.width = self.data.shape
        self.split = int(self.height*self.ratio)

    def get_train_data(self):
        """ 获取训练集数据 """
        x_train = np.array([self.data[i:i+self.window, :-1] for i in range(self.split - self.window)])
        y_train = np.array([self.data[i:i+self.window, -1, [-1]] for i in range(self.split - self.window)])
        return x_train, y_train

    def get_test_data(self):
        x_test = np.array([self.data[i:i+self.window, :-1] for i in range(self.split, self.height - self.window)])
        y_test = np.array([self.data[i:i+self.window, -1, [-1]] for i in range(self.split, self.height - self.window)])
        return x_test, y_test




def vwap(group):
    v = group['volume'].values
    tp = (group['low'] + group['close'] + group['high']).div(3).values
    return group.assign(vwap=(tp * v).cumsum() / v.cumsum())

def qcut(groupDf, scoreClm, layerNum, ascend=False):
    """
    对某一列数据进行分层的函数。如果不能够刚好平分，会将余下的样本分给低层数，低层数包含的样本会更多一些。
    :param groupDf: 包含待分层数据的df
    :param scoreClm: 按哪一列分层，scoreClm是该列的列名
    :param layerNum: 分多少层
    :param ascend: ascend为True，分数越低，所处层数越低；ascend为False，分数越高，所处层数越低。
    """

    """没有nan值的做法与有nan值的差不多，但没有dropna这一步，不会对groupDf本身进行修改，适用于groupby().apply"""

    groupDf.sort_values(by=scoreClm, ascending=ascend, inplace=True)
    total = len(groupDf)
    # eg:total=27，layerNum=5，27/5=5...2，1-5层分别分5个样本，余下的两个名额分别加入到1、2层
    # 确定完各层的样本后，顺序排序，最前的是1层，以此类推
    layerList = sorted(list(range(1, 1+layerNum))*(total//layerNum) + list(range(1, 1+total % layerNum)))
    groupDf['layer'] = layerList

    # 当同一大小的score横跨两层时，判断哪一层的个数比较多，统一归于个数多的那一层。(主要用于离散型的score)
    segPointList = groupDf.loc[groupDf['layer'] != groupDf['layer'].shift(1)].index   # 找到层间的分界点
    segPointList = segPointList[1:]  # 剔除掉第一个分界点(1和shift产生的nan)
    for segPoint in segPointList:
        if groupDf[scoreClm][segPoint] == groupDf[scoreClm].shift(1)[segPoint]:  # 如果分界点前后的score相同，则要判断后统一一层
            thisScore = groupDf[scoreClm][segPoint]
            thisLayer = groupDf['layer'][segPoint]
            lastLayer = groupDf['layer'].shift(1)[segPoint]
            # 同一分数，哪一层的个数多就将这个分数归于哪一层，如果个数一样多，就将其归于低层
            if len(groupDf[(groupDf[scoreClm] == thisScore) & (groupDf['layer'] == thisLayer)]) > \
                    len(groupDf[(groupDf[scoreClm] == thisScore) & (groupDf['layer'] == lastLayer)]):
                groupDf.loc[(groupDf[scoreClm] == thisScore) & (groupDf['layer'] == lastLayer), 'layer'] = thisLayer
            elif len(groupDf[(groupDf[scoreClm] == thisScore) & (groupDf['layer'] == thisLayer)]) <= \
                    len(groupDf[(groupDf[scoreClm] == thisScore) & (groupDf['layer'] == lastLayer)]):
                groupDf.loc[(groupDf[scoreClm] == thisScore) & (groupDf['layer'] == thisLayer), 'layer'] = lastLayer

    # 分数一致的归于同一层，可能会导致层数跳空(样本少层数多的时候会出现这种情况)。以下的操作目的是让层数连续
    segPointList = groupDf.loc[groupDf['layer'] != groupDf['layer'].shift(1)].index  # 找到层间的分界点
    segPointList = segPointList[1:]  # 剔除掉第一个分界点(1和shift产生的nan)
    for segPoint in segPointList:
        thisLayer = groupDf['layer'][segPoint]
        lastLayer = groupDf['layer'].shift(1)[segPoint]
        if thisLayer != lastLayer + 1:
            groupDf.loc[groupDf['layer'] == thisLayer, 'layer'] = lastLayer + 1
    return groupDf


def recognize_pattern(group,threshold1=0.05, threshold2=0.07, threshold3=0.08, threshold10=0.1):
    """
    recognize pattern of kline data
    :param group:
    :return:
    """
    cumret1 = group['close'].pct_change()
    cumret2 = cumret1.rolling(2).apply(empyrical.cum_returns_final)
    cumret3 = cumret1.rolling(3).apply(empyrical.cum_returns_final)
    cumret10 = cumret1.rolling(10).apply(empyrical.cum_returns_final)
    group.loc[(cumret10>-threshold10) & (cumret10<threshold10) & (cumret1<threshold1) & (cumret1>threshold1), 'pattern'] = 0
    group.loc[(cumret10>threshold10) | (cumret3>threshold3) | (cumret2>threshold2) | (cumret1>threshold1), 'pattern'] = 100
    group.loc[(cumret10<-threshold10) | (cumret3<-threshold3) | (cumret2<-threshold2) | (cumret1<-threshold1), 'pattern'] = -100
    group['pattern'] = group['pattern'].fillna(0)
    pattern = group.tail(1).get(['date','security','pattern'])
    return pattern

def accum_by_sign(x,y):
    # 按照符号累加
    if np.sign(y)==np.sign(x) or y==0:
        return x+y
    else:
        return y

def area_(arr):
    """
    计算面积比
    """
    accum = list(accumulate(arr, accum_by_sign))
    last_area = np.NaN
    result = []

    for i in range(len(accum)):
        if i==0:
            result.append(np.NaN)
        else:
            if last_area==np.NaN:
                result.append(np.NaN)
            else:
                if np.sign(accum[i])==np.sign(accum[i-1]):
                    try:
                        ratio = (accum[i] + 1e-5)/(last_area + 1e-5)
                    except ZeroDivisionError:
                        ratio = np.inf
                    result.append(ratio)
                else:
                    last_area = accum[i-1]
                    try:
                        ratio = (accum[i] + 1e-5)/(last_area + 1e-5)
                    except ZeroDivisionError:
                        ratio = np.inf
                    result.append(ratio)
    return np.array(list(map(abs, result)))

def div_(x,y):
    """除法， 排除当除零的时候报错"""
    try:
        ratio = (x + 1e-5) / (y + 1e-5)
    except ZeroDivisionError:
        ratio = np.inf
    return ratio

def filtered(results):
    total = []
    for res in results:
        filter1 = list(filter(None, res))    # 删除 None
        filter2 = list(filter(lambda x: len(x.data), filter1))  # 删除属性 data 为空的ind
        for ind in filter2:
            total.append(ind)
    return total

def process_inf(df):
    '''
    df:目标dataframe
    数据去缺失异常值
    '''
    # 去na和inf
    df = df[~df.isin([np.inf, -np.inf]).any(1)]
    return df

def process_nan(df_):
    """
    部分缺失采用中位数填充，全部缺失删除该特征
    """
    df = df_.copy()
    df = df.fillna(df.median())
    return df

def change_code(code, kind):
    """
    # 该方法用来转换“本地数据库”的股票代码和“JoinQuant”的股票代码
        #kind =6 601377
        #kind =9 601377.SH
        #kind =11 601377.XSHG
        #输入code 和需要返回code的类型 类型数字就是位数
    """
    if kind == 6:
        return code[0:6]
    if kind == 9:
        fistr_char = code[0]
        if fistr_char == '6' and len(code) == 6:
            # 上海
            return code + ".SH"
        if (fistr_char == '3' or fistr_char == '0') and len(code) == 6:
            # 深
            return code + ".SZ"
        if len(code) == 9:
            return code
        if len(code) == 11:
            if fistr_char == '6':
                # 上海
                return code.replace("XSHG", "SH")
            if (fistr_char == '3' or fistr_char == '0'):
                # 深
                return code.replace("XSHE", "SZ")
    if kind == 11:
        fistr_char = code[0]
        if fistr_char == '6' and len(code) == 6:
            # 上海
            return code + ".XSHG"
        if (fistr_char == '3' or fistr_char == '0') and len(code) == 6:
            # 深
            return code + ".XSHE"
        if len(code) == 11:
            return code

        if len(code) == 9:
            if fistr_char == '6':
                # 上海
                return code.replace("SH", "XSHG")
            if (fistr_char == '3' or fistr_char == '0'):
                # 深
                return code.replace("SZ", "XSHE")

def format_date(init_date):
    """格式化日期输出 datetime64[ns], 形如 2022-05-20"""
    # str_date = str(init_date)
    str_date = str(init_date)[:8]
    format_date = str_date[:4] + "-" + str_date[4:6] + "-" + str_date[6:8]
    standard_date = pd.to_datetime(format_date)
    return standard_date

def rank(array, quantiles=5):
    """
    收益率分组，作为分类标准，用于训练器训练
    """
    return pd.qcut(array.rank(method='first'), quantiles, labels=False) + 1

def winsorize_perc(series, edge_low=0.025, edge_up=0.025):
    '''
    前后缩尾n%去极值函数
    因子值value的Series
    std为几倍的标准差，have_negative 为布尔值，是否包括负值
    输出Series
    '''
    # r是一个DataFrame,去na
    series = series.dropna().copy()

    series = pd.Series(winsorize(series, limits=[edge_low, edge_up]))

    return series

def winsorize_nSTD(series, STD=3):
    '''
    将因子值以标准差±nSTD拉回缩尾去极值函数
    因子值为value的Series
    std为几倍的标准差，have_negative 为布尔值，是否包括负值
    输出Series
    '''
    # 去na
    series = series.dropna().copy()

    # 定义上下限
    edge_up = series.mean() + STD * series.std()
    edge_low = series.mean() - STD * series.std()

    # 所有超过上边界大的都设定为上边界， #所有超过下边界大的都设定为下边界
    series[series > edge_up] = edge_up
    series[series < edge_low] = edge_low
    return series

def winsorize_nMAD(series, nl=3, nr=3):
    """
    n倍中位数去极值
    将因子值以中位数±nMAD拉回缩尾去极值函数
    因子值为value的Series
    std为几倍的标准差，have_negative 为布尔值，是否包括负值
    输出Series
    """

    # 求出因子值的中位数
    median = np.median(series)

    # 求出因子值与中位数的差值, 进行绝对值
    mad = np.median(abs(series - median))

    # 定义几倍的中位数上下限
    high = median + (nr * mad)
    low = median - (nl * mad)

    # 替换上下限
    series = np.where(series > high, high, series)
    series = np.where(series < low, low, series)
    return series

def standardize(series, method='z-score'):
    '''
    标准化函数：
    s为Series数据
    ty为标准化类型:1 MinMax,2 Z-score Standard,3 maxabs
    '''
    data = series.copy()
    if method == 'min-max':
        result = (data - data.min()) / (data.max() - data.min())
    elif method == 'z-score':
        result = (data - data.mean()) / data.std()
    elif method == 'log':
        result = data / 10 ** np.ceil(np.log10(data.abs().max()))
    return result

def trending(series, i, j, k=0):
    """判断Series趋势的方法"""
    def leftRollingTrend(windowSeries):
        windowSeries.index = range(len(windowSeries))
        return all(x<y for x, y in zip(windowSeries, windowSeries[1:]))
    
    series = series.shift(k)
    lTrend = series.rolling(i).apply(leftRollingTrend)
    rTrend = series.rolling(j).apply(leftRollingTrend).shift(-j)
    mTrend = lTrend * rTrend
    return lTrend, mTrend, rTrend

def trans_code_form(series, trans_tpye='JQtoEM'):
    '''
    数字格式转换
    version:2022-05-26
    input a series
    '''
    if trans_tpye == 'NUMtoEM':
        series = series.astype('int')
        series = series.astype('str')
        series = series.str.strip().str.zfill(6)
        series = series.apply(lambda x: str('SHSE.')+x if x[0]=='6' else str('SZSE.')+x ) 
    elif trans_tpye == 'NUMtoJQ':
        series = series.astype('int')
        series = series.astype('str')
        series = series.str.strip().str.zfill(6)
        series = series.apply(lambda x: x + str('.XSHG') if x[0]=='6' else x + str('.XSHE') ) 
    elif trans_tpye=='EMtoJQ':
        series = series.apply(lambda x: x.split('.')[-1] + '.' + x.split('.')[0])
        series = series.apply(lambda x: x.replace('SZSE','XSHE'))
        series = series.apply(lambda x: x.replace('SHSE','XSHG'))
    elif trans_tpye=='JQtoEM':
        series = series.apply(lambda x: x.split('.')[-1] + '.' + x.split('.')[0])
        series = series.apply(lambda x: x.replace('XSHE','SZSE'))
        series = series.apply(lambda x: x.replace('XSHG','SHSE'))    
    elif trans_tpye == 'NUMtoSTR':
        series = series.astype('int')
        series = series.astype('str')
        series = series.str.strip().str.zfill(6)
    return series



# ****************  mysql ***************************
def intdate_to_strdate(series_):
    series = series_.copy()
    series = series.astype(str)
    series = series.apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:])
    return series

def get_index_return(code='000001.XSHG'):
    sql = f"""
    select `date`,`open` 
    from `index_day_k_line_data` 
    where `code`='{code}' 
    ORDER BY `date`
    """
    data = get_data(sql)
    data[code] = data['open'].pct_change().shift(-1).dropna()
    data['date'] = intdate_to_strdate(data['date'])
    return data.get(['date',code])

def get_trade_day(start, end):
    """获取交易日历"""
    sql = f"select cast(`date` as date) as `date` from `trade_day` where `date`>={start} and `date`<={end} order by `date`"
    trade_day_df = get_data(sql)
    return trade_day_df

def get_security_info():
    """获取股票基本信息"""
    sql = "select `code` as `security`,DATE(start_date) as start_date, `name` from `securities_stock`"
    security_info = get_data(sql)
    return security_info

def get_securities():
    """ 获取所有股票列表 """
    security_info = get_security_info()
    securities = security_info.security.unique()
    return list(securities)

securities = get_securities()
def get_candles(securities=securities, start=20100101, end=20230531):
    """获取股票历史candles"""
    securities = tuple(securities)
    sql = f"""
    select `date`,`code` as `security`,`open`,`high`,`low`,`close`,`volume`,`money`,`factor`,`high_limit`,`low_limit`,`paused`
    from `stock_day_k_line_data` 
    where `code` in {securities} and `date` between {start} and {end} and `volume`!=0 
    ORDER BY `security`,`date`
    """
    candles = get_data(sql)
    candles['date'] = intdate_to_strdate(candles['date'])
    return candles
    
def get_st_record():
    """ 获取股票st记录 """
    sql = "select `date`, `code` as `security`, 'is_st' from `securities_extras` order by `date`"
    st_record = get_data(sql)
    st_record['is_st'] = True
    st_record['date'] = intdate_to_strdate(st_record['date'])
    return st_record

def get_index_candle(security=['000001.XSHG', '399006.XSHE', '399005.XSHE']):
    """ 获取指数K线行情数据 """
    if isinstance(security, str):
        sql = f"""
        select `date`,`code` as `security`,`open`,`high`,`low`,`close`,`volume`,`money` 
        from `index_day_k_line_data`
        where `code`='{security}'
        and `volume`!=0
        order by `code`,`date`
        """
    elif isinstance(security, list):
        security = tuple(security)
        sql = f"""
        select `date`,`code` as `security`,`open`,`high`,`low`,`close`,`volume`,`money` 
        from `index_day_k_line_data`
        where `code` in {security}
        and `volume`!=0
        order by `code`,`date`
        """
    else:
        raise TypeError('security must be str or list!')
    
    index_candle = get_data(sql)
    index_candle['date'] = intdate_to_strdate(index_candle['date'])
    return index_candle

def get_money_flow(start=20150101, end=20230531, fields=[]):
    """ 获取资金流数据 """
    sql = f"""
    select * 
    from `money_flow`
    where `date` between {start} and {end}
    order by `code`,`date`
    """
    money_flow = get_data(sql)
    if fields:
        money_flow = money_flow.get(fields)
        money_flow.rename(columns={'code':'security'}, inplace=True)
        money_flow['date'] = intdate_to_strdate(money_flow['date'])
        return money_flow
    else:
        money_flow.rename(columns={'code':'security'}, inplace=True)
        money_flow['date'] = intdate_to_strdate(money_flow['date'])
        return money_flow

def get_valuation(start=20100101, end=20230531, fields=[]):
    """ 获取valuation表内容 """
    sql = f"""
    select * from `valuation_data` 
    where `date` between {start} and {end}
    """
    valuation = get_data(sql)
    if fields:
        valuation = valuation.get(fields)
        valuation.rename(columns={'code':'security'}, inplace=True)
        valuation['date'] = intdate_to_strdate(valuation['date'])
        return valuation
    else:
        valuation.rename(columns={'code':'security'}, inplace=True)
        valuation['date'] = intdate_to_strdate(valuation['date'])
        return valuation
    
def get_locked_shares(start=20100101, end=20230531):
    """ 获取股票解禁数据 """
    sql = f"""
    select `date`,`code` as `security`,`rate1` as `rate` from `locked_shares` 
    where `date` between {start} and {end}
    order by `code`,`date`
    """
    locked_shares = get_data(sql)
    locked_shares['date'] = intdate_to_strdate(locked_shares['date'])
    return locked_shares

def get_index():
    """ 获取20100101~20230531期间的股票、日期复核索引列 """
    index_df = pd.read_pickle(r'E:\Candles\index.pkl')
    return index_df

def get_release_share(start=20100101, end=20230531):
    """ 获取股票在半年内是否有超过3%的解禁情况 """
    locked_shares = get_locked_shares(start=start, end=end)
    locked_shares['release_date'] = locked_shares['date']
    index_df = get_index()
    release_share_df = pd.merge(left=index_df, right=locked_shares, on=['date','security'], how='left')
    def bfill_release_df(group):
        group[['rate','release_date']] = group[['rate','release_date']].fillna(method='bfill')
        return group
    release_share_df = release_share_df.groupby('security', group_keys=False).apply(bfill_release_df)  
    release_share_df.dropna(inplace=True)
    release_share_df['release_date'] = release_share_df['release_date'].astype('datetime64')
    release_share_df['date'] = release_share_df['date'].astype('datetime64')
    release_share_df['time_delta'] = release_share_df['release_date'] - release_share_df['date']
    half_year = timedelta(days=125)
    release_share_df['is_timedelta_halfyear'] = release_share_df['time_delta'].apply(lambda dt: True if dt<half_year else False)
    release_share_df['rate'] = release_share_df['rate'].astype(float)
    release_share_df['is_rate'] = release_share_df['rate'].apply(lambda x: True if x>0.03 else False)
    release_share_df.loc[release_share_df['is_timedelta_halfyear'] & release_share_df['is_rate'], 'is_release_share'] = True
    release_share_df['is_release_share'] = release_share_df['is_release_share'].fillna(False)
    release_share_df['date'] = release_share_df['date'].astype(str)
    return release_share_df.get(['date','security','is_release_share'])

def get_base(start=20100101, end=20230531):
    """ 获取分析基础数据 """
    # 行情数据
    candles = pd.read_pickle(r'E:\Candles\candles.pkl')
    # 上市日期数据
    security_info = get_security_info()
    # st 数据
    st_record = get_st_record()
    # 股票解禁数据
    release_share = get_release_share()
    # 市值数据
    market_cap = get_valuation(start=start, end=end, fields=['date','code','market_cap'])
    # 资金流数据
    money_flow = get_data("select `date`,`code` as `security`, `net_pct_main` from `money_flow`")
    money_flow['date'] = intdate_to_strdate(money_flow['date'])

    # 合并数据
    tem = pd.merge(left=candles, right=security_info, on='security', how='left')
    candles=None
    security_info = None
    tem = pd.merge(left=tem, right=st_record, on=['date','security'], how='left')
    st_record = None 
    tem = pd.merge(left=tem, right=release_share, on=['date','security'], how='left')
    release_share = None
    tem = pd.merge(left=tem, right=market_cap, on=['date','security'], how='left')
    market_cap = None
    tem = pd.merge(left=tem, right=money_flow, on=['date','security'], how='left')
    return tem

def get_transaction_record(transactions_, daily_posratio):
    """ 获取交易记录 """
    transactions = transactions_.copy()
    transactions = transactions.reset_index()
    transactions['date'] = transactions['date'].apply(lambda dt: dt.strftime('%Y-%m-%d'))
    transactions.rename(columns={'symbol':'security'},inplace=True)
    security_info = get_security_info()
    # 补充股票名称
    result = pd.merge(left=transactions, right=security_info, on='security', how='left')
    # 补充每日仓位比率
    result = pd.merge(left=result, right=daily_posratio, on='date', how='left')
    result['side'] = result['amount'].apply(lambda x: '买入' if x>0 else '卖出')
    del result['sid']
    del result['start_date']
    def cal_stock_profit(group):
        """计算每只股票平仓时的盈亏情况"""
        group['profit'] = group['value'] + group['value'].shift(1)
        group.loc[group['side']=='买入', 'profit'] = 0
        group['profit'] = group['profit'].fillna(0)
        return group
    result = result.groupby('security', group_keys=False).apply(cal_stock_profit)
    order = ['date','security','name','side','price','amount','value','profit','posratio','cash','asset']
    return result.get(order)

def analyse(return_, daily_posratio, code='000300.XSHG'):
    # 指数收益率
    index_return = get_index_return(code=code)
    # 账户收益率
    account_return = return_.reset_index()
    account_return.rename(columns={'index':'date'},inplace=True)
    account_return['date'] = account_return['date'].apply(lambda dt: dt.strftime('%Y-%m-%d'))
    account_index_return = pd.merge(left=account_return, right=index_return, on='date', how='left')
    account_index_return.columns = ['date','account','index']
    
    # 计算统计指标
    max_drawdown = empyrical.max_drawdown(account_index_return['account'])  # 最大回撤
    alpha, beta = empyrical.alpha_beta(account_index_return['account'], account_index_return['index'])  # 阿尔法、贝塔收益
    sharpe = empyrical.sharpe_ratio(account_index_return['account'])  # 夏普比率
    calmar = empyrical.calmar_ratio(account_index_return['account'])  # 卡玛比率
    annual_return = empyrical.annual_return(account_index_return['account'])  # 年化收益率
    cum_return = empyrical.cum_returns_final(account_index_return['account']) # 累计收益率
    posratio = np.mean(daily_posratio['posratio'])   # 资金使用率
    profit = daily_posratio['profit'].iloc[0]        # 盈利次数
    loss = daily_posratio['loss'].iloc[0] 
    success_rate = daily_posratio['success_rate'].iloc[0]  # 胜率
    etable_data = pd.DataFrame({
        '指标名称':['最大回撤','阿尔法收益','贝塔收益','夏普比率','卡玛比率','年化收益率','累计收益率', '资金使用率', '盈利次数','亏损次数','胜率'],
        '指标值':[f'{round(100*max_drawdown,2)}%',f'{round(alpha*100,2)}%',f'{round(100*beta,2)}%',round(sharpe,2),round(calmar,2),f'{round(100*annual_return,2)}%',f'{round(100*cum_return,2)}%', f'{round(100*posratio,2)}%',profit,loss,f'{round(100*success_rate,2)}%']
        })
    etable = eTable(etable_data,'资产组合表现','')
    
    # 绘制累计收益率曲线
    account_index_return['account'] = empyrical.cum_returns(account_index_return['account'])
    account_index_return['index'] = empyrical.cum_returns(account_index_return['index'])
    account_index_return[['account','index']] = account_index_return[['account','index']].applymap(lambda x: round(x,4))
    account_index_return.columns=['date','资产组合','沪深300']
    eline = eLine(account_index_return, '累计收益率曲线',f'基准收益: 沪深300')
    epage = ePage()
    epage.add(etable)
    epage.add(eline)
    return epage

def get_column_astype(colName, colType):
    INT =1
    LONG = 2
    DOUBLE = 3
    SYMBOL = 4
    TIMESTAMP = 5
    
    res = defaultdict(list)

    for i in range(len(colType)):
        if colType[i]==INT:
            res['int'].append(colName[i])
        elif colType[i]==LONG:
            res['int64'].append(colName[i])
        elif colType[i] == DOUBLE:
            res['float'].append(colName[i])
        elif colType[i] == SYMBOL:
            res['str'].append(colName[i])
        elif colType[i]==TIMESTAMP:
            res['ts2Timestamp'].append(colName[i])
        else:
            continue
    return res