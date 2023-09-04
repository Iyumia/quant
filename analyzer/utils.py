from tqdm import tqdm
from ..utils import *
import statsmodels.api as sm
import scipy.stats as stats
import alphalens.performance as perf
from ..database.dolphin import *
from alphalens import utils, tears



@timeit(1)
def get_factor(database, factor, symbols, start, end):
    if database=='joinquant':
        sql = f"""
        select rq as date, stock_code as symbol, factor_value as {factor} from pt where factor='{factor}' and rq between {start}:{end} and stock_code in {symbols} order by stock_code, rq
        """
        factor = selectBySQL('dfs://Factor','factors',sql)

        
    elif database=='worldquant':
        sql = f"""
        select date, symbol, factor_value as {factor} from pt 
        where factor_name='{factor}' and date between {start}:{end} and symbol in {symbols} order by symbol, date
        """
        factor = selectBySQL('dfs://WorldQuant','alpha101', sql)
    
    elif database=='gtja':
        sql = f"""
        select date, symbol, factor_value as {factor} from pt 
        where factor_name='{factor}' and date between {start}:{end} and symbol in {symbols} order by symbol, date
        """
        factor = selectBySQL('dfs://GTJA','alpha191',sql)
    
    else:
        raise ValueError('database error!')
    factor.drop_duplicates(subset=['date','symbol'],inplace=True)
    factor.set_index(['date', 'symbol'], inplace=True)
    return factor

@timeit(1)
def get_multi_factor(database, factors, symbols, start, end):
    def inner(database, factors, symbols, start, end):
        if database=='joinquant':
            sql = f"""
            select rq as date, stock_code as symbol, factor as factor_name, factor_value from pt 
            where factor in {factors} and rq between {start}:{end} and stock_code in {symbols} 
            order by stock_code,rq"""
            multi_factor = selectBySQL('dfs://Factor','factors',sql)

        elif database=='worldquant':
            sql = f"""
            select * from pt 
            where factor_name in {factors} and date between {start}:{end} and symbol in {symbols} 
            order by symbol, date"""
            multi_factor = selectBySQL('dfs://WorldQuant','alpha101',sql)
        
        elif database=='gtja':
            sql = f"""
            select * from pt 
            where factor_name in {factors} and date between {start}:{end} and symbol in {symbols} 
            order by symbol, date"""
            multi_factor = selectBySQL('dfs://GTJA','alpha191',sql)
            
        else:
            raise ValueError('database error!')    

        multi_factor.drop_duplicates(subset=['date','symbol','factor_name'],inplace=True)
        multi_factor_pivot = multi_factor.pivot_table(index=['date','symbol'], columns='factor_name', values='factor_value')
        return multi_factor_pivot

    factors_list = []
    start_ = 0
    end_ = len(factors)
    while start_ < end_:
        factor = inner(database, factors[start_:min(start_+50,end_)], symbols, start, end)
        factors_list.append(factor)
        start_ += 50
    result = pd.concat(factors_list, axis=1)
    return result

@timeit(1)
def get_price(symbols, start, end, pivot=False):
    sql = f"""
    select date, symbol, adjClose as close from pt
    where date between {start}:{end} and symbol in {symbols} 
    order by symbol,date
    """
    price = selectBySQL('dfs://stockData','Kline',sql)
    price.set_index(['date','symbol'], inplace=True)

    if pivot:
        return price.pivot_table(index=price.index.get_level_values(0), columns=price.index.get_level_values(1), values='close')
    else:
        return price

def get_market_value(symbols, start, end):
    """
    获取市值数据，用于下一步市值中性化
    """
    sql = f"""
    select date,symbol,a_share_market_val as market_value from pt
    where date between {start}:{end} and symbol in {symbols}
    order by symbol,date 
    """
    market_value = selectBySQL('dfs://stockData','Kline',sql, index=['date','symbol'])
    return market_value

@timeit(1)
def preprocess_data(data_, winsorized=winsorize_nMAD, standard=standardize):
    """
    数据预处理， 对因子进行去nan、缩尾、标准化处理, 该设计是用来对因子df 进行数据预处理
    """
    data = data_.copy()
    no_inf_na_data = process_nan(data)
    winsorized_data = no_inf_na_data.apply(winsorized)
    standard_data = winsorized_data.apply(standard)
    return standard_data

def market_neutral(factor_):
    """市值中性化"""

    # 提取参数获取市值因子
    factor = factor_.copy()
    symbols = factor.index.get_level_values(1).unique().to_list()  # 股票列表
    dates = factor.index.get_level_values(0).unique().to_list() 
    start = str(min(dates))[:10].replace('-','.')
    end = str(max(dates))[:10].replace('-','.')
    market= get_market_value(symbols, start, end)                # 去nan

    factor_market = pd.concat([factor, market], axis=1)
    factor_market.dropna(inplace=True)
    result = sm.OLS(factor_market.iloc[:, :-1], factor_market.iloc[:,-1], hasconst=True).fit()
    return pd.DataFrame(result.resid, columns=factor.columns)

def industry_neutral(factor_, industry_name):
    """
    行业中性化
    """
    factor = factor_.copy()
    factor_num = len(factor.columns)
    symbols = factor.index.get_level_values(1).unique().to_list()  # 股票列表
    dates = factor.index.get_level_values(0).unique().to_list() 
    start = str(min(dates))[:10].replace('-','.')
    end = str(max(dates))[:10].replace('-','.')
    sql = f"""
    select date,symbol,{industry_name} as indcode from pt 
    where date between {start}:{end} and symbol in {symbols} order by symbol,date
    """
    indusry = selectBySQL('dfs://stockData', 'Kline', sql, index=['date','symbol'])
    factor_industry = pd.concat([factor, indusry], axis=1)
    factor_industry = factor_industry.dropna()
    factor_industry_hot = pd.get_dummies(factor_industry, prefix='', prefix_sep='')
    # 线性回归
    result = sm.OLS(factor_industry_hot.iloc[:, :factor_num], factor_industry_hot.iloc[:, factor_num:], hasconst=True).fit()
    
    # 返回按照index排序后的结果
    return pd.DataFrame(result.resid.sort_index(), columns=factor_.columns)

def cal_ic_matrix(multi_factor_, period):
    """
    计算 IC 矩阵， 以时间截面求相关系数，累计append 成为 ic_matrix, 最后把index 改回源时间序列的index
    """
    # 复制数据
    multi_factor = multi_factor_.copy()

    # 提取参数
    start = str(multi_factor.index.get_level_values(0)[0]).replace('-', '')[:8]  # 开始时间
    end = str(multi_factor.index.get_level_values(0)[-1]).replace('-', '')[:8]   # 结束时间
    codes = list(multi_factor.index.get_level_values(1).unique())  # 股票代码列表
    price = get_price(codes, start, end)                           # 获取价格数据
    multi_factor = pd.concat([multi_factor, price], axis=1)        # 按照复合索引合并
    multi_factor = multi_factor.dropna()                           # 去除缺失值
    multi_factor[f"return_{period}"] = (multi_factor.close.shift(-period) - multi_factor.close) / multi_factor.close  #  <------- 此处应该groupby处理！！！
    multi_factor = multi_factor.dropna()
    del multi_factor['close']
    obj = multi_factor.copy()


    # 开始计算相关系数
    dates = obj.index.get_level_values(0).unique()    # 日期列表
    ic_matrix = pd.DataFrame()
    for date in dates:
        df = obj.xs(date, level=0)
        ic_row_str = f"df.corr().return_{period}[:-1].T"  # 每一天的因子 IC 值
        ic_row = eval(ic_row_str)
        ic_matrix = ic_matrix.append([ic_row])

    ic_matrix.index = multi_factor.index.get_level_values(0).unique()
    return ic_matrix

def cal_roll_ic(ic_matrix, period):
    """
    滚动计算 IC 值, 即按照timeperiod 滚动计算每个因子IC值得均值。 
    """
    roll_mean = ic_matrix.rolling(period).mean()
    roll_mean = roll_mean.dropna()
    return roll_mean
    
def cal_roll_ir(ic_matrix, period):
    """
    计算 IR 值, 即按照timeperiod 先滚动计算每个因子IC值得均值和标准差, 然后再计算IR 值。
    """
    roll_mean = ic_matrix.rolling(period).mean()
    roll_std = ic_matrix.rolling(period).std()
    roll_ir = roll_mean / roll_std
    roll_ir = roll_ir.dropna()
    return roll_ir

def cal_weight(ic_matrix, by=None, period=5):
    """
    通过因子矩阵计算因子权重矩阵, 默认为因子等权重, 可选IC, IR 滚动计算因子权重
    """
    
    if by=="IC":
        roll_ic = cal_roll_ic(ic_matrix, period)
        roll_ic = roll_ic.dropna()
        sum_ = roll_ic.apply(lambda x: abs(x).sum(), axis=1)
        weight =  roll_ic.apply(lambda d: d / sum_)
        
    elif by=="IR":
        roll_ir = cal_roll_ir(ic_matrix, period)
        roll_ir = roll_ir.dropna()
        sum_ = roll_ir.apply(lambda x: abs(x).sum(), axis=1)
        weight =  roll_ir.apply(lambda d: d / sum_)
        
    else:
        sum_ = len(ic_matrix.columns)
        weight = ic_matrix.applymap(lambda x: 1/sum_)
    
    return weight

def cal_comprehensive_factor(multi_factor_, by=None, period=5):
    """
    计算经典合成因子
    params:
        multi_factor_: 多因子DataFrame
        by: 合成方式
        period: 滚动窗口
    """
    multi_factor = multi_factor_.copy()
    ic_matrix = cal_ic_matrix(multi_factor, period=1)
    weight = cal_weight(ic_matrix, by, period)
    weight_index = weight.index
    multi_factor = multi_factor.loc[weight_index]
    dates = multi_factor.index.get_level_values(0).unique()
    
    comprehensive_factor = pd.DataFrame(index=multi_factor.index)
    comprehensive_factor_list = []
    
    for date in dates:
        factor_matrix = multi_factor.loc[date].values
        weight_matrix = weight.loc[date]
        comprehensive_factor_matrix = np.dot(factor_matrix, weight_matrix)  # 是一个列表
        comprehensive_factor_list += list(comprehensive_factor_matrix)
        
    comprehensive_factor["comprehensive_factor"] = comprehensive_factor_list
    return comprehensive_factor

def get_index_intersection(df1,df2):
    # 获取df1, df2的共同索引
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)
    index1 = set(df1.index)
    index2 = set(df2.index)
    index = index1.intersection(index2)
    return df1.loc[index].sort_index(), df2.loc[index].sort_index()

result_list = []
def get_roll_by_month(df_):
    """
    对因子数据按月进行滚动，返回每个月的因子数据
    """
    df = df_.sort_values(by=['date','code']).copy()
    df.index = range(len(df))
    length = len(df.index)
    
    stage_list = []
    first_year, first_month = df['date'][0].year, df['date'][0].month
    last_year, last_month = df['date'][length-1].year, df['date'][length-1].month
    
    if first_year==last_year and first_month==last_month:
        global result_list
        result_list.append(df)

    for i in range(length):
        if df['date'][i].year == first_year and df['date'][i].month==first_month:
            stage_list.append(i)

        else:
            result_list.append(df.iloc[stage_list])
            remains = df.iloc[i:]
            get_roll_by_month(remains)
            break

    return list(map(lambda x: x.set_index(['date','code']), result_list))

def get_return_rank(price, period=5, quantiles=5):
    """
    输入价格数据，获取收益率分层的结果
    price: multi_index
    """
    df_list = []
    for code in price.index.get_level_values(1).unique():
        df = price.xs(code, level=1)
        df['code'] = code
        df['return'] = (df['close'] / df['close'].shift(period)) - 1
        df.dropna(inplace=True)
        df['return_rank'] = df[['return']].apply(lambda x: rank(x, quantiles))
        df_list.append(df)
    
    target = pd.concat(df_list, axis=0)
    target.set_index([target.index,'code'], inplace=True)
    target.sort_index(inplace=True)
    return_rank = target.get(['return_rank'])
    return_rank.dropna(inplace=True)
    return return_rank

def get_factor_analyse(factor_name, factor_data, long_short=True, equal_weight=False):
    """
    返回因子初步分析结果， 主要是因子IC值统计量
    """
    alpha_beta = perf.factor_alpha_beta(factor_data, demeaned=long_short, equal_weight = equal_weight).get(['1D'])
    
    ic_data = perf.factor_information_coefficient(factor_data).get(['1D'])
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["IR"] = ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)
    
    alpha_beta.index = ['Alpha','Beta']
    analyse = pd.concat([ic_summary_table.T, alpha_beta.iloc[[0]]], axis=0)
    analyse.columns = [factor_name]
    return analyse

def get_batch_factor_analyse(multi_factor_, price, quantiles=5, periods=(1, 5, 10), long_short=True, equal_weight=False):
    """
    批量对多因子数据进行分析，遍历调用get_factor_analyse
    后期需要改成多进程加快计算速度。
    """
    multi_factor = multi_factor_.copy()
    analyse_list = []
    factor_names = multi_factor.columns
    for factor_name in tqdm(factor_names, colour='green'):
        try:
            # 获取 factor, price --> factor_data --> analyse
            factor = multi_factor.get([factor_name])
            pivot_price = price.pivot_table(index=price.index.get_level_values(0), columns=price.index.get_level_values(1), values='close')
            factor_data = utils.get_clean_factor_and_forward_returns(factor, pivot_price, quantiles=quantiles, periods=periods)

            analyse = get_factor_analyse(factor_name, factor_data, long_short=long_short, equal_weight=equal_weight)
            analyse_list.append(analyse)
        
        except utils.MaxLossExceededError:
            # 可能存在数据不足的情况
            print(f"{factor_name}: max_loss (35.0%) exceeded 100.0%, consider increasing it")

    result_analyse = pd.concat(analyse_list, axis=1)
    return result_analyse.T

def get_data_to_sql_ready(data, start,end,equal_weight,long_short,factor_name="haven't_set_factor_name",input_date=None, group_neutral=False, by_group=False):
    # 有关因子回测的全部信息，综合成一个表格
    all_concated_tables = get_all_concated_tables(data,equal_weight = equal_weight,long_short = long_short, group_neutral=group_neutral, by_group=by_group)

    # 装置，方便数据库存储。
    all_concated_tablesT = all_concated_tables.T
    all_concated_tablesT["FACTOR_NAME"] = factor_name
    all_concated_tablesT.reset_index(inplace=True)
    all_concated_tablesT.rename(columns={"index": "Turnover_Days"}, inplace=True)

    all_concated_tablesT["Turnover_Days"] = all_concated_tablesT["Turnover_Days"].apply(lambda x: int(x[:-1]))

    all_concated_tablesT.set_index("FACTOR_NAME", inplace=True)
    all_concated_tablesT.reset_index(inplace=True)

    # 原df的列名进行更改，以便数据库的操作
    new_columns_list = []
    for each in list(all_concated_tablesT.columns):
        new_columns_list.append(each)

    all_concated_tablesT.columns = new_columns_list

    if input_date is not None:
        all_concated_tablesT["input_date"] = input_date

    #20220409 增加两列 start_date , end_date ---
    all_concated_tablesT["start_date"] = start
    all_concated_tablesT["end_date"] = end
    #20220409 增加两列 start_date , end_date ---
    return all_concated_tablesT

def get_all_concated_tables(data,equal_weight,long_short, group_neutral=False, by_group=False):
    mean_quant_rateret_table, returns_table = get_mean_quant_rateret_returns_table(data,long_short)
    inf_tear_sheet_table = get_inf_tear_sheet_data(data).T
    turnover_table, autocorrelation_table = get_turnover_autocorrelation_table(data)

    mean_quant_rateret_table_2 = tears.get_mean_quant_rateret_table_2(data)
    factor_feature_table = tears.get_factor_feature_table(data,equal_weight = equal_weight, long_short=long_short, group_neutral=group_neutral, by_group=by_group)

    all_concated_tables = pd.concat([
        returns_table,
        inf_tear_sheet_table,
        turnover_table,
        autocorrelation_table,
        mean_quant_rateret_table_2,
        factor_feature_table
    ], axis=0)
    return all_concated_tables

def get_mean_quant_rateret_returns_table(factor_data, long_short):
    """
    input：
        factor_data:通过alphalens标准处理后的dataframe
    output:
        mean_quant_rateret_table, returns_table: 包含mean_quant_rateret_table和returns_table中的df
    """
    mean_quant_rateret_table, returns_table = create_return_tear_sheet_data(factor_data, long_short=long_short)
    return mean_quant_rateret_table, returns_table

def create_return_tear_sheet_data(factor_data, long_short, group_neutral=False):
    # Returns Analysis
    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data, demeaned=long_short, group_adjust=group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    return [plot_quantile_statistics_table_data(factor_data).apply(lambda x: x.round(6)),
            plot_returns_table_data(alpha_beta, mean_quant_rateret, mean_ret_spread_quant)]

def plot_quantile_statistics_table_data(factor_data):
    quantile_stats = factor_data.groupby('factor_quantile') \
        .agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] \
                                / quantile_stats['count'].sum() * 100.

    return quantile_stats

def plot_returns_table_data(alpha_beta,mean_ret_quantile,mean_ret_spread_quantile):
    DECIMAL_TO_BPS = 10000
    returns_table = pd.DataFrame()
    returns_table = returns_table.append(alpha_beta)
    returns_table.loc["Mean Period Wise Return Top Quantile (bps)"] = \
        mean_ret_quantile.iloc[-1] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Return Bottom Quantile (bps)"] = \
        mean_ret_quantile.iloc[0] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Spread (bps)"] = \
        mean_ret_spread_quantile.mean() * DECIMAL_TO_BPS
    return returns_table

def get_inf_tear_sheet_data(factor_data):
    """
    input：
        factor_data:通过alphalens标准处理后的dataframe
    output:
        inf_tear_sheet_data: 包含inf_tear_sheet_data中的df
    """
    inf_tear_sheet_data = create_inf_tear_sheet_data(factor_data)
    return inf_tear_sheet_data

def get_turnover_autocorrelation_table(factor_data):
    """
    input：
        factor_data:通过alphalens标准处理后的dataframe
    output:
        turnover_table, autocorrelation_table: 包含turnover_table, autocorrelation_table中的df
    """
    turnover_table, autocorrelation_table = create_turnover_tear_sheet_data(factor_data)
    return turnover_table, autocorrelation_table

def create_inf_tear_sheet_data(factor_data):
    # Information Analysis
    ic = perf.factor_information_coefficient(factor_data)
    table = plot_information_table_data_table(ic)
    return table.T

def plot_information_table_data_table(ic_data):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    #new 20211218 Cal IR=IC/STD_IC
    # 四舍五入处理
    return ic_summary_table.apply(lambda x: x.round(3)).T

def create_turnover_tear_sheet_data(factor_data):
    # Turnover Analysis

    # 从cloumns 当中--> ['1D','5D', '10D']
    periods = utils.get_forward_returns_columns(factor_data.columns)

    # ['1D','5D','10D'] --> [1, 5, 10]
    periods = list(map(lambda p: pd.Timedelta(p).days, periods))

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in range(1, int(quantile_factor.max()) + 1)
            ],
            axis=1,
        )
        for p in periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in periods
        ],
        axis=1,
    )

    return plot_turnover_table_test_table(autocorrelation, quantile_turnover)

def plot_turnover_table_test_table(autocorrelation_data, quantile_turnover):
    turnover_table = pd.DataFrame()
    for period in sorted(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].iteritems():
            turnover_table.loc["Quantile {} Mean Turnover ".format(quantile),
                               "{}D".format(period)] = p_data.mean()
    auto_corr = pd.DataFrame()
    for period, p_data in autocorrelation_data.iteritems():
        auto_corr.loc["Mean Factor Rank Autocorrelation",
                      "{}D".format(period)] = p_data.mean()

    return [turnover_table.apply(lambda x: x.round(3)), auto_corr.apply(lambda x: x.round(3))]

def layerRank_for_oneGroup_noNan(groupDf, scoreClm, layerNum, ascend=False):
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