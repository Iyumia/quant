import warnings
import pandas as pd
import numpy as np
import empyrical
import seaborn as sns
from scipy import stats
from datetime import datetime
import statsmodels.api as sm
from scipy.stats.mstats import winsorize

from ..database.dolphin import selectBySQL, df2Table
from ..database.mysql import get_data
from ..chart import eTable, eLine, ePage

warnings.filterwarnings('ignore')


def winsorize_func(group, name, winsorized='median'):
    """因子数据异常值处理"""
    if winsorized == 'percent':
        group[name]=winsorize(group[name], limits=[0.025, 0.025])
        
    elif winsorized == 'std':
        up = group[name].mean() + 3*group[name].std()
        low = group[name].mean() - 3*group[name].std()
        group[group[name]>up] = up
        group[group[name]<low] = low
        
    elif winsorized == 'median':
        median = np.median(group[name])
        mad = np.median(abs(group[name] - median))
        up = median + (3 * mad)
        low = median - (3 * mad)
        group[name] = np.where(group[name] > up, up, group[name])
        group[name] = np.where(group[name] < low, low, group[name])
    else:
        raise NotImplementedError(f'[INFO]: 暂不支持使用{winsorize}方法处理因子异常值数据!')
    return group

def standard_func(group, name, standard):
    """因子数据标准化"""
    if standard == 'z-score':
        group[name] = (group[name] - group[name].mean()) / group[name].std()
    elif standard == 'min-max':
        group[name] = (group[name] - group[name].min()) / (group[name].max() - group[name].min())
    else:
        raise NotImplementedError(f'[INFO]: 暂不支持使用{standard}方法对因子数据进行标准化!')
    return group

def get_trade_date_shift(present, shift):
    """获取交易日的偏移天数"""
    trade_date_array = get_data("select cast(cast(date as char) as date) as `date`  from `trade_day` order by `date`;").date.values
    present_ix = np.where(trade_date_array==present)[0][0]
    shift_ix = present_ix+shift
    shift_date = trade_date_array[shift_ix]
    return shift_date

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


class Factor:
    """单因子数据预处理类"""
    def __init__(self, data):
        self.data = data
        self.name = data.columns.to_list()
        self.start = data.index.levels[0].min()
        self.end = data.index.levels[0].max()
        self.symbols = data.index.levels[1]
    
    def __call__(self, name:list):
        """对单个、多个因子进行切片"""
        data = self.data.get(name)
        factor = Factor(data)
        return factor

    def preprocess_data(self, nan=['ffill','median','drop'], winsorized='median', standard='z-score', neutral=[True, True]):
        self.preprocess_nan(nan, inplace=True)
        self.preprocess_winsorize(winsorized, inplace=True)
        self.preprocess_standard(standard, inplace=True)
        self.preprocess_neutral(neutral, inplace=True)
        
    def preprocess_nan(self, nan=['ffill','median','drop'], inplace=True):
        """处理缺失值"""
        data = self.data.copy()
        data = data.replace([np.inf,-np.inf],np.nan)
        print(f'[INFO]: 缺失值处理; 处理前数据量:{len(data)}, 数据缺失情况:{data.isnull().sum().sum()}')
        
        for method in nan:
            if method == 'drop':
                data.dropna(inplace=True)
                print(f'[INFO]: 缺失值处理-删除缺失值; 删除后数据量:{len(data)}, 数据缺失情况:{data.isnull().sum().sum()}')
            elif method == 'ffill':
                data = data.groupby(data.index.get_level_values(1), group_keys=False).apply(lambda group: group.fillna(method='ffill'))
                print(f'[INFO]: 缺失值处理-向后填充; 数据缺失情况:{data.isnull().sum().sum()}')
            elif method == 'median':
                data = data.groupby(data.index.get_level_values(0), group_keys=False).apply(lambda group: group.fillna(group.median()))
                print(f'[INFO]: 缺失值处理-中位数填充; 数据缺失情况:{data.isnull().sum().sum()}')
            else:
                raise NotImplementedError(f'[WARNING]: 暂不支持使用{method}方法处理缺失数据!')

        print(f'[INFO]: 缺失值处理完成, 处理后数据量:{len(data)}, 数据缺失情况:{data.isnull().sum().sum()}')
        if inplace==True:
            self.data = data
            self.data.columns = self.name
            self.nan = nan
            return None
        else:
            self.nan = []
            return data
    
    def preprocess_winsorize(self, winsorized='median', inplace=True):
        """处理异常值数据"""
        print(f'[INFO]: 数据缩尾处理-{winsorized}...')
        data = self.data.copy()
        data = data.groupby(data.index.get_level_values(0), group_keys=False).apply(lambda group: winsorize_func(group, self.name, winsorized))
        print(f'[INFO]: 数据缩尾处理完成!')
        if inplace==True:
            self.data = data
            self.data.columns = self.name
            self.winsorized = winsorized
            return None
        else:
            self.winsorized = ''
            return data
    
    def preprocess_standard(self, standard, inplace=True):
        """数据标准化"""
        print(f'[INFO]: 数据标准化-{standard}...')
        data = self.data.copy()
        data = data.groupby(data.index.get_level_values(0), group_keys=False).apply(lambda group:standard_func(group, self.name, standard))
        print('[INFO]: 数据标准化完成!')
        if inplace==True:
            self.data = data
            self.data.columns = self.name
            self.standard = standard
            return None
        else:
            self.standard = ''
            return data
    
    def preprocess_neutral(self, neutral=[False,False], inplace=True):
        """ 因子数据中性化 """
        data = self.data.copy()
        if neutral[0] or neutral[1]:
            market_cap_industry = self.get_marketcap_industry()
            data_market_cap_industry = pd.concat([data, market_cap_industry], axis=1).dropna()
            var_split = np.where(data_market_cap_industry.columns=='market_cap')[0][0] 

            if neutral[0] and not neutral[1]:
                print('[INFO]: 中性化处理-市值中性...')
                market_cap_neutral_result = sm.OLS(data_market_cap_industry.iloc[:, :var_split], np.log(data_market_cap_industry.iloc[:,var_split]), hasconst=True).fit().resid.reset_index()
                market_cap_neutral_result.set_index(['date','security'],inplace=True)
                market_cap_neutral_result.columns = [self.name]
                data = market_cap_neutral_result

            elif neutral[1] and not neutral[0]:
                print('[INFO]: 中性化处理-行业中性化...')
                industry_neutral_result = sm.OLS(data_market_cap_industry.iloc[:, :var_split], data_market_cap_industry.iloc[:,var_split+1:], hasconst=True).fit().resid.reset_index()
                industry_neutral_result.set_index(['date','security'],inplace=True)
                industry_neutral_result.columns = [self.name]
                data = industry_neutral_result

            else:
                print('[INFO]: 中性化处理-市值 & 行业中性化...')
                market_cap_industry_neutral_result = sm.OLS(data_market_cap_industry.iloc[:, :var_split], data_market_cap_industry.iloc[:,var_split:], hasconst=True).fit().resid.reset_index()
                market_cap_industry_neutral_result.set_index(['date','security'],inplace=True)
                market_cap_industry_neutral_result.columns = [self.name]
                data = market_cap_industry_neutral_result
            print('[INFO]: 中性化处理完成!')
        else:
            pass    
        
        if inplace==True:
            self.data = data
            self.data.columns = self.name
            self.neutral=neutral
            return None
        else:
            self.neutral=neutral
            return data
            
    def get_marketcap_industry(self):
        """获取市值、行业数据"""
        sql = "select date, security, market_cap, sw1_code from pt"
        marketcap_industry = selectBySQL('dfs://StockInfoLib','StockInfoTable', sql).set_index(['date','security'])

        # 向后填充市值、行业缺失数据
        marketcap_industry.groupby(marketcap_industry.index.get_level_values(1), group_keys=False).apply(lambda group: group.fillna(method='ffill'))
        # 行业数据转换成哑变量
        industry_one_hat = pd.get_dummies(marketcap_industry['sw1_code'], prefix='', prefix_sep='')
        marketcap_onehatindustry = pd.concat([marketcap_industry, industry_one_hat], axis=1)
        del marketcap_onehatindustry['sw1_code']
        return marketcap_onehatindustry

    def corr(self, period=3):
        if self.data.shape[1]==1:
            # 单因子时，计算自相关系数
            def shift(group):
                group['shift'] = group[self.name].shift(period)
                return group
            autocorrelation = self.data.groupby(self.data.index.get_level_values(1), group_keys=False).apply(shift).dropna().corr().iloc[0,1]
            return round(autocorrelation, 4)
        else:
            # 多因子时，绘制相关系数热力图
            corr = self.corr()
            sns.heatmap(corr)



class Analyst:
    """因子分析管理类"""
    def __init__(self, factor, start='2015.01.01', end='2023.05.31', quantiles=10, periods=(1,5,10), retmethod='o2o', benchmark='000300.XSHG', commission=0.001, cretmethod='cumprovd'):
        self.factor = factor 
        self.name = factor.name[0]
        self.start = start
        self.end = end
        self.quantiles = quantiles
        self.periods = periods 
        self.retmethod = retmethod
        self.cretmethod = cretmethod
        self.benchmark = benchmark
        self.commission = commission
        self.cretmethod = cretmethod
        self.create_time = datetime.now()
        
    def get_stock_forward_return(self):
        """获取股票未来收益率并减去手续费"""
        rets = ['o1o'+str(p+1) for p in self.periods] if self.retmethod=='o2o' else ['c0c'+str(p) for p in self.periods]
        sql = f'select * from pt where date between {self.start}:{self.end} order by security,date'
        stock_return = selectBySQL('dfs://StockReturnLib', 'StockReturnTable', sql).get(['date','security']+rets).set_index(['date','security']) - self.commission
        return stock_return

    def get_benchmark_forward_return(self):
        """ 获取指数未来收益率 """
        rets = ['o1o'+str(p+1) for p in self.periods] if self.retmethod=='o2o' else ['c0c'+str(p) for p in self.periods]
        sql = f"select * from pt where date between {self.start}:{self.end} and code='{self.benchmark}' order by date"
        benchmark_return = selectBySQL('dfs://IndexReturnLib', 'IndexReturnTable', sql).get(['date']+rets).set_index('date')
        return benchmark_return

    def get_stock_hold_forward_return(self):
        """计算每个持仓周期内的股票持仓收益率"""
        stock_forward_return = self.get_stock_forward_return()
        dates = stock_forward_return.index.levels[0]
        hold_forward_return_list = []
        for retcol in stock_forward_return.columns:
            period = int(retcol.split('o')[-1])-1 if 'o' in retcol else int(retcol.split('c')[-1]) 
            date_cuts = dates[::period]  # 时间切片
            hold_forward_return = stock_forward_return.get([retcol]).loc[date_cuts]
            hold_forward_return.columns = ['stock_hold_forward_return']
            hold_forward_return['period'] = period
            hold_forward_return_list.append(hold_forward_return)
        hold_forward_return = pd.concat(hold_forward_return_list, axis=0)
        return hold_forward_return

    def get_benchmark_hold_forward_return(self):
        """获取指数在不同持仓周期的未来收益率"""
        benchmark_return = self.get_benchmark_forward_return()
        dates = benchmark_return.index
        benchmark_hold_forward_return_list = []
        # 计算基准收益在不同周期下的持仓收益率
        for retcol in benchmark_return.columns:
            period = int(retcol.split('o')[-1])-1 if 'o' in retcol else int(retcol.split('c')[-1]) 
            date_cuts = dates[::period]  # 时间切片
            benchmark_hold_forward_return = benchmark_return.get([retcol]).loc[date_cuts]
            benchmark_hold_forward_return.columns = ['benchmark_hold_forward_return']
            benchmark_hold_forward_return['period'] = period
            benchmark_hold_forward_return_list.append(benchmark_hold_forward_return)
            
        benchmark_hold_forward_return = pd.concat(benchmark_hold_forward_return_list, axis=0)
        return benchmark_hold_forward_return 

    def get_analyst_base(self):
        """以下所有分析围绕 base 展开"""
        stock_hold_forward_return = self.get_stock_hold_forward_return()
        benchmark_hold_forward_return = self.get_benchmark_hold_forward_return()
        hold_forward_return = pd.merge(left=stock_hold_forward_return.reset_index(), right=benchmark_hold_forward_return.reset_index(), on=['date','period'], how='inner')
        hold_forward_return['excess_return'] = hold_forward_return['stock_hold_forward_return'] - hold_forward_return['benchmark_hold_forward_return']
        # 删除去nan
        base = pd.merge(left=self.factor.data.reset_index(), right=hold_forward_return, on=['date','security'], how='inner')
        # 排除异常股票
        clean_stock_sql = "select date, security from pt where is_st=false and is_new=false and is_release=false and paused=0 and date>2015.01.01"
        clean_stock = selectBySQL('dfs://StockInfoLib','StockInfoTable', clean_stock_sql)
        clean_base = pd.merge(left=clean_stock, right=base, on=['date','security'], how='left').dropna()
        # 对因子值进行分组
        clean_base_quantile = clean_base.groupby(['period','date'],group_keys=False) \
            .apply(lambda group:qcut(group, self.name, self.quantiles, ascend=True)) \
            .rename(columns={'layer':'quantile'}).sort_values(['period','quantile','date'])
        clean_base_quantile['period'] = clean_base_quantile['period'].astype(int)
        return clean_base_quantile

    def stats_ic(self, base):
        """分组进行 IC 值分析"""
        # 计算不同 period 对应的ic的时间序列df
        ic_df = base.groupby(['period','date'])[[self.name,'stock_hold_forward_return']].apply(lambda group:group.corr().iloc[[0],1:]).rename(columns={'stock_hold_forward_return':'ic'})
        ic_df.index = [ic_df.index.get_level_values(0), ic_df.index.get_level_values(1)]

        def cal_group_ic_stats(group):
            group = group.dropna()
            ic_mean = group['ic'].mean()
            ic_std = group['ic'].std()
            ir = ic_mean / ic_std
            ratio_of_abs_ic_gt_002 = group.query("`ic`>0.02 or `ic`<-0.02").__len__() / group.__len__()
            t,p = stats.ttest_1samp(group['ic'],0.02)
            
            ic_stats_df = pd.DataFrame({
                'ic_mean':[ic_mean],
                'ic_std':[ic_std],
                'ir':[ir],
                'ratio_of_abs_ic_gt_002':[ratio_of_abs_ic_gt_002],
                't_ic':[t],
                'p_ic':[p]
            })
            return ic_stats_df
        
        ic_stats_df = ic_df.groupby(ic_df.index.get_level_values(0)).apply(cal_group_ic_stats)
        ic_stats_df.index = ic_stats_df.index.get_level_values(0)
        ic_stats_df.reset_index(inplace=True)
        return ic_stats_df
    
    def stats_return(self, base):
        """统计不同period, quantile 下的收益率指标"""
        return_df = base.groupby(['period','quantile','date'])[['stock_hold_forward_return']].mean()

        def cal_group_return_stats(group):
            group = group.dropna()
            mean_return = group['stock_hold_forward_return'].mean()
            std_return = group['stock_hold_forward_return'].std()
            t,p = stats.ttest_1samp(group['stock_hold_forward_return'],0.0)
            return_stats_df = pd.DataFrame({
                'mean_return':[mean_return],
                'std_return':[std_return],
                't_return':[t],
                'p_return':[p]
            })
            return return_stats_df
        
        ret_stats_df = return_df.reset_index().groupby(['period','quantile']).apply(cal_group_return_stats)
        ret_stats_df.index = [ret_stats_df.index.get_level_values(0),ret_stats_df.index.get_level_values(1)]
        ret_stats_df.reset_index(inplace=True)
        return ret_stats_df

    def stats_portfolio(self, base):
        """计算不同 period, quantile 下的组合指标"""
        daily_return = base.groupby(['period','quantile','date'])[['stock_hold_forward_return','benchmark_hold_forward_return']].mean()

        def cal_group_portfolio_stats(group):
            max_drawdown = empyrical.max_drawdown(group['stock_hold_forward_return'])
            sharpe_ratio = empyrical.sharpe_ratio(group['stock_hold_forward_return'])
            calmar_ratio = empyrical.calmar_ratio(group['stock_hold_forward_return'])
            cum_return = empyrical.cum_returns_final(group['stock_hold_forward_return'])
            alpha, beta = empyrical.alpha_beta(returns=group['stock_hold_forward_return'], factor_returns=group['benchmark_hold_forward_return'])
            portfolio_stats_df = pd.DataFrame({
                'cum_return':[cum_return],
                'sharpe':[sharpe_ratio],
                'calmar':[calmar_ratio],
                'max_drawdown':[max_drawdown],
                'alpha':[alpha],
                'beta':[beta]
            })
            return portfolio_stats_df
        
        portfolio_stats_df = daily_return.groupby([daily_return.index.get_level_values(0), daily_return.index.get_level_values(1)]).apply(cal_group_portfolio_stats)
        portfolio_stats_df.index = [portfolio_stats_df.index.get_level_values(0), portfolio_stats_df.index.get_level_values(1)]
        portfolio_stats_df.reset_index(inplace=True)
        return portfolio_stats_df

    def get_analyst_summary(self, base, save=False):
        """计算所有统计指标, 并汇总到一个表格中"""
        ic_stats_df = self.stats_ic(base)
        ret_stats_df = self.stats_return(base)
        portfolio_stats_df = self.stats_portfolio(base)
        left_df = pd.merge(left=ret_stats_df, right=ic_stats_df,on='period',how='outer')
        summary = pd.merge(left=left_df, right=portfolio_stats_df, on=['period','quantile'],how='outer')
        summary['mean_return'] = summary['mean_return'] * 10000
        summary.iloc[:,2:] = summary.iloc[:,2:].applymap(lambda x:round(x,4))
        length = len(summary)
        summary.insert(0, 'create_time', [self.create_time]*length)
        summary.insert(0, 'benchmark', [self.benchmark]*length)
        summary.insert(0, 'end', [self.end]*length)
        summary.insert(0, 'start', [self.start]*length)
        summary.insert(0, 'name', [self.name]*length)
        if save:
            summary_copy = summary.copy()
            summary_copy['start'] = summary_copy['start'].astype('datetime64')
            summary_copy['end'] = summary_copy['end'].astype('datetime64')
            summary_copy['create_time'] = pd.to_datetime(summary_copy['create_time'])
            df2Table('dfs://FactorSummaryLib', 'FactorSummaryTable', summary_copy)
        return summary

    def get_net_value(self, base, save=False):
        """绘制收益率净值曲线"""
        cumret = base.groupby(['period','quantile','date'])[['stock_hold_forward_return', 'benchmark_hold_forward_return', 'excess_return']].mean() + 1
        cumret.reset_index(inplace=True)
        cumret[['stock_hold_forward_return','benchmark_hold_forward_return','excess_return']] = cumret.groupby(['period','quantile'], group_keys=False)[['stock_hold_forward_return','benchmark_hold_forward_return','excess_return']].apply(np.cumprod)
        length = len(cumret)
        cumret.insert(0, 'create_time', [self.create_time]*length)
        cumret.insert(0, 'name', [self.name]*length)
        cumret.columns = ['name','create_time','period','quantile','date','stock_net_value','benchmark_net_value','excess_net_value']
        if save:
            cumret_copy = cumret.copy()
            cumret_copy['create_time'] = pd.to_datetime(cumret_copy['create_time'])
            cumret_copy['date'] = cumret_copy['date'].astype('datetime64')
            df2Table('dfs://FactorNetValueLib', 'FactorNetValueTable', cumret_copy)
        return cumret

    def run(self, save=False):
        """生成ePage对象"""
        # 添加汇总表格
        base = self.get_analyst_base()
        self.epage = ePage()
        summary = self.get_analyst_summary(base, save=save)
        summary.columns = [
            'Name','Start','End','Benchmark','Date','Period','Quantile','Return-Mean(bps)','Return-Std','t-stats(Return)','p-value(Return)',
            'IC-Mean','IC-Std','IR','Ratio of abs(IC)>0.02','t-stats(IC)','p-value(IC)',
            'Cumulate Return','Sharpe','Calmar','Max-Drawdown','Alpha','Beta']
        self.etable = eTable(summary, '因子表现汇总', f'因子名称:{self.name}')
        self.epage.add(self.etable)

        # 添加分组回测净值曲线; 超额净值曲线
        net_value = self.get_net_value(base, save=save)
        net_value['date'] = net_value['date'].astype(str)
        for values in ['stock_net_value', 'excess_net_value']:
            for period in self.periods:
                data = net_value.query(f"`period`=={period}").pivot_table(index='date', columns='quantile',values=values)
                data = data.applymap(lambda x: round(x,4))
                data.columns = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
                ## 尝试去除净值中的缺失值情况
                data = data.fillna(method='ffill')
                if values == 'stock_net_value':
                    # 先绘制分组回测累计净值曲线
                    eline = eLine(data.reset_index(), '资产净值曲线', f'调仓周期:{period}天')
                    self.epage.add(eline)
                    # 再绘制多空净值曲线
                    data = data.get(['Q1','Q10'])
                    data['Long-Short'] = data['Q10'] - data['Q1']
                    data['Long-Short'] = data['Long-Short'].apply(lambda x: round(x,4))
                    eline = eLine(data.get(['Long-Short']).reset_index(), '多-空净值曲线', f'调仓周期:{period}天')
                    self.epage.add(eline)
                else:
                    eline = eLine(data.reset_index(), '超额净值曲线', f'调仓周期:{period}天')
                    self.epage.add(eline)
        return self.epage

    def show(self):
        """显示epage"""
        return self.epage.show()
    
    def save(self, html=None, csv=None):
        """保存分析结果"""
        if html is not None:
            self.epage.save(html)
        if csv is not None:
            self.etable.data.to_csv(csv, index=False)
            
        
