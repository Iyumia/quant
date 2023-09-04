import pandas as pd
from tqdm import tqdm
import backtrader as bt
from datetime import datetime
from ..utils import *
from .db import df_2_table
from ..decorator import timer
from .strategy import DataFeed





class Kernel:
    """ 策略模版 """
    def __init__(self, strategy, fromdate, todate, money=2000000, record=False, comm=0.00, slippage=0.00, basic=None):
        """ 完整策略必备属性 """
        self.strategy = strategy
        self.strategy.record = record
        self.strategy.fromdate = fromdate
        self.strategy.todate = todate
        self.money = money
        self.comm = comm
        self.slippage = slippage
        self.basic = basic

    def init(self, test):
        """策略初始化"""
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(self.money)
        self.cerebro.broker.setcommission(self.comm)
        self.cerebro.broker.set_slippage_perc(perc=self.slippage)
        self.cerebro.addstrategy(self.strategy)
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        self.load_data(test=test) 

    def load_data(self, test=False):
        """ 加载数据给cerebro """
        self.datas = self.basic if self.basic is not None else pd.read_pickle(self.strategy.url)
        self.strategy.basic = self.basic
        self.datas[(self.datas['date']<=self.strategy.todate) & (self.datas['date']>=self.strategy.fromdate)]  # 时间维度筛选
        self.datas['date'] = self.datas['date'].astype('datetime64')
        self.datas.set_index('date', inplace=True)
        self.index = pd.DataFrame(index=self.datas.index.unique()) 
        self.securities = self.datas.query("`weight`>0").security.unique()
        self.securities = self.securities if not test else self.securities[:20]
        self.datas = self.datas[self.datas['security'].isin(self.securities)]   # 筛选有交易记录的股票

        # 加载数据给cerebro
        for security in self.securities:
        # for security in tqdm(self.securities, ncols=100, colour='green'):
            data = self.datas.query(f"security=='{security}'")
            data = pd.merge(self.index, data, left_index=True, right_index=True, how='left')
            data[['open','high','low','close']] = data[['open','high','low','close']].fillna(method='ffill')
            data[['open','high','low','close']] = data[['open','high','low','close']].fillna(0)
            data['weight'] = data['weight'].fillna(0)
            data['paused'] = data['paused'].fillna(1)
            datafeed = DataFeed(dataname=data, fromdate=self.strategy.fromdate, todate=self.strategy.todate)
            self.cerebro.adddata(datafeed, name=security) 
        
    def get_transactions(self, transactions):
        transactions.reset_index(inplace=True)
        transactions['date'] = transactions['date'].apply(lambda dt: dt.strftime('%Y-%m-%d'))
        transactions.rename(columns={'symbol':'security'},inplace=True)
        security_info = get_security_info()
        transactions = pd.merge(left=transactions, right=security_info, on='security', how='left')
        transactions = pd.merge(left=transactions, right=self.strategy.structure, on='date', how='left')
        transactions['side'] = transactions['amount'].apply(lambda x: '买入' if x>0 else '卖出')
        del transactions['sid']
        del transactions['start_date']
        def cal_stock_profit(group):
            """计算每只股票平仓时的盈亏情况"""
            group['profit'] = group['value'] + group['value'].shift(1)
            group.loc[group['side']=='买入', 'profit'] = 0
            group['profit'] = group['profit'].fillna(0)
            return group
        transactions = transactions.groupby('security', group_keys=False).apply(cal_stock_profit)
        transactions['策略名称'] = self.strategy.name
        transactions['uuid'] = self.strategy.uuid
        transactions['资金使用率'] = 1- (transactions['cash'] / transactions['asset'])
        commission = self.strategy.commission
        commission['date'] = commission['date'].apply(lambda dt: dt.strftime('%Y-%m-%d'))
        record = pd.merge(left=transactions, right=commission, on=['date','security'], how='outer')
        cols = ['策略名称','uuid','date','security','name','side','price','amount','value','commission','profit','资金使用率','cash','asset']
        record = record.get(cols)
        record.columns = ['策略名称','uuid','日期','股票代码','股票名称','交易方向','成交价格','成交数量','成交金额','手续费','盈利金额','资金使用率','可用资金','账户权益']
        record['测试时间'] = self.testime
        return record


    def get_report_returns(self, returns, code='000300.XSHG'):
        # 指数收益率
        index_return = get_index_return(code=code)
        index_return.columns = ['日期','000300.XSHG']
        # 账户收益率
        account_return = returns.reset_index()
        account_return.rename(columns={'index':'日期'},inplace=True)
        account_return['日期'] = account_return['日期'].apply(lambda dt: dt.strftime('%Y-%m-%d'))
        account_index_return = pd.merge(left=account_return, right=index_return, on='日期', how='left')
        account_index_return.columns = ['日期','资产组合','沪深300']
        account_index_return['超额收益'] = 0
        
        # 计算统计指标
        max_drawdown = empyrical.max_drawdown(account_index_return['资产组合'])  # 最大回撤
        alpha, beta = empyrical.alpha_beta(account_index_return['资产组合'], account_index_return['沪深300'])  # 阿尔法、贝塔收益
        sharpe = empyrical.sharpe_ratio(account_index_return['资产组合'])  # 夏普比率
        calmar = empyrical.calmar_ratio(account_index_return['资产组合'])  # 卡玛比率
        annual_return = empyrical.annual_return(account_index_return['资产组合'])  # 年化收益率
        cum_return = empyrical.cum_returns_final(account_index_return['资产组合']) # 累计收益率

        """|策略名称|uuid|开始日期|结束日期|期初权益|期末权益|最大回撤|阿尔法收益|贝塔收益|夏普比率|卡玛比率|年化收益率|累计收益率|资金使用率|盈利次数|亏损次数|胜率|策略描述|"""
        # 计算 etable
        rowdata = pd.DataFrame(columns=
                                   ['策略名称','uuid','作者','开始日期','结束日期','期初权益','期末权益','交易手续费','交易滑点','最大回撤比','阿尔法收益率',
                                    '贝塔收益率','夏普比率','卡玛比率','年化收益率','累计收益率','资金使用率','盈利次数','亏损次数','胜率','说明'
                                    ])
        row = [
            self.strategy.name,self.strategy.uuid,self.strategy.author,self.strategy.fromdate,self.strategy.todate,self.money,round(self.strategy.finasset,2),
            self.comm,self.slippage, f'{round(max_drawdown*100,2)}%',f'{round(alpha*100,2)}%',f'{round(beta*100,2)}%',round(sharpe,2),
            round(calmar,2), f'{round(annual_return*100,2)}%', f'{round(cum_return*100,2)}%', f'{round(self.strategy.posratio*100,2)}%', 
            self.strategy.procount, self.strategy.loscount, f'{round(self.strategy.winrate*100,2)}%',self.strategy.doc
            ]
        rowdata.loc[0] = row
        rowdata.insert(3, '测试时间', [self.testime])

        # 计算eline
        # 绘制累计收益率曲线
        account_index_return['资产组合'] = empyrical.cum_returns(account_index_return['资产组合'])
        account_index_return['沪深300'] = empyrical.cum_returns(account_index_return['沪深300'])
        account_index_return['超额收益'] = account_index_return['资产组合'] - account_index_return['沪深300']
        account_index_return[['资产组合','沪深300','超额收益']] = account_index_return[['资产组合','沪深300','超额收益']].applymap(lambda x: round(x,4))
        account_index_return.columns=['日期','资产组合','沪深300','超额收益']
        account_index_return.insert(0,'uuid', len(account_index_return)*[self.strategy.uuid])
        account_index_return.insert(0,'策略名称', len(account_index_return)*[self.strategy.name])
        account_index_return['测试时间'] = self.testime
        return rowdata, account_index_return


    # @timer(1)
    def run(self, test=False, save=False, coc=False):
        """ 启动策略回测 """
        self.testime = datetime.now()
        self.init(test=test)

        # 设置交易时机 coo or coc
        if coc:
            self.cerebro.broker.set_coc(True)
            
        print('[INFO]: 账户初始权益: %.2f' % self.cerebro.broker.getvalue())
        stats = self.cerebro.run()
        pyfolio = stats[0].analyzers.pyfolio 
        returns, positions, transactions, gross_lev = pyfolio.get_pf_items()

        # 分析过程,获取成交记录,回测报告,收益率曲线数据。
        self.transactions = self.get_transactions(transactions)
        report, returns = self.get_report_returns(returns)
        # 保存到数据库
        if save:
            df_2_table(report, 'report')
            df_2_table(returns, 'daily_return', index=False, create=False)
            df_2_table(self.transactions, 'trade_record')
        print('[INFO]: 账户期末权益: %.2f' % self.cerebro.broker.getvalue())


