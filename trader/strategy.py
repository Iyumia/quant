import uuid
import warnings
import pandas as pd
import backtrader as bt
from quant.log import logger
from quant.utils import *
from backtrader import  Order
warnings.filterwarnings('ignore')



class DataFeed(bt.feeds.PandasData):
    # 要添加的线, 注意逗号不能胜率
    lines = ('paused', 'is_zt', 'is_dt', 'is_yzzt', 'is_yzdt','weight','is_3index_fall','is_sell',)  
    # 设置 line 在数据源上的列位置
    params = (
        ('paused', -1),
        ('is_zt', -1),
        ('is_dt', -1),
        ('is_yzzt', -1),
        ('is_yzdt', -1),
        ('weight', -1),
        ('is_3index_fall',-1),
        ('is_sell', -1),
        )



class Template(bt.Strategy):
    """ 策略模板 """
    name = ''
    author = ''
    url = ''
    uuid = ''
    doc = ''
    
    
    def __init__(self):
        """策略初始化"""
        # self.stocknum = 10          # 最大持股数
        # self.value = 200000         # 买入股票的价值
        self.record = False
        self.dates = []             # 交易日期列表
        self.posratio = []          # 每日资金使用率列表
        self.cashs = []             # 每日可用资金
        self.assets = []            # 每日资产价值
        self.orders = {}            # 以字典形式记录买入订单的名称、价格
        self.procount = 0           # 盈利次数
        self.loscount = 0           # 亏损次数
        self.active = True          # 回测状态
        self.tradedates = []        # 交易信号日期
        self.tradecodes=[]          # 成交股票代码
        self.comms = []             # 手续费

        
    def log(self, txt):
        """ 输出记录信息 """
        if self.record:
            dt = self.data.datetime.date(0)
            msg = f'[{dt}] {txt}'
            logger.info(msg)

    def notify_order(self, order):
        """ 根据订单状态做出判断 """
        if order.status == Order.Completed:
            if order.isbuy():
                self.log('买入: %s, 数量: %.2f, 价格: %.2f, 成本: %.2f, 手续费 %.2f' % (order.data._name,order.executed.size,order.executed.price,order.executed.value,order.executed.comm))   
                self.orders[order.data._name] = order.executed.price
    
            else: # Sell
                self.log('卖出: %s, 数量: %.2f, 价格: %.2f, 现金: %.2f, 手续费 %.2f' % (order.data._name,order.executed.size,order.executed.price,order.executed.value,order.executed.comm))   
                if order.executed.price >= self.orders[order.data._name]:
                    self.procount += 1
                else:
                    self.loscount += 1
            
            dt = self.datetime.date(0)
            self.tradedates.append(dt)
            self.tradecodes.append(order.data._name)
            self.comms.append(order.executed.comm)



        elif order.status == Order.Canceled:
            if order.isbuy():
                self.log('取消买入: %s, 数量: %.2f, 价格: %.2f !' % (order.data._name,order.executed.size,order.executed.price))   
            else:
                self.log('取消卖出: %s, 数量: %.2f, 价格: %.2f !' % (order.data._name,order.executed.size,order.executed.price))   

        elif order.status == Order.Margin:
            if order.isbuy():
                self.log('资金不足, 买入: %s 失败, 已取消订单!' % order.data._name)   
        else:
            return
                 
    def start(self):
        print(f'[INFO]: 策略回测启动....')

    def stop(self):
        print(f'[INFO]: 策略回测完成!')
        self.__class__.structure = pd.DataFrame({'date':self.dates, 'cash':self.cashs, 'asset':self.assets, 'posratio':self.posratio})
        self.__class__.structure['date'] = self.__class__.structure['date'].astype(str)
        self.__class__.finasset = self.assets[-1]
        self.__class__.posratio = np.mean(self.posratio)
        self.__class__.procount = self.procount
        self.__class__.loscount = self.loscount
        try:
            self.__class__.winrate = self.procount / (self.loscount + self.procount)
        except ZeroDivisionError:
            self.__class__.winrate = 0
        # 手续费
        self.__class__.commission = pd.DataFrame({'date':self.tradedates,'security':self.tradecodes, 'commission':self.comms})
