# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
from quant.update import *
from quant.trader import db
from quant.utils import trans_code_form
from .update import *



class SimulateStrategy:
    """ 模拟交易管理类 """
    def __init__(self, strategy_id, context, factor1, factor2, topN=10):
        """ 初始化一个模拟策略 """
        self.strategy_id = strategy_id
        context.factor1 = factor1
        context.factor2 = factor2 
        context.topN = topN
        self.strategy_name = f'{factor1}&{factor2} top:{topN}'

    def log(self, msg):
        """ 日志输出 """
        logger.info(f'{self.strategy_name} {msg}')

    def schedule_select_stock_daily(self, context):
        """开盘前9:00计算入选的股票并写入数据库"""
        self.log(f'早盘选股....')
        choose_today_stock = select_topN_run_daily(context.factor1, context.factor2, topN=context.topN, test=False)
        forward_trade_date = get_forward_trade_date(test=False)
        choose_today_stock['date'] = forward_trade_date
        choose_today_stock.insert(0, 'strategy', self.strategy_name)
        choose_today_stock['update_time'] =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        choose_today_stock['code'] = trans_code_form(choose_today_stock['code'],trans_tpye='JQtoEM')
        db.df_2_table(choose_today_stock, 'simulate_target_position')
        self.log(f'早盘选股完成!')
        
    def schedule_read_stock_daily(self, context):
        """开盘前9:10读取数据库中的入选股票列表"""
        self.log(f'读取数据库选股列表....')
        today_date =  int(datetime.now().strftime('%Y%m%d'))
        topN = db.get_data(f"select `strategy`,`date`,`code`,`weight`,`update_time` from `simulate_target_position` where `date`={today_date} and `strategy`='{self.strategy_name}'")
        datescan = (topN['update_time'].iloc[0].date().strftime('%Y%m%d') == str(today_date))
        if datescan:
            context.topN = topN
            self.log(f'读取数据库选股列表完成!')
            return topN
        else:
            raise ValueError('选股日期与更新日期结果不一致!')
        
            
    def schedule_trade_daily(self, context):
        """开盘9:30后先卖出不在列表的股票,再买入备选股票"""
        # 获取当下日期并校验
        self.log('进行交易....')
        now = context.now.strftime('%Y%m%d')
        topN_now = str(context.topN.date.iloc[0])
        if now!=topN_now:
            raise TimeoutError('交易日期与选股日期不一致!')
        
        # 获取当下持仓信息
        hold_stock_list = [pos['symbol'] for pos in context.account().positions() if pos['side']==1]
        # 当下选股列表
        select_stock_list = context.topN.query(f"`strategy`=='{self.strategy_name}'")['code'].to_list()
        # 需要卖出股票列表
        sell_stock_list = [stock for stock in hold_stock_list if stock not in select_stock_list]
        # 需要买入的股票
        buy_stock_list = [stock for stock in select_stock_list if stock not in hold_stock_list]
        # 遍历卖出股票
        for symbol in sell_stock_list:
            order_return = order_target_percent(symbol, percent=0, position_side=OrderSide_Buy, order_type=OrderType_Market)
            if order_return[0]['status'] == 3:
                self.log(f"卖出股票:{symbol}成功!")
            else:
                self.log(f"卖出股票:{symbol}失败, error code:{order_return[0]['status']}")
                
        # 买入股票
        time.sleep(30)
        for symbol in buy_stock_list:
            weight = context.topN.query(f"`strategy`=='{self.strategy_name}' and `code`=='{symbol}'")['weight'].iloc[0]
            order_return = order_target_percent(symbol, percent=weight*0.9, position_side=OrderSide_Buy, order_type=OrderType_Market)
            if order_return[0]['status'] == 3:
                self.log(f"买入股票:{symbol}成功!")
            else:
                self.log(f"买入股票:{symbol}失败, error code:{order_return[0]['status']}")
        self.log(f'交易完成!')


    def record_order(self, context):
        """记录当日委托订单数据"""
        self.log(f"当日委托订单数据写入数据库....")
        orders = get_orders()
        orders_df = pd.DataFrame(orders)
        orders_df['update_time'] = context.now
        orders_df['properties'] = orders_df['properties'].astype(str)
        if len(orders_df)>0:
            db.df_2_table(orders_df, 'simulate_order')
            self.log(f"当日委托订单数据写入数据库成功!")
        else:
            self.log('当日委托订单为空!')


    def record_position(self, context):
        """记录当日持仓数据"""
        self.log(f"当日持仓数据写入数据库....")
        positions = context.account().positions()
        positions_df = pd.DataFrame(positions)
        positions_df['update_time'] = context.now
        positions_df['properties'] = positions_df['properties'].astype(str)
        db.df_2_table(positions_df, 'simulate_position')
        self.log(f"当日持仓数据写入数据库成功!")


    def record_cash(self, context):
        """记录当日账户资金数据"""
        self.log(f"当日账户资金数据写入数据库....")
        cash = context.account().cash
        cash_df = pd.DataFrame([cash])
        cash_df['update_time'] = context.now
        db.df_2_table(cash_df, 'simulate_cash')
        self.log(f"当日账户资金数据写入数据库成功!")


    def schedule_record_daily(self, context):
        """收盘15:30 记录当日所有委托订单、持仓信息、账户资金情况"""
        # 读取选股列表
        self.schedule_read_stock_daily(context)
        # 记录当日委托订单数据
        self.record_order(context)
        # 记录当日持仓股票数据
        self.record_position(context)
        # 记录当日账户资金数据
        self.record_cash(context)
    

    

if __name__ == '__main__':
    def init(context):
        simulate_strategy = SimulateStrategy('241cf378-47ee-11ee-b3f3-00ff5a78177a', context, factor1='TVSTD20', factor2='natural_log_of_market_cap', topN=10)
        simulate_strategy.log('策略初始化完成!')
        # schedule(schedule_func=simulate_strategy.schedule_select_stock_daily, date_rule='1d', time_rule='13:59:00')
        schedule(schedule_func=simulate_strategy.schedule_read_stock_daily, date_rule='1d', time_rule='14:22:00')
        schedule(schedule_func=simulate_strategy.schedule_trade_daily, date_rule='1d', time_rule='14:23:00')
        schedule(schedule_func=simulate_strategy.schedule_record_daily, date_rule='1d', time_rule='14:25:00')

    """运行策略,strategy_id策略ID, 由系统生成"""
    run(strategy_id='241cf378-47ee-11ee-b3f3-00ff5a78177a', filename='main.py', mode=MODE_LIVE, token='0882e3145afce5ffcf1c4f5f2f72ccb1495d46b4')

