import pandas as pd
import pymongo
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


# config
URL = "mongodb://FMTER4INTER:FMTER4INTER@192.168.26.248:37201"   # pymongo 连接 mongoDB 的url
mongoClient = pymongo.mongo_client.MongoClient(URL)  
klineDB = mongoClient["hb_jq_stock_daily_cn_base_db"]
infoDB = mongoClient["hb_jq_stock_daily_cn_info_db"]
symbols = klineDB.list_collection_names(session=None)    # 所有股票



class GetDataFromMongodbBase:
    mongodb_setting = {
        "192.168.26.31" : {"host": "192.168.26.31", "port": '27017', "user": "userRead", "password": "userRead123456"},
        "192.168.26.135" : {"host": "192.168.26.135", "port": '37203', "user": "ifind_read", "password": "FMTER_B036B037"}
        }
        

    # -------------------------------------------------- 连接数据库 --------------------------------------------------
    def __init__(self, IP):
        host = self.mongodb_setting[IP]['host']
        port = self.mongodb_setting[IP]['port']
        user = self.mongodb_setting[IP]['user']
        password = self.mongodb_setting[IP]['password']
        try:
            self.client = pymongo.MongoClient(f'mongodb://{user}:{password}@{host}:{port}/?authSource=admin')
        except Exception as e:
            print("初始化数据库连接失败：%s" % e)

    # -------------------------------------------------- 获取单个集合 --------------------------------------------------
    def single_post(self, dbName, postName, getCol=[], start=None, end=None, keyCol='date'):
        """
        dbName：数据库名，str
        postName：集合名，str
        getCol：所需要的列名，list
        """
        #         print('keyCol is ',keyCol)
        dblist = self.client.list_database_names()
        if dbName in dblist:
            # 获取数据库
            db = self.client[dbName]
            # 获取集合
            post = db[postName]
            qurey = self.select_record(getCol=getCol, start=start, end=end, keyCol=keyCol)
            #             print("数据库筛选条件：",qurey)
            data = pd.DataFrame(list(post.find(*qurey)))
            return data
        else:
            print("数据库%s不存在" % (dbName))

    # -------------------------------------------------- 获取多个集合 --------------------------------------------------
    def multi_post(self, dbName, postName=None, getCol=[], start=None, end=None, keyCol='date'):
        """
        postName：集合名,list | str | None，默认为None，取数据库所有数据
        """
        if isinstance(postName, list):
            postNameList = postName
        elif postName == None:
            # 获取数据库所有集合
            postNameList = self.client[dbName].list_collection_names()
        elif isinstance(postName, str):
            postNameList = [postName]
        else:
            print('参数postName不复合要求')
        #         print('所需集合：',postNameList)
        df_list = []
        if len(postNameList) == 1:
            data = self.single_post(dbName=dbName, postName=postNameList[0], getCol=getCol, start=start, end=end,
                                    keyCol=keyCol)
        else:
            for postName in tqdm(postNameList):
                if postName in self.client[dbName].list_collection_names():
                    df = self.single_post(dbName=dbName, postName=postName, getCol=getCol, start=start, end=end,
                                          keyCol=keyCol)
                    df_list.append(df)
                else:
                    print("数据库%s中没有集合%s" % (dbName, postName))
            data = pd.concat(df_list)
        return data

    def select_record(self, getCol=[], start=None, end=None, keyCol='date'):
        # ------------------- 根据某一列的值进行筛选
        if (start == None) & (end != None):
            qurey_date = {keyCol: {'$lte': end}}
        elif (start != None) & (end == None):
            qurey_date = {keyCol: {'$gte': start}}
        elif (start != None) & (end != None):
            qurey_date = {keyCol: {'$gte': start, '$lte': end}}
        else:
            qurey_date = {}
        # ------------------- 取指定列
        get_dict = {'_id': False}
        for i in getCol:
            get_dict[i] = True
        qurey_col = get_dict
        return qurey_date, qurey_col

def GetDataFromMongodb(IP='192.168.26.31',  # 连接数据库的IP
                       dbName='factor_stock_daily_alpha101',  # 数据库名，str
                       postName=None,  # 集合名(数据表名)，None | str | list，默认为None取所有集合
                       getCol=[],  # 需要的列名,默认[]全选
                       keyCol='date',  # 筛选数据所用的列名，同start、end共同使用
                       start=None,  # 筛选区间的初始值（含）
                       end=None  # 筛选区间的末值（含）
                       ):
    class_ = GetDataFromMongodbBase(IP)
    data = class_.multi_post(dbName=dbName, postName=postName, getCol=getCol, start=start, end=end, keyCol=keyCol)
    return data

def get_stock_data(symbol:str, start:str=None, end:str=None, field:list=None, index=None)->pd.DataFrame:
    """获取单只股票日频数据"""
    mongoCollection = klineDB[symbol]
    filters = {'date':{'$gte': start, '$lte':end}} if (start and end) else {}  # 筛选条件
    cursor = mongoCollection.find(filters).sort([("date",pymongo.ASCENDING)])     
    stockData = pd.DataFrame(list(cursor))
    if field:
        choice = stockData.get(field)
    else:
        choice = stockData

    if index:
        result = choice.set_index(index)
        return result
    else:
        return choice

def get_stocks_data(symbols, start:str=None, end:str=None, field:list=None, index=None, multiProcess:bool=False):
    """获取多只股票数据"""
    stockDataList = []
    if multiProcess:
        pool = Pool(12)
        stockDataList = pool.map(partial(get_stock_data, start=start, end=end, field=field, index=index), symbols)
        allStockData = pd.concat(stockDataList, axis=0).drop_duplicates(subset=['date','symbol'])
        allStockData.sort_values(['date','symbol'], inplace=True)
        pool.close()
        return allStockData
    else:
        for symbol in tqdm(symbols, ncols=60, colour='green'):
            stockData = get_stock_data(symbol, start, end, field, index)
            stockDataList.append(stockData)
        allStockData = pd.concat(stockDataList, axis=0).drop_duplicates(subset=['date','symbol'])
        allStockData.sort_values(['date','symbol'], inplace=True)
        return allStockData

def getIndexComponent(symbol, start=None, end=None):
    """获取指数成分股及权重"""
    mongoCollection = infoDB[symbol]
    filters = {'date':{'$gte': start, '$lte':end}} if (start and end) else {}
    columns = {'_id':0, 'date':1, 'stockName':1,'weight':1}
    cursor = mongoCollection.find(filters, columns)
    indexComponent = pd.DataFrame(list(cursor))
    indexComponent.sort_values(['stockName','date'],inplace=True)
    indexComponent.index = range(len(indexComponent))
    return indexComponent

def getStockPool(index=None, date=None):
    """获取股票池列表"""
    if index is None:
        return symbols
    else:
        mongoCollection = infoDB[index]
        cursor = mongoCollection.find({'date': date})
        stockPool = pd.DataFrame(list(cursor))['stockName'].to_list()
        return stockPool

def getIndexComponentData(symbol:str, start:str=None, end:str=None, field:list=None, index=['date','symbol'], multiProcess:bool=False):
    """获取指数成分数据"""
    indexComponent = getIndexComponent(symbol, start, end)
    indexs = indexComponent.index

    symbols = indexs.get_level_values(1).unique()
    data = get_stocks_data(symbols[:200], start=start, end=end, field=field, index=index, multiProcess=multiProcess)
    indexs_ = indexs.intersection(data.index)
    result = data.loc[indexs_].sort_index()
    return result

def getIndexData(symbol, start, end, field=['date','symbol','adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume'], index=None):
    """获取指数行情数据"""
    mongoCollection = infoDB[symbol]
    filters = {'date':{'$gte': start, '$lte':end}} if (start and end) else {}
    cursor = mongoCollection.find(filters)
    indexData = pd.DataFrame(list(cursor))
    if index is None:
        return indexData.get(field).sort_values('date')
    else:
        indexData.set_index(index, inplace=True)
        return indexData.get(field).sort_values('date')

def getTradeDate():
    trade_date_df = GetDataFromMongodb(
        IP='192.168.26.135', 
        dbName='ifind_db', 
        postName='stock_daily_tradeDate',
        keyCol='tradeDate')

    return trade_date_df


