import dolphindb as ddb
import dolphindb.settings as keys


s = ddb.session()
s.connect("localhost", 8848, "admin", "123456", keepAliveTime=999999, reconnect=True)


def getSession(enableAsync=False, isCompress=False):
    s = ddb.session(enableASYNC=enableAsync, compress=isCompress)
    s.connect("localhost", 8848, "admin", "123456")
    return s


def upload(var, aliasName):
    s.upload({aliasName: var})
    

def runScript(*script, enableAsync=False, isCompress=False, clearMemory=False):
    global s
    if enableAsync or isCompress:
        s = getSession(enableAsync, isCompress)
        s.run(*script, clearMemory=clearMemory)
    else:
        result = s.run(*script, clearMemory=clearMemory)
        return result
        

def clearAllCache():
    script="clearAllCache()"
    runScript(script)
    print('All caches have been cleaned up!')


def loadText(filePath, delimiter=','):
    data = s.loadText(filePath, delimiter=delimiter)
    df = data.toDF()
    return df


def ploadText(filePath, delimiter=','):
    data = s.ploadText(filePath, delimiter=delimiter)
    df = data.toDF()
    return df


def loadTable(dbPath, tableName):
    table = s.loadTable(tableName=tableName, dbPath=dbPath)
    return table


def df2Table(dbPath, tableName, df, enableAsync=False):
    """将df采用异步或非异步的方式写入数据库"""
    script = f"append!{{loadTable('{dbPath}', `{tableName})}}"
    runScript(script, df, enableAsync=enableAsync)


def df2TableAppender(df, dbPath, tableName, partitionColName, connectNum=20):
    """ 并发将data写入数据库 """
    pool = ddb.DBConnectionPool("localhost", 8848, connectNum, "admin", "123456")
    appender = ddb.PartitionedTableAppender(dbPath, tableName, partitionColName, pool)
    result = appender.append(df)
    return result


def data2MemoryTable(tableName, *data, enableAsync=False, clearMemory=False):
    """ 将 data: DataFrame or iterable 插入内存表中 """
    script = "tableInsert{%s}" % tableName
    if len(data)>0:
        runScript(script, *data, enableAsync=enableAsync, clearMemory=clearMemory)

def dropStreamTable(tableName):
    """ 删除流数据表 """
    try:
        script = f"dropStreamTable('{tableName}')"
        runScript(script)
        print(f'droped {tableName}!')
    except Exception as e:
        print(f"droped {tableName} error:{e}")

def getSchema(dbPath, table):
    script=f"""
    pt=loadTable('{dbPath}',`{table})
    schema(pt);
    """
    result = runScript(script, existReturn=True)
    return result


def selectBySQL(dbPath, tableName, sql, clearMemory=False, index=None):
    script = f"""
    db=database("{dbPath}")
    pt=loadTable(db,`{tableName})
    {sql};
    """
    result = runScript(script, clearMemory=clearMemory)
    if index is None:
        return result
    else:
        return result.set_index(index)



def dropDatabase(dbPath, force=False):
    warningDropDB = [
    'dfs://Factor','dfs://WorldQuant101','dfs://Index_Kline','dfs://Index_stock',
    'dfs://Industry','dfs://Kline','dfs://Kline_Cap_Ind']

    if dbPath in warningDropDB and force==False:
        raise ValueError(f'This database of {dbPath} needs to be stored for a long time. Please delete it carefully!')
    if s.existsDatabase(dbPath):
        s.dropDatabase(dbPath)
        print(f"{dbPath}: Droped"+" Success!")


def createDatabase(dbName, partitionType, partitions, dbPath, existDrop=True):
    if existDrop:
        dropDatabase(dbPath)

    if partitionType == 'value':
        db = s.database(dbName=dbName, dbPath=dbPath, partitionType=keys.VALUE, partitions=partitions)
    elif partitionType == 'range':
        db = s.database(dbName=dbName, dbPath=dbPath, partitionType=keys.RANGE, partitions=partitions)
    elif partitionType == 'list':
        db = s.database( dbName=dbName, dbPath=dbPath,partitionType=keys.LIST, partitions=partitions)
    elif partitionType == 'hash':
        db = s.database(dbName=dbName, dbPath=dbPath, partitionType=keys.HASH, partitions=partitions)
    elif partitionType == 'compo':
        db = s.database(dbName=dbName, dbPath=dbPath, partitionType=keys.COMPO, partitions=partitions) 
    else:
        db = s.database(dbName=dbName, dbPath=dbPath)
    print(f"{dbPath} creation succeeded!")
    return db


def createPartitionedTable(db, tableName, partitionColumns, data):
    t = s.table(data=data)
    db.createPartitionedTable(table=t, tableName=tableName, partitionColumns=partitionColumns).append(t)
    print(f"{tableName}: Create"+" Success!")






