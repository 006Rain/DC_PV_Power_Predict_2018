import numpy as np
import pandas as pd

def Data_Process( data_train, data_test ):
    '''
    1. 训练集数据处理
    '''
    #提取月、日、时、分时间信息（观察数据知：年份和秒数对预测无影响）
    data_train['时间'] = pd.to_datetime( data_train['时间'], format='%Y-%m-%d %H:%M:%S' ) #转换为时间格式

    data_train['Month'] = data_train['时间'].apply( lambda x: x.month )
    data_train['Day'] = data_train['时间'].apply( lambda x: x.day )
    data_train['Hour'] = data_train['时间'].apply( lambda x: x.hour )
    data_train['Minute'] = data_train['时间'].apply( lambda x: x.minute )

    #将时间列设为索引(即：删除'时间'列)
    data_train = data_train.set_index( '时间', drop=True )

    #数据归一化：均值归一化（经验证，min-max归一化不合适）
    data_train = data_train.apply( lambda x: ( x - np.mean(x) ) / np.std(x) )

    '''
    2. 测试集数据处理
    '''
    #去除无用的id列，并
    data_test = data_test.drop( columns=['id'] )

    #提取月、日、时、分时间信息
    data_test['时间'] =  pd.to_datetime( data_test['时间'], format='%Y-%m-%d %H:%M:%S' )

    data_test['Month'] = data_test['时间'].apply( lambda x: x.month )
    data_test['Day'] = data_test['时间'].apply( lambda x: x.day )
    data_test['Hour'] = data_test['时间'].apply( lambda x: x.hour )
    data_test['Minute'] = data_test['时间'].apply( lambda x: x.minute )

    #将时间列设为索引
    data_test = data_test.set_index( '时间', drop=True )

    #数据归一化
    data_test = data_test.apply( lambda x: ( x - np.mean(x) ) / np.std(x) )
    
    return data_train, data_test