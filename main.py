from Load_Save_Data import Load_Data, Save_Data
from Data_Process import Data_Process
from Train_Predict import Train_Predict

import numpy as np
from datetime import datetime

if __name__ == "__main__":
    #1. 加载数据
    print( '加载数据=======', datetime.now() )
    data_train, data_test = Load_Data()
    #print( data_train.shape, data_test.shape )
    #print( data_train.head(), data_test.head() )
    #print( data_train.info() ) #可以看出，训练集不存在数据缺失
    #print( data_test.info() )  #可以看出，测试集不存在数据缺失
    #print( data_train.describe() ) #可以看出，训练集不存在明显异常值
    #print( data_test.describe() )  #可以看出，测试集不存在明显异常值
    power_mean = np.mean( data_train['实际功率'] )
    power_std = np.std( data_train['实际功率'] )
    # data = data_train['实际功率'].values
    # print( 'statistic: ', data[data>20].size ) #num: 15814
    # print( 'statistic2: ', data[data<0.0].size ) #num: 47837
    
    #2. 特征工程
    print( '处理数据=======', datetime.now() )
    data_train, data_test = Data_Process( data_train, data_test )
    #print( data_train.head() )
    #print( data_test.head() )

    #测试代码 begin
    #data_train = data_train.iloc[ :10000, : ] #test
    #data_test = data_test.iloc[ :10000, : ] #test
    #测试代码 end

    #3. 训练模型, 预测功率
    print( '启动预测=======', datetime.now() )
    power_predict = Train_Predict( data_train, data_test )
    #将归一化数据恢复为正常值
    power_predict = power_predict * power_std + power_mean
    print( '预测完成=======', datetime.now() )

    #4. 保存结果数据
    print( '保存预测结果=======', datetime.now() )
    Save_Data( power_predict )
    print( '保存完成，退出程序=======', datetime.now() )
