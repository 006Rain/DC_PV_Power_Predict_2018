import pandas as pd
from sklearn import svm
from datetime import datetime

def Train_Predict( data_train, data_test ):
    '''
    1. 先预测实发辐照度
    '''
    #切分特征和标签
    print( '1.1 切分特征和标签=======', datetime.now() )
    x_train = data_train.drop( columns = ['实发辐照度', '实际功率'] )
    y_train = data_train['实发辐照度']
    
    #预测实发辐照度
    print( '1.2 预测实发辐照度=======', datetime.now() )
    svm_SVR = svm.SVR( gamma='auto' )
    svm_SVR.fit( x_train, y_train )
    y_predict_irradiance = svm_SVR.predict( data_test )
    
    #将预测的实发辐照度放入测试集中
    data_test['实发辐照度'] = y_predict_irradiance

    '''
    2. 再预测功率
    '''
    #切分特征和标签
    print( '2.1 切分特征和标签=======', datetime.now() )
    x_train = data_train.drop( columns = ['实际功率'] )
    y_train = data_train['实际功率']
 
    #预测功率
    print( '2.2 预测功率=======', datetime.now() )
    svm_SVR.fit( x_train, y_train )
    y_predict_power = svm_SVR.predict( data_test )

    return y_predict_power
