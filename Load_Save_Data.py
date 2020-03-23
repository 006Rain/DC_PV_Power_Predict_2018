import pandas as pd

data_dir = './DC_Data/'

def Load_Data():
    '''
    读取训练数据和测试数据，并返回
    '''
    data_train1 = pd.read_csv( data_dir + 'train_1.csv' )
    data_train2 = pd.read_csv( data_dir + 'train_2.csv' )
    data_train3 = pd.read_csv( data_dir + 'train_3.csv' )
    data_train4 = pd.read_csv( data_dir + 'train_4.csv' )
    data_train = pd.concat( [data_train1, data_train2, data_train3, data_train4] ) 
    
    data_test1 = pd.read_csv( data_dir + 'test_1.csv' )
    data_test2 = pd.read_csv( data_dir + 'test_2.csv' )
    data_test3 = pd.read_csv( data_dir + 'test_3.csv' )
    data_test4 = pd.read_csv( data_dir + 'test_4.csv' )
    data_test = pd.concat( [data_test1, data_test2, data_test3, data_test4] )
    
    return data_train, data_test

def Save_Data( data_save ):
    '''
    按要求保存数据
    '''
    data_save = pd.DataFrame( data_save )
    data_save.columns = ['prediction']
    data_save.to_csv( data_dir + 'power_predict.csv', float_format='%.7f', index_label=['id'] )
    