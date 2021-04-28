# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:56:20 2021
1、Input：train.csv, test.csv, data.csv, delta_threshold
2、Input：train_new.csv, test_new.csv, user-poi, time-poi, user-catg, time-catg, catg-catg， poi-category-location_train(地点包含的所有GPS位置)

@author: Administrator
"""
import pandas as pd
import os
import numpy as np
delta_threshold = 6*3600


def graphGenerate(df_train):
    df_user2poi = pd.DataFrame(columns = ['user','poi'])
    df_time2poi = pd.DataFrame(columns = ['timeshot','poi'])
    df_user2catg = pd.DataFrame(columns =['user', 'category'])
    df_time2catg = pd.DataFrame(columns =['timeshot', 'category'])
    
    df_user2poi['user'] = df_train['user']
    df_user2poi['poi'] = df_train['poi']
    
    df_time2poi['timeshot'] = df_train['timeValue']
    df_time2poi['poi'] = df_train['poi']
    
    df_user2catg['user'] = df_train['user']
    df_user2catg['category'] = df_train['category']
    
    df_time2catg['timeshot'] = df_train['timeValue']
    df_time2catg['category'] = df_train['category']
    
    return df_user2poi, df_time2poi, df_user2catg, df_time2catg
    

data='NYC'
basepath = r"data/preprocessedData/"
datapath_train = os.path.join(basepath, "{}_train.csv".format(data))
datapath_test = os.path.join(basepath, "{}_test.csv".format(data))
datapath_data = os.path.join(basepath, "{}_data.csv".format(data))

df_train = pd.read_csv(datapath_train, sep='\t')
df_train = df_train.sort_values(by=['user','timeConvert'])

df_test = pd.read_csv(datapath_test, sep='\t')
df_test = df_test.sort_values(by=['user','timeConvert'])

df_data = pd.read_csv(datapath_data, sep='\t')
df_data = df_data.sort_values(by=['user','timeConvert'])

# built category_category graph construct
df_catg2catg= pd.DataFrame(columns = ['user', 'category_source', 'category_target'])
category_user = []
delta_time = []
category_source = []
category_target = []
df_train = df_train.set_index('user')
users = list(set(df_train.index))
for user in users:
    temp = df_train.loc[user]
    temp_len = len(temp)
    temp.index = range(temp_len)
    source = list(temp['category'][:temp_len-1])
    target = list(temp['category'][1:])
    timeStamp_forward = list(temp['timeConvert'][:temp_len-1])
    timeStamp_backward = list(temp['timeConvert'][1:])
    delta_t = list(np.array(timeStamp_backward)-np.array(timeStamp_forward))
    user_list = [user] * (temp_len-1)
    
    category_user = category_user + user_list
    delta_time = delta_time + delta_t
    category_source = category_source + source
    category_target = category_target + target

df_catg2catg['user'] = category_user
df_catg2catg['delta_time'] = delta_time
df_catg2catg['category_source'] = category_source
df_catg2catg['category_target'] = category_target
index_true = df_catg2catg['delta_time'] <= 21600
df_catg2catg = df_catg2catg.loc[index_true]


catg_set = set(list(df_catg2catg['category_source'])+ list(df_catg2catg['category_target']))
catg_set_train = set(df_train['category'])
catg_subset_train = catg_set_train - catg_set
df_train = df_train.reset_index()
df_train = df_train.set_index('category')

for i in catg_subset_train:
    if i in catg_set_train:
        df_train = df_train.drop(i)
df_train = df_train.reset_index()
df_train = df_train.sort_values(by=['user','timeConvert'])

catg_set_data = set(df_data['category'])
catg_subset_test = catg_set_data - catg_set

df_test = df_test.set_index('category')
catg_set_test = set(df_test.index)
for i in catg_subset_test:
    if i in catg_set_test:
        df_test = df_test.drop(i)
df_test = df_test.reset_index()

poi_set_train = set(df_train['poi'])
poi_set = set(df_data['poi'])
poi_subset_test = list(poi_set - poi_set_train) 

df_test = df_test.set_index(['poi'])
poi_set_test = set(df_test.index)
for i in poi_subset_test:
    if i in poi_set_test:
        df_test = df_test.drop(i)
df_test = df_test.reset_index()
df_test = df_test.sort_values(by=['user','timeConvert'])

print('graph generate and save')
df_user2poi, df_time2poi, df_user2catg, df_time2catg = graphGenerate(df_train)
df_user2poi.to_csv(r'./data/preprocessedData/{}/graph_user2poi.csv'.format(data),sep = '\t', index = False, header = True)
df_time2poi.to_csv(r'./data/preprocessedData/{}/graph_time2poi.csv'.format(data),sep = '\t', index = False, header = True)
df_user2catg.to_csv(r'./data/preprocessedData/{}/graph_user2catg.csv'.format(data),sep = '\t', index = False, header = True)
df_time2catg.to_csv(r'./data/preprocessedData/{}/graph_time2catg.csv'.format(data),sep = '\t', index = False, header = True)
df_catg2catg.to_csv(r'./data/preprocessedData/{}/graph_catg2catg.csv'.format(data),sep = '\t', index = False, header = True)
     
#保存data_new.csv, train_new.csv, test_new.csv
df_dataNew = pd.DataFrame()
columns_list = list(set(df_train.columns))
for i in columns_list:
    df_dataNew[i] = list(df_train[i]) + list(df_test[i])

print('save df_dataNew, df_trainNew and df_testNew!')
df_dataNew = df_dataNew.set_index('user')
df_dataNew.to_csv(r'./data/preprocessedData/{}_dataNew.csv'.format(data),sep = '\t', index = True, header = True)
df_train = df_train.set_index('user')
df_train.to_csv(r'./data/preprocessedData/{}_trainNew.csv'.format(data),sep = '\t', index = True, header = True)
df_test = df_test.set_index('user')
df_test.to_csv(r'./data/preprocessedData/{}_testNew.csv'.format(data),sep = '\t', index = True, header = True)
df_dataNew = df_dataNew.reset_index()

print('save poi_category_location')
dataCopy = df_dataNew.copy()
temp = pd.DataFrame()
temp['poi'] = dataCopy['poi']
temp['category'] = dataCopy['category']
temp['lat'] = dataCopy['lat']
temp['lon'] = dataCopy['lon']
temp['categoryName'] = dataCopy['categoryName'] 
temp = temp.sort_values(by=['poi','lat'])
temp1 = temp.copy()
temp2 = temp.copy()
temp1 = temp1.drop_duplicates(subset=['poi','category'],keep='first')
temp2 = temp2.drop_duplicates(subset=['poi','category','lat'],keep='first')
temp1 = temp1.set_index('poi')
temp2 = temp2.set_index('poi')
temp1.to_csv(r'./data/preprocessedData/{}_poi_category_location_train.csv'.format(data),sep = '\t', index = True, header = True)
temp2.to_csv(r'./data/preprocessedData/{}_poi_category_location_train1.csv'.format(data),sep = '\t', index = True, header = True)

print('hello world!')
