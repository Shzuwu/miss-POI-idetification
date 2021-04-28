# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:52:31 2021

@author: Administrator
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
import time
import pandas as pd


# for testset 
def datasetMaking(data_df):
    df_result = pd.DataFrame()
    user_list = list(set(data_df.index))
    users, t1, t2, tc, t3, l1, l2, l3, p1, p2, p3, c1, c2, c3 = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
    
    print('length of user_list:',len(user_list))
    for user in user_list:
        temp = data_df.loc[user]
        temp_len = len(temp)
        temp.index = range(len(temp))
        temp_left = temp.iloc[:temp_len-2]
        temp_middle = temp.iloc[1:temp_len-1]
        temp_right = temp.iloc[2:] 
        print('user:', user)
        
        t1 = t1 + list(temp_left['timeConvert'])
        t2 = t2 + list(temp_middle['timeConvert'])
        tc = tc + list(temp_middle['timeValue'])
        t3 = t3 + list(temp_right['timeConvert'])
        
        p1 = p1 + list(temp_left['poi'])
        p2 = p2 + list(temp_middle['poi'])
        p3 = p3 + list(temp_right['poi'])
        
        c1 = c1 + list(temp_left['category'])
        c2 = c2 + list(temp_middle['category'])
        c3 = c3 + list(temp_right['category'])

        l1_lat = list(temp_left['lat'])
        l2_lat = list(temp_middle['lat'])
        l3_lat = list(temp_right['lat'])
        l1_lon = list(temp_left['lon'])
        l2_lon = list(temp_middle['lon'])
        l3_lon = list(temp_left['lon'])
        l1 = l1 + list(zip(l1_lat, l1_lon))
        l2 = l2 + list(zip(l2_lat, l2_lon))
        l3 = l3 + list(zip(l3_lat, l3_lon))
        
        users = users + [user]*(temp_len-2)

    df_result['users'] = users
    df_result['t1'] = t1
    df_result['t2'] = t2
    df_result['tc'] = tc
    df_result['t3'] = t3
    df_result['l1'] = l1
    df_result['l2'] = l2
    df_result['l3'] = l3
    df_result['p1'] = p1
    df_result['p3'] = p3
    df_result['c1'] = c1
    df_result['c3'] = c3
    df_result['p2'] = p2
    df_result['c2'] = c2
        
    return df_result

data='NYC'
basepath = r"data/preprocessedData/"
datapath_train = os.path.join(basepath, "{}_trainNew.csv".format(data))
datapath_test = os.path.join(basepath, "{}_testNew.csv".format(data))
df_train = pd.read_csv(datapath_train, sep='\t')
df_test = pd.read_csv(datapath_test, sep='\t')

if set(df_test.columns) == set(df_train.columns):
    print('hello world')

df_train = df_train.sort_values(by=['user','timeConvert'])
df_train = df_train.set_index('user')
df_train = datasetMaking(df_train)
df_train.to_csv(r'./data/preprocessedData/{}/df_trainFinal.csv'.format(data),sep = '\t', index = False, header = True)
print('df_testFinal set has been constructed')

df_test = df_test.sort_values(by=['user','timeConvert'])
df_test = df_test.set_index('user')
df_test = datasetMaking(df_test)
df_test.to_csv(r'./data/preprocessedData/{}/df_testFinal.csv'.format(data),sep = '\t', index = False, header = True)
print('df_testFinal set has been constructed')