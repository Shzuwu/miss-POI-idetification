# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:03:35 2021

@author: Administrator
将数据集划分为训练集和测试集
1、输入：dataset_TSMC2014_NYC.txt或者dataset_TSMC2014_TKY.txt, 最少人数minimun_threshold
2、输出：train.csv，test.csv, data_set.csv
"""

print('hello world')

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
import time
from collections import Counter

def timeValue(day_hourminute):
    day = eval(day_hourminute.split(':')[0])
    hourminute = day_hourminute.split(':')[1]
    date_divide = 6
    if '0200' <= hourminute < '0600':
        index = 0
    elif '0600' <= hourminute < '1000':
        index = 1
    elif '1000' <= hourminute < '1400':
        index = 2
    elif '1400' <= hourminute < '1800':
        index = 3
    elif '1800' <= hourminute < '2200':
        index = 4
    elif hourminute >= '2200' or hourminute < '0200':
        index = 5
    else:
        print('time error:', hourminute)
    return day*date_divide+index

'''
step1 读数据
'''
data='NYC'
Minimun_Threshold = 10
print ('preprocess {} data'.format(data))
basepath = r"data/original data/dataset_TSMC2014_{}.txt".format(data)
df_data = pd.read_csv(basepath, header=None, sep='\t')

'''
step2 时间转换
'''
df_data[8] = [time.mktime(time.strptime(
            df_data[7][index], '%a %b %d %X +0000 %Y')) + int(df_data[6][index]) * 60 for index in df_data[7].index]

'''
step3 按照[用户，时间线]进行排序，data reindex and sort the dataframe by person's timeline
'''
columns = ['user','poi','category','categoryName','lat','lon','timeOffset','timestamp','timeConvert']
df_data.columns = columns
df_data = df_data.sort_values(by=['user','timeConvert'])
df_data = df_data.set_index('user')

temp = list(df_data['timestamp'])
df_data['timeValue'] = [timeValue(time.strftime("%w:%H%M",time.strptime(i,'%a %b %d %X +0000 %Y'))) for i in temp]

'''
step5 用户少于10个签到的需要进行删除
 '''
user_list = list(df_data.index)
counter = Counter(user_list)
user_less = [i for i in counter.keys() if counter[i] < Minimun_Threshold]
for i in user_less:
    df_data.drop(i)
    

'''
step6 提取用户、时间和类别user, poi and poi_categories
'''
users = list(set(df_data.index))
df_train = pd.DataFrame()
df_test = pd.DataFrame()


'''
step7 按照80%训练集，20%测试集进行数据划分。divide trainset and testset with 80% and 20%
'''
for user in users:
    temp = df_data.loc[user]
    tempNum = len(temp)
    trainNum = int(tempNum*0.8)
    trainElement = temp.iloc[:trainNum]
    testElement = temp.iloc[trainNum:]
    df_train = pd.concat([df_train, trainElement])
    df_test = pd.concat([df_test, testElement])
    
'''
step8 保存各项数据
'''
df_train.to_csv(r'./data/preprocessedData/{}_train.csv'.format(data),sep = '\t', index = True, header = True)
df_test.to_csv(r'./data/preprocessedData/{}_test.csv'.format(data),sep = '\t', index = True, header = True)
df_data.to_csv(r'./data/preprocessedData/{}_data.csv'.format(data),sep = '\t', index = True, header = True)
