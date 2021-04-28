# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:19:03 2020

@author: Administrator
"""

import math
import random
import copy
import os

import time
import tensorflow as tf
import numpy as np
from numpy import *
import networkx as nx
import pandas as pd
from preprocess4 import LOC2INDEX

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.models import Model

def MinMaxNorm(score_array):
    min_value = score_array.min()
    max_value = score_array.max()
    result = (score_array - min_value)/(max_value - min_value)
    return result

def StandNorm(score_array):
    x_averange = score_array.mean()
    x_std = np.std(x_averange)
    result = (score_array - x_averange)/x_std
    return result
    
def MaxAbsNorm(score_array):
    maxAbsValue = np.abs(score_array).max()
    result = score_array/maxAbsValue
    return result

def DisCalculation(current_coordinate, Index_coordinate_lat, Index_coordinate_lon):
    lat = current_coordinate[0]
    lon = current_coordinate[1]
    dist = abs(Index_coordinate_lat-lat) + abs(Index_coordinate_lon-lon)
    dist_distribution = 2/(1+np.exp(parameter_a*dist))
    temp = (dist_distribution < 0.1)*0.1
    dist_distribution = np.array((dist_distribution, temp)).max(axis=0)
    return dist_distribution
    
    
#Margin = 0.1
DELTA_THRESHOLD = 6*3600
GRID_LENGTH = 10
sigma = 0.01   
beta = 0.7
sess = tf.Session()

# user | t1 | t2 | tc | t3 | l1 | l2 | l3 | p1 | p3 | c1 | c3————>p2 | c2
data = "NYC"
path_test =   r'./data/preprocessedData/{}/df_testFinal.csv'.format(data)
df_test = pd.read_csv(path_test,sep='\t')


# POI | category | lat | lon | corrdinate | index
path_dataSet= r"./data/preprocessedData/{}_poi_category_location_train1.csv".format(data) 
loc2index = LOC2INDEX(path_dataSet, GRID_LENGTH)
df_dataSet = loc2index.generateIndexCoordinate()
grid_width = (loc2index.x_max - loc2index.x_min)/GRID_LENGTH
#(grid_width*50, 0.1)  y=2/(1+np.exp(a*x))
x_input = grid_width
y_result = 0.95
parameter_a = np.log(2/y_result-1)/(x_input)

user_txt = open('{}_embedding_user.txt'.format(data),'r')
poi_txt = open('{}_embedding_poi.txt'.format(data),'r')
time_txt = open('{}_embedding_time.txt'.format(data),'r')
catg_txt = open('{}_embedding_catg.txt'.format(data),'r')
catgTarget_txt = open('{}_embedding_catgTarget.txt'.format(data),'r')

emb_poi = eval(poi_txt.read())
emb_user = eval(user_txt.read())
emb_time = eval(time_txt.read())
emb_catg = eval(catg_txt.read())
emb_catgTarget = eval(catgTarget_txt.read())

'''
prepare for test
'''
test_num = len(df_test)
candidate_num = len(df_dataSet)
users = list(df_test['users'])
c1 = list(df_test['c1'])
c3 = list(df_test['c3'])
t1 = list(df_test['t1'])
t2 = list(df_test['t2'])
tc = list(df_test['tc'])
t3 = list(df_test['t3'])
l2 = [eval(i) for i in df_test['l2']]
ground_truth = list(df_test['p2'])

candidate_poi = list(df_dataSet['poi'])
candidate_poiCatg = list(df_dataSet['category'])
Index_coordinate = list(df_dataSet['coordinate'])
Index_Number = list(df_dataSet['index'])

candidate_poi_array = np.array([emb_poi[i] for i in candidate_poi])
candidate_poiCatg_array = np.array([emb_catg[i] for i in candidate_poiCatg])
candidate_poiCatgTarget_array = np.array([emb_catgTarget[i] for i in candidate_poiCatg])
Index_coordinate_lat = np.array([i[0] for i in Index_coordinate])
Index_coordinate_lon = np.array([i[1] for i in Index_coordinate])
poi_catg_array = candidate_poi_array + candidate_poiCatg_array

print('Begin to take the test!')

delta_T1_T2 = np.array(t2) - np.array(t1)
delta_T2_T3 = np.array(t3) - np.array(t2)

top_1_result = []
top_5_result = []
top_10_result = []
MAP_result = []
count =0

test_index = np.random.permutation(test_num)
start =time.clock()
#for i in range(test_num/20):
for i in test_index:
    count = count + 1
    
    truth = ground_truth[i]
    embedding_user = emb_user[users[i]]
    embedding_time = emb_time[tc[i]]
    embedding_c1 = emb_catg[c1[i]]
    embedding_c3 = emb_catgTarget[c3[i]]
    
    delta_t1_t2 = delta_T1_T2[i]
    delta_t2_t3 = delta_T2_T3[i]
    
    # Preference Influence calculation
    score_preference = np.dot(poi_catg_array, (embedding_user + embedding_time))
    score_preference = MinMaxNorm(score_preference)
    
    flag = 0
    
    # Sequence Influence calculation
    if delta_t1_t2 > DELTA_THRESHOLD and delta_t2_t3 > DELTA_THRESHOLD:
        flag = 0
        sequence = np.zeros(len(score_preference))
    elif delta_t1_t2 <= DELTA_THRESHOLD and delta_t2_t3 > DELTA_THRESHOLD:
        flag = 1
        sequence = np.dot(candidate_poiCatgTarget_array, embedding_c1)
    elif delta_t1_t2 > DELTA_THRESHOLD and delta_t2_t3 <= DELTA_THRESHOLD:
        flag = 2
        sequence = np.dot(candidate_poiCatg_array, embedding_c3)
    else:
        flag = 3
        sequence_t1t2 = np.dot(candidate_poiCatgTarget_array, embedding_c1)
        sequence_t2t3 = np.dot(candidate_poiCatg_array, embedding_c3)
        sequence = np.array((sequence_t1t2, sequence_t2t3)).max(axis=0)
    
    if flag == 0:
        score_sequence = np.zeros(len(score_preference))
    else:
        score_sequence = np.dot(candidate_poiCatg_array, ((embedding_user + embedding_time)))*sequence
        score_sequence = MinMaxNorm(score_sequence)
    
    # Geographic Influence calculation
    current_coordinate, current_index = loc2index.findIndex(l2[i][0], l2[i][1])
    dist_distribution = DisCalculation(current_coordinate, Index_coordinate_lat, Index_coordinate_lon)
    dist_distribution = dist_distribution + 1
    
    total_score = (beta*score_preference + (1-beta)*score_sequence)*dist_distribution
    
    sort_index = np.argsort(-total_score)  #从大到小
    top1 = sort_index[0]
    top5 = sort_index[:5]
    top10 = sort_index[:10]
    top50 = sort_index[:50]
    
    poi_top1 = candidate_poi[top1]
    poi_top5 = [candidate_poi[i] for i in top5]
    poi_top10 = [candidate_poi[i] for i in top10]
    poi_top50 = [candidate_poi[i] for i in top50]
    
    if truth == poi_top1:
        top_1_result.append(1)
    else:
        top_1_result.append(0)
    
    if truth in poi_top5:
        top_5_result.append(1)
    else:
        top_5_result.append(0)
    
    if truth in poi_top10:
        top_10_result.append(1)
    else:
        top_10_result.append(0)
    
    if truth in poi_top50:
        index = 1/(1 + poi_top50.index(truth))
        MAP_result.append(index)
    else:
        MAP_result.append(0.01)
    
    if count%1000 == 0:
        print('count:',count)
        print('top1_acc:', np.mean(top_1_result))
        print('top5_acc:', np.mean(top_5_result))
        print('top10_acc:', np.mean(top_10_result))
        print('MAP_value:',np.mean(MAP_result))
        

print('Final result:')
print('top1_acc:', np.mean(top_1_result))
print('top5_acc:', np.mean(top_5_result))
print('top10_acc:', np.mean(top_10_result))
print('MAP_value:',np.mean(MAP_result))

end = time.clock()
print('Running time: %s Seconds'%(end-start))
    
        
        

