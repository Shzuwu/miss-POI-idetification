# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:13:24 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter
from pylab import scatter
import pylab
import time
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_catg2catg_draw(matrix, top_list):
    sns.set()
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = top_list
    df_matrix.columns = top_list
    ax = sns.heatmap(df_matrix, fmt='d', linewidths=.5, cmap='YlGnBu')
    plt.xlabel('Category_target', fontproperties = 'Times New Roman',fontsize=15, color='k') #x轴label的文本和字体大小
    plt.ylabel('Category_source', fontproperties = 'Times New Roman',fontsize=15, color='k') #y轴label的文本和字体大小
    plt.xticks(fontsize=12, color='k') #x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=12, color='k')
    plt.savefig('./figureMaking/catg2catg.pdf', bbox_inches='tight')
    plt.show()

def scatterPlot(item,str_item):
    df_poi = df.copy()
    df_poi = df_poi.set_index(item)
    df_poi = df_poi.sort_values(by=[item])
    
    poi_list = list(df_poi.index)
    poi_counter = list(Counter(poi_list).values())
    poi_counter = Counter(poi_counter)
    
    if item == 'categoryID':
        poi_counter_x = np.log10(list(poi_counter.keys()))
        poi_counter_y = np.log10(list(poi_counter.values()))
    elif item == 'user':
        poi_counter_x = np.log10(list(poi_counter.keys()))
        poi_counter_y = np.log10(list(poi_counter.values()))
        x = [0,1,2,3,4]
        plt.xticks(x, [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'], fontsize=12, color='k')
        y = [0,0.4,0.8,1.2,1.6]
        plt.yticks(y, [r'$10^{0}$', r'$10^{0.4}$', r'$10^{0.8}$', r'$10^{1.2}$', r'$10^{1.6}$'], fontsize=12, color='k')
        
    else:
        poi_counter_x = np.log10(list(poi_counter.keys()))
        poi_counter_y = np.log10(list(poi_counter.values()))
        x = [0,1,2,3,4]
        plt.xticks(x, [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'],fontsize=12, color='k')
        y = [0,1,2,3,4]
        plt.yticks(y, [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'],fontsize=12, color='k')
    
    plt.scatter(poi_counter_x, poi_counter_y, c='blue')
    plt.xlabel('Check-ins', fontproperties = 'Times New Roman',fontsize=15, color='k') #x轴label的文本和字体大小
    plt.ylabel('Number of ' + str_item, fontproperties = 'Times New Roman',fontsize=15, color='k') #y轴label的文本和字体大小
    plt.savefig('./figureMaking/' + str_item+'.pdf', bbox_inches='tight')
    plt.show()
    return poi_counter_x, poi_counter_y

 
def heatmap_draw(matrix, time, name, time_str, top_list):
    sns.set()
    time_str = [str(i) for i in time_str]
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = top_list
    df_matrix.columns = time_str
    ax = sns.heatmap(df_matrix, fmt='d', linewidths=.5, cmap='YlGnBu')
    plt.xlabel(time, fontproperties = 'Times New Roman',fontsize=15, color='k') #x轴label的文本和字体大小
    plt.ylabel('Category', fontproperties = 'Times New Roman',fontsize=15, color='k') #y轴label的文本和字体大小
    plt.xticks(fontsize=12, color='k') #x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=12, color='k')
    plt.savefig('./figureMaking/'+ name + '.pdf', bbox_inches='tight')
    plt.show()



data = 'NYC'
#data = 'TKY'
#data = 'checkins-gowalla1'
#data = 'checkins-4sq'

if data == 'NYC' or data == 'TKY':
    path = r'data/original data/dataset_TSMC2014_{}.txt'.format(data)
    df = pd.read_csv(path, header=None, sep='\t',encoding='unicode_escape')
    df_columns = ['user', 'poi', 'categoryID', 'category_name', 'lat', 'lon', 'off_set', 'timestamp']
    df.columns = df_columns
elif data == 'checkins-gowalla1' or data == 'checkins-4sq':
    path = r'data/original data/{}.txt'.format(data)
    df = pd.read_csv(path, header=None, sep='\t',encoding='unicode_escape')
    df_columns = ['user','timestamp','lat','lon','poi']
    df.columns = df_columns

df_NYC_category = pd.read_excel('NYC_category1.xlsx')
small_category = list(df_NYC_category['small category'])
big_category = list(df_NYC_category['big category'])
dict_category = dict()
for i in range(len(small_category)):
    dict_category[small_category[i]] = big_category[i]
       
x,y = scatterPlot('poi','POIs')
x,y = scatterPlot('user','Users')
#x,y = scatterPlot('categoryID')
print('data is:', data)

# hit-map 
if data == 'NYC' or data == 'TKY': 
    df_category = pd.DataFrame()
    df_category['category_name'] = df['category_name']
    df_category['timestamp'] = df['timestamp']
    
    list_catg = list(df_category['category_name'])
    list_time = list(df_category['timestamp'])
    time_day = [int(time.strftime("%w", time.strptime(i,'%a %b %d %X +0000 %Y'))) for i in list_time]
    time_hour = [int(time.strftime("%H", time.strptime(i,'%a %b %d %X +0000 %Y'))) for i in list_time]
    
    df_category['time_day'] = time_day
    df_category['time_hour'] = time_hour
    
    df_category = df_category.set_index('category_name')
    df_category = df_category.sort_values(by=['category_name'])
    
    poi_counter = Counter(list_catg)
    top_eight = list(poi_counter.most_common(7))
    top_list = [i[0] for i in top_eight]
    
    df_top = df_category.loc[top_list]
    
    divide = 12
    divide_len = int(24 / divide)
    matrix_weekdays = np.zeros((len(top_eight),24))
    matrix_weekends = np.zeros((len(top_eight),24))
    
    matrix_weekdays_session = np.zeros((len(top_eight),divide))
    matrix_weekends_session = np.zeros((len(top_eight),divide))
    
    
    for i in range(len(top_list)):
        df_temp = df_top.loc[top_list[i]]
        temp_day =  list(df_temp['time_day'])
        temp_hour = list(df_temp['time_hour'])
        for j in range(len(df_temp)):
            if 5 >= temp_day[j] >0:
                matrix_weekdays[i, temp_hour[j]] += 1
                matrix_weekdays_session[i, (temp_hour[j]//divide_len)] += 1
            else:
                matrix_weekends[i, temp_hour[j]] += 1
                matrix_weekends_session[i, (temp_hour[j]//divide_len)] += 1
    
    heatmap_draw(matrix_weekdays, "Hour_weekdays", 'matrix_weekdays',list(range(0,24)), top_list)
    heatmap_draw(matrix_weekends, "Hour_weekends", 'matrix_weekends',list(range(0,24)), top_list)
    heatmap_draw(matrix_weekdays_session, "Hour_weekdays", 'matrix_weekdays_session',list(range(0,24,divide_len)), top_list)
    heatmap_draw(matrix_weekends_session, "Hour_weekends", 'matrix_weekends_session',list(range(0,24,divide_len)), top_list)
    
    list_big_catg = [dict_category[i] for i in list_catg]    
    df_category['big_catg'] = list_big_catg
    df_category = df_category.set_index('big_catg')
    df_category = df_category.sort_values(by=['big_catg'])
    
    big_catg_list = list(set(list_big_catg))
    matrix_weekdays_bigcatg = np.zeros((len(big_catg_list),24))
    matrix_weekends_bigcatg = np.zeros((len(big_catg_list),24))
    matrix_weekdays_bigcatg_session = np.zeros((len(big_catg_list),divide))
    matrix_weekends_bigcatg_session = np.zeros((len(big_catg_list),divide))
    
    for i in range(len(big_catg_list)):
        df_temp = df_category.loc[big_catg_list[i]]
        temp_day =  list(df_temp['time_day'])
        temp_hour = list(df_temp['time_hour'])
        for j in range(len(df_temp)):
            if 5 >= temp_day[j] >0:
                matrix_weekdays_bigcatg[i, temp_hour[j]] += 1
                matrix_weekdays_bigcatg_session[i, (temp_hour[j]//divide_len)] += 1
            else:
                matrix_weekends_bigcatg[i, temp_hour[j]] += 1
                matrix_weekends_bigcatg_session[i, (temp_hour[j]//divide_len)] += 1
    
#    heatmap_draw(matrix_weekdays_bigcatg, "hour_weekdays", 'matrix_weekdays_bigcatg',list(range(0,24)), big_catg_list)
#    heatmap_draw(matrix_weekends_bigcatg, "hour_weekends", 'matrix_weekends_bigcatg',list(range(0,24)), big_catg_list)
#    heatmap_draw(matrix_weekdays_bigcatg_session, "hour_weekdays", 'matrix_weekdays_bigcatg_session',list(range(0,24,divide_len)), big_catg_list)
#    heatmap_draw(matrix_weekends_bigcatg_session, "hour_weekends", 'matrix_weekends_bigcatg_session',list(range(0,24,divide_len)), big_catg_list)
#    
if data == 'NYC' or data == 'TKY': 
    top_list1 = top_list.copy()
    top_list1 = ['Bar','Home (private)','Office','Subway','Gym / Fitness Center','Coffee Shop','Food & Drink Shop']
    
    big_catg_list1 = big_catg_list.copy()
    df_catg2catg = pd.DataFrame()
    df_catg2catg['user'] = df['user']
    df_catg2catg['category_name'] = df['category_name']
    df_catg2catg['big_categoty'] = [dict_category[i] for i in list(df['category_name'])]
    df_catg2catg['timestamp'] = df['timestamp']
    
    
    df_catg2catg = df_catg2catg.set_index('category_name')
    df_catg2catg = df_catg2catg.loc[top_list1]
    df_catg2catg = df_catg2catg.reset_index()
    
    df_catg2catg['timeConvert'] = [time.mktime(time.strptime(
        df_catg2catg['timestamp'][index], '%a %b %d %X +0000 %Y')) for index in df_catg2catg['timestamp'].index]

    df_catg2catg = df_catg2catg.sort_values(by=['user','timeConvert'])
    df_catg2catg = df_catg2catg.set_index('user')
    
    user_list = list(set(df_catg2catg.index))
    matrix_catg2catg = np.zeros((len(top_list1),len(top_list1)))
    
    time_threshold = 3600*6
    
    for user in user_list:
        df_temp = df_catg2catg.loc[user]
        catg_list= list(df_temp['category_name'])
        timeCovert_array = np.array(df_temp['timeConvert'])
        catg_forward = catg_list[:-1]
        catg_backward= catg_list[1:]
        
        if type(df_temp) == pd.core.series.Series:
            continue
        
        time_forward = timeCovert_array[:-1]
        time_backward = timeCovert_array[1:]
        time_delta = time_backward - time_forward
        for index in range(len(catg_forward)):
            if catg_forward[index] not in top_list1 or catg_backward[index] not in top_list1:
                print(catg_forward[index], catg_backward[index])
                continue
            
            if catg_forward[index] == catg_backward[index]:
                continue
            
            if time_delta[index] > time_threshold:
                continue
            
            i = top_list1.index(catg_forward[index])
            j = top_list1.index(catg_backward[index])
            matrix_catg2catg[i,j] += 1
    heatmap_catg2catg_draw(matrix_catg2catg, top_list1)
    print()





    
   
    
        


    
    
    
    








    
    
