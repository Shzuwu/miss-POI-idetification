# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 19:16:02 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os


class LOC2INDEX:
    def __init__(self, path, GRID_LENGTH):
        self.path = path
        self.GRID_LENGTH = GRID_LENGTH
        self.data = pd.read_csv(path, sep='\t')
        self.draw_loc()
        self.gridArgument()
        self.loc2grid(self.grid_length, self.grid_width)
    
    def draw_loc(self):
        self.Lat = self.data['lat']
        self.Lon = self.data['lon']
        plt.scatter(self.Lat, self.Lon)
        
    def gridArgument(self):
        p_loc_x = np.array(self.Lat)
        p_loc_y = np.array(self.Lon)  
        self.x_min = p_loc_x.min()
        self.x_max = p_loc_x.max()
        self.y_min = p_loc_y.min()
        self.y_max = p_loc_y.max()
        
        self.grid_length = self.GRID_LENGTH
        self.grid_width = int(self.GRID_LENGTH*(self.y_max-self.y_min)/(self.x_max-self.x_min))
    
    def loc2grid(self, grid_length, grid_width):
        x_range = np.linspace(self.x_min, self.x_max, grid_length+1)
        y_range = np.linspace(self.y_min, self.y_max, grid_width+1)
        self.x_dict_values = [(x_range[i]+x_range[i+1])/2 for i in range(len(x_range)-1)]   
        self.y_dict_values = [(y_range[i]+y_range[i+1])/2 for i in range(len(y_range)-1)]
        
    def findIndex(self, lat, lon):
        x_dict_index = int((lat - self.x_min)/(self.x_max - self.x_min)*self.grid_length)
        y_dict_index = int((lon - self.y_min)/(self.y_max - self.y_min)*self.grid_width)
        if x_dict_index == self.grid_length:
            x_dict_index = self.grid_length-1
            
        if y_dict_index == self.grid_width:
            y_dict_index = self.grid_width-1
        
        index_corrdinate = (self.x_dict_values[x_dict_index], self.y_dict_values[y_dict_index])
        index = y_dict_index * self.grid_length + x_dict_index
        
        return index_corrdinate, index
    
    def generateIndexCoordinate(self):
        list_index, list_coordinate = [],[]
        df_data = self.data
        temp_lat = list(df_data['lat'])
        temp_lon = list(df_data['lon'])
        for i in range(len(temp_lat)):
            corrdinate, index = self.findIndex(temp_lat[i], temp_lon[i])
            list_coordinate.append(corrdinate)
            list_index.append(index)
        df_data['coordinate'] = list_coordinate
        df_data['index'] = list_index
        print('hello world')
        return df_data
    
#data='NYC'       
#basepath = r"data/preprocessedData/"
#path = os.path.join(basepath, "{}_poi_category_location_train.csv".format(data))
#path1 = os.path.join(basepath, "{}_poi_category_location_train1.csv".format(data))
#loc2index = LOC2INDEX(path)   
#data_indexCoordinate = loc2index.generateIndexCoordinate()
#
#loc2index1 = LOC2INDEX(path1)
#data_indexCoordinate1 = loc2index1.generateIndexCoordinate()
#
#data_indexCoordinate.to_csv(r'./data/preprocessedData/{}_Index_Coordinate.csv'.format(data),sep = '\t', index = False, header = True)
#data_indexCoordinate1.to_csv(r'./data/preprocessedData/{}_Index_Coordinate1.csv'.format(data),sep = '\t', index = False, header = True)
