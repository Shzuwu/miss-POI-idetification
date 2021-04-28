# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:49:47 2021
将category相关信息进行修改
具体包括，重新新建图，具体包括：
user-poi
poi-time
user-catg
catg-catgTarget

其他测试：
1：u2c和c2c不要负采样测试(修改包括iteration部分和train部分)
2: c2c中6h间隔测试(负采样)
3: c2c中6h间隔测试(不含有负采样)
@author: Administrator
"""

import math
import random
import copy

import numpy as np
import tensorflow as tf
import networkx as nx
import pandas as pd

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.models import Model


from alias import create_alias_table, alias_sample
from utils import graphConstruct_user2poi, graphConstruct_time2poi, graphConstruct_user2catg, graphConstruct_catg2catg

def list2dict(node_list):
    idx2node = node_list
    node2idx = dict()
    for idx,node in enumerate(idx2node):
        node2idx[node] = idx
    return idx2node, node2idx

def time2str(timeValue):
    return 'time'+str(timeValue)

def c2c_target(catg):
    return 't_'+catg

def preprocess_Allgraph(user_list, poi_list, time_list, catg_list):
    node2idx = dict()
    idx2node = list()
    time_list = [time2str(i) for i in time_list]
    catgTarget_list = [c2c_target(i) for i in catg_list]
    for i in [poi_list, user_list, time_list, catg_list, catgTarget_list]:
        idx2node = idx2node + i
    for idx, node in enumerate(idx2node):
        node2idx[node] = idx
    return idx2node, node2idx

def ge_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_true*y_pred)))
    
class GE:
    def __init__(self, user_list, poi_list, time_list, catg_list, g_user2poi, g_time2poi, g_user2catg, g_catg2catg,\
                 g_u2pNotDirect, g_t2pNotDirect, g_u2cNotDirect, g_c2cNegativeDirect, embedding_size, negative_ratio, batch_size):
        self.g_user2poi = g_user2poi
        self.g_time2poi = g_time2poi
        self.g_user2catg = g_user2catg
        self.g_catg2catg = g_catg2catg
        self.g_u2pNotDirect = g_u2pNotDirect
        self.g_t2pNotDirect = g_t2pNotDirect
        self.g_u2cNotDirect = g_u2cNotDirect
        self.g_c2cNegativeDirect = g_c2cNegativeDirect
        
        # generate idx2node and node2idx for each node——————>需要修改
        self.user_idx2node, self.user_node2idx = list2dict(user_list)
        self.poi_idx2node, self.poi_node2idx = list2dict(poi_list)
        self.time_idx2node, self.time_node2idx = list2dict(time_list)
        self.catg_idx2node, self.catg_node2idx = list2dict(catg_list)
        
        # generate idx2node and node2idx for all nodes together——————>需要修改
        self.idx2node, self.node2idx = preprocess_Allgraph(user_list, poi_list, time_list, catg_list)
        self.nodeNum = len(self.idx2node)
       
        self._embedding = {}
        self.embedding_size = embedding_size
        self.negative_ratio = negative_ratio
        self.batch_size = batch_size
        
        # generate graphs' number of edges and nodes——————>需要修改
        self.u2p_nodesNum = g_user2poi.number_of_nodes()
        self.u2p_edgesNum = g_user2poi.number_of_edges()
        self.t2p_nodesNum = g_time2poi.number_of_nodes()
        self.t2p_edgesNum = g_time2poi.number_of_edges()
        self.u2c_nodesNum = g_user2catg.number_of_nodes()
        self.u2c_edgesNum = g_user2catg.number_of_edges()
        self.c2c_nodesNum = g_catg2catg.number_of_nodes()
        self.c2c_edgesNum = g_catg2catg.number_of_edges()
        
        # parameters initialization
        self.weight = [np.random.rand(len(self.idx2node), self.embedding_size)*2-1] 
        
        # sampling model initialization: edges sampling and nodes sampling——————>需要修改
        self.sample_prob()
        
        # training model initialization
        self.reset_model()
    
    def reset_model(self, opt='adam'):
        # ————》 generate batch_iter input
        self.batch_it = self.batch_iter()
        
        #  ————》construct training model with input, output and loss function
        self.model, self._embedding_dict = self.create_model()
        
        #  ————》compile training model
        self.model.compile(opt,ge_loss)
        
    def create_model(self):
        #————》input，
        x = Input(shape = (1,))
        y = Input(shape = (1,))
        
        nodeEmb = Embedding(self.nodeNum, self.embedding_size, weights = self.weight, name = "node_emb")
        
        x_emb = nodeEmb(x)
        y_emb = nodeEmb(y)
        
        result = Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=-1, keep_dims=False), name='output_sampling')([x_emb, y_emb])
        
        output_list = [result]
        model = Model(inputs = [x, y], outputs = output_list)
        
        return model, {'nodeEmb' : nodeEmb}
        
    def batch_iter(self):
        print('generate dataset iteration')
        
        u2p_edgesNum = self.u2p_edgesNum
        t2p_edgesNum = self.t2p_edgesNum
        u2c_edgesNum = self.u2c_edgesNum
        c2c_edgesNum = self.c2c_edgesNum 
        
        mod = 0
        mode_size = self.negative_ratio + 1
        
        u2p_batchSize = self.batch_size
        t2p_batchSize = int(self.batch_size * t2p_edgesNum / u2p_edgesNum)
        u2c_batchSize = int(self.batch_size * u2c_edgesNum / u2p_edgesNum)
        c2c_batchSize = int(self.batch_size * c2c_edgesNum / u2p_edgesNum)
        
        u2p_shuffle = np.random.permutation(range(u2p_edgesNum))
        u2p_start = 0
        u2p_end = min(u2p_start + u2p_batchSize, u2p_edgesNum)
        
        t2p_shuffle = np.random.permutation(range(t2p_edgesNum))
        t2p_start = 0
        t2p_end = min(t2p_start + t2p_batchSize, t2p_edgesNum)
        
        u2c_shuffle = np.random.permutation(range(u2c_edgesNum))
        u2c_start = 0
        u2c_end = min(u2c_start + u2c_batchSize, u2c_edgesNum)
        
        c2c_shuffle = np.random.permutation(range(c2c_edgesNum))
        c2c_start = 0
        c2c_end = min(c2c_start + c2c_batchSize, c2c_edgesNum)
        
        while True:
            if mod == 0:
                
                # Postive sampling for user2poi
                u2p_u = []
                u2p_p = []
                for i in range(u2p_start, u2p_end):
                    if random.random() >= self.u2p_edgesAccept[u2p_shuffle[i]]:
                        u2p_shuffle[i] = self.u2p_edgesAlias[u2p_shuffle[i]]        
                    cur_h = self.u2p_edgesIdx[u2p_shuffle[i]][0]
                    cur_t = self.u2p_edgesIdx[u2p_shuffle[i]][1]
                    u2p_u.append(cur_h)
                    u2p_p.append(cur_t)    
                sign = np.ones(len(u2p_u))
                yield([np.array(u2p_u), np.array(u2p_p)], [sign])
                
                 #Postive sampling for tiem2poi
                t2p_t = []
                t2p_p = []
                for i in range(t2p_start, t2p_end):
                    if random.random() >= self.t2p_edgesAccept[t2p_shuffle[i]]:
                        t2p_shuffle[i] = self.t2p_edgesAlias[t2p_shuffle[i]]
                    cur_h = self.t2p_edgesIdx[t2p_shuffle[i]][0]
                    cur_t = self.t2p_edgesIdx[t2p_shuffle[i]][1]
                    t2p_t.append(cur_h)
                    t2p_p.append(cur_t)
                sign = np.ones(len(t2p_t))                
                yield([np.array(t2p_t), np.array(t2p_p)], [sign])
                
                # Postive sampling for user2poi
                u2c_u = []
                u2c_c = []
                for i in range(u2c_start, u2c_end):
                    if random.random() >= self.u2c_edgesAccept[u2c_shuffle[i]]:
                        u2c_shuffle[i] = self.u2c_edgesAlias[u2c_shuffle[i]]
                    cur_h = self.u2c_edgesIdx[u2c_shuffle[i]][0]
                    cur_t = self.u2c_edgesIdx[u2c_shuffle[i]][1]
                    u2c_u.append(cur_h)
                    u2c_c.append(cur_t)
                sign = np.ones(len(u2c_u))
                yield([np.array(u2c_u), np.array(u2c_c)], [sign])
                
                #postive sampling for catg2catg
                c2c_c1 = []
                c2c_c2 = []
                for i in range(c2c_start, c2c_end):
                    if random.random() >= self.c2c_edgesAccept[c2c_shuffle[i]]:
                        c2c_shuffle[i] = self.c2c_edgesAlias[c2c_shuffle[i]]
                    cur_h = self.c2c_edgesIdx[c2c_shuffle[i]][0]
                    cur_t = self.c2c_edgesIdx[c2c_shuffle[i]][1]
                    c2c_c1.append(cur_h)
                    c2c_c2.append(cur_t)
                sign = np.ones(len(c2c_c1))            
                yield ([np.array(c2c_c1), np.array(c2c_c2)], [sign])
   
                
            else:
                # negative sampling for user-poi
                sign = np.ones(len(u2p_u)) * -1
                u2p_p = []
                for i in range(len(u2p_u)):
                    temp = alias_sample(self.u2p_poiNodeAccept, self.u2p_poiNodeAlias)
                    temp_index = self.u2p_poiIdx[temp]
                    u2p_p.append(temp_index)    
                yield([np.array(u2p_u), np.array(u2p_p)], [sign])

                
                # negative sampling for time-poi
                sign = np.ones(len(t2p_t)) * -1
                t2p_p = []
                for i in range(len(t2p_t)):
                    temp = alias_sample(self.t2p_poiNodeAccept, self.t2p_poiNodeAlias)
                    temp_index = self.t2p_poiIdx[temp]   
                    t2p_p.append(temp_index)  
                yield([np.array(t2p_t), np.array(t2p_p)], [sign])
                
                
                # negative sampling for user-catg
                sign = np.ones(len(u2c_u)) * -1
                u2c_c = []
                for i in range(len(u2c_u)):
                    temp = alias_sample(self.u2c_catgNodeAccept, self.u2c_catgNodeAlias)
                    temp_index = self.u2c_catgIdx[temp]
                    u2c_c.append(temp_index)
                yield([np.array(u2c_u), np.array(u2c_c)],[sign])
                
                # negative sampling for catg-catg
                sign = np.ones(len(c2c_c1)) * -1
                c2c_c2 = []
                for i in range(len(c2c_c1)):
                    temp = alias_sample(self.c2c_catgTargetAccept, self.c2c_catgTargetAlias)
                    temp_index = self.c2c_catgTargetIdx[temp]
                    c2c_c2.append(temp_index)
                yield([np.array(c2c_c1), np.array(c2c_c2)], [sign])    
                    
            mod = mod + 1
            mod %= mode_size
            
            if mod == 0:
                u2p_start = u2p_end
                u2p_end = min(u2p_start + u2p_batchSize, u2p_edgesNum)
                
                t2p_start = t2p_end
                t2p_end = min(t2p_start + t2p_batchSize, t2p_edgesNum)
                
                u2c_start = u2c_end
                u2c_end = min(u2c_start + u2c_batchSize, u2c_edgesNum)
                
                c2c_start = c2c_end
                c2c_end = min(c2c_start + c2c_batchSize, c2c_edgesNum)
            
            if u2p_start >= u2p_edgesNum:
                u2p_shuffle = np.random.permutation(range(u2p_edgesNum))
                u2p_start = 0
                u2p_end = min(u2p_start + u2p_batchSize, u2p_edgesNum)
                
                t2p_shuffle = np.random.permutation(range(t2p_edgesNum))
                t2p_start = 0
                t2p_end = min(t2p_start + t2p_batchSize, t2p_edgesNum)
                
                u2c_shuffle = np.random.permutation(range(u2c_edgesNum))
                u2c_start = 0
                u2c_end = min(u2c_start + u2c_batchSize, u2c_edgesNum)
                
                c2c_shuffle = np.random.permutation(range(c2c_edgesNum))
                c2c_start = 0
                c2c_end = min(c2c_start + c2c_batchSize, c2c_edgesNum)
                  
    def edges_sampling(self, graph):        
        temp = [graph[edge[0]][edge[1]].get('weight',1) for edge in graph.edges()]
        total_sum = sum(temp)
        norm = [i/total_sum for i in temp]
        edgesAccept, edgesAlias = create_alias_table(norm)     
        return edgesAccept, edgesAlias
    
    def Node_sampling(self, assignedGraph, item_idx2node, item_node2idx):
        power = 0.75
        numNodes = len(item_idx2node)
        
        node_degree = np.zeros(numNodes)
        for node in item_idx2node:
            for sub_node in assignedGraph[node]:
                node_degree[item_node2idx[node]] += assignedGraph[node][sub_node].get('weight', 1.0)
        
        total_sum = sum([math.pow(node_degree[i], power) for i in range(numNodes)])
        norm_prob = [float(math.pow(node_degree[i], power))/total_sum for i in range(numNodes)]
        
        nodeAccept, nodeAlias = create_alias_table(norm_prob)
        return nodeAccept, nodeAlias
        
    def sample_prob(self):
      
        # edges sampling for user-poi (user,poi)
        self.u2p_edgesAccept, self.u2p_edgesAlias = self.edges_sampling(self.g_user2poi)
        self.u2p_edgesIdx = [(self.node2idx[edge[0]], self.node2idx[edge[1]]) for edge in self.g_user2poi.edges()]

        # edges sampling for time-poi (time,poi)
        self.t2p_edgesAccept, self.t2p_edgesAlias = self.edges_sampling(self.g_time2poi)
        self.t2p_edgesIdx = [(self.node2idx[time2str(edge[0])], self.node2idx[edge[1]]) for edge in self.g_time2poi.edges()]
        
        # edges sampling for user-catg(user, catg)
        self.u2c_edgesAccept, self.u2c_edgesAlias = self.edges_sampling(self.g_user2catg)
        self.u2c_edgesIdx = [(self.node2idx[edge[0]], self.node2idx[edge[1]]) for edge in self.g_user2catg.edges()]
        
        # edges sampling for catg-catg ( catg_source, catg_target)
        self.c2c_edgesAccept, self.c2c_edgesAlias = self.edges_sampling(self.g_catg2catg)
        self.c2c_edgesIdx = [(self.node2idx[edge[0]], self.node2idx[c2c_target(edge[1])]) for edge in self.g_catg2catg.edges()]
        
        # poi nodes negative sampling for user-poi
        self.u2p_poiNodeAccept, self.u2p_poiNodeAlias = self.Node_sampling(self.g_u2pNotDirect, self.poi_idx2node, self.poi_node2idx)
        self.u2p_poiIdx = [self.node2idx[i] for i in self.poi_idx2node]      

        # poi nodes negative sampling for time-poi
        self.t2p_poiNodeAccept, self.t2p_poiNodeAlias = self.Node_sampling(self.g_t2pNotDirect, self.poi_idx2node, self.poi_node2idx)
        self.t2p_poiIdx = [self.node2idx[i] for i in self.poi_idx2node]
        
#        category negative sampling for user-catg
        self.u2c_catgNodeAccept, self.u2c_catgNodeAlias = self.Node_sampling(self.g_u2cNotDirect, self.catg_idx2node, self.catg_node2idx)
        self.u2c_catgIdx = [self.node2idx[i] for i in self.catg_idx2node]

#       category negative samplinf for catg-catg     
        self.c2c_catgTargetAccept, self.c2c_catgTargetAlias = self.Node_sampling(self.g_c2cNegativeDirect, self.catg_idx2node, self.catg_node2idx)
        self.c2c_catgTargetIdx = [self.node2idx[c2c_target(i)] for i in self.catg_idx2node]
       
    def train(self, epochs=5, initial_epoch = 0, verbose=1, times=1):
        
        steps = self.u2p_edgesNum // self.batch_size + 1
        self.steps_per_epoch = steps * ((1 + self.negative_ratio)*4)
#        self.steps_per_epoch = 10
        hist = self.model.fit_generator(self.batch_it, epochs = epochs, initial_epoch=initial_epoch, steps_per_epoch = self.steps_per_epoch, verbose=verbose)
        return hist
    
    def get_embedding_dataframe(self):
        print('generate embedding with dataframe')
        embeddings_user = pd.DataFrame()
        embeddings_poi = pd.DataFrame()
        embeddings_time = pd.DataFrame()
        embeddings_catg = pd.DataFrame()
        embeddings_catg_target = pd.DataFrame()
        
        embeddings = self._embedding_dict['nodeEmb'].get_weights()[0]
        
        temp = []
        for i in self.user_idx2node:
            index = self.node2idx[i]
            temp.append([embeddings[index]])
        embeddings_user['user'] = self.user_idx2node
        embeddings_user['embedding'] = temp
        
        temp = []
        for i in self.poi_idx2node:
            index = self.node2idx[i]
            temp.append([embeddings[index]])
        embeddings_poi['poi'] = self.poi_idx2node
        embeddings_poi['embedding'] = temp
        
        temp = []
        for i in self.time_idx2node:
            index = index = self.node2idx[time2str(i)]
            temp.append([embeddings[index]])
        embeddings_time['time'] = self.time_idx2node
        embeddings_time['embedding'] = temp
        
        temp = []
        temp1= []
        for i in self.catg_idx2node:
            index = self.node2idx[i]
            index_target = self.node2idx[c2c_target(i)]
            
            temp.append([embeddings[index]])
            temp1.append(embeddings[index_target])
        embeddings_catg['category'] = self.catg_idx2node
        embeddings_catg['embedding'] = temp
        
        embeddings_catg_target['category'] = self.catg_idx2node
        embeddings_catg_target['embedding'] = temp1
            
    def get_embedding_dict(self):
        print('generate embedding with dictionary')
        embeddings_user = {}
        embeddings_poi = {}
        embeddings_time = {}
        embeddings_catg = {}
        embeddings_catg_target = {}
        
        embeddings = self._embedding_dict['nodeEmb'].get_weights()[0]
        
        for i in self.user_idx2node:
            index = self.node2idx[i]
            embeddings_user[i] = embeddings[index]
        
        for i in self.poi_idx2node:
            index = self.node2idx[i]
            embeddings_poi[i] = embeddings[index]
        
        for i in self.time_idx2node:
            index = self.node2idx[time2str(i)]
            embeddings_time[i] = embeddings[index]
        
        for i in self.catg_idx2node:
            index = self.node2idx[i]
            index_target = self.node2idx[c2c_target(i)]
            
            embeddings_catg[i] = embeddings[index]
            embeddings_catg_target[i] = embeddings[index_target]
        
        poi_txt = open('embedding_poi.txt','w')
        user_txt = open('embedding_user.txt','w')
        time_txt = open('embedding_time.txt','w')
        catg_txt = open('embedding_catg.txt','w')
        catgTarget_txt = open('embedding_catgTarget.txt','w')
        
        user_txt.write(str(embeddings_user))
        poi_txt.write(str(embeddings_poi))
        time_txt.write(str(embeddings_time))
        catg_txt.write(str(embeddings_catg))
        catgTarget_txt.write(str(embeddings_catg_target))
        
        poi_txt.close()
        user_txt.close()
        time_txt.close()
        catg_txt.close()
        catgTarget_txt.close()
        
        return embeddings_user, embeddings_poi, embeddings_time, embeddings_catg, embeddings_catg_target
            
    def get_embeddingSaved(self):
        print('generate embedding') 
        self.embeddings = self._embedding_dict['nodeEmb'].get_weights()[0]
       
if __name__ =="__main__":
    
    data = 'NYC'
    path_user2poi = r'./data/preprocessedData/{}/graph_user2poi.csv'.format(data)
    df_user2poi, g_user2poi, g_u2pNotDirect = graphConstruct_user2poi(path_user2poi)
    print('graph user-poi has been contructed completely!')
        
    path_time2poi = r'./data/preprocessedData/{}/graph_time2poi.csv'.format(data)
    df_time2poi, g_time2poi, g_t2pNotDirect = graphConstruct_time2poi(path_time2poi)
    print('graph time-poi has been contructed completely!')
    
    path_user2catg = r'./data/preprocessedData/{}/graph_user2catg.csv'.format(data)
    df_user2catg, g_user2catg, g_u2cNotDirect = graphConstruct_user2catg(path_user2catg)
    print('graph user-catg has been contructed completely!')
    
        
    path_catg2catg = r'./data/preprocessedData/{}/graph_catg2catg.csv'.format(data)
    df_catg2catg, g_catg2catg, g_c2cNegativeDirect= graphConstruct_catg2catg(path_catg2catg)
    print('graph catg-catg has been contructed completely!')
    
    user_list = sorted(list(set(df_user2poi['user'])))
    poi_list = sorted(list(set(df_user2poi['poi'])))
    time_list = sorted(list(set(df_time2poi['timeshot'])))
    catg_list = sorted(list(g_catg2catg.nodes))
    
    model = GE(user_list, poi_list, time_list, catg_list, g_user2poi, g_time2poi, g_user2catg, g_catg2catg, \
               g_u2pNotDirect, g_t2pNotDirect, g_u2cNotDirect, g_c2cNegativeDirect, embedding_size = 128, negative_ratio = 5, batch_size = 512)
    
    model.train(epochs=10, verbose=2)
    
    embeddings_user, embeddings_poi, embeddings_time, embeddings_catg, embeddings_catg_target = model.get_embedding_dict()
   
    
    
    
    
    
