# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:39:52 2020

@author: Administrator
"""
import pandas as pd
import networkx as nx
import numpy as np


def preprocess_node(node_list):
    node2idx = dict()
    for index, node in enumerate(node_list):
        node2idx[node] = index
    return node2idx

# 图的预处理
def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx

# ****************************************************
# 节点整理
def preprocess_graph(g_user2catg, g_user2poi):
    node2idx = {}
    idx2node = []
    node_size = 0
    for graph in [g_user2catg, g_user2poi]:
        for node in graph.nodes():
            idx2node.append(node)
            node_size += 1 
    idx2node = list(set(idx2node))
    for idx,node in enumerate(idx2node):
        node2idx[node] = idx
    return idx2node, node2idx

# 构建用户-地点二部图
def graphConstruct_user2poi (path_user2poi):
    df = pd.read_csv(path_user2poi,sep='\t')
    df['weight'] = np.ones(len(df),dtype=np.float32)
    df = df['weight'].groupby([df['user'],df['poi']]).sum()
    df = df.reset_index()
    g_user2poi = nx.from_pandas_edgelist(df,source='user',target='poi',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    g_u2pNotDirect = nx.from_pandas_edgelist(df,source='user',target='poi',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return df, g_user2poi, g_u2pNotDirect

# 构建用户-类别二部图
def graphConstruct_user2catg (path_user2catg):
    df = pd.read_csv(path_user2catg,sep='\t')
    df['weight'] = np.ones(len(df),dtype=np.float32)
    df = df['weight'].groupby([df['user'],df['category']]).sum()
    df = df.reset_index()
    g_user2catg = nx.from_pandas_edgelist(df,source='user',target='category',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    g_u2cNotDirect = nx.from_pandas_edgelist(df,source='user',target='category',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return df, g_user2catg, g_u2cNotDirect

def timeValue(day_hourminute):
    day = eval(day_hourminute.split(':')[0])
    hourminute = day_hourminute.split(':')[1]
        # divide a day into six part()
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

def graphConstruct_time2poi(path_time2poi):
    # timeshot, poi
    df = pd.read_csv(path_time2poi, sep = '\t')
    df['weight'] = np.ones(len(df),dtype=np.float)
    df= df['weight'].groupby([df['timeshot'],df['poi']]).sum()
    df = df.reset_index()
    g_time2poi = nx.from_pandas_edgelist(df, source='timeshot',target='poi',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    g_t2pNotDirect = nx.from_pandas_edgelist(df, source='timeshot',target='poi',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return df, g_time2poi, g_t2pNotDirect

def graphConstruct_time2catg(path_time2catg):
    # timeshot, poi
    df = pd.read_csv(path_time2catg, sep = '\t')
    df['weight'] = np.ones(len(df),dtype=np.float)
    df= df['weight'].groupby([df['timeshot'],df['category']]).sum()
    df = df.reset_index()
    g_time2catg = nx.from_pandas_edgelist(df, source='timeshot',target='category',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    g_t2cNotDirect = nx.from_pandas_edgelist(df, source='timeshot',target='category',edge_attr=['weight'],\
                                         create_using=nx.Graph())
    return df, g_time2catg, g_t2cNotDirect

def graphConstruct_catg2catg(path_catg2catg):
    df_new = pd.read_csv(path_catg2catg, sep='\t')
    df = pd.DataFrame()
    df['category_source'] = df_new['category_source']
    df['category_target'] = df_new['category_target']  
    df['weight'] = np.ones(len(df),dtype=np.float32)        
    df=df['weight'].groupby([df['category_source'],df['category_target']]).sum()
    df = df.reset_index()
    g_catg2catg = nx.from_pandas_edgelist(df,source='category_source',target='category_target',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    g_c2cNegativeDirect = nx.from_pandas_edgelist(df,source='category_target',target='category_source',edge_attr=['weight'],\
                                         create_using=nx.DiGraph())
    return df, g_catg2catg, g_c2cNegativeDirect


#def sample_prob(self):
#    """
#    output:
#    u2c_nodes——>list, u2c_nodes_prob——>list
#    u2p_nodes——>list, u2p_nodes_prob——>list
#    u2c_edges——>list, u2c_edges_prob——>list
#    u2p_edges——>list, u2p_edges_prob——>list
#    """
#    power = 0.75
#    node_degree = dict()
#    edge_degree = dict()
#    
#    # nodes sampling for user2catg
#    for node in self.g_user2catg.nodes():
#        for sub_node in self.g_user2catg[node]:
#            node_degree[node] += self.g_user2catg[node][sub_node].get('weight',1)
#    total_sum = sum([math.pow(i, power) for i in node_degree.values()])
#    self.u2c_nodes_prob = [float(math.pow(node_degree[i], power))/total_sum for i in node_degree]
#    self.u2c_nodes = list(node_degree.keys())
#    node_degree.clear()
#    
#    #nodes sampling for user2poi
#    for node in self.g_user2poi.nodes():
#        for sub_node in self.g_user2poi[node]:
#            node_degree[node] = self.g_user2poi[node][sub_node].get('weight',1)
#    total_sum = sum([math.pow(i,power) for i in node_degree.values()])
#    self.u2p_nodes_prob = [float(math.pow(node_degree[i], power))/total_sum for i in node_degree]  
#    self.u2p_nodes = list(node_degree.keys())
#    node_degree.clear()
#    
#    # edges sampling for user2catg
#    for edge in self.g_user2catg.edges():
#        edge_degree[edge] = self.g_user2catg[edge[0]][edge[1]].get('weight',1)
#    total_sum = sum(edge_degree.values())
#    self.u2c_edges_prob = [float(edge_degree[i])/total_sum for i in edge_degree.keys()]
#    self.u2c_edges = list(edge_degree.keys())
#    edge_degree.clear()
#    
#    # edges sampling for user2poi
#    for edge in self.g_user2poi.edges():
#        edge_degree[edge] = self.g_user2poi[edge[0]][edge[1]].get('weight',1)
#    total_sum = sum(edge_degree.values())
#    self.u2p_edges_prob = [float(edge_degree[i])/total_sum for i in edge_degree.keys()]
#    self.u2p_edges = list(edge_degree.keys())
#    edge_degree.clear()
#    self.total_sum = total_sum
#
##    #nodes sampling for user2poi
##    self.u2p_nodes = [node2idx[node] for node in self.g_user2poi.nodes()]
##    u2p_nodes_origin = [node for node in self.g_user2poi.nodes()]
##    for node in self.g_user2poi.nodes():
##        u2p_nodesDegree[node] = 0
##        for subnode in self.g_user2poi[node]:
##            u2p_nodesDegree[node] += self.g_user2poi[node][subnode].get('weight', 1.0)
##
##    total_sum = sum([math.pow(u2p_nodesDegree[node], power) for node in self.g_user2poi.nodes()])
##    u2p_norm = [float(math.pow(u2p_nodesDegree[node], power))/total_sum for node in self.g_user2poi.nodes()]
##    # sequence checking
##    if u2p_nodes_origin == list(u2p_nodesDegree.keys()):
##        print('hello world u2p!')
##    self.u2p_nodeAccept, self.u2p_nodeAlias = create_alias_table(u2p_norm) 