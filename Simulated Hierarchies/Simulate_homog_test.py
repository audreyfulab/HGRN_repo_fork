# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:51:09 2023

@author: Bruin
"""


import random
import torch
import argparse
import numpy as np
import networkx as nx
import seaborn as sbn
import matplotlib.pyplot as plt
import sys
import pandas as pd
sys.path.append('/mnt/ceph/jarredk/scGNN_for_genes/HC-GNN/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
sys.path.append('/mnt/ceph/jarredk/scGNN_for_genes/gen_data')
# sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/gen_data')
# sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/HC-GNN/')
# sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
# sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from Simulate import simulate_graph
from simulation_utilities import compute_graph_STATs
#import os
#os.chdir('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Bethe Hessian Tests/')
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
# general
import random as rd
from itertools import product
from tqdm import tqdm
import time
#set seed
rd.seed(123)
import pdb

# simulation default arguments
parser.add_argument('--connect', dest='connect', default='disc', type=str)
parser.add_argument('--connect_prob', dest='connect_prob', default=0.05, type=float)
parser.add_argument('--toplayer_connect_prob', dest='toplayer_connect_prob', default=0.3, type=float)
parser.add_argument('--top_layer_nodes', dest='top_layer_nodes', default=10, type=int)
parser.add_argument('--subgraph_type', dest='subgraph_type', default='small world', type=str)
parser.add_argument('--subgraph_prob', dest='subgraph_prob', default=0.05, type=float)
parser.add_argument('--nodes_per_super2', dest='nodes_per_super2', default=(10,20), type=tuple)
parser.add_argument('--nodes_per_super3', dest='nodes_per_super3', default=(5,10), type=tuple)
parser.add_argument('--node_degree', dest='node_degree', default=5, type=int)
parser.add_argument('--sample_size',dest='sample_size',default = 500, type=int)
parser.add_argument('--layers',dest='layers',default = 2, type=int)
parser.add_argument('--SD',dest='SD',default = 0.1, type=float)
args = parser.parse_args()


# args.connect = 'full'
args.toplayer_connect_prob = 0.3
args.connect_prob = 0.01
# args.top_layer_nodes = 5
# args.subgraph_type = 'small world'
args.subgraph_prob=0.01
args.nodes_per_super2=(5,5)
args.nodes_per_super3=(5,5)
#args.layers = 3
args.sample_size = 500
# args.SD = 0.1
args.node_degree = 5

#mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
mainpath = '/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/'

structpath = ['small_world/','scale_free/','random_graph/']
connectpath = ['disconnected/', 'fully_connected/']
layerpath = ['2_layer/', '3_layer/']
noisepath = ['SD01/','SD05/']


struct_nm = ['smw_','sfr_','rdg_']
connect_nm =['disc_', 'full_']
layer_nm = ['2_layer_','3_layer_']
noise_nm = ['SD01','SD05']

struct = ['small world','scale free','random graph']
connect = ['disc', 'full']
layers = [2, 3]
noise = [0.1, 0.5]



grid1 = product(structpath, connectpath, layerpath, noisepath)
grid2 = product(struct_nm, connect_nm, layer_nm, noise_nm)
grid3 = product(struct, connect, layers, noise)

#simulate

info_table = pd.DataFrame(columns = ['subgraph_type', 'connection_prob','layers','StDev',
                                     'nodes_per_layer', 'edges_per_layer', 'subgraph_prob',
                                     'sample_size','modularity_bottom','avg_node_degree_bottom',
                                     'avg_connect_within_bottom','avg_connect_between_bottom',
                                     'modularity_middle',
                                     'avg_node_degree_middle','avg_connect_within_middle',
                                     'avg_connect_between_middle',
                                     ])


for idx, value in tqdm(enumerate(zip(grid1, grid2, grid3)), desc="Simulating hierarchies…", ascii=False, ncols=75):
    if idx == 2:
        #pdb.set_trace()
        args.subgraph_type = value[2][0]
        args.connect = value[2][1]
        args.layers = value[2][2]
        args.SD = value[2][3]
        
        print('-'*60)
        args.savepath = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/'
        print('saving hierarchy to {} '.format(args.savepath))
        pe, gexp, nodes, edges, nx_all, adj_all, args.savepath, nodelabs = simulate_graph(args)
        print('done')
        print('-'*60)
        print('computing statistics....')
        pdb.set_trace()
        mod, node_deg, deg_within, deg_between = compute_graph_STATs(A_all = adj_all, 
                                                                     comm_assign = nodelabs, 
                                                                     layers = args.layers)
        
        print('*'*25+'top layer stats'+'*'*25)
        print('modularity = {:.4f}, mean node degree = {:.4f}'.format(
            mod[0].detach().numpy(), node_deg[0]
            )) 
        print('mean within community degree = {:.4f}, mean edges between communities = {:.4f}'.format(
            deg_within[0], deg_between[0] 
            ))
        print('*'*25+'middle layer stats'+'*'*25)
        print('modularity = {:.4f}, mean node degree = {:.4f}'.format(
            mod[1].detach().numpy(), node_deg[1]
            )) 
        print('mean within community degree = {}, mean edges between communities = {}'.format(
            deg_within[1], deg_between[1] 
            ))
        print('*'*60)
        if args.layers == 3:
            
            row_info = [args.subgraph_type, args.connect, args.layers, args.SD,
                       tuple(nodes),tuple(edges),args.subgraph_prob, args.sample_size,
                       mod[0].detach().numpy(), node_deg[0], deg_within[0], deg_between[0], 
                       mod[1].detach().numpy(), node_deg[1], deg_within[1], deg_between[1]]
        else:
            row_info = [args.subgraph_type, args.connect, args.layers, args.SD,
                       tuple(nodes),tuple(edges), args.subgraph_prob, args.sample_size,
                       mod[0], node_deg[0], deg_within[0], deg_between[0], 
                       'NA', 'NA', 'NA', 'NA']
        
        print('done')
        print('saving hierarchy statistics...')
        info_table.loc[idx] = row_info
        info_table.to_csv(mainpath+'network_statistics.csv')
        print('done')
    print('Simulations Complete')
    
        
        
