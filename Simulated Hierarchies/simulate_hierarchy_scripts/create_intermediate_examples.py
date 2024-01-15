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
#sys.path.append('/mnt/ceph/jarredk/scGNN_for_genes/HC-GNN/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
#sys.path.append('/mnt/ceph/jarredk/scGNN_for_genes/gen_data')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/gen_data')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/HC-GNN/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
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
#rd.seed(333)


# simulation default arguments
parser.add_argument('--connect', dest='connect', default='disc', type=str)
parser.add_argument('--connect_prob', dest='connect_prob', default='use_baseline', type=str)
parser.add_argument('--toplayer_connect_prob', dest='toplayer_connect_prob', default=0.3, type=float)
parser.add_argument('--top_layer_nodes', dest='top_layer_nodes', default=5, type=int)
parser.add_argument('--subgraph_type', dest='subgraph_type', default='small world', type=str)
parser.add_argument('--subgraph_prob', dest='subgraph_prob', default=0.05, type=float)
parser.add_argument('--nodes_per_super2', dest='nodes_per_super2', default=(3,3), type=tuple)
parser.add_argument('--nodes_per_super3', dest='nodes_per_super3', default=(20,20), type=tuple)
parser.add_argument('--node_degree', dest='node_degree', default=5, type=int)
parser.add_argument('--sample_size',dest='sample_size', default = 500, type=int)
parser.add_argument('--layers',dest='layers', default = 2, type=int)
parser.add_argument('--SD',dest='SD', default = 0.1, type=float)
parser.add_argument('--seed_number', dest='seed_number',default = 555, type=int)
args = parser.parse_args()


# args.connect = 'full'
# args.toplayer_connect_prob = 0.3
args.connect_prob = 0.01
# args.top_layer_nodes = 5
# args.subgraph_type = 'small world'
# args.nodes_per_super2=(5,5)
# args.nodes_per_super3=(5, 5)
# args.layers = 3
# args.sample_size = 500
# args.SD = 0.1
# args.node_degree = 5

mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/Toy_examples/Intermediate_examples/'
#mainpath = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/'

structpath = ['small_world/','scale_free/','random_graph/']
connectpath = ['disconnected/', 'fully_connected/']
layerpath = ['3_layer/']
#noisepath = ['SD01/','SD05/']


struct_nm = ['smw_','sfr_','rdg_']
connect_nm =['disc_', 'full_']
layer_nm = ['3_layer_']
#noise_nm = ['SD01','SD05']

struct = ['small world','scale free','random graph']
connect = ['disc', 'full']
layers = [3]
#noise = [0.1, 0.5]



grid1 = product(structpath, connectpath, layerpath)
grid2 = product(struct_nm, connect_nm, layer_nm)
grid3 = product(struct, connect, layers)

#simulate

info_table = pd.DataFrame(columns = ['subgraph_type', 'connection_prob','layers','StDev',
                                     'nodes_per_layer', 'edges_per_layer', 'subgraph_prob',
                                     'sample_size','modularity_top','avg_node_degree_top',
                                     'avg_connect_within_bottom','avg_connect_between_top',
                                     'modularity_middle','avg_node_degree_middle',
                                     'avg_connect_within_middle','avg_connect_between_middle',
                                     ])

for idx, value in tqdm(enumerate(zip(grid1, grid2, grid3)), desc="Simulating hierarchies…", ascii=False, ncols=75):
    
    
    args.subgraph_type = value[2][0]
    
    args.connect = value[2][1]
    #args.connect = 'full'
    args.layers = value[2][2]
    args.SD = 0.1
    args.node_degree = 3
    args.force_connect = True
    
    #if args.subgraph_type == 'small world':
    #    if args.connect == 'full':
    print('='*60)
    print(args)
    print('-'*60)
    args.savepath = mainpath+''.join(value[0])+''.join(value[1])
    print('saving hierarchy to {} '.format(args.savepath))
    pe, gexp, nodes, edges, nx_all, adj_all, args.savepath, nodelabs = simulate_graph(args)
    print('done')
    print('-'*60)
    print('computing statistics....')
    
    mod, node_deg, deg_within, deg_between = compute_graph_STATs(A_all = adj_all, 
                                                                 comm_assign = nodelabs, 
                                                                 layers = args.layers,
                                                                 sp = args.savepath,
                                                                 node_size = 60,
                                                                 font_size = 9,
                                                                 add_labels=True)
        

        
    print('*'*25+'top layer stats'+'*'*25)
    print('modularity = {:.4f}, mean node degree = {:.4f}'.format(
        mod[0], node_deg[0]
        )) 
    print('mean within community degree = {:.4f}, mean edges between communities = {:.4f}'.format(
        deg_within[0], deg_between[0] 
        ))
    if args.layers > 2:
        print('*'*25+'middle layer stats'+'*'*25)
        print('modularity = {:.4f}, mean node degree = {:.4f}'.format(
            mod[1], node_deg[1]
        )) 
        print('mean within community degree = {}, mean edges between communities = {}'.format(
            deg_within[1], deg_between[1] 
            ))
        print('*'*60)
    if args.layers == 3:
        
        row_info = [args.subgraph_type, args.connect, args.layers, args.SD,
                    tuple(nodes),tuple(edges),args.subgraph_prob, args.sample_size,
                    mod[0], node_deg[0], deg_within[0], deg_between[0], 
                    mod[1], node_deg[1], deg_within[1], deg_between[1]]
    else:
        row_info = [args.subgraph_type, args.connect, args.layers, args.SD,
                    tuple(nodes),tuple(edges), args.subgraph_prob, args.sample_size,
                    mod[0], node_deg[0], deg_within[0], deg_between[0], 
                    'NA', 'NA', 'NA', 'NA']
        
    print(pd.DataFrame(row_info))

    print('done')
    print('saving hierarchy statistics...')
    info_table.loc[idx] = row_info
    info_table.to_csv(mainpath+'toy_examples_network_statistics.csv')
    np.savez(mainpath+'toy_examples_network_statistics.npz', data = info_table.to_numpy())
    print('done')
    
print('Simulation Complete')       
    