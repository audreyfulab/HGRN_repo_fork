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
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
sys.path.append('C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/HGRN_software/')
from simulation_software.Simulate import simulate_graph
from simulation_software.simulation_utilities import compute_graph_STATs
import warnings
import random as rd
from itertools import product
from tqdm import tqdm
import time
import os
#set seed
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
rd.seed(978)


# simulation default arguments
parser.add_argument('--connect', dest='connect', default='disc', type=str)
parser.add_argument('--connect_prob_middle', dest='connect_prob_middle', default='use_baseline', type=str)
parser.add_argument('--connect_prob_bottom', dest='connect_prob_bottom', default='use_baseline', type=str)
parser.add_argument('--toplayer_connect_prob', dest='toplayer_connect_prob', default=0.3, type=float)
parser.add_argument('--top_layer_nodes', dest='top_layer_nodes', default=5, type=int)
parser.add_argument('--subgraph_type', dest='subgraph_type', default='small world', type=str)
parser.add_argument('--subgraph_prob', dest='subgraph_prob', default=[0.5, 0.2], type=float)
parser.add_argument('--nodes_per_super2', dest='nodes_per_super2', default=(5, 5), type=tuple)
parser.add_argument('--nodes_per_super3', dest='nodes_per_super3', default=(60, 70), type=tuple)
parser.add_argument('--node_degree_middle', dest='node_degree_middle', default=3, type=int)
parser.add_argument('--node_degree_bottom', dest='node_degree_bottom', default=5, type=int)
parser.add_argument('--sample_size',dest='sample_size', default = 500, type=int)
parser.add_argument('--layers',dest='layers', default = 2, type=int)
parser.add_argument('--SD',dest='SD', default = 0.1, type=float)
parser.add_argument('--common_dist', dest='common_dist',default = True, type=bool)
parser.add_argument('--seed_number', dest='seed_number',default = 555, type=int)
parser.add_argument('--within_edgeweights', dest='within_edgeweights',default = (0.5, 0.8), type=tuple)
parser.add_argument('--between_edgeweights', dest='between_edgeweights',default = (0, 0.2), type=tuple)
parser.add_argument('--use_weighted_graph', dest='use_weighted_graph',default = False, type=bool)
args = parser.parse_args()


# args.connect = 'full'
args.connect_prob_middle = [0.1, 0.1] #first value is for within community, second value is for between community
args.connect_prob_bottom = [0.002, 0.001]
args.common_dist = False
# args.top_layer_nodes = 5
# args.subgraph_type = 'small world'
# args.nodes_per_super2=(5,5)
# args.nodes_per_super3=(5, 5)
# args.layers = 3
#args.sample_size = 500
# args.SD = 0.1
#args.node_degree = 5

path = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/'
#path = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/DATA/'
location = 'complex_networks/'
mainpath = os.path.join(path, location)
if not os.path.exists(mainpath):
    os.makedirs(mainpath)

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
                                     'sample_size','modularity_top','avg_node_degree_top',
                                     'avg_connect_within_top','avg_connect_between_top',
                                     'modularity_middle','avg_node_degree_middle',
                                     'avg_connect_within_middle','avg_connect_between_middle',
                                     ])

#net = 22

for idx, value in tqdm(enumerate(zip(grid1, grid2, grid3)), desc="Simulating hierarchies…", ascii=False, ncols=75):
    
    
    args.subgraph_type = value[2][0]
    args.connect = value[2][1]
    args.layers = value[2][2]
    args.SD = value[2][3]
    args.force_connect = True
    
    print('-'*60)
    args.savepath = mainpath+''.join(value[0])+''.join(value[1])
    
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    print('saving hierarchy to {} '.format(args.savepath))
    
    if args.layers == 2:
        args.nodes_per_super2 = (50, 100)
    else:
        args.nodes_per_super2 = (3, 3)
        
    pe, gexp, nodes, edges, nx_all, adj_all, args.savepath, nodelabs, ori = simulate_graph(args)
    print('done')
    print('-'*60)
    print('computing statistics....')
    
    mod, node_deg, deg_within, deg_between = compute_graph_STATs(A_all = adj_all, 
                                                                 comm_assign = nodelabs, 
                                                                 layers = args.layers,
                                                                 sp = args.savepath,
                                                                 node_size=12,
                                                                 font_size = 0,
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
    info_table.to_csv(mainpath+'network_statistics.csv')
    np.savez(mainpath+'network_statistics.npz', data = info_table.to_numpy())
    print('done')
    
    plt.close('all')
    
print('Simulation Complete')
        
        
