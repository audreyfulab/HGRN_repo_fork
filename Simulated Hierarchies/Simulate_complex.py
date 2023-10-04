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
sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/gen_data')
sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/HC-GNN/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
from Simulate import simulate_graph
#import os
#os.chdir('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Bethe Hessian Tests/')
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
# general
from random import randint as rd   

# model
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


args.connect = 'full'
args.toplayer_connect_prob = 0.3
args.connect_prob = 0.01
args.top_layer_nodes = 10
args.subgraph_type = 'small world'
args.subgraph_prob=0.01
args.nodes_per_super2=(10,20)
args.nodes_per_super3=(5, 10)
args.layers = 3
args.sample_size = 500
args.SD = 0.1
# 2 layer path
#args.savepath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/small_world/fully_connected/2_layer/SD01/sm_2full_connect01_sd01.npz'

#3 layer path
args.savepath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/small_world/fully_connected/3_layer/SD01/sm_3full_connect01_sd01.npz'
#simulate
pe, nodes_per_layer = simulate_graph(args)
