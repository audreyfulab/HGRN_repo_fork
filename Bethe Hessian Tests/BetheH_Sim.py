# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 10:14:56 2023

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
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Bethe Hessian Tests/')
from BH_simulate import simulate_graph
from utils_modded import get_input_graph
import os
os.chdir('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Bethe Hessian Tests/')
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
args.top_layer_nodes = 10
args.subgraph_type = 'small world'
args.subgraph_prob=0.01
args.nodes_per_super2=(10,20)
args.nodes_per_super3=(5, 10)
args.layers = 2
args.sample_size = 500


degvals = []



corrcuts = np.arange(0.25, 0.85, 0.1)
connect_prob = [0.05, 0.02, 0.01, 0.005]
fig, (ax1, ax2) = plt.subplots(2,2, figsize = (12,10))
for c in range(0, len(connect_prob)):
    avgD = []
    acc = []
    pred = []
    nodes_per_graph = []
    stats = []
    for j in range(0,40):
        Dvec = []
        predvec = []
        nodevec = []
        accvec = []
        #args.connect = ['disc','full'][1]
        print(args, '\n')   
        args.connect_prob = connect_prob[c]
        pe, nodes = simulate_graph(args)
        nodes_per_graph.append(nodes)
        for i in range(0, len(corrcuts)):
        #for i in range(0, len(degvals)):
        #for i in range(0, 20):
        
            G, A = get_input_graph(X = pe, 
                                   method = 'Correlation', 
                                   r_cutoff = corrcuts[i])
            N = A.shape[0]
        

            Deg = np.diag(np.matmul(A, np.ones((N, 1))).reshape(N))
            avg_degree = np.matmul(np.matmul(np.ones((N,1)).T, A), np.ones((N, 1)))/N
            eta = np.sqrt(avg_degree)
    
            Dvec.append(avg_degree)
            #print(Dvec)
            Bethe_Hessian = (np.square(eta)-1)*np.diag(np.ones(N))+Deg - eta*A
        
            eigvals = np.linalg.eigh(Bethe_Hessian)[0]
        
            k = np.sum(eigvals<0)
            predvec.append(k)
            accvec.append((k/args.top_layer_nodes))
        
        
        
    
            print("="*60)
            print('Average Degree ={}, Number of communities detected = {}'.format(avg_degree,k))
            print("="*60)
        
        
        acc.append(accvec)
        avgD.append(Dvec)
        pred.append(predvec)
        nodes_per_graph.append(nodevec)
        
    pred_array = np.array(pred)
    avgD_array = np.array(avgD).reshape(40,6)
    print(avgD)
    cols = np.arange(0, len(corrcuts)) 



    for i in range(0, len(corrcuts)):
        print('now constructing plot!!!!!!!!!!1')
        if c<2:
            print(c)
            ax1[c].scatter(pred_array[:,i], avgD_array[:,i], label = 'r = '+str(np.round(corrcuts[i], 2)))
            ax1[c].legend(loc="upper right")
            ax1[c].set_xlabel('Number of predicted communities', size = 12)
            ax1[c].set_ylabel('Average node degree of input graph ', size = 12)
            #ax1[c].tick_params(axis='both', which='major', labelsize=10)
            #ax1[c].set_title('Connection Prob ='+str(connect_prob[c]))
        elif c>1:
            ax2[c-2].scatter(pred_array[:,i], avgD_array[:,i], label = 'r = '+str(np.round(corrcuts[i], 2)))
            ax2[c-2].legend(loc="upper right")
            ax2[c-2].set_xlabel('Number of predicted communities', size = 12)
            ax2[c-2].set_ylabel('Average node degree of input graph ', size = 12)
            #ax2[c-2].tick_params(axis='both', which='major', labelsize=10)(fontsize=12)
            #ax2[c-2].set_title('Connection Prob ='+str(connect_prob[c]))
    

fig.savefig('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Bethe Hessian Tests/sim_result_scatter_full_combined.pdf')
   
    