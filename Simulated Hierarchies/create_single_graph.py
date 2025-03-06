# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:21:08 2024

@author: Bruin
"""



import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from simulation_software.Simulate import simulate_graph
from simulation_software.simulation_utilities import compute_graph_STATs, sort_labels
from model.utilities import get_input_graph
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
from sklearn.manifold import TSNE
import pickle
from sklearn.decomposition import PCA
from model.model_layer import gaeGAT_layer as GAT
from model.model import GATE, CommClassifer, HCD
from model.train import CustomDataset, batch_data, fit
from simulation_software.simulation_utilities import compute_modularity, post_hoc_embedding, compute_beth_hess_comms
from model.utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df, LoadData, build_true_graph, get_input_graph, plot_nodes, plot_clust_heatmaps
import seaborn as sbn
import matplotlib.pyplot as plt
from community import community_louvain as cl
from itertools import product, chain
from tqdm import tqdm
import pdb
import ast
import random as rd
from sklearn.manifold import TSNE
import pickle
from sklearn.decomposition import PCA

# rd.seed(123)
# torch.manual_seed(123)



# simulation default arguments
parser.add_argument('--connect', dest='connect', default='disc', type=str)
parser.add_argument('--connect_prob', dest='connect_prob', default='use_baseline', type=str)
parser.add_argument('--toplayer_connect_prob', dest='toplayer_connect_prob', default=0.3, type=float)
parser.add_argument('--top_layer_nodes', dest='top_layer_nodes', default=5, type=int)
parser.add_argument('--subgraph_type', dest='subgraph_type', default='small world', type=str)
parser.add_argument('--subgraph_prob', dest='subgraph_prob', default=0.05, type=float)
parser.add_argument('--nodes_per_super2', dest='nodes_per_super2', default=(3,3), type=tuple)
parser.add_argument('--nodes_per_super3', dest='nodes_per_super3', default=(20,20), type=tuple)
parser.add_argument('--node_degree_middle', dest='node_degree_middle', default=5, type=int)
parser.add_argument('--node_degree_bottom', dest='node_degree_bottom', default=5, type=int)
parser.add_argument('--sample_size',dest='sample_size', default = 500, type=int)
parser.add_argument('--layers',dest='layers', default = 2, type=int)
parser.add_argument('--SD',dest='SD', default = 0.1, type=float)
parser.add_argument('--common_dist', dest='common_dist',default = True, type=bool)
parser.add_argument('--seed_number', dest='seed_number',default = None, type=int)
parser.add_argument('--within_edgeweights', dest='within_edgeweights',default = (0.4, 0.8), type=tuple)
parser.add_argument('--between_edgeweights', dest='between_edgeweights',default = (0, 0.2), type=tuple)
parser.add_argument('--use_weighted_graph', dest='use_weighted_graph',default = False, type=bool)
args = parser.parse_args()
args.connect_prob = 0.05
args.common_dist = False
args.subgraph_prob = 0.1
args.SD = 0.1
args.node_degree_bottom = 15
args.node_degree_middle = 3
args.force_connect = True
args.layers = 3
args.connect = 'disc'
args.subgraph_type = 'small world'
args.use_weighted_graph = True
#args.savepath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/Single_sim_dpd_identity/'

args.savepath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/'
#args.savepath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_4_15_2024/example7_output/'



def run_simulations(args, use_gpu = False):
    
    device = 'cuda:'+str(0) if use_gpu and torch.cuda.is_available() else 'cpu'
    
    print('*********** using DEVICE: {} **************'.format(device))
    
    
    pe, gexp, nodes, edges, nx_all, adj_all, args.savepath, nodelabs, orin = simulate_graph(args)
    
    true_adj_undi = build_true_graph(args.savepath+'.npz')
    
    pdadj = pd.DataFrame(adj_all[-1])
    pdadj.to_csv(args.savepath+'example_adjacency_matrix.csv')


    in_graph05, in_adj05 = get_input_graph(X = pe, 
                                           method = 'Correlation', 
                                           r_cutoff = 0.5)

    pdadj2 = pd.DataFrame(in_adj05)
    pdadj2.to_csv(args.savepath+'example_adjacency_matrix_corrgraph.csv')

    indices_top, indices_mid, labels_df, sorted_true_labels_top, sorted_true_labels_middle = sort_labels(nodelabs)


    #C = F.one_hot(torch.tensor(sorted_true_labels_top).to(torch.int64)).to(torch.float32)
    #X = torch.tensor(pe_sorted).requires_grad_()
    
    
    origin_nodes = [i[0] for i in orin]
    if args.layers > 2:
        pe_sorted = pe[indices_mid,:]
        idx = [indices_mid.index(i) for i in origin_nodes]
    else:
        pe_sorted = pe[indices_top,:]
        idx = [indices_mid.index(i) for i in origin_nodes]
        
    fig, ax = plt.subplots(figsize = (12, 10))
    sbn.heatmap(np.corrcoef(pe_sorted), ax = ax)
    ax.set_title('Pseudoexpression correlations')

    labels = [sorted_true_labels_middle, sorted_true_labels_top]
    TSNE_embed=TSNE(n_components=3, 
                    learning_rate='auto',
                    init='random', 
                    perplexity=3).fit_transform(pe_sorted)
    PCs = PCA(n_components=3).fit_transform(pe_sorted)

    which_labels = ['Middle', 'Top']
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    for i in range(0, len(labels)):
        
        ax[0][i].scatter(TSNE_embed[:,0], TSNE_embed[:,1], 
                         s = 150.0, c = labels[i], cmap = 'plasma')
        ax[0][i].scatter(TSNE_embed[idx,0], TSNE_embed[idx,1],
                     c='red', s = 300, marker = '.')
        ax[0][i].set_xlabel('Dimension 1')
        ax[0][i].set_ylabel('Dimension 2')
        ax[0][i].set_title( 't-SNE '+which_labels[i])
        #adding node labels
        ax[1][i].scatter(PCs[:,0], PCs[:,1], 
                         s = 150.0, c = labels[i], cmap = 'plasma')
        ax[1][i].scatter(PCs[idx,0], PCs[idx,1], 
        c='red', s = 300, marker = '.')
        ax[1][i].set_xlabel('Dimension 1')
        ax[1][i].set_ylabel('Dimension 2')
        ax[1][i].set_title( 'PCA '+which_labels[i])
        
        
        
        
    fig3d = plt.figure(figsize=(12,10))
    ax3d = plt.axes(projection='3d')
    ax3d.scatter3D(PCs[:,0], PCs[:,1], PCs[:,2], 
                  c=labels[1], cmap='plasma')
    ax3d.scatter3D(PCs[idx,0], PCs[idx,1], PCs[idx,2], 
                 c='red', s = 300, marker = '.',
                 depthshade = False)

    ax3d.set_xlabel('Dimension 1')
    ax3d.set_ylabel('Dimension 2')
    ax3d.set_zlabel('Dimension 3')




    fig3d = plt.figure(figsize=(12,10))
    ax3d = plt.axes(projection='3d')
    ax3d.scatter3D(TSNE_embed[:,0], TSNE_embed[:,1], TSNE_embed[:,2], 
                  c=labels[0], cmap='plasma')
    ax3d.scatter3D(TSNE_embed[idx,0], TSNE_embed[idx,1], TSNE_embed[idx,2], 
                 c='red', s = 300, marker = '.',
                 depthshade = False)

    ax3d.set_xlabel('Dimension 1')
    ax3d.set_ylabel('Dimension 2')
    ax3d.set_zlabel('Dimension 3')
    
    
    
    
run_simulations(args)