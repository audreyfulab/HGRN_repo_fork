# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:27:13 2024

@author: Bruin
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
import networkx as nx
from sklearn.metrics import roc_auc_score, f1_score, normalized_mutual_info_score,homogeneity_score, completeness_score
from sklearn.neighbors import kneighbors_graph
import scipy as spy
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from sklearn.decomposition import PCA

from utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df, LoadData, get_input_graph, plot_nodes

loadpath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/Toy_examples/Intermediate_examples/small_world/disconnected/3_layer/smw_disc_3_layer_'

pe, true_adj_undi, indices_top, indices_middle, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle = LoadData(filename=loadpath_main)

 
def my_kmeans(X, Niter, k, tol = 0.001):
    
    """
    """
    TSNE_embed=TSNE(n_components=3, 
                    learning_rate='auto',
                    init='random', 
                    perplexity=3).fit_transform(X.detach().numpy())
    PCs = PCA(n_components=3).fit_transform(X.detach().numpy())
    
    N = X.shape[0]
    ss = X.shape[1]
    seeds = np.random.randint(0, N, k)
    M = X[seeds, :]
    nodes = torch.arange(0, N)
    loss_hist = []
    change_loss = 1
    for i in range(0, Niter):
        dists = torch.zeros((N, k))
        for j in range(0, k):
            dists[:,j] = torch.square((X - M[j,:])).sum(dim = 1)
            
        loss_hist.append(dists.mean(dim = 0).sum()*(1/k))
        if i > 2:
            change_loss = abs(loss_hist[-2] - loss_hist[-1])
        S = dists.argmin(1)
        n_k = torch.diag(torch.mm(F.one_hot(S).T, F.one_hot(S)))
        clust_IDs = torch.unique(S)
        for m in range(0, k):
            ix = nodes[S == clust_IDs[m]]
            M[m,:] = torch.index_select(X, 0, ix).mean(dim = 0)/n_k[m]
            
        
        if change_loss < tol:
            return(S)
        fig, ax = plt.subplots(1,2, figsize = (10,10))
        ax[0].scatter(TSNE_embed[:,0], TSNE_embed[:,1], 
                         s = 150.0, c = S, cmap = 'plasma')
        ax[0].set_xlabel('Dimension 1')
        ax[0].set_ylabel('Dimension 2')
        ax[0].set_title( 't-SNE ')
        #adding node labels
        ax[1].scatter(PCs[:,0], PCs[:,1], 
                         s = 150.0, c = S, cmap = 'plasma')
        ax[1].set_xlabel('Dimension 1')
        ax[1].set_ylabel('Dimension 2')
        ax[1].set_title( 'PCA ')       
        
    
    
    
    
my_kmeans(torch.Tensor(pe), 150, 5, tol = 0.0001)   
    
    
    
    
    
    
    
    