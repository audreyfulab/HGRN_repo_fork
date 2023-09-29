# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:55:09 2023

@author: Bruin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from model_layer import gaeGAT_layer as GAT
from model import GATE, CommClassifer, HGRNgene
from train import CustomDataset, batch_data, fit
from utilities import resort_graph
sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/HC-GNN/')
from utils_modded import Load_Simulation_Data, get_input_graph, node_clust_eval
import seaborn as sbn
import matplotlib.pyplot as plt

paths = ['C:/Users/Bruin/Documents/GitHub/hierarchicalGRN/hgnn_software/data/disconnected/sm/sm_disc_0.1/',
         'C:/Users/Bruin/Documents/GitHub/hierarchicalGRN/hgnn_software/data/full/sm/sm_full_0.1/']
pe, true_adj_undi, flat_list_indices, true_labels, sort_true_labels = Load_Simulation_Data(paths[0], 
                                                                              data='sm',
                                                                              #connectivity='full',
                                                                              connectivity=['disc','full'][0],
                                                                              SD='0.1')

#get the input graph 
in_graph, in_adj = get_input_graph(X = pe, 
                                   method = 'Correlation', 
                                   r_cutoff = 0.5)

in_sorted1 = in_adj[flat_list_indices,:]
in_adj_sorted = in_sorted1[:, flat_list_indices]
nodes = pe.shape[0]
attrib = pe.shape[1]
epochs = 20
encoder = GATE(in_nodes = nodes, in_attrib = attrib, attention_act='Sigmoid')
decoder = GATE(in_nodes = nodes, in_attrib = 64, hid_sizes=[128, 256, attrib], attention_act='Sigmoid')


# Z, A = encoder.forward(torch.Tensor(pe), torch.Tensor(in_adj))

# X_hat, A = decoder.forward(Z, A)

# communityDetector = CommClassifer(in_nodes = nodes, in_attrib = Z.shape[1])
# X_top, A_top, S, A_all = communityDetector.forward(Z, A)



#define HGRNgene model
HGRN_model = HGRNgene(nodes, attrib, comm_sizes=[10],attn_act='Sigmoid')

#convert data to tensors and allow self loops in graph
X = torch.Tensor(pe).requires_grad_()
A = torch.Tensor(in_adj).requires_grad_()+torch.eye(nodes)



#fit model
out = fit(HGRN_model, X, A, optimizer='Adam', epochs = 200, update_interval=50, 
        lr = 1e-4, prop_train = 0.8, gamma = 0.5, delta = 1, comm_loss='Clustering',
        true_labels=true_labels.clustlabs.to_numpy(), verbose=False)








A_pred = resort_graph(out[1].detach().numpy(), flat_list_indices)

fig, ax1 = plt.subplots(1,2, figsize=(12,10))
sbn.heatmap(A_pred, ax = ax1[0])
sbn.heatmap(out[-1][0].detach().numpy()[flat_list_indices, None], ax = ax1[1])










