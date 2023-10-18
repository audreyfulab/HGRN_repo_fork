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
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from model_layer import gaeGAT_layer as GAT
from model import GATE, CommClassifer, HCD
from train import CustomDataset, batch_data, fit
from simulation_utilities import compute_modularity
from utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df
sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/HC-GNN/')
from utils_modded import Load_Simulation_Data, get_input_graph
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


comm_sizes = [60,10]
#define HGRNgene model
HCD_model = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act='LeakyReLU')

#convert data to tensors and allow self loops in graph
X = torch.Tensor(pe).requires_grad_()
A = torch.Tensor(in_adj).requires_grad_()+torch.eye(nodes)



#fit model
out = fit(HCD_model, X, A, optimizer='Adam', epochs = 100, update_interval=20, 
        lr = 1e-4, gamma = 0.5, delta = 1, comm_loss='Modularity',
        true_labels=true_labels.clustlabs.to_numpy(), verbose=False)



S_sub, S_layer, S_all = trace_comms(out[4], comm_sizes)



A_pred = resort_graph(out[1].detach().numpy(), flat_list_indices)
A_true = resort_graph(in_adj, flat_list_indices)
fig, (ax1,ax2) = plt.subplots(2,2, figsize=(12,10))
sbn.heatmap(A_pred, ax = ax1[0])
sbn.heatmap(A_true, ax = ax1[1])


df = gen_labels_df(S_layer, sort_true_labels, flat_list_indices)


sbn.heatmap(df, ax = ax2[0])

print('-'*50)
print('true network modularity = {:.4f}'.format(
    compute_modularity(A_true, sort_true_labels)
    ))





