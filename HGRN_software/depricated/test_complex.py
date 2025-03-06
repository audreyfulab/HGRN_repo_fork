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
from utilities import resort_graph, trace_comms, node_clust_eval, Load_Simulation_Data, get_input_graph
import seaborn as sbn
import matplotlib.pyplot as plt

paths = ['C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/small_world/fully_connected/2_layer/SD01/',
         'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/small_world/fully_connected/3_layer/SD01/']
pe, true_adj_undi, flat_list_indices, true_labels, sort_true_labels = Load_Simulation_Data(paths[0], 
                                                                                           data = 'sm', 
                                                                                           layers='2',
                                                                                           connectivity = ['full','disc'][0],
                                                                                           connect_prob = '01',
                                                                                           SD = ['01','05'][0])

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


# Z, A = encoder.forward(torch.tensor(pe), torch.tensor(in_adj))

# X_hat, A = decoder.forward(Z, A)

# communityDetector = CommClassifer(in_nodes = nodes, in_attrib = Z.shape[1])
# X_top, A_top, S, A_all = communityDetector.forward(Z, A)



#define HGRNgene model
HGRN_model = HGRNgene(nodes, attrib, comm_sizes=[150,10],attn_act='LeakyReLU')

#convert data to tensors and allow self loops in graph
X = torch.tensor(pe).requires_grad_()
A = torch.tensor(in_adj).requires_grad_()+torch.eye(nodes)



#fit model
out = fit(HGRN_model, X, A, optimizer='Adam', epochs = 200, update_interval=50, 
        lr = 1e-3, prop_train = 0.8, gamma = 0.5, delta = 0.8, comm_loss='Modularity',
        true_labels=true_labels.clustlabs.to_numpy(), verbose=False)




S_all = trace_comms(out[-1], [150,10])
S_middle = S_all[0]
S_top = S_all[1]

Middle_2_Top = torch.mm(F.one_hot(S_middle), F.one_hot(S_top)).argmax(1)


A_pred = resort_graph(out[1].detach().numpy(), flat_list_indices)

fig, ax1 = plt.subplots(1,2, figsize=(12,10))
sbn.heatmap(A_pred, ax = ax1[0])

df = pd.DataFrame(np.array([Middle_2_Top.detach().numpy()[flat_list_indices, None].reshape(pe.shape[0],),
                      sort_true_labels]).transpose(), columns = ['Predicted','Truth'])
sbn.heatmap(df, ax = ax1[1])










