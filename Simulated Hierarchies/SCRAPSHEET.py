# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:24:53 2023

@author: Bruin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
from simulation_utilities import post_hoc_embedding
from run_simulations_test import run_simulations
from simulation_utilities import post_hoc_embedding

ep = 20
out, res, graphs, data, truth, preds, louv_pred = run_simulations(save_results=True,
                                                       which_net=0,
                                                       which_ingraph=0,
                                                       reso=[1,1],
                                                       hd=[256, 128, 64],
                                                       gam=0,
                                                       delt=1, 
                                                       learn_rate=1e-4,
                                                       epochs = ep,
                                                       updates = 2,
                                                       loss_fn='Modularity',
                                                       activation = 'LeakyReLU',
                                                       TOAL=True)


sp = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/run_sim_test/'
fig, ax = plt.subplots(figsize = (14,10))
G = nx.from_numpy_array((graphs[0]- torch.eye(data.shape[0])).detach().numpy())
nx.draw_networkx(G, pos=nx.shell_layout(G), 
                 with_labels = True,
                 font_size = 10,
                 node_color = truth[0], 
                 ax = ax,
                 node_size = 100,
                 cmap = 'rainbow')
fig2, ax2 = plt.subplots(figsize = (14,10))
nx.draw_networkx(G, pos=nx.shell_layout(G), 
                 with_labels = True,
                 font_size = 10,
                 node_size = 100,
                 node_color = truth[1], 
                 ax = ax2,
                 cmap = 'rainbow')

fig.savefig(sp+'Circular_layout_truegraph_topclusts.png', dpi = 300)
fig2.savefig(sp+'Circular_layout_truegraph_midclusts.png', dpi = 300)

epoch = ep-1
#Top layer TSNE and PCA
post_hoc_embedding(graph=out[0][epoch][3][0]-torch.eye(data.shape[0]), 
                        input_X = data,
                        embed=out[0][epoch][2][0], 
                        probabilities=out[0][epoch][4],
                        size = 150.0,
                        labels = preds, 
                        truth = truth,
                        fs=10,
                        path = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/run_sim_test/', 
                        save = True,
                        node_size = 25, font_size = 10)
