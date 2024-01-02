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
import sys
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')

from run_simulations_test import run_simulations

out, res, graphs = run_simulations(save_results=True, 
                                   loss_fn='Modularity',
                                   which_graph=3,
                                   which_network = 14,
                                   true_comm_layers=True,
                                   comm_sizes=[146, 10],
                                   resolu=[150, 3],
                                   epochs = 300,
                                   updates = 300,
                                   gam = 1, 
                                   delt = 1,
                                   hidden_dims = [256, 128, 64])