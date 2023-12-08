# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:44:12 2023

@author: Bruin
"""

import numpy as np
import networkx as nx
#from networkx.community import modularity
import seaborn as sbn
import matplotlib.pyplot as plt
import itertools  
import sys
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
from utilities import Modularity
from simulation_utilities import compute_modularity
import torch.nn.functional as F
import random as rd

def apply_edge(mat, node1, node2):
    
    mat[node1, node2] = 1
    mat[node2, node1] = 1
    
    return mat


def easy_modularity(adj, nd_list):
    nl = [0]+nd_list
    idx = np.cumsum(nl)
    deg_seq = adj.sum(axis = 1)
    sig_totals = []
    sig_in = []
    L = adj.sum()
    for i in range(0, len(nl)-1):
        sig_totals.append(deg_seq[idx[i]:idx[i+1]].sum())
        sig_in.append(adj[idx[i]:idx[i+1], idx[i]:idx[i+1]].sum())
    
    mod = (sig_in/L) - np.square(sig_totals/L)
    return mod, np.array(sig_in), np.array(sig_totals)




