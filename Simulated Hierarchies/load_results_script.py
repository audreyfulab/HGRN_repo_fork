# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:58:03 2024

@author: Bruin
"""

import pickle
#preamble
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import sys
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from model_layer import gaeGAT_layer as GAT
from model import GATE, CommClassifer, HCD
from train import CustomDataset, batch_data, fit
from simulation_utilities import compute_modularity, post_hoc_embedding, compute_beth_hess_comms
from utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df, LoadData, get_input_graph, plot_nodes
import seaborn as sbn
import matplotlib.pyplot as plt
from community import community_louvain as cl
from itertools import product, chain
from tqdm import tqdm
import pdb
import ast
import random as rd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from community import community_louvain as cl




def load_output(graph = 0, struct = 0, connect = 0, layers = 0, noise = 0, resolution = 0, 
                gamma = 1, delta = 1, TOAL = True):
    
    case = ['A_ingraph_true/','A_corr_no_cutoff/','A_ingraph02/', 'A_ingraph05/', 'A_ingraph07/'][graph]
    structpath = ['small_world/','scale_free/','random_graph/'][struct]
    connectpath = ['disconnected/', 'fully_connected/'][connect]
    layerpath = ['2_layer/', '3_layer/'][layers]
    noisepath = ['SD01/','SD05/'][noise]
    resolu = [[1,1],[100, 5]][resolution]
    
    struct_nm = ['smw_','sfr_','rdg_'][struct]
    connect_nm =['disc_', 'full_'][connect]
    layer_nm = ['2_layer_','3_layer_'][layers]
    noise_nm = ['SD01','SD05'][noise]
    
    nm = '_gam_'+str(gamma)+'_delt_'+str(delta)+'_reso_'+str(resolu[0])+'_'+str(resolu[1])+'_TOAL_'+str(TOAL)
    mp_res = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/Simulation_Results/Simulation_Results_'
    mp_data = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/DATA/'
    
    tab = pd.read_csv(mp_res+case[0:len(case)-1]+'_'+nm+'.csv')
    #Simulation_Results_smw_full_3_layer_SD05A_ingraph07__gam_1_delt_1_reso_100_5_TOAL_False_OUTPUT
    #Simulation_Results_smw_full_3_layer_SD01A_ingraph_true__gam_1_delt_1_reso_1_1_TOAL_True_OUTPUT.pkl'
    #Simulation_Results_A_ingraph07__gam_1_delt_1_reso_100_5_TOAL_True
    fn = mp_res +struct_nm+connect_nm+layer_nm+noise_nm+case[0:len(case)-1]+'_'+nm+'_OUTPUT.pkl'
    with open(fn, 'rb') as f:
        out = pickle.load(f)
        
    lp = mp_data+structpath+connectpath+layerpath+noisepath+struct_nm+connect_nm+layer_nm+noise_nm
    # pe, true_adj_undi, indices_top, indices_middle, new_true_labels, 
    # sorted_true_labels_top, sorted_true_labels_middle
    data = LoadData(filename=lp)
    stats = pd.read_csv(mp_data+'network_statistics.csv')
    
    return tab, out, data, stats

tab, out, data, stats = load_output()


def get_preds(tab, out, data, stats, network = 0):
    
    print('='*20)
    print(stats.loc[network])
    print('='*20)
    print('*'*20)
    print(tab.loc[network])
    print('*'*20)
    A = out[0][0][3][0]
    nodes = A.shape[0]
    G = (A-torch.eye(nodes)).detach().numpy()
    comms = cl.best_partition(nx.from_numpy_array(G))
    #extract cluster labels
    louv_preds = list(comms.values())
    best_epoch_preds_top = out[0][tab.loc[network][2]][-2][1]
    best_epoch_preds_mid = out[0][tab.loc[network][2]][-2][0]
    truth_top = data[-2]
    truth_mid = data[-1]
    
    return best_epoch_preds_top, best_epoch_preds_mid, louv_preds, truth_top, truth_mid

prd_top, prd_mid, lprd, ttop, tmid = get_preds(network = 2)