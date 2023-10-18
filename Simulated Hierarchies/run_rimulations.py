# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:46:39 2023

@author: Bruin
"""

#preamble
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from model_layer import gaeGAT_layer as GAT
from model import GATE, CommClassifer, HCD
from train import CustomDataset, batch_data, fit
from simulation_utilities import compute_modularity
from utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df, LoadData, get_input_graph
import seaborn as sbn
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import pdb


def run_simulations():
    
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    loadpath_main = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/'
    savepath_main ='/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/Simulation_Results/'
    #loadpath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    #savepath_main ='C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/Simulation_Results/'
    structpath = ['small_world/','scale_free/','random_graph/']
    connectpath = ['disconnected/', 'fully_connected/']
    layerpath = ['2_layer/', '3_layer/']
    noisepath = ['SD01/','SD05/']
    
    
    struct_nm = ['smw_','sfr_','rdg_']
    connect_nm =['disc_', 'full_']
    layer_nm = ['2_layer_','3_layer_']
    noise_nm = ['SD01','SD05']
    
    struct = ['small world','scale free','random graph']
    connect = ['disc', 'full']
    layers = [2, 3]
    noise = [0.1, 0.5]
    
    #read in network statistics 
    stats = pd.read_csv(loadpath_main+'network_statistics.csv')
    #combine pathname and filename pieces
    grid1 = product(structpath, connectpath, layerpath, noisepath)
    grid2 = product(struct_nm, connect_nm, layer_nm, noise_nm)
    grid3 = product(struct, connect, layers, noise)
    
    #preallocate results table
    res_table = pd.DataFrame(columns = ['Modularity_True_ingraph', 'Modularity_r05_ingraph',
                                         'Modularity_r08_igraph', 'Recon_A_true_ingraph',
                                         'Recon_A_r05_ingraph', 'Recon_A_r08_ingraph',
                                         'Recon_X_true_ingraph','Recon_X_r05_ingraph',
                                         'Recon_X_r08_ingraph', 'true_ingraph_metrics',
                                         'r05_ingraph_metrics', 'r08_ingraph_metrics'])
    
    #run simulations
    for idx, value in enumerate(zip(grid1, grid2, grid3)):
        
        #extract and use true community sizes
        if len(stats.nodes_per_layer[idx]) == 9:    
            comm_sizes = [int(stats.nodes_per_layer[idx][1:3]), int(stats.nodes_per_layer[idx][5:8])][::-1]
        else:
            comm_sizes = [int(stats.nodes_per_layer[idx][1:3]), int(stats.nodes_per_layer[idx][5:9])][::-1]
            
        #set pathnames and read in simulated network
        print('-'*25+'loading in data'+'-'*25)
        loadpath = loadpath_main+''.join(value[0])+''.join(value[1])
        savepath = savepath_main+''.join(value[0])+''.join(value[1])
        pe, true_adj_undi, sort_indices, true_labels, sort_true_labels = LoadData(filename=loadpath)
        #sort nodes in expression table 
        pe_sorted = pe[sort_indices,:]
        #generate input graphs for correlations r > 0.5 and r > 0.8
        in_graph05, in_adj05 = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = 0.5)
        
        in_graph08, in_adj08 = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = 0.8)
        #print network statistics
        print('network statistics:')
        print(stats.loc[idx])
        print('...done')
        
        #nodes and attributes
        nodes = pe_sorted.shape[0]
        attrib = pe_sorted.shape[1]
        #set up three separate models for true input graph, r > 0.5 input graph, and
        #r > 0.8 input graph scenarios
        print('-'*25+'setting up and fitting models'+'-'*25)
        HCD_model_truth = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act='LeakyReLU')
        HCD_model_r05 = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act='LeakyReLU')
        HCD_model_r08 = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act='LeakyReLU')
        
        #set attribute and input graph(s) to torch tensors with grad attached
        X = torch.Tensor(pe_sorted).requires_grad_()
        #three input graph scenarios -- add self loops
        A_truth = torch.Tensor(true_adj_undi[:nodes,:nodes]).requires_grad_()+torch.eye(nodes)
        A_r05 = torch.Tensor(in_adj05).requires_grad_()+torch.eye(nodes)
        A_r08 = torch.Tensor(in_adj08).requires_grad_()+torch.eye(nodes)
        
        #combine input items into lists for iteration
        Mods = [HCD_model_truth, HCD_model_r05, HCD_model_r08]
        Graphs = [A_truth, A_r05, A_r08]
        printing = ['fitting model using true input graph',
                    'fitting model using r > 0.5 input graph',
                    'fitting model using r > 0.8 input graph']
        
        #preallocate metrics
        metrics = []
        modularity = []
        recon_A = []
        recon_X = []
        print('...done')
        #fit the three models
        for i in range(0, 3):
            print("*"*80)
            print(printing[i])
            out = fit(Mods[i], X, Graphs[i], optimizer='Adam', epochs = 500, update_interval=50, 
                      lr = 1e-4, gamma = 0.5, delta = 1, comm_loss='Modularity',
                      true_labels=sort_true_labels, verbose=False, 
                      save_output=True, output_path=savepath)
            
            #record best losses and best performances
            total_loss = (np.array(out[-3])+np.array(out[-2]))-np.array(out[-4])
            best_loss_idx = total_loss.tolist().index(min(total_loss))
            perf = np.array(out[-1])
            best_perf_idx = perf.sum(axis = 1).tolist().index(max(perf.sum(axis = 1)))
            
            #update lists
            modularity.append(np.round(out[-4][best_loss_idx], 4))
            recon_A.append(np.round(out[-3][best_loss_idx], 4))
            recon_X.append(np.round(out[-2][best_loss_idx], 4))
            metrics.append(out[-1][best_perf_idx])
            
            
            #output assigned labels for all layers
            S_sub, S_layer, S_all = trace_comms(out[4], comm_sizes)
            
            #compare true and predicted graph adjacency 
            A_pred = resort_graph(out[1].detach().numpy(), sort_indices)
            A_true = resort_graph(Graphs[i].detach().numpy(), sort_indices)
            fig, (ax1,ax2) = plt.subplots(2,2, figsize=(12,10))
            sbn.heatmap(A_pred, ax = ax1[0])
            sbn.heatmap(A_true, ax = ax1[1])
            
            df = gen_labels_df(S_layer, sort_true_labels, sort_indices)
            
            sbn.heatmap(df, ax = ax2[0])
            
            fig.savefig(savepath+'_heatmaps.pdf')
            
        #update performance table
        row_add = [modularity[0], modularity[1], modularity[2],
                   recon_A[0], recon_A[1], recon_A[2],
                   recon_X[0], recon_X[1], recon_X[2],
                   tuple(np.round(metrics[0], 4)), 
                   tuple(np.round(metrics[1], 4)),
                   tuple(np.round(metrics[2], 4))]
        
        print('saving performance statistics...')
        res_table.loc[idx] = row_add
        res_table.to_csv(savepath_main+'simulation_results.csv')
        print('done')
        
        
        
        
        
run_simulations()