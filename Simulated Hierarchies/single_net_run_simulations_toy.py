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
import networkx as nx
import sys
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from model_layer import gaeGAT_layer as GAT
from model import GATE, CommClassifer, HCD
from train import CustomDataset, batch_data, fit
from simulation_utilities import compute_modularity, post_hoc_embedding
from utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df, LoadData, get_input_graph, plot_nodes
import seaborn as sbn
import matplotlib.pyplot as plt
from community import community_louvain as cl
from itertools import product, chain
from tqdm import tqdm
import pdb
import ast
import random as rd
rd.seed(123)
torch.manual_seed(123)



def run_simulations(save_results = False, which_net = 0, which_ingraph=1, gam = 1, delt = 1, 
                    learn_rate = 1e-4, epochs = 10, updates = 10, reso = [1,1,], 
                    loss_fn = ['Modularity', 'Clustering'], activation = 'LeakyReLU'):
    
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    #loadpath_main = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/'
    #savepath_main ='/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/'
    loadpath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/Toy_examples/'
    savepath_main ='C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/test/'
    #structpath = ['small_world/','scale_free/','random_graph/']
    connectpath = ['disconnected/', 'fully_connected/']
    layerpath = ['2_layer/', '3_layer/']
    #noisepath = ['SD01/','SD05/']


    #struct_nm = ['smw_','sfr_','rdg_']
    connect_nm =['disc_', 'full_']
    layer_nm = ['2_layer_','3_layer_']
    #noise_nm = ['SD01','SD05']

    #struct = ['small world','scale free','random graph']
    connect = ['disc', 'full']
    layers = [2, 3]
    #noise = [0.1, 0.5]



    #read in network statistics 
    stats = pd.read_csv(loadpath_main+'toy_examples_network_statistics.csv')
    #combine pathname and filename pieces
    grid1 = product(connectpath, layerpath)
    grid2 = product(connect_nm, layer_nm)
    grid3 = product(connect, layers)
    
    #preallocate results table
    res_table = pd.DataFrame(columns = ['Communities_Upper_Limit',
                                        'Max_Modularity',
                                        'Modularity',
                                        'Reconstruction_A',
                                        'Reconstruction_X', 
                                        'Metrics',
                                        'Number_Predicted_Comms',
                                        'Louvain_Modularity',
                                        'Louvain_Metrics',
                                        'Louvain_Predicted_comms'])
    tables = [res_table.copy(),res_table.copy(),res_table.copy(),res_table.copy(),
              res_table.copy()]
    
    case = ['A_ingraph_true/','A_corr_no_cutoff/','A_ingraph02/', 
            'A_ingraph05/', 'A_ingraph08/']
    case_nm = ['A_ingraph_true','A_corr_no_cutoff','A_ingraph02', 
               'A_ingraph05', 'A_ingraph08']
    
    #run simulations
    for idx, value in enumerate(zip(grid1, grid2, grid3)):
        
        if idx == which_net:
            #pdb.set_trace()
            lays = value[2][1]
            #extract and use true community sizes
            npl = np.array(ast.literal_eval(stats.nodes_per_layer[idx])).tolist()
            if len(npl) == 2:    
                comm_sizes = npl[::-1][1:]
            else:
                comm_sizes = npl[::-1][1:]
                    
            #pdb.set_trace()
            #set pathnames and read in simulated network
            print('-'*25+'loading in data'+'-'*25)
            loadpath = loadpath_main+''.join(value[0])+''.join(value[1])
            #pdb.set_trace()
            pe, true_adj_undi, indices_top, indices_middle, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle = LoadData(filename=loadpath)
           
            #combine target labels into list
            print('Read in expression data of dimension = {}'.format(pe.shape))
            if lays == 2:
                target_labels = [sorted_true_labels_top, []]
                #sort nodes in expression table 
                pe_sorted = pe[indices_top,:]
            else:
                target_labels = [sorted_true_labels_top, 
                                 sorted_true_labels_middle]
                #sort nodes in expression table 
                pe_sorted = pe[indices_middle,:]
            

            #generate input graphs for correlations r > 0.2, r > 0.5 and r > 0.8
            in_graph02, in_adj02 = get_input_graph(X = pe_sorted, 
                                                   method = 'Correlation', 
                                                   r_cutoff = 0.2)
            
            in_graph05, in_adj05 = get_input_graph(X = pe_sorted, 
                                                   method = 'Correlation', 
                                                   r_cutoff = 0.5)
                
            in_graph08, in_adj08 = get_input_graph(X = pe_sorted, 
                                               method = 'Correlation', 
                                               r_cutoff = 0.6)
            
            rmat = np.absolute(np.corrcoef(pe_sorted))
            in_graph_rmat = nx.from_numpy_array(rmat)
            #print network statistics
            print('network statistics:')
            print(stats.loc[idx])
            print('...done')
                
            #nodes and attributes
            nodes = pe.shape[0]
            attrib = pe.shape[1]
            #set up three separate models for true input graph, r > 0.5 input graph, and
            #r > 0.8 input graph scenarios
            print('-'*25+'setting up and fitting models'+'-'*25)
            HCD_model_truth = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act=activation)
            HCD_model_rmat = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act=activation)
            HCD_model_r02 = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act=activation)
            HCD_model_r05 = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act=activation)
            HCD_model_r08 = HCD(nodes, attrib, comm_sizes=comm_sizes, attn_act=activation)
            
            #set attribute and input graph(s) to torch tensors with grad attached
            X = torch.Tensor(pe_sorted).requires_grad_()
            #three input graph scenarios -- add self loops
            A_truth = torch.Tensor(true_adj_undi[:nodes,:nodes]).requires_grad_()+torch.eye(nodes)
            A_rmat = torch.Tensor(rmat).requires_grad_()
            A_r02 = torch.Tensor(in_adj02).requires_grad_()+torch.eye(nodes)
            A_r05 = torch.Tensor(in_adj05).requires_grad_()+torch.eye(nodes)
            A_r08 = torch.Tensor(in_adj08).requires_grad_()+torch.eye(nodes)
            
            #combine input items into lists for iteration
            Mods = [HCD_model_truth, HCD_model_rmat, HCD_model_r02, 
                    HCD_model_r05, HCD_model_r08]
            Graphs = [A_truth, A_rmat, A_r02, A_r05, A_r08]
            printing = ['fitting model using true input graph',
                        'fitting model using r matrix as input graph',
                        'fitting model using r > 0.2 input graph',
                        'fitting model using r > 0.5 input graph',
                        'fitting model using r > 0.8 input graph']
            
            #preallocate metrics
            metrics = []
            comm_loss = []
            recon_A = []
            recon_X = []
            predicted_comms = []
            louv_mod = []
            louv_num_comms = []
            print('...done')
            sp = savepath_main
            #fit the three models
            for i in range(0, 5):
                
                if i == which_ingraph:
                    print("*"*80)
                    print(printing[i])
                    out = fit(Mods[i], X, Graphs[i], optimizer='Adam', epochs = epochs, 
                              update_interval=updates, layer_resolutions=reso,
                              lr = learn_rate, gamma = gam, delta = delt, comm_loss=loss_fn,
                              true_labels = target_labels, verbose=False, 
                              save_output=save_results, 
                              output_path=sp,
                              ns = 25,
                              fs = 10)
                        
                    #record best losses and best performances
                    #pdb.set_trace()
                    total_loss = np.array(out[-5])
                    best_loss_idx = total_loss.tolist().index(min(total_loss))
                    perf = np.array([list(chain.from_iterable(i)) for i in out[-1]])
                    best_perf_idx = perf.sum(axis = 1).tolist().index(max(perf.sum(axis = 1)))
                    
                    #update lists
                    comm_loss.append(np.round(out[-4][best_loss_idx], 4))
                    recon_A.append(np.round(out[-3][best_loss_idx], 4))
                    recon_X.append(np.round(out[-2][best_loss_idx], 4))
            
                    
                    #compute the upper limit of communities and modularity
                    upper_limit = torch.sqrt(torch.sum(Graphs[i]-torch.eye(nodes)))
                    max_modularity = 1 - (2/upper_limit)
                    
                    #output assigned labels for all layers
                    S_sub, S_layer, S_all = trace_comms(out[6], comm_sizes)
                    predicted_comms.append(tuple([len(np.unique(i)) for i in S_layer]))
                    
                    #get prediction using louvain method
                    comms = cl.best_partition(nx.from_numpy_array((Graphs[i]-torch.eye(nodes)).detach().numpy()))
                    louv_mod = cl.modularity(comms, nx.from_numpy_array((Graphs[i]-torch.eye(nodes)).detach().numpy()))
                    #extract cluster labels
                    louv_preds = list(comms.values())
                    louv_num_comms = len(np.unique(louv_preds))
                    #make heatmap for louvain results and get metrics
                    fig, ax = plt.subplots()
                    #compute performance based on layers
                    if lays == 2:
                        metrics.append({'Top': tuple(np.round(out[-1][best_perf_idx][0], 4))})
                        louv_metrics = {'Top': tuple(np.round(node_clust_eval(target_labels[0], 
                                                                     np.array(louv_preds), 
                                                                     verbose=False), 4))}
                        sbn.heatmap(pd.DataFrame(np.array([louv_preds,  
                                                           target_labels[0].tolist()]).T,
                                                 columns = ['Louvain','Truth_Top']),
                                    ax = ax)
                    else:
                        metrics.append({'Top': tuple(np.round(out[-1][best_perf_idx][0], 4)),
                                        'Middle': tuple(np.round(out[-1][best_perf_idx][-1], 4))})
                        lnm=['Top','Middle']
                        louv_metrics = []
                        for j in range(0, 2):
                            louv_metrics.append({lnm[j]: tuple(np.round(node_clust_eval(target_labels[j], 
                                                                                 np.array(louv_preds), 
                                                                                 verbose=False), 4))})
                            sbn.heatmap(pd.DataFrame(np.array([louv_preds, 
                                                               target_labels[1].tolist(), 
                                                               target_labels[0].tolist()]).T,
                                                     columns = ['Louvain','Truth_Middle','Truth_Top']),
                                        ax = ax)
                    #make heatmap for louvain results
                    fig.savefig(savepath_main+'Louvain_results.pdf')
                    plot_nodes((Graphs[i]-torch.eye(nodes)).detach().numpy(), 
                               labels = np.array(louv_preds), 
                               path = savepath_main+'Louvain_graph_'+case_nm[i], 
                               node_size = 25, 
                               font_size = 10, 
                               add_labels = True,
                               save = True)
                    #update performance table
                    row_add = [np.round(upper_limit.detach().numpy()),
                               np.round(max_modularity.detach().numpy(),4),
                               tuple(comm_loss[-1].tolist()), 
                               recon_A[-1], 
                               recon_X[-1],
                               metrics[-1], 
                               predicted_comms[-1],
                               np.round(louv_mod, 4),
                               louv_metrics,
                               louv_num_comms]
                    print(row_add)
                    print('updating performance statistics...')
                    tables[i].loc[idx] = row_add
                    print('*'*80)
                    print(tables[i].loc[idx])
                    print('*'*80)
                    if save_results == True:
                        tables[i].to_csv(savepath_main+'Toy_sim_results_'+case_nm[i]+'.csv')
    

    
    print('done')
    return out, tables, Graphs, X, target_labels
            
    


out, res, graphs, data, truth = run_simulations(save_results=True,
                                   which_net=3,
                                   which_ingraph=0,
                                   reso=[1,1],
                                   gam=1,
                                   delt=0, 
                                   learn_rate=1e-4,
                                   epochs = 50,
                                   updates = 50,
                                   loss_fn='Modularity')

#max_epoch = 10
#iteri = range(0, max_epoch)
epoch_step = 5
iteri = np.arange(0, 30, epoch_step)
print(iteri)
for epoch in iteri:
    post_hoc_embedding(graph=out[0][epoch][3][0]-torch.eye(18), 
                       input_X = data,
                       embed=out[0][epoch][2][0], 
                       probabilities=out[0][epoch][4],
                       size = 150.0,
                       labels = out[0][epoch][-3][0], 
                       truth = truth[1],
                       fs=18,
                       path = '', node_size = 25, font_size = 10)


