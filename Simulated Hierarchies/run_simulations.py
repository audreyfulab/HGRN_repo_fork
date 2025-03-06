# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:46:39 2023

@author: Bruin
"""
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
#import sys
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from model_layer import gaeGAT_layer as GAT
from model import GATE, CommClassifer, HCD
from train import CustomDataset, batch_data, fit
from simulation_utilities import compute_modularity, compute_beth_hess_comms, post_hoc_embedding
from utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df, LoadData, get_input_graph, plot_nodes
import seaborn as sbn
import matplotlib.pyplot as plt
from community import community_louvain as cl
from itertools import product, chain
from tqdm import tqdm
import pickle
import pdb
import ast
import random as rd
import gc
rd.seed(123)
torch.manual_seed(123)


def run_simulations(dataset = ['complex', 'intermediate','toy', 'cora', 'pubmed'],
                    parent_dist = ['equal','unequal'],
                    readpath = '/mnt/ceph/jarredk/',
                    save_results = False, gam = 1, delt = 1, lam = [1,1], learn_rate = 1e-4, 
                    epochs = 10, updates = 10, reso = [1,1], hd = [256, 128, 64], cms = [],
                    attn_heads = 1, activation = 'LeakyReLU', TOAL = False, true_comm_layers = True, 
                    sp = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/Simulation_Results/', 
                    use_gpu = True, **kwargs):
    
    device = 'cuda:'+str(0) if use_gpu and torch.cuda.is_available() else 'cpu'
    print('***** Using device {} ********'.format(device))
    if use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # set filename conventions
    nm = '_gam_'+str(gam)+'_delt_'+str(delt)+'_reso_'+str(reso[0])+'_'+str(reso[1])+'_lamda_'+str(lam[0])+'_'+str(lam[1])+'_TOAL_'+str(TOAL)

    savepath_main = sp
    
    if dataset in ['complex', 'intermediate', 'toy']
    # set filepath and settings grid
    if dataset == 'complex':
        #set data path
        loadpath_main = readpath+'HGRN_repo/Simulated_Hierarchies/DATA/'
        #set filepaths
        structpath = ['small_world/','scale_free/','random_graph/']
        connectpath = ['disconnected/', 'fully_connected/']
        layerpath = ['2_layer/', '3_layer/']
        noisepath = ['SD01/','SD05/']
        
        #set nm conventions
        struct_nm = ['smw_','sfr_','rdg_']
        connect_nm =['disc_', 'full_']
        layer_nm = ['2_layer_','3_layer_']
        noise_nm = ['SD01','SD05']
        
        #set parameters
        struct = ['small world','scale free','random graph']
        connect = ['disc', 'full']
        layers = [2, 3]
        noise = [0.1, 0.5]
        
        #set grids
        grid1 = product(structpath, connectpath, layerpath, noisepath)
        grid2 = product(struct_nm, connect_nm, layer_nm, noise_nm)
        grid3 = product(struct, connect, layers, noise)
        
        #read in network statistics 
        stats = pd.read_csv(loadpath_main+'network_statistics.csv')
        
    elif dataset == 'intermediate':
        if parent_dist == 'equal':
            loadpath_main = readpath+'HGRN_repo/Simulated_Hierarchies/DATA/Intermediate_examples/'
        else:
            loadpath_main = readpath+'HGRN_repo/Simulated_Hierarchies/DATA/Intermediate_examples_2/'
        structpath = ['small_world/','scale_free/','random_graph/']
        connectpath = ['disconnected/', 'fully_connected/']
        layerpath = ['3_layer/']

        struct_nm = ['smw_','sfr_','rdg_']
        connect_nm =['disc_', 'full_']
        layer_nm = ['3_layer_']
        
        
        struct = ['small world','scale free','random graph']
        connect = ['disc', 'full']
        layers = [3]
        
        
        #read in network statistics 
        stats = pd.read_csv(loadpath_main+'intermediate_examples_network_statistics.csv')
        #combine pathname and filename pieces
        grid1 = product(structpath, connectpath, layerpath)
        grid2 = product(struct_nm, connect_nm, layer_nm)
        grid3 = product(struct, connect, layers)
        
    elif dataset == 'toy':
        loadpath_main = readpath+'HGRN_repo/Simulated_Hierarchies/DATA/Toy_examples/'
        
        connect_nm =['disc_', 'full_']
        layer_nm = ['2_layer_','3_layer_']

        connect = ['disc', 'full']
        layers = [2, 3]

        #read in network statistics 
        stats = pd.read_csv(loadpath_main+'toy_examples_network_statistics.csv')
        #combine pathname and filename pieces
        grid1 = product(connectpath, layerpath)
        grid2 = product(connect_nm, layer_nm)
        grid3 = product(connect, layers)
    
    
    #preallocate results table
    res_table = pd.DataFrame(columns = ['Best Loss Epoch',
                                        'Best Perf Epoch',
                                        'Beth_Hessian_Comms',
                                        'Communities_Upper_Limit',
                                        'Max_Modularity',
                                        'Loss_Modularity',
                                        'Loss_Clustering',
                                        'Reconstruction_A',
                                        'Reconstruction_X', 
                                        'Metrics',
                                        'Number_Predicted_Comms',
                                        'Louvain_Modularity',
                                        'Louvain_Metrics',
                                        'Louvain_Predicted_comms'])
    tables = [res_table.copy(),res_table.copy(),res_table.copy(),res_table.copy(),
              res_table.copy()]
    
    #set filepath for input graphs
    case = ['A_ingraph_true/','A_corr_no_cutoff/','A_ingraph02/', 
            'A_ingraph05/', 'A_ingraph07/']
    
    #set naming conventions for input graphs
    case_nm = ['A_ingraph_true','A_corr_no_cutoff','A_ingraph02', 
               'A_ingraph05', 'A_ingraph07']
    
    #run simulations - enumerates over all networks in specific dataset
    for idx, value in enumerate(zip(grid1, grid2, grid3)):
        
        #extract network number of layers
        lays = value[2][2]
        #extract and use true community sizes
        npl = np.array(ast.literal_eval(stats.nodes_per_layer[idx])).tolist()
        if true_comm_layers: 
            comm_sizes = npl[::-1][1:]
        #use set community sizes
        else:
            comm_sizes = cms

        #set pathnames and read in simulated network
        print('-'*25+'loading in data'+'-'*25)
        loadpath = loadpath_main+''.join(value[0])+''.join(value[1])
        pe, true_adj_undi, indices_top, indices_middle, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle = LoadData(filename=loadpath)
        
        #combine target labels into list
        print('*** Read in expression data of dimension = {} ***'.format(pe.shape))
        if lays == 2:
            target_labels = [sorted_true_labels_top, []]
            #sort nodes in expression table 
            pe_sorted = pe[indices_top,:]
        else:
            target_labels = [sorted_true_labels_top, 
                             sorted_true_labels_middle]
            #sort nodes in expression table 
            pe_sorted = pe[indices_middle,:]

        #generate input graphs for correlations r, r > 0.2, r > 0.5 and r > 0.8
        in_graph02, in_adj02 = get_input_graph(X = pe_sorted, 
                                               method = 'Correlation', 
                                               r_cutoff = 0.2)
        
        in_graph05, in_adj05 = get_input_graph(X = pe_sorted, 
                                               method = 'Correlation', 
                                               r_cutoff = 0.5)
            
        in_graph07, in_adj07 = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = 0.7)
        #get correlation matrix
        rmat = np.absolute(np.corrcoef(pe_sorted))
        in_graph_rmat = nx.from_numpy_array(rmat)
        #print network statistics
        print('*** network statistics: ***')
        print(stats.loc[idx])
        print('*'*30)
        
        #nodes and attributes
        nodes = pe.shape[0]
        attrib = pe.shape[1]
        #set up 5 separate models for true input graph, r, r > 0.2, r > 0.5 and 
        #r > 0.8 input graph scenarios
        print('-'*25+'setting up and fitting models'+'-'*25)
        HCD_model_truth = HCD(nodes, attrib, hidden_dims=hd, 
                              comm_sizes=comm_sizes, attention_heads = attn_heads,
                              attn_act=activation).to(device)
        HCD_model_rmat = HCD(nodes, attrib, hidden_dims=hd,
                             comm_sizes=comm_sizes, attention_heads = attn_heads,
                             attn_act=activation).to(device)
        HCD_model_r02 = HCD(nodes, attrib, hidden_dims=hd, 
                            comm_sizes=comm_sizes, attention_heads = attn_heads,
                            attn_act=activation).to(device)
        HCD_model_r05 = HCD(nodes, attrib, hidden_dims=hd, 
                            comm_sizes=comm_sizes, attention_heads = attn_heads,
                            attn_act=activation).to(device)
        HCD_model_r07 = HCD(nodes, attrib, hidden_dims=hd, 
                            comm_sizes=comm_sizes, attention_heads = attn_heads,
                            attn_act=activation).to(device)
        
        #set attribute and input graph(s) to torch tensors with gradient
        X = torch.tensor(pe_sorted).requires_grad_()
        #-- add in self loops to input adjacency matrices
        A_truth = torch.tensor(true_adj_undi[:nodes,:nodes]).requires_grad_()+torch.eye(nodes)
        A_rmat = torch.tensor(rmat).requires_grad_()
        A_r02 = torch.tensor(in_adj02).requires_grad_()+torch.eye(nodes)
        A_r05 = torch.tensor(in_adj05).requires_grad_()+torch.eye(nodes)
        A_r07 = torch.tensor(in_adj07).requires_grad_()+torch.eye(nodes)
        
        #combine input items into lists for iteration:
        #models
        Mods = [HCD_model_truth, HCD_model_rmat, HCD_model_r02, 
                HCD_model_r05, HCD_model_r07]
        #graphs
        Graphs = [A_truth, A_rmat, A_r02, A_r05, A_r07]
        #updates
        printing = ['fitting model using true input graph',
                    'fitting model using r matrix as input graph',
                    'fitting model using r > 0.2 input graph',
                    'fitting model using r > 0.5 input graph',
                    'fitting model using r > 0.7 input graph']
        
        #preallocate lists for storing model fitting statistics 
        metrics = []
        comm_loss_mod = []
        comm_loss_clust = []
        recon_A = []
        recon_X = []
        predicted_comms = []
        louv_mod = []
        louv_num_comms = []
        print('...done')
        #fit the models
        for i in range(0, len(case_nm)):
            out = []
            print("*"*80)
            print(printing[i])
            savepath = savepath_main+''.join(value[0])+case[i]+''.join(value[1])
            out = fit(Mods[i], X, Graphs[i], optimizer='Adam', epochs = epochs, 
                      update_interval=updates, 
                      layer_resolutions=reso,
                      lr = learn_rate, 
                      gamma = gam, 
                      delta = delt, 
                      lamb = lam,
                      true_labels = target_labels, 
                      verbose=False, 
                      save_output=save_results,
                      turn_off_A_loss= TOAL,
                      output_path=savepath+nm+'_'+case_nm[i]+'_',
                      ns = 25,
                      fs = 10)
                
            #record best losses and best performances
            #pdb.set_trace()
            total_loss = np.array(out[-5])
            best_loss_idx = total_loss.tolist().index(min(total_loss))
            perf = np.array([list(chain.from_iterable(i)) for i in out[-1]])
            best_perf_idx = perf.sum(axis = 1).tolist().index(max(perf.sum(axis = 1)))
            
            #update lists
            comm_loss_mod.append(np.round(out[-6][best_loss_idx], 4))
            comm_loss_clust.append(np.round(out[-4][best_loss_idx], 4))
            recon_A.append(np.round(out[-3][best_loss_idx], 4))
            recon_X.append(np.round(out[-2][best_loss_idx], 4))
    
            
            #compute the upper limit of communities, the beth hessian, and max modularity
            upper_limit = torch.sqrt(torch.sum(Graphs[i]-torch.eye(nodes)))
            beth_hessian = compute_beth_hess_comms((Graphs[i]-torch.eye(nodes)).cpu().detach().numpy())
            max_modularity = 1 - (2/upper_limit)
            
            #output assigned labels for all layers
            S_sub, S_layer, S_all = trace_comms([i.cpu().clone() for i in out[6]], comm_sizes)
            predicted_comms.append(tuple([len(np.unique(i)) for i in S_layer]))
            
            #get prediction using louvain method
            comms = cl.best_partition(nx.from_numpy_array((Graphs[i]-torch.eye(nodes)).cpu().detach().numpy()))
            louv_mod = cl.modularity(comms, nx.from_numpy_array((Graphs[i]-torch.eye(nodes)).cpu().detach().numpy()))
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
            fig.savefig(savepath+'_'+case_nm[i]+'_Louvain_results.pdf')
            plot_nodes((Graphs[i]-torch.eye(nodes)).cpu().detach().numpy(), 
                       labels = np.array(louv_preds), 
                       path = savepath+'_Louvain_graph_'+case_nm[i], 
                       node_size = 25, 
                       font_size = 10, 
                       add_labels = True,
                       save = True)
            #update performance table
            row_add = [best_loss_idx,
                       best_perf_idx,
                       beth_hessian,
                       np.round(upper_limit.cpu().detach().numpy()),
                       np.round(max_modularity.cpu().detach().numpy(),4),
                       tuple(comm_loss_mod[-1].tolist()),
                       tuple(comm_loss_clust[-1].tolist()),
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
            #all_output[i].append(out)
            print('*'*80)
            print(tables[i].loc[idx])
            print('*'*80)
            if save_results == True:
                tables[i].to_csv(savepath_main+'Simulation_Results_'+case_nm[i]+'_'+nm+'.csv')
                with open(savepath_main+'Simulation_Results_'+''.join(value[1])+case_nm[i]+'_'+nm+'_OUTPUT'+'.pkl', 'wb') as f:
                    pickle.dump(out, f)
              
            print('Additional Plotting; TSNE and PCA')
            fig, ax = plt.subplots(figsize = (14,10))
            G = nx.from_numpy_array((Graphs[0]- torch.eye(X.shape[0])).cpu().detach().numpy())
            nx.draw_networkx(G, pos=nx.shell_layout(G), 
                             with_labels = True,
                             font_size = 10,
                             node_color = target_labels[0], 
                             ax = ax,
                             node_size = 100,
                             cmap = 'rainbow')
            
            fig2, ax2 = plt.subplots(figsize = (14,10))
            if len(target_labels)> 1:
                nx.draw_networkx(G, pos=nx.shell_layout(G), 
                                 with_labels = True,
                                 font_size = 10,
                                 node_size = 100,
                                 node_color = target_labels[1], 
                                 ax = ax2,
                                 cmap = 'rainbow')
            
            if save_results == True:
                fig.savefig(savepath+'_'+case_nm[i]+'_Circular_layout_truegraph_topclusts.png', dpi = 300)
                fig2.savefig(savepath+'_'+case_nm[i]+'_Circular_layout_truegraph_midclusts.png', dpi = 300)

            epoch = best_perf_idx
            #Top layer TSNE and PCA
            if lays > 2:
                tl = target_labels[::-1]
            else:
                tl = target_labels
            post_hoc_embedding(graph=out[0][epoch][3][0]-torch.eye(X.shape[0]), 
                                    input_X = X,
                                    embed=out[0][epoch][2][0], 
                                    probabilities=out[0][epoch][4],
                                    size = 150.0,
                                    labels = S_all, 
                                    truth = tl,
                                    fs=10,
                                    path = savepath+'_'+case_nm[i]+'_Best_perfromance_',
                                    save = save_results,
                                    node_size = 25, font_size = 10)
            

            del fig, ax, fig2, ax2, out, G
            plt.close('all')
            
        del Mods, HCD_model_truth, HCD_model_rmat, HCD_model_r02, HCD_model_r05, HCD_model_r07, X
        del pe, true_adj_undi, indices_top, indices_middle, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle
        del Graphs, A_truth, A_rmat, A_r02, A_r05, A_r07, in_graph02, in_adj02, in_graph05, in_adj05, in_graph07, in_adj07, rmat, in_graph_rmat
        gc.collect()
        torch.cuda.empty_cache()
    
    print('............................done...................................')
    #return out, tables, Graphs, X, target_labels, S_all, louv_preds, 




