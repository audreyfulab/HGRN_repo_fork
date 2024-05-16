# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:54:24 2024

@author: Bruin
"""


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import networkx as nx
import seaborn as sbn
import matplotlib.pyplot as plt
import sys
import pandas as pd
#sys.path.append('/mnt/ceph/jarredk/scGNN_for_genes/HC-GNN/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
#sys.path.append('/mnt/ceph/jarredk/scGNN_for_genes/gen_data')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/gen_data')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/scGNN_for_genes/HC-GNN/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from Simulate import simulate_graph
from simulation_utilities import compute_graph_STATs, sort_labels
from utilities import get_input_graph
#import os
#os.chdir('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Bethe Hessian Tests/')
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
# general
import random as rd
from itertools import product
from tqdm import tqdm
import time
from sklearn.manifold import TSNE
import pickle
from sklearn.decomposition import PCA
# from model_layer import gaeGAT_layer as GAT
# from model import GATE, CommClassifer, HCD
from model import GATE, CommClassifer, HCD
from train import CustomDataset, batch_data, fit
from simulation_utilities import compute_modularity, post_hoc_embedding, compute_beth_hess_comms
from utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df, LoadData, build_true_graph, get_input_graph, plot_nodes, plot_clust_heatmaps
import seaborn as sbn
import matplotlib.pyplot as plt
from community import community_louvain as cl
from itertools import product, chain
from tqdm import tqdm
import pdb
import ast
import random as rd
from sklearn.manifold import TSNE
import pickle
from sklearn.decomposition import PCA

rd.seed(123)
torch.manual_seed(123)
#set seed
#rd.seed(333)






def run_simulations(args, save_results = False, which_net = 0, which_ingraph=0, gam = 1, delt = 1, 
                    lam = 1, learn_rate = 1e-4, epochs = 10, updates = 10, reso = [1,1], 
                    hd = [256, 128, 64], use_true_comms =True, cms = [], savepath = '',
                    activation = 'LeakyReLU', use_gpu = True, verbose = True,
                    TOAL = False, return_result = ['best_perf_top', 'best_perf_mid'],**kwargs):
    
    device = 'cuda:'+str(0) if use_gpu and torch.cuda.is_available() else 'cpu'
    
    print('*********** using DEVICE: {} **************'.format(device))
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    #loadpath_main = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/'
    #savepath_main ='/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/'
    
    
    pe, gexp, nodes, edges, nx_all, adj_all, args.savepath, nodelabs, orin = simulate_graph(args)
    
    true_adj_undi = build_true_graph(args.savepath+'.npz')
    
    pdadj = pd.DataFrame(adj_all[-1])
    pdadj.to_csv(args.savepath+'example_adjacency_matrix.csv')


    in_graph05, in_adj05 = get_input_graph(X = pe, 
                                           method = 'Correlation', 
                                           r_cutoff = 0.5)

    pdadj2 = pd.DataFrame(in_adj05)
    pdadj2.to_csv(args.savepath+'example_adjacency_matrix_corrgraph.csv')

    indices_top, indices_mid, labels_df, sorted_true_labels_top, sorted_true_labels_middle = sort_labels(nodelabs)


    #C = F.one_hot(torch.Tensor(sorted_true_labels_top).to(torch.int64)).to(torch.float32)
    #X = torch.Tensor(pe_sorted).requires_grad_()
    origin_nodes = [i[0] for i in orin]
    if args.layers > 2:
        pe_sorted = pe[indices_mid,:]
        idx = [indices_mid.index(i) for i in origin_nodes]
    else:
        pe_sorted = pe[indices_top,:]
        idx = [indices_mid.index(i) for i in origin_nodes]

    labels = [sorted_true_labels_middle, sorted_true_labels_top]
    TSNE_embed=TSNE(n_components=3, 
                    learning_rate='auto',
                    init='random', 
                    perplexity=3).fit_transform(pe_sorted)
    PCs = PCA(n_components=3).fit_transform(pe_sorted)

    which_labels = ['Middle', 'Top']
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    for i in range(0, len(labels)):
        
        ax[0][i].scatter(TSNE_embed[:,0], TSNE_embed[:,1], 
                         s = 150.0, c = labels[i], cmap = 'plasma')
        ax[0][i].scatter(TSNE_embed[idx,0], TSNE_embed[idx,1],
                     c='red', s = 300, marker = '.')
        ax[0][i].set_xlabel('Dimension 1')
        ax[0][i].set_ylabel('Dimension 2')
        ax[0][i].set_title( 't-SNE '+which_labels[i])
        #adding node labels
        ax[1][i].scatter(PCs[:,0], PCs[:,1], 
                         s = 150.0, c = labels[i], cmap = 'plasma')
        ax[1][i].scatter(PCs[idx,0], PCs[idx,1], 
        c='red', s = 300, marker = '.')
        ax[1][i].set_xlabel('Dimension 1')
        ax[1][i].set_ylabel('Dimension 2')
        ax[1][i].set_title( 'PCA '+which_labels[i])
        fig.savefig(args.savepath+'TSNE_PCA_GT_'+which_labels[i]+'.png', dpi = 300)
        
        
        
    if(args.common_dist == True):
        cdsetting = 'Common_dist'
    else:
        cdsetting = ''
    
    fig3dPCA = plt.figure(figsize=(14,12))
    ax3dPCA = plt.axes(projection='3d')
    ax3dPCA.scatter3D(PCs[:,0], PCs[:,1], PCs[:,2], 
                  c=labels[1], cmap='plasma')
    ax3dPCA.scatter3D(PCs[idx,0], PCs[idx,1], PCs[idx,2], 
                 c='red', s = 300, marker = '.',
                 depthshade = False)

    ax3dPCA.set_xlabel('Dimension 1')
    ax3dPCA.set_ylabel('Dimension 2')
    ax3dPCA.set_zlabel('Dimension 3')
    
    fig3dPCA.savefig(args.savepath+'3D_PCA_GT'+cdsetting+'.png', dpi = 300)




    fig3dTSNE = plt.figure(figsize=(14,12))
    ax3dTSNE = plt.axes(projection='3d')
    ax3dTSNE.scatter3D(TSNE_embed[:,0], TSNE_embed[:,1], TSNE_embed[:,2], 
                  c=labels[0], cmap='plasma')
    ax3dTSNE.scatter3D(TSNE_embed[idx,0], TSNE_embed[idx,1], TSNE_embed[idx,2], 
                 c='red', s = 300, marker = '.',
                 depthshade = False)

    ax3dTSNE.set_xlabel('Dimension 1')
    ax3dTSNE.set_ylabel('Dimension 2')
    ax3dTSNE.set_zlabel('Dimension 3')
    

    fig3dTSNE.savefig(args.savepath+'3D_TSNE_GT'+cdsetting+'.png', dpi = 300)
    
    #loadpath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/Toy_examples/Intermediate_examples/OLD_1_23_2024/'
    #loadpath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/Single_sim_dpd_identity/'
    #savepath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/Single_sim_dpd_identity/'

    loadpath_main = args.savepath
    savepath_main = sp

    #preallocate results table
    res_table = pd.DataFrame(columns = ['Beth_Hessian_Comms',
                                        'Communities_Upper_Limit',
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
            'A_ingraph05/', 'A_ingraph07/']
    case_nm = ['A_ingraph_true','A_corr_no_cutoff','A_ingraph02', 
               'A_ingraph05', 'A_ingraph07']
    
    #run simulations
    #pdb.set_trace()
    lays = args.layers
    #extract and use true community sizes
    comm_sizes = cms
    #comm_sizes =[40,5]
 
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
        pe_sorted = pe[indices_mid,:]
            

    #generate input graphs for correlations r > 0.2, r > 0.5 and r > 0.8
    in_graph02, in_adj02 = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = 0.2)
    
    in_graph05, in_adj05 = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = 0.5)
    
    in_graph08, in_adj07 = get_input_graph(X = pe_sorted, 
                                       method = 'Correlation', 
                                       r_cutoff = 0.7)
    
    rmat = np.absolute(np.corrcoef(pe_sorted))
    in_graph_rmat = nx.from_numpy_array(rmat)
    #print network statistics
    
    #nodes and attributes
    nodes = pe.shape[0]
    attrib = pe.shape[1]
    #set up three separate models for true input graph, r > 0.5 input graph, and
    #r > 0.8 input graph scenarios
    print('-'*25+'setting up and fitting models'+'-'*25)

    HCD_model_truth = HCD(nodes, attrib, hidden_dims=hd, 
                          comm_sizes=comm_sizes, attn_act=activation, **kwargs).to(device)
    HCD_model_rmat = HCD(nodes, attrib, hidden_dims=hd,
                         comm_sizes=comm_sizes, attn_act=activation, **kwargs).to(device)
    HCD_model_r02 = HCD(nodes, attrib, hidden_dims=hd, 
                        comm_sizes=comm_sizes, attn_act=activation, **kwargs).to(device)
    HCD_model_r05 = HCD(nodes, attrib, hidden_dims=hd, 
                        comm_sizes=comm_sizes, attn_act=activation, **kwargs).to(device)
    HCD_model_r07 = HCD(nodes, attrib, hidden_dims=hd, 
                        comm_sizes=comm_sizes, attn_act=activation, **kwargs).to(device)
    
    #set attribute and input graph(s) to torch tensors with grad attached
    X = torch.Tensor(pe_sorted).requires_grad_()
    #three input graph scenarios -- add self loops
    A_truth = torch.Tensor(true_adj_undi[:nodes,:nodes]).requires_grad_()+torch.eye(nodes)
    A_rmat = torch.Tensor(rmat).requires_grad_()
    A_r02 = torch.Tensor(in_adj02).requires_grad_()+torch.eye(nodes)
    A_r05 = torch.Tensor(in_adj05).requires_grad_()+torch.eye(nodes)
    A_r07 = torch.Tensor(in_adj07).requires_grad_()+torch.eye(nodes)
    
    #combine input items into lists for iteration
    Mods = [HCD_model_truth, HCD_model_rmat, HCD_model_r02, 
            HCD_model_r05, HCD_model_r07]
    Graphs = [A_truth, A_rmat, A_r02, A_r05, A_r07]
    printing = ['fitting model using true input graph',
                'fitting model using r matrix as input graph',
                'fitting model using r > 0.2 input graph',
                'fitting model using r > 0.5 input graph',
                'fitting model using r > 0.7 input graph']
    
    #preallocate metrics
    metrics = []
    comm_loss = []
    recon_A = []
    recon_X = []
    predicted_comms = []
    louv_mod = []
    louv_num_comms = []
    print('...done')
    #fit the three models
    for i in range(0, 5):
        
        if i == which_ingraph:
            print("*"*80)
            print(printing[i])
            out = fit(Mods[i], X, Graphs[i], 
                      optimizer='Adam', 
                      epochs = epochs, 
                      update_interval=updates, 
                      layer_resolutions=reso,
                      lr = learn_rate, 
                      gamma = gam, 
                      delta = delt, 
                      lamb = lam, 
                      true_labels = target_labels, 
                      verbose=verbose, 
                      save_output=save_results, 
                      turn_off_A_loss= TOAL,
                      output_path=savepath_main, 
                      ns = 25, 
                      fs = 10)
            
            #record best losses and best performances
            #pdb.set_trace()
            total_loss = np.array(out[-5])
            best_loss_idx = total_loss.tolist().index(min(total_loss))
            #perf_top = np.array([list(chain.from_iterable(i)) for i in temp_top])
            #perf_mid = np.array([list(chain.from_iterable(i)) for i in temp_mid])
            
            
            if return_result == 'best_perf_top':
                temp_top = [i[0] for i in out[-1]]
                perf_top = np.array(temp_top)
                best_perf_idx = perf_top[:,2].tolist().index(max(perf_top[:,2]))
                print('Best Performance Top Layer: Epoch = {}, \nHomogeneity = {},\nCompleteness = {}, \nNMI = {}'.format(
                best_perf_idx,      
                perf_top[best_perf_idx, 0],
                perf_top[best_perf_idx, 1],
                perf_top[best_perf_idx, 2]
                ))
            else:
                temp_mid = [i[1] for i in out[-1]]
                perf_mid = np.array(temp_mid)
                best_perf_idx = perf_mid[:,2].tolist().index(max(perf_mid[:,2]))
                print('Best Performance Middle Layer: Epoch = {}, \nHomogeneity = {},\nCompleteness = {}, \nNMI = {}'.format(
                    best_loss_idx,
                    perf_mid[best_perf_idx, 0],
                    perf_mid[best_perf_idx, 1],
                    perf_mid[best_perf_idx, 2]
                    ))
            #update lists
            comm_loss.append(np.round(out[-4][best_loss_idx], 4))
            recon_A.append(np.round(out[-3][best_loss_idx], 4))
            recon_X.append(np.round(out[-2][best_loss_idx], 4))
    
            
            #compute the upper limit of communities, the beth hessian, and max modularity
            upper_limit = torch.sqrt(torch.sum(Graphs[i]-torch.eye(nodes)))
            beth_hessian = compute_beth_hess_comms((Graphs[i]-torch.eye(nodes)).cpu().detach().numpy())
            max_modularity = 1 - (2/upper_limit)
    
            #output assigned labels for all layers
            S_sub, S_layer, S_all = trace_comms(out[6], comm_sizes)
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
            fig.savefig(savepath_main+'Louvain_results.pdf')
            plot_nodes((Graphs[i]-torch.eye(nodes)).cpu().detach().numpy(), 
                       labels = np.array(louv_preds), 
                       path = savepath_main+'Louvain_graph_'+case_nm[i], 
                       node_size = 25, 
                       font_size = 10, 
                       add_labels = True,
                       save = True)
            #update performance table
            row_add = [beth_hessian,
                       np.round(upper_limit.cpu().detach().numpy()),
                       np.round(max_modularity.cpu().detach().numpy(),4),
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
            tables[i].loc[0] = row_add
            print('*'*80)
            print(tables[i].loc[0])
            print('*'*80)
            if save_results == True:
                tables[i].to_csv(savepath_main+'Simulation_Results_'+case_nm[i]+'.csv')
                with open(savepath_main+'Simulation_Results_'+'OUTPUT'+'.pkl', 'wb') as f:
                    pickle.dump(out, f)
                
    print('done')
    return out, tables, Graphs, X, target_labels, S_all, S_sub, louv_preds, [best_perf_idx, best_loss_idx]
            


# simulation default arguments
parser.add_argument('--connect', dest='connect', default='disc', type=str)
parser.add_argument('--connect_prob', dest='connect_prob', default='use_baseline', type=str)
parser.add_argument('--toplayer_connect_prob', dest='toplayer_connect_prob', default=0.3, type=float)
parser.add_argument('--top_layer_nodes', dest='top_layer_nodes', default=5, type=int)
parser.add_argument('--subgraph_type', dest='subgraph_type', default='small world', type=str)
parser.add_argument('--subgraph_prob', dest='subgraph_prob', default=0.05, type=float)
parser.add_argument('--nodes_per_super2', dest='nodes_per_super2', default=(3,3), type=tuple)
parser.add_argument('--nodes_per_super3', dest='nodes_per_super3', default=(20,20), type=tuple)
parser.add_argument('--node_degree', dest='node_degree', default=5, type=int)
parser.add_argument('--sample_size',dest='sample_size', default = 500, type=int)
parser.add_argument('--layers',dest='layers', default = 2, type=int)
parser.add_argument('--SD',dest='SD', default = 0.1, type=float)
parser.add_argument('--common_dist', dest='common_dist',default = True, type=bool)
parser.add_argument('--seed_number', dest='seed_number',default = 555, type=int)
args = parser.parse_args()
rd.seed(555)
torch.manual_seed(555)

# args.connect = 'full'
# args.toplayer_connect_prob = 0.3
args.connect_prob = 0.01
args.common_dist = False
args.subgraph_prob = 0.05

args.SD = 0.1
args.node_degree = 3
args.force_connect = True
args.layers = 3
args.connect = 'disc'
args.subgraph_type = 'small world'
#args.savepath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/Single_sim_dpd_identity/'

args.savepath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/unequal_dists/'
#args.savepath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_4_15_2024/example8_output/'


simulation_dict = {'SD': args.SD,
                   'Layers': args.layers,
                   'Connection': args.connect,
                   'Type': args.subgraph_type,
                   'same_parent_distribution': args.common_dist,
                   'Node_degree': args.node_degree,
                   'subgraph_connect_prob': args.subgraph_prob,
                   'top_layer_connect_prob': args.connect_prob,
                   'top_layer_nodes': args.top_layer_nodes,
                   'mid_layer_nodes': 15,
                   'bottom_layer_nodes': 300,
                   'sample_size': args.sample_size,
                   'Seed': 555}

sim_df = pd.DataFrame.from_dict([simulation_dict])
sim_df.to_csv(args.savepath+'simulation_params.csv')
    
#sp = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_4_15_2024/example6_output/'
sp = args.savepath
# smw simple data: [true clusts]
# smw complex: []
# sfr simple data: [128, 5] atn: 20 heads, lam = [0.1, 0.1]
ep = 500
wn = 0
wg = 0
saveit = True
#sp = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/'
#sp = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test2/'


#disconnected settings
param_dict = {'epochs': ep, 
              'hidden': [256, 128, 64, 32],
              'gamma': 1e-4,
              'delta': 1e-3,
              'lambda': [1e-4, 1e-5],
              'input_graph': wg,
              'network': wn,
              'learning_rate': 1e-5,
              'true_comms': True,
              'cms': [15, 5],
              'use_attn': True,
              'attn_heads': 10,
              'normalize': True}

out, res, graphs, data, truth, preds, preds_sub, louv_pred, idx = run_simulations(args = args,
    save_results=saveit,
    savepath = sp,
    which_net=wn,
    which_ingraph=wg,
    reso=[1,1,1,1],
    hd=param_dict['hidden'],
    gam = param_dict['gamma'],
    delt = param_dict['delta'],
    lam = param_dict['lambda'],
    learn_rate=param_dict['learning_rate'],
    use_true_comms=param_dict['true_comms'],
    cms=param_dict['cms'],
    epochs = param_dict['epochs'],
    updates = param_dict['epochs'],
    activation = 'Sigmoid',
    TOAL=False,
    use_multi_head=param_dict['use_attn'],
    attn_heads=param_dict['attn_heads'],
    verbose = True,
    return_result = 'best_perf_top',
    normalize_inputs=param_dict['normalize'])

bp, bl = idx


params_df = pd.DataFrame([['Hidden Layer Dimensions: ', str(param_dict['hidden'])],
                          ['X_loss: ', param_dict['gamma']],
                          ['Modularity_loss: ', param_dict['delta']],
                          ['Clustering_loss: ', str(param_dict['lambda'])],
                          ['Learning_rate: ', str(param_dict['learning_rate'])],
                          ['Comms Layers: ', str(param_dict['cms'])],
                          ['Epochs: ', str(param_dict['epochs'])],
                          ['Multi Head Attn: ', str(True)],
                          ['Use Attention:', str(param_dict['use_attn'])],
                          ['Attn. Heads: ', str(param_dict['attn_heads'])],
                          ['Normalize_inputs: ',str(param_dict['normalize'])]],
                          columns=['Parameter','Value'])

params_df.to_csv(sp+'param_settings.csv')


print('Louvain compared to middle')
homo_l2m, comp_l2m, nmi_l2m = node_clust_eval(truth[1], louv_pred)
print('Louvain compared to top')
homo_l2t, comp_l2t, nmi_l2t =node_clust_eval(truth[0], louv_pred)

print('*'*50)
print('*'*50)
print('*'*50)


print('----------Middle preds to middle truth----------')
homo_m2m, comp_m2m, nmi_m2m = node_clust_eval(true_labels=truth[::-1][0], 
                                  pred_labels = out[0][bp][-3][0], verbose=False)
print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
    homo_m2m, comp_m2m, nmi_m2m
    ))


print('----------Middle preds to top truth----------')
homo_m2t, comp_m2t, nmi_m2t = node_clust_eval(true_labels=truth[::-1][1], 
                                  pred_labels = out[0][bp][-3][0], verbose=False)
print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
    homo_m2t, comp_m2t, nmi_m2t
    ))


# print('----------top preds to middle truth----------')
# homo, comp, nmi = node_clust_eval(true_labels=truth[::-1][1], 
#                                   pred_labels = out[0][bp][-2][1], verbose=False)
# print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
#     homo, comp, nmi
#     ))


print('----------top preds to top truth----------')
homo_t2t, comp_t2t, nmi_t2t = node_clust_eval(true_labels=truth[::-1][1], 
                                  pred_labels = out[0][bp][-3][1], verbose=False)
print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
    homo_t2t, comp_t2t, nmi_t2t
    ))

print('*'*50)
print('*'*50)
print('*'*50)


best_iter_metrics = pd.DataFrame([['louvain vs middle']+np.round([homo_l2m, comp_l2m, nmi_l2m],4).tolist(),
                                  ['louvain vs top']+np.round([homo_l2t, comp_l2t, nmi_l2t],4).tolist(),
                                  ['HCD: middle vs middle']+np.round([homo_m2m, comp_m2m, nmi_m2m],4).tolist(),
                                  ['HCD: middle vs top']+np.round([homo_m2t, comp_m2t, nmi_m2t],4).tolist(),
                                  ['HCD: top vs top']+np.round([homo_t2t, comp_t2t, nmi_t2t],4).tolist()],
                                 columns=['Comparison','Homogeneity','Completeness','NMI'])

if saveit:
    best_iter_metrics.to_csv(sp+'best_iteration_toplayer_metrics.csv')


# G = nx.from_numpy_array((out[0][bp][3][0]-torch.eye(data.shape[0])).cpu().detach().numpy())
# templabs = np.arange(0, data.shape[0])
# clust_labels = {list(G.nodes)[k]: templabs.tolist()[k] for k in range(len(truth[1]))}
# nx.draw_networkx(G, node_color = truth[1], 
#                  node_size = 30,
#                  font_size = 8,
#                  with_labels = False,
#                  labels = clust_labels,
#                  cmap = 'plasma')
#Top layer TSNE and PCA
post_hoc_embedding(graph=out[0][bp][3][0]-torch.eye(data.shape[0]), 
                        input_X = data,
                        data=data, 
                        probabilities=out[0][bp][4],
                        size = 150.0,
                        labels = out[0][bp][-3],
                        truth = truth[::-1],
                        fs=10,
                        node_size = 25, 
                        cm = 'plasma',
                        font_size = 10,
                        include_3d_plots = True,
                        save = saveit,
                        path = sp)


plot_clust_heatmaps(A = graphs[wg], 
                    A_pred = out[0][bp][3][0]-torch.eye(data.shape[0]), 
                    true_labels = truth, 
                    pred_labels = out[0][bp][-3], 
                    layers = 3, 
                    epoch = bp, 
                    save_plot = saveit, 
                    sp = sp+'best_iteration_'+str(bp))




   

#tsne
TSNE_data=TSNE(n_components=3, 
               learning_rate='auto',
               init='random', 
               perplexity=3).fit_transform(out[0][bp][2][0].detach().numpy())
#pca
PCs = PCA(n_components=3).fit_transform(out[0][bp][2][0].detach().numpy())


fig, (ax1, ax2) = plt.subplots(2,2, figsize = (12,10))
#tsne plot
ax1[0].scatter(TSNE_data[:,0], TSNE_data[:,1], s = 25, c = out[0][bp][-3][1], cmap = 'plasma')
ax1[0].set_xlabel('Dimension 1')
ax1[0].set_ylabel('Dimension 2')
ax1[0].set_title(' t-SNE Embedding Bottleneck (Predicted)')
#adding node labels
    
#PCA plot
ax1[1].scatter(PCs[:,0], PCs[:,1], s = 25, c = out[0][bp][-3][1], cmap = 'plasma')
ax1[1].set_xlabel('Dimension 1')
ax1[1].set_ylabel('Dimension 2')
ax1[1].set_title(' PCA Embedding-Bottleneck_(Predicted)')



x1 = torch.mm(out[0][bp][2][0], out[0][bp][2][1].transpose(0,1))

TSNE_data2=TSNE(n_components=3, 
               learning_rate='auto',
               init='random', 
               perplexity=3).fit_transform(x1.detach().numpy())
#pca
PCs2 = PCA(n_components=3).fit_transform(x1.detach().numpy())

#tsne plot
ax2[0].scatter(TSNE_data2[:,0], TSNE_data2[:,1], s = 25, c = out[0][bp][-3][1], cmap = 'plasma')
ax2[0].set_xlabel('Dimension 1')
ax2[0].set_ylabel('Dimension 2')
ax2[0].set_title(' t-SNE Embedding Comm1-projection (Predicted)')
#adding node labels
    
#PCA plot
ax2[1].scatter(PCs2[:,0], PCs2[:,1], s = 25, c = out[0][bp][-3][1], cmap = 'plasma')
ax2[1].set_xlabel('Dimension 1')
ax2[1].set_ylabel('Dimension 2')
ax2[1].set_title(' PCA Embedding-Comm1-projection (Predicted)')


