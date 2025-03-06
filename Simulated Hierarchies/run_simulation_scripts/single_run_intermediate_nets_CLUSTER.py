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
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from model_layer import gaeGAT_layer as GAT
from model import GATE, CommClassifer, HCD
from train import CustomDataset, batch_data, fit
from simulation_utilities import compute_modularity, post_hoc_embedding, compute_beth_hess_comms
from utilities import resort_graph, trace_comms, node_clust_eval, gen_labels_df, LoadData, get_input_graph, plot_nodes, plot_clust_heatmaps
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



def run_simulations(save_results = False, which_net = 0, which_ingraph=1, gam = 1, delt = 1, 
                    lam = 1, learn_rate = 1e-4, epochs = 10, updates = 10, reso = [1,1], 
                    hd = [256, 128, 64], use_true_comms =True, cms = [], 
                    activation = 'LeakyReLU', use_gpu = True, verbose = True,
                    TOAL = False, return_result = ['best_perf_top', 'best_perf_mid'],**kwargs):
    
    device = 'cuda:'+str(0) if use_gpu and torch.cuda.is_available() else 'cpu'
    
    print('*********** using DEVICE: {} **************'.format(device))
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    loadpath_main = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/DATA/Toy_examples/Intermediate_examples/'
    savepath_main ='/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/DATA/Toy_examples/Intermediate_examples/Results/test/'
    
    #loadpath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/Toy_examples/Intermediate_examples/OLD_1_23_2024/'
    #loadpath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/'
    #savepath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/'
    
    structpath = ['small_world/','scale_free/','random_graph/']
    connectpath = ['disconnected/', 'fully_connected/']
    layerpath = ['3_layer/']
    #noisepath = ['SD01/','SD05/']


    struct_nm = ['smw_','sfr_','rdg_']
    connect_nm =['disc_', 'full_']
    layer_nm = ['3_layer_']
    #noise_nm = ['SD01','SD05']

    struct = ['small world','scale free','random graph']
    connect = ['disc', 'full']
    layers = [3]
    #noise = [0.1, 0.5]



    #read in network statistics 
    #stats = pd.read_csv('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/intermediate_examples_network_statistics.csv')
    stats = pd.read_csv('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/intermediate_examples_network_statistics.csv')
    #combine pathname and filename pieces
    grid1 = product(structpath, connectpath, layerpath)
    grid2 = product(struct_nm, connect_nm, layer_nm)
    grid3 = product(struct, connect, layers)

    
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
    for idx, value in enumerate(zip(grid1, grid2, grid3)):
        
        if idx == which_net:
            #pdb.set_trace()
            lays = value[2][1]
            #extract and use true community sizes
            if use_true_comms == True:
                npl = np.array(ast.literal_eval(stats.nodes_per_layer[idx])).tolist()
                comm_sizes = npl[::-1][1:]
            else:
                comm_sizes = cms
            #comm_sizes =[40,5]
                    
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
                
            in_graph08, in_adj07 = get_input_graph(X = pe_sorted, 
                                               method = 'Correlation', 
                                               r_cutoff = 0.7)
            
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
            X = torch.tensor(pe_sorted).requires_grad_()
            #three input graph scenarios -- add self loops
            A_truth = torch.tensor(true_adj_undi[:nodes,:nodes]).requires_grad_()+torch.eye(nodes)
            A_rmat = torch.tensor(rmat).requires_grad_()
            A_r02 = torch.tensor(in_adj02).requires_grad_()+torch.eye(nodes)
            A_r05 = torch.tensor(in_adj05).requires_grad_()+torch.eye(nodes)
            A_r07 = torch.tensor(in_adj07).requires_grad_()+torch.eye(nodes)
            
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
            sp = savepath_main+''.join(value[1])
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
                              output_path=sp, 
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
                    tables[i].loc[idx] = row_add
                    print('*'*80)
                    print(tables[i].loc[idx])
                    print('*'*80)
                    if save_results == True:
                        tables[i].to_csv(savepath_main+'Simulation_Results_'+case_nm[i]+'.csv')
                        with open(savepath_main+'Simulation_Results_'+'OUTPUT'+'.pkl', 'wb') as f:
                            pickle.dump(out, f)

    
    print('done')
    return out, tables, Graphs, X, target_labels, S_all, S_sub, louv_preds, [best_perf_idx, best_loss_idx]
            
    
#cms:
# smw simple data: [true clusts]
# smw complex: []
# sfr simple data: [128, 5] atn: 20 heads, lam = [0.1, 0.1]
ep = 500
wn = 1
wg = 0
out, res, graphs, data, truth, preds, preds_sub, louv_pred, idx = run_simulations(
    save_results=False,
    which_net=wn,
    which_ingraph=wg,
    reso=[1,1],
    hd=[256, 128, 64, 32],
    gam = 1,
    delt = 1,
    lam = [0.09,0.0001],
    learn_rate=1e-5,
    use_true_comms=True,
    cms=[50, 10],
    epochs = ep,
    updates = ep,
    activation = 'LeakyReLU',
    TOAL=False,
    use_multi_head=False,
    attn_heads=5,
    verbose = True,
    return_result = 'best_perf_top',
    normalize_inputs=True)

bp, bl = idx





print('Louvain compared to middle')
node_clust_eval(truth[1], louv_pred)
print('Louvain compared to top')
node_clust_eval(truth[0], louv_pred)

print('*'*50)
print('*'*50)
print('*'*50)


print('----------Middle preds to middle truth----------')
homo, comp, nmi = node_clust_eval(true_labels=truth[::-1][0], 
                                  pred_labels = out[0][bp][-2][0], verbose=False)
print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
    homo, comp, nmi
    ))


print('----------Middle preds to top truth----------')
homo, comp, nmi = node_clust_eval(true_labels=truth[::-1][1], 
                                  pred_labels = out[0][bp][-2][0], verbose=False)
print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
    homo, comp, nmi
    ))


print('----------top preds to middle truth----------')
homo, comp, nmi = node_clust_eval(true_labels=truth[::-1][1], 
                                  pred_labels = out[0][bp][-2][1], verbose=False)
print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
    homo, comp, nmi
    ))


print('----------top preds to top truth----------')
homo, comp, nmi = node_clust_eval(true_labels=truth[::-1][0], 
                                  pred_labels = out[0][bp][-2][1], verbose=False)
print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
    homo, comp, nmi
    ))

print('*'*50)
print('*'*50)
print('*'*50)
# fig, ax = plt.subplots(figsize = (12, 10))
# df1 = pd.DataFrame(np.array([preds[0].cpu().detach().numpy(), preds[1].cpu().detach().numpy(), 
#                     louv_pred, truth[1], truth[0]]).T, 
#                     columns = ['HCD Middle', 'HCD Top', 'Louvain', 'Truth Middle', 'Truth Top'])

# sbn.heatmap(df1, ax = ax)

# fig.savefig('')

# fig, ax = plt.subplots(figsize = (14,10))
# G = nx.from_numpy_array((graphs[0]- torch.eye(data.shape[0])).cpu().detach().numpy())
# nx.draw_networkx(G, pos=nx.shell_layout(G), 
#                   with_labels = True,
#                   font_size = 10,
#                   node_color = truth[0], 
#                   ax = ax,
#                   node_size = 100,
#                   cmap = 'rainbow')
# fig2, ax2 = plt.subplots(figsize = (14,10))
# nx.draw_networkx(G, pos=nx.shell_layout(G), 
#                   with_labels = True,
#                   font_size = 10,
#                   node_size = 150,
#                   node_color = truth[1], 
#                   ax = ax2,
#                   cmap = 'tab20')

epoch = ep-1
#Top layer TSNE and PCA
# post_hoc_embedding(graph=out[0][epoch][3][0]-torch.eye(data.shape[0]), 
#                         input_X = data,
#                         embed=out[0][epoch][2][0], 
#                         probabilities=out[0][epoch][4],
#                         size = 150.0,
#                         labels = preds,
#                         truth = truth[::-1],
#                         fs=10,
#                         path = '', 
#                         node_size = 25, 
#                         font_size = 10)


G = nx.from_numpy_array((out[0][bp][3][0]-torch.eye(data.shape[0])).cpu().detach().numpy())
templabs = np.arange(0, data.shape[0])
clust_labels = {list(G.nodes)[k]: templabs.tolist()[k] for k in range(len(truth[1]))}
nx.draw_networkx(G, node_color = truth[1], 
                 node_size = 30,
                 font_size = 8,
                 with_labels = False,
                 labels = clust_labels,
                 cmap = 'plasma')
#Top layer TSNE and PCA
post_hoc_embedding(graph=out[0][bp][3][0]-torch.eye(data.shape[0]), 
                        input_X = data,
                        data=data, 
                        probabilities=out[0][bp][4],
                        size = 150.0,
                        labels = out[0][bp][-2],
                        truth = truth[::-1],
                        fs=10,
                        node_size = 25, 
                        cm = 'plasma',
                        font_size = 10,
                        save = True,
                        path = '/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/')


plot_clust_heatmaps(A = graphs[wg], 
                    A_pred = out[0][bp][3][0]-torch.eye(data.shape[0]), 
                    true_labels = truth, 
                    pred_labels = out[0][bp][-2], 
                    layers = 3, 
                    epoch = bp, 
                    save_plot = True, 
                    sp = '/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/clust_maps_bp_')
#using louvain predictions
# post_hoc_embedding(graph=out[0][epoch][3][0]-torch.eye(data.shape[0]), 
#                         input_X = data,
#                         embed=data, 
#                         probabilities=out[0][epoch][4],
#                         size = 150.0,
#                         labels = [torch.tensor(louv_pred), torch.tensor(louv_pred)],
#                         truth = truth[::-1],
#                         fs=10,
#                         path = '', 
#                         node_size = 25, 
#                         font_size = 10)



# adj = (out[0][epoch][3][0]-torch.eye(data.shape[0])).cpu().detach().numpy()
# TSNE_embed=TSNE(n_components=2, 
#                 learning_rate='auto',
#                 init='random', 
#                 perplexity=3).fit_transform(data)
# PCs = PCA(n_components=2).fit_transform(data.cpu().detach().numpy())

# plt.scatter(TSNE_embed[:,0], TSNE_embed[:,1], s = 150.0, c = truth[0], cmap = 'plasma')
# plt.scatter(PCs[:,0], PCs[:,1], s = 150.0, c = preds[0], cmap = 'plasma')
# #pca
# PCs = PCA(n_components=2).fit_transform(adj)
# #node labels
# nl = np.arange(TSNE_embed.shape[0])
# #figs
# fig, (ax1, ax2) = plt.subplots(2,2, figsize = (12,10))
# #tsne plot
# ax1[0].scatter(TSNE_embed[:,0], TSNE_embed[:,1], s = 150.0, c = preds[0], cmap = 'plasma')
# ax1[0].set_xlabel('Dimension 1')
# ax1[0].set_ylabel('Dimension 2')
# ax1[0].set_title( 'TSNE Embeddings (Predicted)')
# #adding node labels
    
# #PCA plot
# ax1[1].scatter(PCs[:,0], PCs[:,1], s = 150.0, c = preds[0], cmap = 'plasma')
# ax1[1].set_xlabel('Dimension 1')
# ax1[1].set_ylabel('Dimension 2')
# ax1[1].set_title(' PCA Embeddings (Predicted)')
# #adding traced_labels
    

# #TSNE and PCA plots using true cluster labels
# #tsne truth
# ax2[0].scatter(TSNE_embed[:,0], TSNE_embed[:,1], s = 150.0, c = truth[::-1][0], cmap = 'plasma')
# ax2[0].set_xlabel('Dimension 1')
# ax2[0].set_ylabel('Dimension 2')
# ax2[0].set_title(layer_nms[i]+' TSNE Embeddings (Truth)')

    
# #pca truth
# ax2[1].scatter(PCs[:,0], PCs[:,1], s = size, c = truth[i], cmap = cm)
# ax2[1].set_xlabel('Dimension 1')
# ax2[1].set_ylabel('Dimension 2')
# ax2[1].set_title(layer_nms[i]+' PCA Embeddings (Truth)')

# if len(preds) > 1:
#     #middle layer TSNE and PCA
#     post_hoc_embedding(graph=out[0][epoch][3][0]-torch.eye(data.shape[0]), 
#                         input_X = data,
#                         embed=out[0][epoch][2][0], 
#                         probabilities=out[0][epoch][4][0],
#                         size = 150.0,
#                         labels = out[0][epoch][-3][0], 
#                         truth = truth[1],
#                         fs=10,
#                         path = '', node_size = 25, font_size = 8)

# #max_epoch = 10
# #iteri = range(0, max_epoch)
# epoch_step = 20
# iteri = np.arange(0, 100, epoch_step)
# print(iteri)
# for epoch in iteri:
#     post_hoc_embedding(graph=out[0][epoch][3][0]-torch.eye(data.shape[0]), 
#                        input_X = data,
#                        embed=out[0][epoch][2][0], 
#                        probabilities=out[0][epoch][4],
#                        size = 150.0,
#                        labels = out[0][epoch][-3][0], 
#                        truth = truth[1],
#                        fs=18,
#                        path = '', node_size = 25, font_size = 10)

# titles = ['True Graph','Correlation Matrix', 'r > 0.2', 'r > 0.5', 'r > 0.7']
# fig, axes = plt.subplots(3, 2, figsize = (12, 14))
# plt.subplots_adjust(wspace=0.01, hspace=0.1)
# for idx, ax in enumerate(axes.flat):
#     if idx < 5:
#         im = sbn.heatmap((graphs[idx]-torch.eye(data.shape[0])).cpu().detach().numpy(), 
#                          vmin=0, vmax=1, cmap = 'hot', yticklabels=False, 
#                          xticklabels=False, cbar = False, ax = ax)
#         ax.set_title(titles[idx], fontsize = 16)

# fig.delaxes(axes[2][1])
# #fig.subplots_adjust(right=0.8)
# #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# #fig.colorbar(np.random.uniform(size = 10), cax=cbar_ax)
# #fig.delaxes()
# fig.savefig('C:/Users/Bruin/Desktop/graph_sparsity.png', dpi = 500)

# fig, ax = plt.subplots(figsize = (14, 12))
# sbn.heatmap((graphs[1]-torch.eye(data.shape[0])).cpu().detach().numpy(),ax = ax)










