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
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import os
from sklearn.preprocessing import LabelBinarizer


rd.seed(123)
torch.manual_seed(123)



def run_simulations(save_results = False, gam = 1, delt = 1, 
                    lam = 1, learn_rate = 1e-4, epochs = 10, updates = 10, reso = [1,1], 
                    hd = [256, 128, 64], cms = [], verbose = False,
                    activation = 'LeakyReLU', use_gpu = True,
                    TOAL = False, return_result = ['best_perf_top', 'best_perf_mid'],**kwargs):
    
    device = 'cuda:'+str(0) if use_gpu and torch.cuda.is_available() else 'cpu'
    
    print('*********** using DEVICE: {} **************'.format(device))
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    #pathnames and filename conventions
    #mainpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/'
    #loadpath_main = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/'
    #savepath_main ='/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/'
    

    savepath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/cora/test/'
    
    data_dir = "C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/cora"
    #os.makedirs(data_dir, exist_ok=True)

    #dataset = Planetoid(root=data_dir, name='cora')
    #data = dataset[0]
    
    print('Reading In Cora dataset...')
    edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
    edgelist["label"] = "cites"
    print(edgelist.head())
    
    graph = nx.to_numpy_array(nx.from_pandas_edgelist(edgelist))
    
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names =  feature_names + ["subject"]
    node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
    
    
    #encode labels
    labels = node_data['subject']
    lb = LabelBinarizer()
    lb.fit(labels)
    target_labels = [lb.transform(labels).argmax(1)]
    
    x_array = np.array(node_data[node_data.columns[:-1]])
    X = torch.Tensor(x_array).requires_grad_()
    
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
        
    comm_sizes = cms
    lays = len(comm_sizes)
                
    #nodes and attributes
    nodes = x_array.shape[0]
    attrib = x_array.shape[1]


    HCD_model = HCD(nodes, attrib, hidden_dims=hd, 
                    comm_sizes=comm_sizes, 
                    attn_act=activation, 
                    **kwargs).to(device)

    A = torch.Tensor(graph).requires_grad_()+torch.eye(nodes)

    
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
    sp = savepath_main
    print("*"*80)
    out = fit(HCD_model, X, A, 
              optimizer='Adam', 
              epochs = epochs, 
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
    upper_limit = torch.sqrt(torch.sum(A-torch.eye(nodes)))
    beth_hessian = compute_beth_hess_comms((A-torch.eye(nodes)).cpu().detach().numpy())
    max_modularity = 1 - (2/upper_limit)
    
    #output assigned labels for all layers
    S_sub, S_layer, S_all = trace_comms(out[6], comm_sizes)
    predicted_comms.append(tuple([len(np.unique(i)) for i in S_layer]))
    
    #get prediction using louvain method
    comms = cl.best_partition(nx.from_numpy_array((A-torch.eye(nodes)).cpu().detach().numpy()))
    louv_mod = cl.modularity(comms, nx.from_numpy_array((A-torch.eye(nodes)).cpu().detach().numpy()))
    #extract cluster labels
    louv_preds = list(comms.values())
    louv_num_comms = len(np.unique(louv_preds))
    #make heatmap for louvain results and get metrics
    fig, ax = plt.subplots()
    #compute performance based on layers
    if lays == 1:
        metrics.append({'Top': tuple(np.round(out[-1][best_perf_idx][0], 4))})
        louv_metrics = {'Top': tuple(np.round(node_clust_eval(target_labels[0], 
                                                              np.array(louv_preds), 
                                                              verbose=False), 4))}
        sbn.heatmap(pd.DataFrame(np.array([louv_preds,  
                                           target_labels[0].tolist()]).T,
                                 columns = ['Louvain','Truth_Top']),
                    ax = ax)

    #make heatmap for louvain results
    fig.savefig(savepath_main+'Louvain_results.pdf')

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
    res_table.loc[0] = row_add
    print('*'*80)
    print(res_table)
    print('*'*80)
    print('done')
    return out, res_table, A, X, target_labels, S_all, S_sub, louv_preds, [best_perf_idx, best_loss_idx]
            
    

ep = 100
wn = 2
wg = 0
out, res, graphs, data, truth, preds, preds_sub, louv_pred, idx = run_simulations(
    save_results=False,
    reso=[1,1],
    hd=[512, 256, 128],
    gam = 1,
    delt = 1,
    lam = [0.5, 0.5],
    learn_rate=1e-3,
    cms=[6],
    epochs = ep,
    updates = ep,
    activation = 'LeakyReLU',
    TOAL=False,
    use_multi_head=False,
    attn_heads=3,
    return_result = 'best_perf_top')

bp, bl = idx





print('Louvain compared to middle')
node_clust_eval(truth[1], louv_pred)
print('Louvain compared to top')
node_clust_eval(truth[0], louv_pred)

print('*'*50)
print('*'*50)
print('*'*50)


print('----------Middle preds to middle truth----------')
homo, comp, nmi = node_clust_eval(true_labels=truth[::-1][1], 
                                  pred_labels = out[0][bp][-2][0], verbose=False)
print('Homogeneity = {}, \nCompleteness = {}, \nNMI = {}'.format(
    homo, comp, nmi
    ))


print('----------Middle preds to top truth----------')
homo, comp, nmi = node_clust_eval(true_labels=truth[::-1][0], 
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
                        embed=data, 
                        probabilities=out[0][bp][4],
                        size = 150.0,
                        labels = out[0][bp][-2],
                        truth = truth[::-1],
                        fs=10,
                        node_size = 25, 
                        cm = 'plasma',
                        font_size = 10,
                        save = False,
                        path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples/Results/test/')


plot_clust_heatmaps(A = graphs[wg], 
                    A_pred = out[0][bp][3][0]-torch.eye(data.shape[0]), 
                    true_labels = truth, 
                    pred_labels = out[0][bp][-2][::-1], 
                    layers = 3, 
                    epoch = bp, 
                    save_plot = False, 
                    sp = '')
#using louvain predictions
# post_hoc_embedding(graph=out[0][epoch][3][0]-torch.eye(data.shape[0]), 
#                         input_X = data,
#                         embed=data, 
#                         probabilities=out[0][epoch][4],
#                         size = 150.0,
#                         labels = [torch.Tensor(louv_pred), torch.Tensor(louv_pred)],
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










