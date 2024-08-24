# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:17:44 2024

@author: Bruin
"""

import sys
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')

from utilities import pickle_data, open_pickled, LoadData
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

figsp = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_6_5_2024/'

#ex12path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_6_5_2024/example15_output/'
#ex13path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_6_5_2024/example16_output/'
#ex14path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_6_5_2024/example17_output/'

ex12path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_6_5_2024/example15_output'
ex13path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_6_5_2024/example16_output'
ex14path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Reports/Report_6_5_2024/example17_output'


loadpath_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/Toy_examples/Intermediate_examples_unique_dist/'

ex12name = 'small_world/fully_connected/3_layer/smw_full_3_layer_'
ex13name = 'scale_free/fully_connected/3_layer/sfr_full_3_layer_'
ex14name = 'random_graph/fully_connected/3_layer/rdg_full_3_layer_'

#load simulated networks
pe12, true_adj_undi12, indices_top12, indices_middle12, new_true_labels12, sorted_true_labels_top12, sorted_true_labels_middle12 = LoadData(filename=loadpath_main+ex12name)
pe13, true_adj_undi13, indices_top13, indices_middle13, new_true_labels13, sorted_true_labels_top13, sorted_true_labels_middle13 = LoadData(filename=loadpath_main+ex13name)
pe14, true_adj_undi14, indices_top14, indices_middle14, new_true_labels14, sorted_true_labels_top14, sorted_true_labels_middle14 = LoadData(filename=loadpath_main+ex14name)

#target labels and reording
target_labels12 = [sorted_true_labels_top12, 
                 sorted_true_labels_middle12]

target_labels13 = [sorted_true_labels_top13, 
                 sorted_true_labels_middle13]

target_labels14 = [sorted_true_labels_top14, 
                 sorted_true_labels_middle14]
#sort nodes in expression table 
pe_sorted12 = pe12[indices_middle12,:]
pe_sorted13 = pe13[indices_middle13,:]
pe_sorted14 = pe14[indices_middle14,:]

#load training output for examples
#example 12 - r >0.5 corr input graph, small world 3-layer-disconnected
#example 13 - r >0.5 corr input graph, scale free 3-layer-disconnected
#example 14 - r >0.5 corr input graph, random graph 3-layer-disconnected 
ex12data = open_pickled(ex12path+'/Simulation_Results_OUTPUT.pkl')
ex13data = open_pickled(ex13path+'/Simulation_Results_OUTPUT.pkl')
ex14data = open_pickled(ex14path+'/Simulation_Results_OUTPUT.pkl')

#pickled output is list with following items:
#[ all_out, X_final, A_final, X_all_final, A_all_final, P_all_final, S_final, 
# mod_loss_hist, loss_history, clust_loss_hist, A_loss_hist, X_loss_hist, perf_hist]

# all_out - (list) epoch specific outputs - contains the following elements:
#---------------------------------------------------------------------
#   [X_hat, A_hat, X_all, A_all, P_all, S_relab, S_all, S_sub, k]
#   X_hat - (tensor) epoch reconstructed attributes
#   A_hat - (tensor) epoch reconstructed graph
#   X_all - (list) all attributes: [input, reconstructed, community layers]
#   A_all - (list) all graphs: [input, reconstructed, community layers]
#   P_all - (list) all prediction probabilities: [middle layer, upper layer]
#   S_relab - (list) relabeled community predictions [middle layer, upper layer]
#   S_all - (list) community predictions (before retracing) [middle layer, upper layer]
#   k - (list) number of predicted communities: [middle layer, upper layer]

# X_final - (tensor) final reconstructed attributes at last epoch
# A_final - (tensor) final reconstructed adjacency at last epoch
# X_all_final - (list) final attributes (input + all layers) at last epoch
# A_all_final - (list) final adjancencies (input + all layers) at last epoch
# P_all_final - (list) final assignment probabilities (all_layers) at last epoch
# S_fianl - (list) final community assignments (all_layers) at last epoch
# mod_loss_history - (list) modularity training loss values (all epochs)
# loss_history - (list) training total loss history (all epochs)
# clust_loss_hist - (list) clustering loss history (all epochs)
# A_loss_history - (list) graph reconstruction loss history (all epochs)
# X_loss_history - (list) attribute reconstruction history (all epochs)
# perf_hist - (list) performance history (all epochs/all layers)



# index for best scoring epoch (on top layer)
bp12 = 50
bp13 = 22
bp14 = 25

#associated indices
allout_idx = 0
predgraph_idx = 1
all_graphs_idx = 3
preds_idx = 5
all_attr_idx = 2

#extract true graph adjacency matrix (located in all graphs list as first entry)
true_graph12 = ex12data[allout_idx][bp12][all_graphs_idx][0].detach().numpy()
true_graph13 = ex13data[allout_idx][bp13][all_graphs_idx][0].detach().numpy()
true_graph14 = ex14data[allout_idx][bp14][all_graphs_idx][0].detach().numpy()

# reconstructed graph at best epoch
pred_graph12 = ex12data[allout_idx][bp12][predgraph_idx].detach().numpy()
pred_graph13 = ex13data[allout_idx][bp13][predgraph_idx].detach().numpy()
pred_graph14 = ex14data[allout_idx][bp14][predgraph_idx].detach().numpy()

#embeddedings at bottleneck for best epoch
Xin12 = ex12data[allout_idx][bp12][all_attr_idx][0].detach().numpy()
Xin13 = ex13data[allout_idx][bp13][all_attr_idx][0].detach().numpy()
Xin14 = ex14data[allout_idx][bp14][all_attr_idx][0].detach().numpy()

#correlation matrices on embedded attributes
corrsembed12 = np.corrcoef(Xin12)
corrsembed13 = np.corrcoef(Xin13)
corrsembed14 = np.corrcoef(Xin14)


#graphs
fig, (ax1, ax2)  = plt.subplots(2, 3, figsize=(18,10))
plt.tight_layout(rect=[0, 0, 1, 0.95])
sbn.heatmap(true_graph12, yticklabels='none', xticklabels='none', ax = ax1[0])
sbn.heatmap(pred_graph12, yticklabels='none', xticklabels='none', ax = ax2[0])
ax1[0].set_title('Small World: True Graph')
ax2[0].set_title('Reconstructed Graph')

sbn.heatmap(true_graph13, yticklabels='none', xticklabels='none', ax = ax1[1])
sbn.heatmap(pred_graph13, yticklabels='none', xticklabels='none', ax = ax2[1])
ax1[1].set_title('Scale Free: True Graph')
ax2[1].set_title('Reconstructed Graph')

sbn.heatmap(true_graph14, yticklabels='none', xticklabels='none', ax = ax1[2])
sbn.heatmap(pred_graph14, yticklabels='none', xticklabels='none', ax = ax2[2])
ax1[2].set_title('Random Graph: True Graph')
ax2[2].set_title('Reconstructed Graph')

fig.suptitle('True Adjacency (Top) and Reconstructed Adjacency (Bottom)', fontsize=16)
fig.savefig(figsp+'Adjacency_Heatmaps_combined.png', dpi = 300)


#correlations
fig2, (ax1, ax2)  = plt.subplots(2, 3, figsize=(18,10))
plt.tight_layout(rect=[0, 0, 1, 0.95])
sbn.heatmap(np.corrcoef(pe_sorted12), yticklabels='none', xticklabels='none', ax = ax1[0])
sbn.heatmap(np.corrcoef(Xin12), yticklabels='none', xticklabels='none', ax = ax2[0])
ax1[0].set_title('Small World: Attributes')
ax2[0].set_title('Embeddings (Bottleneck)')

sbn.heatmap(np.corrcoef(pe_sorted13), yticklabels='none', xticklabels='none', ax = ax1[1])
sbn.heatmap(np.corrcoef(Xin13), yticklabels='none', xticklabels='none', ax = ax2[1])
ax1[1].set_title('Scale Free: Attributes')
ax2[1].set_title('Embeddings (Bottleneck)')

sbn.heatmap(np.corrcoef(pe_sorted14), yticklabels='none', xticklabels='none', ax = ax1[2])
sbn.heatmap(np.corrcoef(Xin14), yticklabels='none', xticklabels='none', ax = ax2[2])
ax1[2].set_title('Random Graph: Attributes')
ax2[2].set_title('Embeddings (Bottleneck)')

fig2.suptitle('Correlations On Node Attributes and Embeddings', fontsize=16)
fig2.savefig(figsp+'Correlation_graphs_combined.png', dpi = 300)






#--------------------------------------------------------------------------
# a simple function for plotting the performance curves during training
#----------------------------------------------------------------
def plot_perf(update_time, performance_list):
    #evaluation metrics
    layers = 2
    titles = ['Top Layer', 'Middle Layer']
    ax_titles = ['Small World','Scale Free', 'Random Graph']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(18,10))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(hspace=0.4, wspace=0.1)

    axlist = [ax1, ax2, ax3]
    for idx, (ax, hist, title) in enumerate(zip(axlist, performance_list, ax_titles)):
        for i in range(0, layers):
            layer_hist = [j[i] for j in hist]
            #homogeneity
            ymin = 0
            ymax = 0.9
            ax[i].plot(np.arange(update_time), np.array(layer_hist)[:,0][:update_time], label = 'Homogeneity')
            ax[i].plot(np.arange(update_time), np.array(layer_hist)[:,1][:update_time], label = 'Completeness')
            ax[i].plot(np.arange(update_time), np.array(layer_hist)[:,2][:update_time], label = 'NMI')
            ax[i].set_xlabel('Training Epochs')
            ax[i].set_ylabel('Performance')
            ax[i].set_title(title+':'+ ' '+titles[i]+' Network Performance')
            ax[i].set_ylim(ymin, ymax)
            ax[i].set_yticks(np.arange(ymin, ymax, 0.2))  # Adjust as needed
            ax[i].set_aspect('auto')  # Ensure aspect ratio is not forced
            if idx == 0:
                ax[i].legend()
    
    fig.suptitle('Performance Curves For All Networks', fontsize=16)
    return fig


figperf = plot_perf(50, [ex12data[-1], ex13data[-1], ex14data[-1]])

figperf.savefig(figsp+'Performance combined.png', dpi = 300)
















# a simple function to plot the loss curves during training
#----------------------------------------------------------------
def plot_loss(epoch, tlosslist, alosslist, xlosslist, modlosslist, clustlosslist):
    
    layers = 3
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,3, figsize=(18,10))
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    #axlist = [ax1, ax2, ax3, ax4, ax5]
    networks = ['Small World','Scale Free','Random Graph']
    for idx, (thist, ahist, xhist, modhist, clusthist, net) in enumerate(zip(tlosslist, alosslist, xlosslist, modlosslist, clustlosslist, networks)):
        #total loss
        ax1[idx].plot(range(0, epoch+1), thist, label = 'Total Loss')
        #ax1[idx].set_xlabel('Training Epochs')
        if idx == 0:
            ax1[idx].set_ylabel('Total Loss')
        ax1[idx].set_title(net+' Training Losses')
        #reconstruction of graph adjacency
        ax2[idx].plot(range(0, epoch+1), ahist, label = 'Graph Reconstruction Loss')
        #ax2[idx].set_xlabel('Training Epochs')
        if idx == 0:
            ax2[idx].set_ylabel('Graph Reconstruction')
        #reconstruction of node attributes
        ax3[idx].plot(range(0, epoch+1), xhist, label = 'Attribute Reconstruction Loss')
        #ax3[idx].set_xlabel('Training Epochs')
        if idx == 0:
            ax3[idx].set_ylabel('Attribute Reconstruction')
        #community loss using modularity
        ax4[idx].plot(range(0, epoch+1), np.array(modhist))
        #ax4[idx].set_xlabel('Training Epochs')
        if idx == 0:
            ax4[idx].set_ylabel('Modularity')
        #community loss using kmeans
        ax5[idx].plot(range(0, epoch+1), np.array(clusthist))
        ax5[idx].set_xlabel('Training Epochs')
        if idx == 0:
            ax5[idx].set_ylabel('Clustering')
    

        ax4[idx].legend(labels = [i for i in ['middle','top'][:layers]], loc = 'lower right')
        ax5[idx].legend(labels = [i for i in ['middle','top'][:layers]], loc = 'lower right')
       
    fig.suptitle('Loss Curves For All Networks', fontsize=16)
    return fig


# mod_loss_hist, loss_history, clust_loss_hist, A_loss_hist, X_loss_hist, perf_hist]
lossfig = plot_loss(epoch = 99, 
                           tlosslist = [ex12data[-5], ex13data[-5], ex14data[-5]], 
                           alosslist = [ex12data[-3], ex13data[-3], ex14data[-3]], 
                           xlosslist = [ex12data[-2], ex13data[-2], ex14data[-2]], 
                           modlosslist =[ex12data[-6], ex13data[-6], ex14data[-6]],
                           clustlosslist = [ex12data[-4], ex13data[-4], ex14data[-4]])

lossfig.tight_layout(pad=3.0)
lossfig.savefig(figsp+'Losses combined.png', dpi = 300)






