# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:35:58 2024

@author: Bruin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:18:41 2024

@author: Bruin
"""

import argparse
import sys
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
sys.path.append('C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/HGRN_software/')
#sys.path.append('C:/Users/Bruin/OneDrive/Documents/GitHub/torch_kmeans/')
from run_simulations_utils import load_application_data_regulon
import os
#import torch_kmeans
import matplotlib.pyplot as plt
import torch
import networkx as nx
import seaborn as sbn
from model.utilities import resort_graph
import numpy as np
import pandas as pd
import json

parser = argparse.ArgumentParser(description='Model Parameters')

parser.add_argument('--dataset', type=str, choices=['complex', 'intermediate', 'toy', 'cora', 'pubmed', 'regulon.EM', 'regulon.DM'], required=False, help='Dataset selection')
parser.add_argument('--parent_distribution', type=str, choices=['same_for_all', 'unequal'], required=False, help='Parent distribution selection')
parser.add_argument('--read_from', type=str, default='local', choices = ['local','cluster'], help='Path to read data from')
parser.add_argument('--which_net', type=int, default=0, help='Network selection')
parser.add_argument('--use_true_graph', type=bool, default=True, help='Sets the input graph to be the true topology')
parser.add_argument('--correlation_cutoff', type=float, default=0.5, help='Use true graph or float giving graph with specified correlation cutoff')
parser.add_argument('--use_method', type=str, default='top_down', choices=['top_down','bottom_up'], help='method for uncovering the hierarchy')
parser.add_argument('--use_softKMeans_top', type=bool, default=False, help='If true, the top layer is inferred with a softKMeans layer')
parser.add_argument('--use_softKMeans_middle', type=bool, default=False, help='If true, the middle layer is inferred with a softKMeans layer')
parser.add_argument('--gamma', type=float, default=1, help='Gamma value')
parser.add_argument('--delta', type=float, default=1, help='Delta value')
parser.add_argument('--lambda_', nargs='+', type=float, default=[1,1], help='Lambda value')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--dropout_rate', type=float, default = 0.2, help='Dropout rate')
parser.add_argument('--use_batch_learning', type=bool, default=False, help='If true data is divided into batches for training according to batch_size')
parser.add_argument('--batch_size', type = int, default=64, help='size of batches used when use_batch_learning == True')
parser.add_argument('--training_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--steps_between_updates', type=int, default=10, help='Number of updates')
parser.add_argument('--resolution', nargs='+', type=int, default=None, help='Resolution')
parser.add_argument('--AE_hidden_size', nargs='+', type=int, default=[256, 128, 64], help='Hidden layer sizes for GATE')
parser.add_argument('--LL_hidden_size', nargs='+', type=int, default = [64, 64], help='hidden layer sizes learning layers on AE embedding')
parser.add_argument('--AE_operator', type=str, choices=['GATConv', 'GATv2Conv', 'SAGEConv'], default='GATv2Conv', help='The type of layer that should be used in the graph autoencoder architecture')
parser.add_argument('--COMM_operator', type=str, choices=['Linear', 'GATConv', 'GATv2Conv', 'SAGEConv'], default='Linear', help='The type of layer that should be used in the community detection module')
parser.add_argument('--use_true_communities', type=bool, default=True, help='Use true communities')
parser.add_argument('--community_sizes', nargs='+', type=int, default=[15, 5], help='Community sizes')
parser.add_argument('--activation', type=str, choices=['LeakyReLU', 'Sigmoid'], required=False, help='Activation function')
parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose output')
parser.add_argument('--return_result', type=str, choices=['best_perf_top', 'best_perf_mid', 'best_loss'], required=False, help='Return result type')
parser.add_argument('--save_results', type=bool, default=False, help='Save results')
parser.add_argument('--set_seed', type=bool, default=True, help='Set random seed')
parser.add_argument('--sp', type=str, default='/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/Simulation_Results/', help='Save path')
parser.add_argument('--plotting_node_size', type=int, default=25, help='Plotting node size')
parser.add_argument('--fs', type=int, default=10, help='Font size')
parser.add_argument('--run_louvain', type=bool, default=True, help='Run Louvain for comparison')
parser.add_argument('--run_kmeans', type=bool, default=True, help='Run KMeans for comparison')
parser.add_argument('--run_hc', type=bool, default=True, help='Run hierarchical clustering for comparison')
parser.add_argument('--use_multihead_attn', type=bool, default=True, help='Use attentional graph layers')
parser.add_argument('--attn_heads', type = int, default = 10, help='The number of attention heads')
parser.add_argument('--normalize_layers', type=bool, default = True, help='Should layers normalize their output')
parser.add_argument('--normalize_input', type=bool, default=True, help='Should the input features be normalized')
parser.add_argument('--split_data', type=bool, default=False, help='split data in training, testing, and validation sets')
parser.add_argument('--train_test_size', nargs='+', type=float, default = [0.8, 0.1], help='fraction of data in training and testing sets')
parser.add_argument('--post_hoc_plots', type=bool, default=True, help='Boolean - should additional plots of results be made')
parser.add_argument('--add_output_layers', type=bool, default=False, help ='should extra layers be added between the embedding and prediction layers?')
parser.add_argument('--make_directories', type=bool, default=False, help='If true directors are created using os.makedir()')
parser.add_argument('--use_graph_updating', type=bool, default=False, help='Graph is updated each epoch during training')
parser.add_argument('--save_model', type=bool, default=False, help='If true model is saved at args.sp as model')
args = parser.parse_args()



#output save settings
#args.sp = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/benchmarks/test/'
#args.sp = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_10_18_2024/Output/application/'
args.sp = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_10_18_2024/Output/application/'
args.save_results = True
args.make_directories = True
args.save_model = True
args.set_seed = 555
args.read_from = 'local'
#args.resolution = [1,1]
#model settings
#args.use_method = 'bottom_up'
args.use_method = "top_down"
args.use_batch_learning = True
args.batch_size = 64
args.use_softKMeans_top = True
args.use_softKMeans_middle = False
args.add_output_layers = False
args.AE_operator = 'GATv2Conv'
args.COMM_operator = 'Linear'
args.attn_heads = 10
args.dropout_rate = 0.2
args.normalize_input = True
args.normalize_layers = True
args.AE_hidden_size = [1024, 512]
args.LL_hidden_size = [128, 64] 
args.gamma = 1
args.delta = 1
args.lambda_ = [1, 1]
#args.lambda_ = [1e-2, 1e-1]
args.learning_rate = 1e-4
args.use_true_communities = False
args.community_sizes = [64, 5]

#training settings
args.dataset = 'regulon.EM'
args.parent_distribution = 'unequal'
args.which_net = 0
args.training_epochs = 25
args.steps_between_updates = 5
args.use_true_graph = False
args.correlation_cutoff = 0.2
args.return_result = 'best_loss'
args.verbose = False
args.run_louvain = False
args.run_kmeans = False
args.run_hc = False

args.split_data = False
args.train_test_size = [0.7, 0.3]


train, test, gene_labels = load_application_data_regulon(args)
X, A, [] = train
model_trained = torch.load(args.sp+'MODEL_2.pth')
model_trained.eval()
output = model_trained.forward(X, A)

X_hat, A_hat, X_all, A_all, P_all, S_all, AW = output

node_lookup_dict = {name: index for index, name in enumerate(gene_labels)}

G = nx.from_numpy_array(A.detach().numpy())
new_G = nx.Graph()

edgelist = []
nodelist = []

attent_weights = AW['encoder'][0][1].mean(dim=1)

for index, edge in enumerate(G.edges()):
    if edge[0] != edge[1]:
        attn_weight = attent_weights[index].detach().numpy()
        if attn_weight >= 0.1:
            edgelist.append([(gene_labels[edge[0]], gene_labels[edge[1]]), attn_weight])
            

for index, node in enumerate(gene_labels):
    nodelist.append((str(gene_labels[index]), {'name': str(gene_labels[index]),
                     'top_label': int(S_all[0].unsqueeze(1)[index]),
                     'middle_label': int(S_all[1].unsqueeze(1)[index])}))
    
edges = [i[0] for i in edgelist]
weights = [float(i[1]) for i in edgelist]
new_G.add_nodes_from(nodelist)
for edge, weight in zip(edges, weights):
            new_G.add_edge(edge[0], edge[1], weight = weight)

nx.write_gexf(new_G, args.sp+"gene_network.gexf")

weighted_adj = nx.to_numpy_array(new_G)

fig, ax = plt.subplots(figsize=(12, 10))

top_sort = np.argsort(S_all[0])
mid_sort = np.argsort(S_all[1])

comm1_idx = [idx for idx, i in enumerate(S_all[0]) if i == 0]
#sbn.heatmap(A.detach().numpy(), xticklabels=gene_labels, yticklabels=gene_labels, ax = ax[0])

sbn.heatmap(resort_graph(weighted_adj, comm1_idx), xticklabels=np.array(gene_labels)[comm1_idx], 
            yticklabels=np.array(gene_labels)[comm1_idx], ax = ax)
ax.set_title('Heatmap of Attention Coefficients')
ax.tick_params(axis='both', which='major', labelsize=3)
ax.tick_params(axis='both', which='minor', labelsize=3)



fig, ax = plt.subplots(figsize=(12, 10))
sbn.heatmap(S_all[0].unsqueeze(1).detach().numpy()[top_sort].T, ax = ax, xticklabels=np.array(gene_labels)[top_sort])
ax.tick_params(axis='both', which='major', labelsize=8)


stripped_lables = np.array([i.strip('(+)') for i in gene_labels])
top_sort = np.argsort(S_all[0].detach().numpy())
df = pd.DataFrame(np.array([stripped_lables, S_all[0].detach().tolist(), S_all[1].detach().tolist()]).T,
                  columns = ['Regulon', 'Top Assignment', 'Middle Assignment'])


df.to_csv(args.sp+'gene_data.csv')



graph_data = nx.json_graph.node_link_data(new_G)


with open(args.sp+'graph_data.json', 'w') as json_file:
    json.dump(graph_data, json_file, indent=4)




