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
from MAIN_run_simulations_single_net import run_single_simulation
from run_simulations_utils import set_up_model_for_simulation_inplace, plot_embeddings_heatmap, generate_attention_graph, load_application_data_regulon
from model.utilities import resort_graph, node_clust_eval
from model.train import evaluate
import os
#import torch_kmeans
import matplotlib.pyplot as plt
import torch
import networkx as nx
import numpy as np
import seaborn as sbn
import pandas as pd
import json

parser = argparse.ArgumentParser(description='Model Parameters')
parser.add_argument('--dataset', type=str, choices=['complex', 'intermediate', 'toy', 'cora', 'pubmed', 'regulon.EM', 'regulon.DM'], required=False, help='Dataset selection')
parser.add_argument('--parent_distribution', type=str, choices=['same_for_all', 'unequal'], required=False, help='Parent distribution selection')
parser.add_argument('--read_from', type=str, default='local', choices = ['local','cluster'], help='Path to read data from')
parser.add_argument('--which_net', type=int, default=0, help='Network selection')
parser.add_argument('--use_true_graph', type=bool, default=True, help='Sets the input graph to be the true topology')
parser.add_argument('--correlation_cutoff', type=float, default=0.5, help='Use true graph or float giving graph with specified correlation cutoff')
parser.add_argument('--mse_method', type=str, default='mse', choices=['mse', 'msenz'], help='Method for computing attribute reconstruction loss mse is classic Mean squared Error, while msenz is MSE computed over all nonzero values (recommended for sparse datasets such as scRNA)')
parser.add_argument('--use_method', type=str, default='top_down', choices=['top_down','bottom_up'], help='method for uncovering the hierarchy')
parser.add_argument('--use_softKMeans_top', type=bool, default=False, help='If true, the top layer is inferred with a softKMeans layer')
parser.add_argument('--use_softKMeans_middle', type=bool, default=False, help='If true, the middle layer is inferred with a softKMeans layer')
parser.add_argument('--gamma', type=float, default=1, help='Gamma hyperparameter value')
parser.add_argument('--delta', type=float, default=1, help='Delta hyperparameter value')
parser.add_argument('--lambda_', nargs='+', type=float, default=[1,1], help='Lambda hyperparameter value')
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
parser.add_argument('--early_stopping', type=bool, default=True, help='If true early stopping is used during training')
parser.add_argument('--patience', type=int, default=5, help='Number of consecutive epochs with no improvement after which training will stop (only relevant if early stopping is enabled).')
parser.add_argument('--train_test_size', nargs='+', type=float, default = [0.8, 0.1], help='fraction of data in training and testing sets')
parser.add_argument('--post_hoc_plots', type=bool, default=True, help='Boolean - should additional plots of results be made')
parser.add_argument('--add_output_layers', type=bool, default=False, help ='should extra layers be added between the embedding and prediction layers?')
parser.add_argument('--make_directories', type=bool, default=False, help='If true directors are created using os.makedir()')
parser.add_argument('--save_model', type=bool, default=False, help='If true model is saved at args.sp as model')
parser.add_argument('--compute_optimal_clusters', type=bool, default=True, help='If number of top layer communities should be determined via one of the methods in --kappa_method')
parser.add_argument('--kappa_method', type=str, default='elbow', choices = ['bethe_hessian', 'elbow', 'silouette'], help='Method for determining the optimal number of communities for the top layer of the hierarchy')
args = parser.parse_args()



parser2 = argparse.ArgumentParser(description='Simulation Parameters')
parser2.add_argument('--connect', dest='connect', default='disc', type=str)
parser2.add_argument('--connect_prob_middle', dest='connect_prob_middle', default=0.1, type=str)
parser2.add_argument('--connect_prob_bottom', dest='connect_prob_bottom', default=0.01, type=str)
parser2.add_argument('--toplayer_connect_prob', dest='toplayer_connect_prob', default=0.3, type=float)
parser2.add_argument('--top_layer_nodes', dest='top_layer_nodes', default=5, type=int)
parser2.add_argument('--subgraph_type', dest='subgraph_type', default='small world', type=str)
parser2.add_argument('--subgraph_prob', dest='subgraph_prob', default=[0.5, 0.2], type=float)
parser2.add_argument('--nodes_per_super2', dest='nodes_per_super2', default=(3,3), type=tuple)
parser2.add_argument('--nodes_per_super3', dest='nodes_per_super3', default=(20,20), type=tuple)
parser2.add_argument('--node_degree_middle', dest='node_degree_middle', default=3, type=int)
parser2.add_argument('--node_degree_bottom', dest='node_degree_bottom', default=5, type=int)
parser2.add_argument('--sample_size',dest='sample_size', default = 500, type=int)
parser2.add_argument('--layers',dest='layers', default = 2, type=int)
parser2.add_argument('--SD',dest='SD', default = 0.1, type=float)
parser2.add_argument('--common_dist', dest='common_dist',default = True, type=bool)
parser2.add_argument('--seed_number', dest='seed_number',default = 555, type=int)
parser2.add_argument('--within_edgeweights', dest='within_edgeweights',default = (0.5, 0.8), type=tuple)
parser2.add_argument('--between_edgeweights', dest='between_edgeweights',default = (0, 0.2), type=tuple)
parser2.add_argument('--use_weighted_graph', dest='use_weighted_graph',default = False, type=bool)
parser2.add_argument('--set_seed', dest='set_seed', default=False, type=bool)
parser2.add_argument('--force_connect', dest='force_connect', default=True, type=bool)
parser2.add_argument('--savepath', dest='savepath', default='./', type=str)
parser2.add_argument('--mixed_graph', dest='mixed_graph', default=False, type=str)
sim_args = parser2.parse_args()



#simulation settings
sim_args.subgraph_type = 'random graph'
sim_args.connect = 'full'
#global simulation settings
sim_args.top_layer_nodes = 5
sim_args.nodes_per_super2 = (3,3)
#sim_args.nodes_per_super3 = (40,90)
sim_args.common_dist = False
sim_args.force_connect = True
sim_args.connect_prob_middle = [np.random.uniform(0.01, 0.15), 
                                np.random.uniform(0.01, 0.15)
                                ] 
sim_args.connect_prob_bottom = [np.random.uniform(0.01, 0.1), 
                                np.random.uniform(0.01, 0.1)
                                ]
sim_args.set_seed = False
sim_args.layers = 3
sim_args.SD = 0.5
sim_args.mixed_graph = False
#sim_args.mixed_graph = True
sim_args.savepath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_2_13_2025/jobs/testing/graph/'
#sim_args.savepath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_1_3_2025/Example_graph/'

#output save settings
#args.sp = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/benchmarks/test/'
#args.sp = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_1_3_2025/debug_results/'
args.sp = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_2_13_2025/jobs/testing/'
args.save_results = True
args.make_directories = False
args.use_gpu = False
args.save_model = False
args.set_seed = 555
args.read_from = 'local'
args.mse_method = 'mse'
args.load_from_existing = True
args.early_stopping = True
args.patience = 20
args.use_method = "top_down"
args.use_batch_learning = True
args.batch_size = 64
args.use_softKMeans_top = False
args.use_softKMeans_middle = False
args.add_output_layers = False
args.AE_operator = 'GATv2Conv'
#args.COMM_operator = 'Linear'
args.COMM_operator = 'None'
args.attn_heads = 1
args.dropout_rate = 0.2
args.normalize_input = True
args.normalize_layers = True
args.AE_hidden_size = [32]
args.LL_hidden_size = [128, 64] 
args.gamma = 1
args.delta = 1
args.lambda_ = [1, 1] # [top, middle]
args.learning_rate = 1e-3
args.use_true_communities = False
args.community_sizes = [15, 5]
args.compute_optimal_clusters = True
args.kappa_method = 'bethe_hessian'

#training settings
args.dataset = 'regulon.DM.activity'
args.parent_distribution = 'unequal'
args.which_net = 1
args.training_epochs = 500
args.steps_between_updates = 10
args.use_true_graph = False
args.correlation_cutoff = 0.35
args.return_result = 'best_loss'
args.verbose = False
args.run_louvain = True
args.run_kmeans = False
args.run_hc = True

args.split_data = True
args.train_test_size = [0.8, 0.2]

# mainpath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_12_11_2024/Output/testing/'
# #dirname = sim_args.subgraph_type.replace(' ', '_')+'_'+sim_args.connect+str(sim_args.SD).replace('.', '')+'_Case_'+str(0)
# sim_args.savepath = '/'.join([mainpath, 'graph/'])
# args.sp = '/'.join([mainpath, 'results/'])
import time

device = 'cuda:'+str(0) if args.use_gpu and torch.cuda.is_available() else 'cpu'

start = time.time()
results = run_single_simulation(args, simulation_args = sim_args, return_model = False, heads = 1)
end = time.time()

print(f'TOTAL TIME {end-start}')
plt.close('all')


# X, A, target_labels = set_up_model_for_simulation_inplace(args, sim_args, load_from_existing = True)

# model = torch.load(args.sp+'checkpoint.pth')
# # model.eval()
# # output = model.forward(X, A)

# # X_hat, A_hat, X_all, A_all, P_all, S_all, AW = output

# perf_layers, output, S_relab = evaluate(model, X, A, 2, true_labels = target_labels)
# def clear_workspace():
#     global_vars = list(globals().keys())
#     exclude = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__annotations__', '__builtins__', '__file__', '__cached__', 'clear_workspace', 'args', 'run_single_simulation']
    
#     for var in global_vars:
#         if var not in exclude:
#             del globals()[var]


# mainpath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_10_18_2024/Output/wo_fine_tuning_updated_linear_top/'
# subpaths = ['smw_0/', 'smw_1/', 'sf_0/', 'sf_1/', 'rg_0/', 'rg_1/']
# networks = range(0, 6)
# # #networks = []

# for index, (net, _dir) in enumerate(zip(networks, subpaths)):
    
#     print('+'*65)
#     print('+'*65)
#     print('+'*65)
#     args.sp = os.path.join(mainpath, _dir)
#     args.which_net = net
#     results = run_single_simulation(args, return_model=False, heads = 1)
#     bpi, bli = indices
    
args_dict = vars(args)
simargs_dict = vars(sim_args)

dfargs1 = pd.DataFrame(list(args_dict.items()), columns=['Parameter', 'Value'])
dfargs2 = pd.DataFrame(list(simargs_dict.items()), columns=['Parameter', 'Value'])

dfargs1.to_csv(args.sp+'Model_Parameters.csv')
dfargs2.to_csv(sim_args.savepath+'Simulation_Parameters.csv')

#args.split_data = False
#X, A, target_labels = set_up_model_for_simulation_inplace(args, sim_args, load_from_existing = True)

args.split_data = False
X, A, gene_labels, gt = load_application_data_regulon(args)
target_labels = [np.array(gt['regulon_kmeans'].tolist()), np.array(gt['regulon_kmeans'].tolist())]
#args.split_data = False
#train, test, gene_labels = load_application_data_regulon(args)
#X, A, [] = train
model = torch.load(args.sp+'checkpoint.pth', weights_only=False)
perf_layers, output, S_relab = evaluate(model.to(device), X, A, 2, true_labels = target_labels)
                
X_hat, A_hat, X_all, A_all, P_all, S_all, AW = output

S_all = [i.cpu() for i in S_all]

print('='*60)
print('-'*10+'final top'+'-'*10)
final_top_res=node_clust_eval(target_labels[0], [i.cpu() for i in S_relab[0]], verbose = True)
print('-'*10+'final middle'+'-'*10)
final_middle_res=node_clust_eval(target_labels[1], [i.cpu() for i in S_relab[1]], verbose = True)
print('='*60)

#generate graph 

gene_labels = [str(i) for i in range(0, X.shape[0])]
node_lookup_dict = {name: index for index, name in enumerate(gene_labels)}

new_G, weighted_adj = generate_attention_graph(args, A, AW, gene_labels, S_all)

fig, ax = plt.subplots(figsize=(12, 10))

top_sort = np.argsort(S_all[0])
mid_sort = np.argsort(S_all[1])

#comm1_idx = [idx for idx, i in enumerate(S_all[0]) if i == 0]


sbn.heatmap(weighted_adj, xticklabels=np.array(gene_labels), 
            yticklabels=np.array(gene_labels), ax = ax)
ax.set_title('Heatmap of Attention Coefficients')
ax.tick_params(axis='both', which='major', labelsize=5)
ax.tick_params(axis='both', which='minor', labelsize=5)
fig.savefig(args.sp+'attention_weights.pdf')


fig, ax = plt.subplots(figsize=(12, 10))
sbn.heatmap(S_all[0].unsqueeze(1).detach().numpy()[top_sort].T, ax = ax, xticklabels=np.array(gene_labels)[top_sort])
ax.tick_params(axis='both', which='major', labelsize=8)
fig.savefig(args.sp+'predicted_labels.pdf')


stripped_labels = np.array([i.strip('(+)') for i in gene_labels])
top_sort = np.argsort(S_all[0].detach().numpy())
df = pd.DataFrame(np.array([stripped_labels[top_sort], S_all[0][top_sort].detach().tolist(), S_all[1][top_sort].detach().tolist()]).T,
                  columns = ['Regulon', 'Top Assignment', 'Middle Assignment'])


df.to_csv(args.sp+'gene_data.csv')

top_resorted_X = X[top_sort,:].cpu().detach().numpy()
top_resorted_A = resort_graph(A, top_sort).cpu().detach().numpy()

mid_resorted_X = X[mid_sort,:].cpu().detach().numpy()
mid_resorted_A = resort_graph(A, mid_sort).cpu().detach().numpy()

fig, (ax1, ax2) = plt.subplots(2,2, figsize = (12,12))

sbn.heatmap(np.corrcoef(top_resorted_X), ax = ax1[0])
sbn.heatmap(top_resorted_A, ax = ax1[1])
sbn.heatmap(np.corrcoef(mid_resorted_X), ax = ax2[0])
sbn.heatmap(mid_resorted_A, ax = ax2[1])

ax1[0].set_title('Top layer correlation matrix and graph (sorted)')
ax2[0].set_title('Middle layer correlation matrix and graph (sorted)')

fig.savefig(args.sp+'sorted_correlation_matrices.pdf')

graph_data = nx.json_graph.node_link_data(new_G)

with open(args.sp+'graph_data.json', 'w') as json_file:
    json.dump(graph_data, json_file, indent=4)

# embeds = [i[2][0].cpu().detach().numpy() for i in out[0]]
# atw = [i[-1] for i in out[0]]
# slist = [i[-4] for i in out[0]]

# gl = [str(i) for i in range(0, Xres.shape[0])]
# weighted_graphs = [generate_attention_graph(args, Ares, AW, gl, S, cutoff='none')[1] for index, (AW, S) in enumerate(zip(atw, slist))]


# plyfig1 = plot_embeddings_heatmap(embeds, use_correlations=True, verbose=True)
# plyfig2 = plot_embeddings_heatmap(weighted_graphs, verbose = True)

# if args.save_results:
#     plyfig1.write_html(args.sp+"embeddings_interactive.html")
#     plyfig2.write_html(args.sp+"attention_weights_interactive.html")

del results
plt.close('all')







