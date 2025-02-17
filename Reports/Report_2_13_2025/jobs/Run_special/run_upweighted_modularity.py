# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:41:31 2024

@author: Bruin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:18:41 2024

@author: Bruin
"""

import argparse
import sys
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
#sys.path.append('C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
#sys.path.append('C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/HGRN_software/')
from MAIN_run_simulations_single_net import run_single_simulation
from run_simulations_utils import set_up_model_for_simulation_inplace, plot_embeddings_heatmap, generate_attention_graph
from model.utilities import resort_graph, node_clust_eval
from model.train import evaluate
import os
import matplotlib.pyplot as plt
import torch
import networkx as nx
import numpy as np
import seaborn as sbn
import pandas as pd
import json
from itertools import product
from tqdm import tqdm

# Model arguments
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
parser.add_argument('--early_stopping', type=bool, default=True, help='If true early stopping is used during training')
parser.add_argument('--patience', type=int, default=5, help='Number of consecutive epochs with no improvement after which training will stop (only relevant if early stopping is enabled).')
parser.add_argument('--train_test_size', nargs='+', type=float, default = [0.8, 0.1], help='fraction of data in training and testing sets')
parser.add_argument('--post_hoc_plots', type=bool, default=True, help='Boolean - should additional plots of results be made')
parser.add_argument('--add_output_layers', type=bool, default=False, help ='should extra layers be added between the embedding and prediction layers?')
parser.add_argument('--make_directories', type=bool, default=False, help='If true directors are created using os.makedir()')
parser.add_argument('--save_model', type=bool, default=False, help='If true model is saved at args.sp as model')
parser.add_argument('--use_beth_hessian', type=bool, default=False, help='If number of top layer communities should be determined via Beth Hessian (Spectral) means')
parser.add_argument('--compute_optimal_clusters', type=bool, default=True, help='If number of top layer communities should be determined via one of the methods in --kappa_method')
parser.add_argument('--kappa_method', type=str, default='bethe_hessian', choices = ['bethe_hessian', 'elbow', 'silouette'], help='Method for determining the optimal number of communities for the top layer of the hierarchy')
args = parser.parse_args()


# Simulation arguments
parser2 = argparse.ArgumentParser(description='Simulation Parameters')
parser2.add_argument('--connect', dest='connect', default='disc', type=str)
parser2.add_argument('--connect_prob_middle', dest='connect_prob_middle', default='use_baseline', type=str)
parser2.add_argument('--connect_prob_bottom', dest='connect_prob_bottom', default='use_baseline', type=str)
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



#model settings
args.save_results = True
args.make_directories = True
args.save_model = False
args.set_seed = 555
args.read_from = 'cluster'
args.early_stopping = True
args.patience = 20
args.use_method = "top_down"
args.use_batch_learning = False
args.batch_size = 64
args.load_from_existing = True

#Adjusted Parameters
args.use_softKMeans_top = False
args.COMM_operator = 'None'
args.community_sizes = [15, 5]
args.compute_optimal_clusters = True
args.kappa_method = 'bethe_hessian'

#Unchanged Parameters
args.use_softKMeans_middle = False
args.add_output_layers = False
args.AE_operator = 'GATv2Conv'
args.attn_heads = 5
args.dropout_rate = 0.2
args.normalize_input = True
args.normalize_layers = True
args.AE_hidden_size = [256, 128]
args.LL_hidden_size = [128, 64] 
args.gamma = 1
args.delta = 1e3
args.lambda_ = [1, 1]
args.learning_rate = 1e-3
args.use_true_communities = False
args.dataset = 'generated'
args.parent_distribution = 'unequal'
args.which_net = 1
args.training_epochs = 500
args.steps_between_updates = 25
args.use_true_graph = False
args.correlation_cutoff = 0.2
args.return_result = 'best_loss'
args.verbose = False
args.run_louvain = True
args.run_kmeans = False
args.run_hc = True
args.use_gpu = False
args.split_data = False
args.train_test_size = [0.8, 0.2]

#global simulation settings
sim_args.top_layer_nodes = 5
sim_args.nodes_per_super2 = (3,3)
sim_args.nodes_per_super3 = (20,20)
sim_args.common_dist = False
sim_args.force_connect = True
sim_args.set_seed = False
sim_args.layers = 3 
sim_args.mixed_graph = False

#PATH to redirect output
mainpath_read = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_1_27_2025/Output/Intermediate_applications/SET_MASTER'
mainpath_save = '/mnt/ceph/jarredk/HGRN_repo/Reports/Report_2_13_2025/Output/Intermediate_applications/SET_SPECIAL'
cnames = ['Method', 'Homogeneity', 'Completeness', 'NMI', 'ARI']

#simulated networks
iters = range(0, 25)
graph_types = ['small world','scale free', 'random graph']
stdev = [0.1, 0.5]
connect_types = ['full', 'disc']
    
if args.use_softKMeans_top:
    mp = '/'.join([mainpath_save, '_'.join(['upmod', 'Kmeans', args.COMM_operator, '_'.join([str(i) for i in args.community_sizes]), 'opt_clusts', str(args.compute_optimal_clusters)])])
else:
    mp = '/'.join([mainpath_save, '_'.join(['upmod', args.COMM_operator, '_'.join([str(i) for i in args.community_sizes]), 'opt_clusts', str(args.compute_optimal_clusters)])])
    

for _type in graph_types:
    
    sim_args.subgraph_type = _type
    
    for sd in stdev:
        
        sim_args.SD = sd
        
        sim_args.connect_prob_middle = [np.random.uniform(0.01, 0.15), 
                                        np.random.uniform(0.01, 0.15)
                                        ] 
        sim_args.connect_prob_bottom = [np.random.uniform(0.001, 0.01), 
                                        np.random.uniform(0.001, 0.01)
                                        ]
        for connect in connect_types:
            
            sim_args.connect = connect

            for index2, it in enumerate(iters):
                    
                dirname = '_'.join([sim_args.subgraph_type.replace(' ', '_'),sim_args.connect,str(sim_args.SD).replace('.', ''),'Case',str(index2)])
                sim_args.savepath = '/'.join([mainpath_read, 'graphs', dirname])+'/'
                args.sp = '/'.join([mp, dirname])+'/'
                
                results = run_single_simulation(args, simulation_args = sim_args, return_model = False, heads = 1)
                plt.close('all')
                
                #save parameters
                args_dict = vars(args)
                simargs_dict = vars(sim_args)
                
                dfargs1 = pd.DataFrame(list(args_dict.items()), columns=['Parameter', 'Value'])
                dfargs2 = pd.DataFrame(list(simargs_dict.items()), columns=['Parameter', 'Value'])
                
                dfargs1.to_csv(args.sp+'Model_Parameters.csv')
                dfargs2.to_csv(sim_args.savepath+'Simulation_Parameters.csv')
                
                out, res_table, Ares, Xres, target_labels, S_all, S_sub, louv_preds, indices, model, pbmt = results
                
                X, A, target_labels = set_up_model_for_simulation_inplace(args, sim_args, load_from_existing = True)
                
                model = torch.load(args.sp+'checkpoint.pth')
                perf_layers, output, S_relab = evaluate(model, X, A, 2, true_labels = target_labels)
                
                X_hat, A_hat, X_all, A_all, P_all, S_all, AW = output
                
                print('='*60)
                print('-'*10+'final top'+'-'*10)
                final_top_res=node_clust_eval(target_labels[0], S_relab[0], verbose = True)
                print('-'*10+'final middle'+'-'*10)
                final_middle_res=node_clust_eval(target_labels[1], S_relab[1], verbose = True)
                print('='*60)
                
                gstats = pd.read_csv(sim_args.savepath+'intermediate_examples_network_statistics.csv')
                gstat_columns = gstats.columns
                
                gstat_rep = pd.DataFrame(np.concatenate([np.array(gstats)]*6),
                                        columns = gstat_columns).iloc[:, 1:]
                
                npres = np.array(pbmt)
                combined_results = pd.DataFrame([['Louvain Middle']+npres[0,:].tolist(), 
                                                ['Louvain Top']+npres[1,:].tolist(), 
                                                ['HC Middle']+npres[4,:].tolist(), 
                                                ['HC Top']+npres[5,:].tolist(),
                                                ['HCD Middle']+final_middle_res.tolist(), 
                                                ['HCD Top']+final_top_res.tolist()],
                                                columns = cnames)
                
                final_table =  pd.concat([combined_results, gstat_rep], axis=1)
                final_table.to_csv(mp+'/'+dirname+'.csv')

                #generate graph 
                gene_labels = [str(i) for i in range(0, X.shape[0])]
                
                node_lookup_dict = {name: index for index, name in enumerate(gene_labels)}
                
                new_G, weighted_adj = generate_attention_graph(args, A, AW, gene_labels, S_all)
                
                top_sort = np.argsort(S_all[0])
                mid_sort = np.argsort(S_all[1])

                fig, ax = plt.subplots(figsize=(12, 10))

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
                                    columns = ['Gene', 'Top Assignment', 'Middle Assignment'])


                df.to_csv(args.sp+'gene_data.csv')

                top_resorted_X = X[top_sort,:].detach().numpy()
                top_resorted_A = resort_graph(A, top_sort).detach().numpy()

                mid_resorted_X = X[mid_sort,:].detach().numpy()
                mid_resorted_A = resort_graph(A, mid_sort).detach().numpy()

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

                plt.close('all')
                
                to_delete = [var for var in list(locals().keys()) 
                            if var not in ['args', 'sim_args', 'mainpath_read', 'mainpath_save', 'it', 'index1', 'iters', 
                                        'run_single_simulation', 'set_up_model_for_simulation_inplace', 'plt',
                                        'torch', 'plot_embeddings_heatmap', 'generate_attention_graph',
                                        'resort_graph', 'node_clust_eval', 'evaluate', 'nx', 'np', 'sbn',
                                        'pd', 'json', 'cnames', 'graph_types', 'stdev', 'sd', 'connect', 'connect_types',
                                        '_iterable', 'index2', 'mp', 'm_m', 'kmi', 'o_s', 'coc']]

                for var in to_delete:
                    del locals()[var]
    
        




