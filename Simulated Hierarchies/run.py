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
from MAIN_run_simulations_single_net import run_single_simulation
#import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Model Parameters')

parser.add_argument('--dataset', type=str, choices=['complex', 'intermediate', 'toy', 'cora', 'pubmed'], required=False, help='Dataset selection')
parser.add_argument('--parent_distribution', type=str, choices=['same_for_all', 'unequal'], required=False, help='Parent distribution selection')
parser.add_argument('--read_from', type=str, default='local', choices = ['local','cluster'], help='Path to read data from')
parser.add_argument('--which_net', type=int, default=0, help='Network selection')
parser.add_argument('--use_true_graph', type=bool, default=True, help='Sets the input graph to be the true topology')
parser.add_argument('--correlation_cutoff', type=float, default=0.5, help='Use true graph or float giving graph with specified correlation cutoff')
parser.add_argument('--use_method', type=str, default='top_down', choices=['top_down','bottom_up'], help='method for uncovering the hierarchy')
parser.add_argument('--use_softKMeans', type=bool, default=False, help='If true, the top layer is inferred with a softKMeans layer')
parser.add_argument('--gamma', type=float, default=1, help='Gamma value')
parser.add_argument('--delta', type=float, default=1, help='Delta value')
parser.add_argument('--lambda_', nargs='+', type=float, default=[1,1], help='Lambda value')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--dropout_rate', type=float, default = 0.2, help='Dropout rate')
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
parser.add_argument('--remove_graph_loss', type=bool, default=False, help='Remove graph loss')
parser.add_argument('--return_result', type=str, choices=['best_perf_top', 'best_perf_mid'], required=False, help='Return result type')
parser.add_argument('--save_results', type=bool, default=False, help='Save results')
parser.add_argument('--set_seed', type=bool, default=True, help='Set random seed')
parser.add_argument('--sp', type=str, default='/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/Simulation_Results/', help='Save path')
parser.add_argument('--plotting_node_size', type=int, default=25, help='Plotting node size')
parser.add_argument('--fs', type=int, default=10, help='Font size')
parser.add_argument('--run_louvain', type=bool, default=True, help='Run Louvain algorithm')
parser.add_argument('--run_kmeans', type=bool, default=True, help='Run KMeans algorithm')
parser.add_argument('--run_cmeans', type=bool, default=True, help='Run Fuzzy CMeans algorithm')
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
args = parser.parse_args()



#output save settings
#args.sp = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/benchmarks/test/'
args.sp = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Reports/Report_8_23_2024/Output/'
args.save_results = False
args.make_directories = False
#args.resolution = [1,1]
#model settings
#args.use_method = 'bottom_up'
args.use_method = "top_down"
args.use_softKMeans = True
args.add_output_layers = False
args.AE_operator = 'GATv2Conv'
args.COMM_operator = 'Linear'
args.attn_heads = 5
args.dropout_rate = 0.3
args.normalize_input = True
args.normalize_layers = True
args.AE_hidden_size = [256, 256]
args.LL_hidden_size = [128, 64] 
args.gamma = 1e-1
args.delta = 1
#args.lambda_ = [1e-2, 1e-2]
args.lambda_ = [1e-3,1e-3]
args.learning_rate = 1e-4
args.remove_graph_loss = False
args.use_true_communities = False
args.community_sizes = [[10,10,10,10,10], 5]
#args.community_sizes = [15, 5]

#training settings
args.dataset = 'intermediate'
args.parent_distribution = 'unequal'
args.which_net = 0
args.training_epochs = 100
args.steps_between_updates = 100
args.use_true_graph = False
args.correlation_cutoff = 0.2
args.return_result = 'best_perf_top'
args.verbose = False
args.run_louvain = True
args.run_kmeans = False

args.split_data = False
args.train_test_size = [0.8, 0.1]

results = run_single_simulation(args, heads = 1)

def clear_workspace():
    global_vars = list(globals().keys())
    exclude = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__annotations__', '__builtins__', '__file__', '__cached__', 'clear_workspace', 'args', 'run_single_simulation']
    
    for var in global_vars:
        if var not in exclude:
            del globals()[var]

clear_workspace()














