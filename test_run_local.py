import argparse
import sys
import os
#expose paths to necessary files
base_dir = "/Users/jordandavis/Desktop/HGRN_repo"

# Add necessary folders to the Python path
sys.path.append(os.path.join(base_dir, "HGRN_software"))
sys.path.append(os.path.join(base_dir, "Simulated Hierarchies"))
sys.path.append(os.path.join(base_dir, "Simulated Hierarchies", "run_simulation_utils"))
sys.path.append(os.path.join(base_dir, "model"))

from MAIN_run_simulations_single_net import run_single_simulation
from run_simulations_utils import set_up_model_for_simulation_inplace
from model.utilities import node_clust_eval
from model.train import evaluate
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from typing import Literal
parser = argparse.ArgumentParser(description='Model Parameters')
parser.add_argument('--dataset', type=str, choices=['complex', 'intermediate', 'toy', 'cora', 'pubmed', 'regulon.EM', 'regulon.DM', 'generated'], required=False, default='generated', help='Dataset selection. When dataset is "generated" the data are simulated accord to simulation argments')
parser.add_argument('--parent_distribution', type=str, choices=['same_for_all', 'unequal'], required=False, help='Parent distribution. when "same_for_all" all parent nodes are simulated from a N(0,1) distribution')
parser.add_argument('--read_from', type=str, default='local', choices = ['local','cluster'], help='Path to read data from. if "local" data is read from the "path/to/repo/Simulated Hierarchies/DATA/')
parser.add_argument('--which_net', type=int, default=0, help='Network selection. Used for pre-generated datasets only')
parser.add_argument('--use_true_graph', type=bool, default=True, help='Sets the input graph to be the true graph')
parser.add_argument('--correlation_cutoff', type=float, default=0.2, help='The minimum correlation required for an edge to be added between two nodes')
parser.add_argument('--use_method', type=str, default='top_down', choices=['top_down','bottom_up'], help='method for uncovering the hierarchy. "top_down" = divisive clustering while "bottom_up" = additive clustering' )
parser.add_argument('--use_softKMeans_top', type=bool, default=False, help='If true, the top layer is inferred with a softKMeans layer')
parser.add_argument('--use_softKMeans_middle', type=bool, default=False, help='If true, the middle layer is inferred with a softKMeans layer (currently broken)')
parser.add_argument('--gamma', type=float, default=1, help='Attribute reconstruction loss hyperparameter')
parser.add_argument('--delta', type=float, default=1, help='Modularity loss hyperparameter')
parser.add_argument('--lambda_', nargs='+', type=Literal[float], default=[1.0, 1.0], help='Clustering loss hyperparameters. First value is for middle layer, second value is for top layer')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--dropout_rate', type=float, default = 0.2, help='Dropout rate')
parser.add_argument('--use_batch_learning', type=bool, default=False, help='If true data is divided into batches for training according to batch_size')
parser.add_argument('--batch_size', type = int, default=64, help='size of batches used when use_batch_learning == True')
parser.add_argument('--training_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--steps_between_updates', type=int, default=10, help='Number of training epochs before each update')
parser.add_argument('--resolution', nargs='+', type=Literal[float], default=None, help='The resolution regularization parameter in the modularity loss calculation')
parser.add_argument('--AE_hidden_size', nargs='+', type=Literal[int], default=[256, 128, 64], help='Hidden layer sizes for GATE')
parser.add_argument('--LL_hidden_size', nargs='+', type=Literal[int], default = [64, 64], help='Sizes for additional learning layers applied to GATE embedding before inference. Only used when "')
parser.add_argument('--AE_operator', type=str, choices=['GATConv', 'GATv2Conv', 'SAGEConv'], default='GATv2Conv', help='The type of layer that should be used in the graph autoencoder architecture')
parser.add_argument('--COMM_operator', type=str, choices=['None', 'Conv2d', 'Linear', 'GATConv', 'GATv2Conv', 'SAGEConv'], default='Linear', help='The type of layer that should be used in the community detection module')
parser.add_argument('--use_true_communities', type=bool, default=True, help='Use true communities')
parser.add_argument('--community_sizes', nargs='+', type=Literal[int], default=[15, 5], help='Specifies the (max) number of communities to be inferred in the middle and top layers respectively')
parser.add_argument('--activation', type=str, choices=['LeakyReLU', 'Sigmoid'], required=False, help='Activation function (ignore)')
parser.add_argument('--use_gpu', type=bool, default=True, help='When True, HCD defaults to device="gpu" if torch.cuda.is_available() is True else device is set to "cpu" ')
parser.add_argument('--verbose', type=bool, default=True, help='when True, additional plots are output in training updates')
parser.add_argument('--return_result', type=str, choices=['best_perf_top', 'best_perf_mid', 'best_loss'], required=False, default=True, help='When True, all model training results are returned')
parser.add_argument('--save_results', type=bool, default=False, help='When True, results are saved to path at "sp" ')
parser.add_argument('--set_seed', type=bool, default=True, help='Sets a random seed for training the model')
parser.add_argument('--seed', dest='seed', type=int, default=555, help='Seed number for training model')
parser.add_argument('--sp', type=str, default='/my/save/path/', help='Specified path to save directory')
parser.add_argument('--plotting_node_size', type=int, default=25, help='Node size - used in plotting')
parser.add_argument('--fs', type=int, default=10, help='Font size - used in plotting')
parser.add_argument('--run_louvain', type=bool, default=True, help='When True, runs Louvain method on input graph for comparison')
parser.add_argument('--run_kmeans', type=bool, default=True, help='When True, runs KMeans on gene expression for comparison')
parser.add_argument('--run_hc', type=bool, default=True, help='When True, runs hierarchical clustering on gene expression for comparison')
parser.add_argument('--use_multihead_attn', type=bool, default=True, help='When True, Multihead attention is applied to all GATE graph attention layers')
parser.add_argument('--attn_heads', type = int, default = 5, help='The number of attention heads for multihead attention')
parser.add_argument('--normalize_layers', type=bool, default = True, help='When True, the output of each neural layer is normalized before activation')
parser.add_argument('--normalize_input', type=bool, default=True, help='When True, the input features are normalized before model fitting')
parser.add_argument('--split_data', type=bool, default=False, help='When True, data is split into training, testing, and validation sets (currently validation splitting doesnt work)')
parser.add_argument('--early_stopping', type=bool, default=True, help='If True, early stopping is used during training')
parser.add_argument('--patience', type=int, default=5, help='Number of consecutive epochs with no improvement after which training will stop (only relevant if early stopping is enabled).')
parser.add_argument('--train_test_size', nargs='+', type=Literal[float], default = [0.8, 0.2], help='Specifies the fraction of data in training and testing sets respectively')
parser.add_argument('--post_hoc_plots', type=bool, default=True, help='When True, additional plots of results are made')
parser.add_argument('--add_output_layers', type=bool, default=False, help ='When True, extra neural layers are added between the embedding and prediction layers')
parser.add_argument('--make_directories', type=bool, default=False, help='If true directories are created using os.makedir()')
parser.add_argument('--save_model', type=bool, default=False, help='When True, model is serialized and saved at args.sp as model.pth file')
parser.add_argument('--compute_optimal_clusters', type=bool, default=True, help='When True, values of "community_sizes" is estimated using the method specified in "kappa_method"')
parser.add_argument('--kappa_method', type=str, default='bethe_hessian', choices = ['bethe_hessian', 'elbow', 'silouette'], help='Method for determining the optimal number of communities for the top and middle layers of the hierarchy (currently "elbow" method does not work)')
parser.add_argument('--load_from_existing', type=bool, default=False, help='When --dataset = "generated" and argument is True, data is read from same directory where previous graph was saved')
args = parser.parse_args()



# Simulation arguments
parser2 = argparse.ArgumentParser(description='Simulation Parameters')
parser2.add_argument('--connect', dest='connect', choices=['disc', 'full'], default='disc', type=str, help='Sets the top layer of the simulated hierarchy to be either all connected or all disconnected')
parser2.add_argument('--connect_prob_middle', dest='connect_prob_middle', default=0.1, type=float, help='Sets the probability for edge creation between two nodes in the middle layer of the hierarchy')
parser2.add_argument('--connect_prob_bottom', dest='connect_prob_bottom', default=0.01, type=float, help='Sets the probability for edge creation between two nodes in the bottom layer of the hierarchy')
parser2.add_argument('--top_layer_nodes', dest='top_layer_nodes', default=5, type=int, help='Sets the number of nodes (i.e communities) in the top layer of the hierarchy')
parser2.add_argument('--subgraph_type', dest='subgraph_type', default='small world', type=str, help='Sets the type of subgraph used to generate communities in the middle and bottom layers of the hierarchy')
parser2.add_argument('--subgraph_prob', nargs='+', dest='subgraph_prob', default=(0.5, 0.2), type=tuple[float], help='Sets the probability of edge creating within small world subgraphs') #controls p value for subgraphs
parser2.add_argument('--nodes_per_super2', nargs='+', dest='nodes_per_super2', default=(3,3), type=tuple[float], help='sets the number of offspring for each node in the top layer of the hierarchy')
parser2.add_argument('--nodes_per_super3', nargs='+', dest='nodes_per_super3', default=(20,20), type=tuple[float], help='Sets the number of offspring for each node in the bottom layer of the hierarchy')
parser2.add_argument('--node_degree_middle', dest='node_degree_middle', default=3, type=int, help='sets the node degree for random graph subgraphs in the middle layer')
parser2.add_argument('--node_degree_bottom', dest='node_degree_bottom', default=5, type=int, help='sets the node degree for random graph subgraphs in the middle layer')
parser2.add_argument('--sample_size',dest='sample_size', default = 500, type=int, help='Sample size (people/cells) for simulated gene expression')
parser2.add_argument('--layers',dest='layers', default = 3, type=int, help='Sets the number of layers in the hierarchy. Note that HCD currently only supports two or three layer hierarchies')
parser2.add_argument('--SD',dest='SD', default = 0.1, type=float, help='Sets the standard deviation for all simulated genes')
parser2.add_argument('--common_dist', dest='common_dist',default = False, type=bool, help='When True, all parent nodes of the network are simulated from a N(0,sigma) distribution. When False, all parent nodes are simulated from N(mu_k, sigma)')
parser2.add_argument('--use_weighted_graph', dest='use_weighted_graph',default = False, type=bool, help='When True, a weigted network is generated according --within_edgeweights and --between_edgeweights')
parser2.add_argument('--within_edgeweights', dest='within_edgeweights', nargs='+', default = (0.5, 0.8), type=tuple[float], help='Tuple giving the minimum and maximum weight for an edge between nodes in the same community')
parser2.add_argument('--between_edgeweights', dest='between_edgeweights', nargs='+', default = (0, 0.2), type=tuple[float], help='Tuple giving the minimum and maximum weight for an edge between nodes in different communities')
parser2.add_argument('--set_seed', dest='set_seed', default=False, type=bool, help='When True, a random seed is set for generating the network')
parser2.add_argument('--seed_number', dest='seed_number',default = 555, type=int, help='random seed for network generation')
parser2.add_argument('--force_connect', dest='force_connect', default=True, type=bool, help='Enforces connectivity between communities: If True, ensures at least one edge exists between nodes from communities that were connected in the previous layer')
parser2.add_argument('--savepath', dest='savepath', default='./', type=str, help='PATH name specifying where the simulated network should be saved')
parser2.add_argument('--mixed_graph', dest='mixed_graph', default=False, type=str, help='When True, a graph with mixed small world, scale free, and random graph topolgies is generated')
sim_args = parser2.parse_args()
args_dict = vars(args)
simargs_dict = vars(sim_args)
'''
def scale_connection_prob(num_nodes, within_between_probs=(4, 12)):
    within_prob, between_prob = within_between_probs
    within_prob = within_prob / num_nodes
    between_prob = between_prob / num_nodes
    return [np.random.uniform(within_prob, between_prob) for _ in range(2)]
'''
def scale_connection_prob(num_nodes, within_between_probs=(4, 12)):
    within_prob, between_prob = within_between_probs
    within_prob = within_prob / num_nodes
    between_prob = between_prob / num_nodes
    #within_prob = np.abs(np.random.normal(within_prob))
    #between_prob = np.abs(np.random.normal(between_prob))
    probs = [within_prob,between_prob]
    return probs
bottom_prob_within =.013
bottom_prob_between = .0027
middle_prob_between = .002
for i in range(1):
    
    bottom_prob_between = bottom_prob_between + .0002
    middle_prob_between = middle_prob_between + .001
    

    sim_args.savepath = f'/Users/jordandavis/Desktop/HGRN_repo/very_small_graph_150/'
    sim_args.connect = 'full'
    sim_args.force_connect = False
    sim_args.top_layer_nodes = 3
    sim_args.nodes_per_super2 = (5,5) #output to sim params
    sim_args.nodes_per_super3 = (10,10) #output to sim params
    n1 = sim_args.top_layer_nodes
    k2 = (sim_args.nodes_per_super2[0]+sim_args.nodes_per_super2[1])/2
    k3 = (sim_args.nodes_per_super3[0]+sim_args.nodes_per_super3[1])/2

    n2 = n1 * k2  # Middle layer
    n3 = n2 * k3  # Bottom layer


    sim_args.connect_prob_middle = [.008,middle_prob_between]
    sim_args.connect_prob_bottom = [bottom_prob_within,bottom_prob_between]
    X,A, target_labels = set_up_model_for_simulation_inplace(args, sim_args)
    params_output_path = os.path.join(sim_args.savepath, 'simulation_params.txt')

    # Ensure the save directory exists
    os.makedirs(sim_args.savepath, exist_ok=True)

    # Collect parameters to save
    params_to_save = {
        "connect": sim_args.connect,
        "nodes_per_super2": sim_args.nodes_per_super2,
        "nodes_per_super3": sim_args.nodes_per_super3,
        "top_layer_nodes": sim_args.top_layer_nodes,
        "middle_layer_nodes": n2,
        "bottom_layer_nodes": n3,
        "node_degree_middle": sim_args.node_degree_middle,
        "node_degree_bottom": sim_args.node_degree_bottom,
        "connect_prob_middle": sim_args.connect_prob_middle,
        "connect_prob_bottom": sim_args.connect_prob_bottom,
        "sim_args.subgraph_type": sim_args.subgraph_type 
    }

    with open(params_output_path, 'w') as f:
        for key, value in params_to_save.items():
            f.write(f"{key}: {value}\n")

    print(f"Parameters saved to {params_output_path}")
    #uotput number of nodes
    #try 10k nodes