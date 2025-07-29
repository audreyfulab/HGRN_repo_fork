# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:19:08 2024

@author: Bruin
"""

"""
this script contains helpers for MAIN_run_simulations_single_net.py
and MAIN_run_simulations_all.py

"""
from itertools import product
import pandas as pd
import ast
import numpy as np
from model.utilities import LoadData, get_input_graph, trace_comms, node_clust_eval, resort_graph, plot_clust_heatmaps
from simulation_software.simulation_utilities import compute_beth_hess_comms, plot_nodes, post_hoc_embedding, compute_graph_STATs
from simulation_software.Simulate import simulate_graph
import torch
import networkx as nx
from community import community_louvain as cl
import matplotlib.pyplot as plt
import seaborn as sbn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.model_selection import train_test_split
from colorama import Fore, Style
import plotly.graph_objects as go
import os
from typing import Optional, Union, List,  Literal


#function which splits training and testing data
def train_test(dataset, prop_train):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train, test, train_size, test_size








def load_simulated_data(args):
    
    if args.read_from == 'local':
        readpath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/'
    elif args.read_from == 'cluster':
        readpath = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/DATA/'
        
    # set filepath and settings grid
    if args.dataset == 'complex':
        #set data path
        loadpath_main = os.path.join(readpath+'complex_networks/')
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
        stats = pd.read_csv(os.path.join(loadpath_main+'network_statistics.csv'))
    
    elif args.dataset == 'intermediate':
        if args.parent_distribution == 'same_for_all':
            loadpath_main = os.path.join(readpath+'Toy_examples/Intermediate_examples/OLD_DATA_5_2_2024/')
        else:
            loadpath_main = os.path.join(readpath+'Toy_examples/Intermediate_examples_unique_dist_sparse/')
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
        stats = pd.read_csv(os.path.join(loadpath_main+'intermediate_examples_network_statistics.csv'))
        #combine pathname and filename pieces
        grid1 = product(structpath, connectpath, layerpath)
        grid2 = product(struct_nm, connect_nm, layer_nm)
        grid3 = product(struct, connect, layers)
    
    elif args.dataset == 'toy':
        loadpath_main = readpath+'Toy_examples/'
        
        connectpath = ['disconnected/', 'fully_connected/']
        layerpath = ['2_layer/', '3_layer/']
        
        connect_nm =['disc_', 'full_']
        layer_nm = ['2_layer_','3_layer_']

        connect = ['disc', 'full']
        layers = [2, 3]

        #read in network statistics 
        stats = pd.read_csv(os.path.join(loadpath_main+'toy_examples_network_statistics.csv'))
        #combine pathname and filename pieces
        grid1 = product(connectpath, layerpath)
        grid2 = product(connect_nm, layer_nm)
        grid3 = product(connect, layers)
        
        stats = pd.read_csv(os.path.join(readpath+'/Toy_examples/toy_examples_network_statistics.csv'))

    return loadpath_main, grid1, grid2, grid3, stats
    
    



def format_regulon_data(args, data, nodes):
    
    #filter out Zero columns
    X = torch.Tensor(data)
    
    
    in_graph, in_adj = get_input_graph(X = X.cpu().detach().numpy(), 
                                       method = 'Correlation', 
                                       r_cutoff = args.correlation_cutoff)
    
    A = torch.Tensor(in_adj)+torch.eye(X.shape[0])
    
    return X, A


def load_application_data_regulon(args):
    
    if args.read_from == 'local':
        readpath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/'
    elif args.read_from == 'cluster':
        readpath = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/DATA/'
        
    data = pd.read_csv(os.path.join(readpath+'Applications/Regulon_DMEM_organoid.csv'))

    nodes, samples = data.shape
    gene_labels = data['Unnamed: 0'].tolist() 
    
    
    if args.dataset == 'regulon.EM':
        EM_index = [idx for idx, i in enumerate(data.columns) if "EM" in i]
        regulon_data = data.to_numpy()[:, EM_index].astype('float64')
        
    if args.dataset == 'regulon.DM':
        DM_index = [idx for idx, i in enumerate(data.columns) if "DM" in i]
        regulon_data = data.to_numpy()[:, DM_index].astype('float64')
        
    #nonzero_cols = [idx for idx, i in enumerate((regulon_data > 0.0).sum(0)) if i > 80]
    nonzero_rows = [idx for idx, i in enumerate((regulon_data > 0.0).sum(1)) if i > 40]
    
    #X_temp = regulon_data[:, nonzero_cols]
    X_reduced = regulon_data[nonzero_rows, :]
    
    if args.split_data:
        x_train, x_test = train_test_split(X_reduced, 
                                           train_size=args.train_test_size[0],
                                           test_size=args.train_test_size[1],
                                           shuffle=True)
        
        X_train, A_train = format_regulon_data(args, x_train, x_train.shape[0]) 
        X_test, A_test = format_regulon_data(args, x_test, x_test.shape[0])
        
        test = [X_test, A_test, []]
    else:
        
        X_train, A_train = format_regulon_data(args, data = X_reduced, nodes = nodes)
        test = None
                
    train = [X_train, A_train, []]
    
    gene_labels_final = np.array(gene_labels)[nonzero_rows]
   
    
    return train, test, gene_labels_final








def load_application_data_Dream5(args):
    
    if args.read_from == 'local':
        readpath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/'
    elif args.read_from == 'cluster':
        readpath = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/DATA/'
        
    set_name = args.dataset.split('.')[1]
        
    gexp = pd.read_csv(os.path.join(readpath+f'Applications/Dream5/{set_name}/expression_data.tsv'), sep = '\t').to_numpy().T
    tfs = pd.read_csv(os.path.join(readpath+f'Applications/Dream5/{set_name}/transcription_factors.tsv'), sep = '\t')
    chip_feats = pd.read_csv(os.path.join(readpath+f'Applications/Dream5/{set_name}/chip_features.tsv'), sep = '\t')
    gene_labels = pd.read_csv(os.path.join(readpath+f'Applications/Dream5/{set_name}/gene_ids.tsv'), sep = '\t')["Name"].tolist()
    
    
    nodes, samples = gexp.shape
    
    nonzero_rows = [idx for idx, i in enumerate(range(0, gexp.shape[0])) if gexp[i, :].sum() != 0]
    
    if args.split_data:
        x_train, x_test = train_test_split(gexp, 
                                           train_size=args.train_test_size[0],
                                           test_size=args.train_test_size[1],
                                           shuffle=True)
        
        X_train, A_train = format_regulon_data(args, x_train, x_train.shape[0]) 
        X_test, A_test = format_regulon_data(args, x_test, x_test.shape[0])
        
        test = [X_test, A_test, []]
    else:
        
        X_train, A_train = format_regulon_data(args, data = gexp, nodes = nodes)
        test = None
                
    train = [X_train, A_train, []]
    
    gene_labels_final = np.array(gene_labels)[nonzero_rows]
   
    
    return train, test, gene_labels_final



def set_up_model_for_simulation_inplace(args, simargs, load_from_existing = False):
    
    if not load_from_existing:
        if not os.path.exists(simargs.savepath):
            os.makedirs(simargs.savepath)
            
        
        
        info_table = pd.DataFrame(columns = ['subgraph_type', 'connection_type', 'connection_prob','layers','StDev',
                                            'nodes_per_layer', 'edges_per_layer', 'subgraph_prob',
                                            'sample_size','modularity_top','avg_node_degree_top',
                                            'avg_connect_within_top','avg_connect_between_top',
                                            'modularity_middle','avg_node_degree_middle',
                                            'avg_connect_within_middle','avg_connect_between_middle'
                                            ])
        
        print(f'saving hierarchy to {simargs.savepath}')
        pe, gexp, nodes, edges, nx_all, adj_all, path, nodelabs, ori = simulate_graph(simargs)
        print('done.')
        print('computing statistics....')
        
        mod, node_deg, deg_within, deg_between = compute_graph_STATs(A_all = adj_all, 
                                                                    comm_assign = nodelabs, 
                                                                    layers = simargs.layers,
                                                                    sp = simargs.savepath,
                                                                    node_size = 60,
                                                                    font_size = 9,
                                                                    add_labels=True)
        print('*'*25+'top layer stats'+'*'*25)
        print('modularity = {:.4f}, mean node degree = {:.4f}'.format(
            mod[0], node_deg[0]
            )) 
        print('mean within community degree = {:.4f}, mean edges between communities = {:.4f}'.format(
            deg_within[0], deg_between[0] 
            ))
        if simargs.layers > 2:
            print('*'*25+'middle layer stats'+'*'*25)
            print('modularity = {:.4f}, mean node degree = {:.4f}'.format(
                mod[1], node_deg[1]
            )) 
            print('mean within community degree = {}, mean edges between communities = {}'.format(
                deg_within[1], deg_between[1] 
                ))
            print('*'*60)
            
            
        cp_dict = {'middle': simargs.connect_prob_middle, 'bottom': simargs.connect_prob_bottom}
        
        if simargs.layers == 3:
            
            row_info = [simargs.subgraph_type, simargs.connect, cp_dict, simargs.layers, simargs.SD,
                        tuple(nodes),tuple(edges),simargs.subgraph_prob, simargs.sample_size,
                        mod[0], node_deg[0], deg_within[0], deg_between[0], 
                        mod[1], node_deg[1], deg_within[1], deg_between[1]]
        else:
            row_info = [simargs.subgraph_type, simargs.connect, cp_dict, simargs.layers, simargs.SD,
                        tuple(nodes),tuple(edges), simargs.sample_size,
                        mod[0], node_deg[0], deg_within[0], deg_between[0], 
                        'NA', 'NA', 'NA', 'NA']
            
        print(pd.DataFrame(row_info))
        
        print('done')
        print('saving hierarchy statistics...')
        info_table.loc[0] = row_info
        info_table.to_csv(simargs.savepath+'intermediate_examples_network_statistics.csv')
        np.savez(simargs.savepath+'intermediate_examples_network_statistics.npz', data = info_table.to_numpy())
        print('done')
        
        print('Simulation Complete')       
        print(f'Loading simulated data from directory: {simargs.savepath}')
    pe, true_adj_undi, indices_top, indices_middle, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle = LoadData(filename=simargs.savepath)
    
    #combine target labels into list
    print('Read in expression data of dimension = {}'.format(pe.shape))
    if simargs.layers == 2:
        target_labels = [sorted_true_labels_top, []]
        #sort nodes in expression table 
        pe_sorted = pe[indices_top,:]
    else:
        target_labels = [sorted_true_labels_top, 
                            sorted_true_labels_middle]
        #sort nodes in expression table 
        pe_sorted = pe[indices_middle,:]
     
    #nodes and attributes
    nodes, attrib = pe.shape
    X = torch.Tensor(pe_sorted)
    #generate input graphs
    if args.use_true_graph:
        A = (torch.Tensor(true_adj_undi[:nodes,:nodes])+torch.eye(nodes))
    else:    
        in_graph, in_adj = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = args.correlation_cutoff)

        A = torch.Tensor(in_adj)+torch.eye(nodes)
            
    return X, A, target_labels
    


def set_up_model_for_simulated_data(args, loadpath_main, grid1, grid2, grid3, stats, **kwargs):
    
    #run simulations
    for idx, value in enumerate(zip(grid1, grid2, grid3)):
        
        if idx == args.which_net:
            
            #print network statistics
            print(f'summary of simulated network statistics for collection {args.dataset}:network {idx}')
            #print(stats.loc[idx])
            print('='*55)
            
            #pdb.set_trace()
            layers = value[2][2]
            #extract and use true community sizes
            if args.use_true_communities == True:
                npl = np.array(ast.literal_eval(stats.nodes_per_layer[idx])).tolist()
                comm_sizes = npl[::-1][1:]
            else:
                comm_sizes = args.community_sizes
            #comm_sizes =[40,5]
                    
            #pdb.set_trace()
            #set pathnames and read in simulated network
            print('-'*25+'loading in data'+'-'*25)
            loadpath = loadpath_main+''.join(value[0])+''.join(value[1])
            #pdb.set_trace()
            pe, true_adj_undi, indices_top, indices_middle, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle = LoadData(filename=loadpath)
           
            #combine target labels into list
            print('Read in expression data of dimension = {}'.format(pe.shape))
            if layers == 2:
                target_labels = [sorted_true_labels_top, []]
                #sort nodes in expression table 
                pe_sorted = pe[indices_top,:]
            else:
                target_labels = [sorted_true_labels_top, 
                                 sorted_true_labels_middle]
                #sort nodes in expression table 
                pe_sorted = pe[indices_middle,:]
     
    #nodes and attributes
    nodes, attrib = pe.shape
    X = torch.Tensor(pe_sorted)
    #generate input graphs
    if args.use_true_graph:
        A = (torch.Tensor(true_adj_undi[:nodes,:nodes])+torch.eye(nodes))
    else:    
        in_graph, in_adj = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = args.correlation_cutoff)

        A = torch.Tensor(in_adj)+torch.eye(nodes)
            
    return X, A, target_labels, comm_sizes
    
    
    
    
    
    
    
def handle_output(args, output, comm_sizes):
    # Initialize default values for all possible outputs
    nodes = output.training_data['X_train'].shape[0]
    A = output.training_data['A_train']
    
    # Initialize all outputs with safe defaults
    defaults = {
        'beth_hessian': None,
        'comm_loss': [None],
        'recon_A': [None],
        'recon_X': [None],
        'perf_mid': [],
        'perf_top': [],
        'upper_limit': None,
        'max_mod': None,
        'best_indices': (None, None),
        'metrics': [{'Top': None, 'Middle': None}],
        'predicted_comms': [tuple()],
        'community_assignments': ([], [], [])
    }
    
    try:
        # Basic metrics that should always exist
        total_loss = np.array([i['Total Loss'] for i in (
            output.test_loss_history if args.split_data 
            else output.train_loss_history
        )])
        best_loss_idx = total_loss.tolist().index(min(total_loss))
        
        # Update reconstruction metrics
        comm_loss = [np.round(output.train_loss_history[best_loss_idx]['Clustering'], 4)]
        recon_A = [np.round(output.train_loss_history[best_loss_idx]['A Reconstruction'], 4)]
        recon_X = [np.round(output.train_loss_history[best_loss_idx]['X Reconstruction'], 4)]
        
        # Graph theory metrics
        upper_limit = torch.sqrt(torch.sum(A-torch.eye(nodes).cpu()))
        beth_hessian = compute_beth_hess_comms((A-torch.eye(nodes).cpu()).detach().numpy())
        max_mod = 1 - (2/upper_limit)
        
        # Community assignments
        if args.use_method == 'bottom_up':
            S_sub, S_layer, S_all = trace_comms(
                [i for i in list(output.predicted_train.values())], 
                comm_sizes
            )
        else:
            S_layer = [i for i in list(output.predicted_train.values())]
            S_sub, S_all = [], []
        predicted_comms = [tuple([len(np.unique(i)) for i in S_layer])]
        
        # Performance metrics handling
        best_perf_idx = None
        metrics = [{'Top': None, 'Middle': None}]
        perf_top = []
        perf_mid = []
        
        if (args.return_result != 'best_loss' and 
            hasattr(output, 'performance_history') and 
            output.performance_history):
            
            # Filter out None values and empty entries
            valid_perf_history = [
                p for p in output.performance_history 
                if p is not None and len(p) > 0
            ]
            
            if valid_perf_history:
                if args.return_result == 'best_perf_top' and len(valid_perf_history[0]) > 0:
                    temp_top = [i[0] for i in valid_perf_history if len(i) > 0]
                    perf_top = np.array(temp_top)
                    if len(perf_top) > 0:
                        best_perf_idx = perf_top[:,2].tolist().index(max(perf_top[:,2]))
                
                elif args.return_result == 'best_perf_mid' and len(valid_perf_history[0]) > 1:
                    temp_mid = [i[1] for i in valid_perf_history if len(i) > 1]
                    perf_mid = np.array(temp_mid)
                    if len(perf_mid) > 0:
                        best_perf_idx = perf_mid[:,2].tolist().index(max(perf_mid[:,2]))
                
                # Get metrics for best epoch
                if best_perf_idx is not None and best_perf_idx < len(valid_perf_history):
                    perf_entry = valid_perf_history[best_perf_idx]
                    metrics = [{
                        'Top': tuple(np.round(perf_entry[0], 4)) if len(perf_entry) > 0 else None,
                        'Middle': (
                            tuple(np.round(perf_entry[1], 4)) 
                            if len(perf_entry) > 1 else None
                        )
                    }]
        
        return (
            beth_hessian,
            comm_loss,
            recon_A,
            recon_X,
            perf_mid,
            perf_top,
            upper_limit,
            max_mod,
            (best_perf_idx, best_loss_idx),
            metrics,
            predicted_comms,
            (S_sub, S_layer, S_all)
        )
        
    except Exception as e:
        print(f"Warning: Error in handle_output - {str(e)}. Returning default values.")
        return tuple(defaults.values())
    
def run_louvain(args, A, labels, layers, savepath):
    
    #preallocate
    louv_mod = []
    louv_num_comms = []
    
    nodes = A.shape[0]
    #get prediction using louvain method
    comms = cl.best_partition(nx.from_numpy_array((A-torch.eye(nodes)).cpu().detach().numpy()))
    louv_mod = cl.modularity(comms, nx.from_numpy_array((A-torch.eye(nodes)).cpu().detach().numpy()))
    #extract cluster labels
    louv_preds = list(comms.values())
    louv_num_comms = len(np.unique(louv_preds))
    #make heatmap for louvain results and get metrics
    fig, ax = plt.subplots()
    #compute performance based on layers
    if layers == 2:
        louv_metrics = {'Top': tuple(np.round(node_clust_eval(labels[0], 
                                                              np.array(louv_preds), 
                                                              verbose=False), 4))}
        sbn.heatmap(pd.DataFrame(np.array([louv_preds,  
                                           labels[0].tolist()]).T,
                                 columns = ['Louvain','Truth_Top']),
                    ax = ax)
    else:
        lnm=['Top','Middle']
        louv_metrics = []
        for j in range(0, 2):
            louv_metrics.append({lnm[j]: tuple(np.round(node_clust_eval(labels[j], 
                                                                        np.array(louv_preds), 
                                                                        verbose=False), 4))})
            sbn.heatmap(pd.DataFrame(np.array([louv_preds, 
                                               labels[1].tolist(), 
                                               labels[0].tolist()]).T,
                                     columns = ['Louvain','Truth_Middle','Truth_Top']),
                        ax = ax)
    
    #make heatmap for louvain results
    if args.save_results:
        fig.savefig(savepath+'Louvain_results.pdf')
        plot_nodes((A-torch.eye(nodes)).cpu().detach().numpy(), 
                   labels = np.array(louv_preds), 
                   path = savepath+'Louvain_graph', 
                   node_size = args.plotting_node_size, 
                   font_size = args.fs, 
                   add_labels = True,
                   save = True)
        
        
    return louv_metrics, louv_mod, louv_num_comms, louv_preds
    
    
    

    
# run Kmeans, Fuzzy Cmeans

def run_kmeans(args, X, labels, layers, sizes):
    """
    
    """
    #make heatmap for louvain results and get metrics
    fig, ax = plt.subplots(figsize = (12, 10))
    os.environ['OMP_NUM_THREADS'] = "2"
    
    
    if layers == 2:
        result = KMeans(n_clusters=sizes[0], random_state=0, n_init="auto").fit(X)
        top_labels = result.labels_
        mid_labels = []
        
        sbn.heatmap(pd.DataFrame(np.array([top_labels,  
                                           labels[0].tolist()]).T,
                                 columns = ['KMeans Middle','KMeans Top', 'Truth_Top']),
                    ax = ax)
        
    else:
        result1 = KMeans(n_clusters=sizes[0], random_state=0, n_init="auto").fit(X)
        result2 = KMeans(n_clusters=sizes[1], random_state=0, n_init="auto").fit(X)
        mid_labels = result1.labels_
        top_labels = result2.labels_
        
        sbn.heatmap(pd.DataFrame(np.array([mid_labels,
                                           top_labels, 
                                           labels[1].tolist(), 
                                           labels[0].tolist()]).T,
                                 columns = ['KMeans Middle', 'KMeans Top','Truth Middle','Truth Top']),
                    ax = ax)
        
        
    fig.savefig(args.sp+'KMeans_results.png')
        
        
        
    return [mid_labels, top_labels]




def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    cluster_colors = plt.cm.get_cmap('tab10', model.n_clusters_)
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, 
               color_threshold=0,
               #link_color_func=lambda k: cluster_colors(model.labels_[k % len(model.labels_)]),
               **kwargs)
    
    
    
    

def run_trad_hc(args, X, labels, layers, sizes):
    """
    
    """
    #make heatmap for louvain results and get metrics
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize = (14, 10))
    
    
    if layers == 2:
        model_top = AgglomerativeClustering(n_clusters=sizes[0], metric='euclidean', linkage='ward', compute_distances=True)
        result_top = model_top.fit(X)
        top_labels = result_top.labels_
        
        sbn.heatmap(pd.DataFrame(np.array([top_labels,  
                                           labels[0].tolist()]).T,
                                 columns = ['Traditional HC Top', 'Truth_Top']),
                    ax = ax1[0])
        
        ax2[0].set_xlabel = 'Dendrogram Top'
        # plot the top three levels of the dendrogram
        plot_dendrogram(model_top, truncate_mode=None, p=3, ax = ax2[0], no_labels=True)
        
    else:
        model_top = AgglomerativeClustering(n_clusters = sizes[0], metric='euclidean', linkage='ward', compute_distances=True) 
        model_mid = AgglomerativeClustering(n_clusters = sizes[1], metric='euclidean', linkage='ward', compute_distances=True)
        result_top = model_top.fit(X) 
        result_mid = model_mid.fit(X)
        top_labels = result_top.labels_
        mid_labels = result_mid.labels_
        
        sbn.heatmap(pd.DataFrame(np.array([top_labels,  
                                           labels[0].tolist()]).T,
                                 columns = ['Traditional HC Top','Truth Top']),
                    ax = ax1[0])
        
        sbn.heatmap(pd.DataFrame(np.array([mid_labels, 
                                           labels[1].tolist()]).T,
                                 columns = ['Traditional HC Middle', 'Truth Middle']),
                    ax = ax1[1])
        
        ax2[0].set_xlabel = 'Dendrogram Top'
        ax2[1].set_xlable = 'Dendrogram Middle'
        # plot the top three levels of the dendrogram
        plot_dendrogram(model_top, truncate_mode=None, p=3, ax = ax2[0], no_labels=True)
        
    if args.save_results:
        fig.savefig(args.sp+'Classical_hierarchical_result.pdf')
        fig.savefig(args.sp+'Classical_hierarchical_result.png', dpi = 500)
        
    return [mid_labels, top_labels]
    
    
    
def read_benchmark_CORA(args, PATH, use_split = True, percent_train = 0.8, percent_test = 0.2):
    
    testing_set = []
    valid_set = []
    if args.read_from == 'local':
        readpath = 'C:/Users/Bruin/OneDrive/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/'
    elif args.read_from == 'cluster':
        readpath = '/mnt/ceph/jarredk//HGRN_repo/Simulated_Hierarchies/DATA/'
    
    dataset = Planetoid(root=os.path.join(readpath+PATH), name = 'Cora')
    data = dataset[0]
    
    if use_split:
        
        data = split_benchmark_data(data, percent_train, percent_test)
        
        # Get train, validation, and test splits
        train_data = format_split_data(data, data.train_mask)
        val_data = format_split_data(data, data.val_mask)
        test_data = format_split_data(data, data.test_mask)
    else:
        train_data = format_split_data(data, 
                                       mask = torch.tensor(np.repeat(True, data.x.shape[0])))
        val_data = []
        test_data = []
        
    #format training data
    training_set = format_data(args, train_data)
    #format testing_data
    if test_data:
        testing_set = format_data(args, test_data)
    #format validation data
    if val_data:
        valid_set = format_data(args, val_data)
 
    
    return (training_set, testing_set, valid_set)
    
    
    
    
    
def split_benchmark_data(data, train_size: float = 0.8, test_size: float = 0.1):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    test_size = test_size/2
    
    train_end = int(train_size * num_nodes)
    test_end = train_end + int(test_size * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    test_mask[indices[train_end:test_end]] = True
    val_mask[indices[test_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

# Extract node features and edges for each split
def format_split_data(data, mask):
    node_idx = mask.nonzero(as_tuple=False).view(-1)
    edge_index, edge_attr = subgraph(node_idx, 
                                     data.edge_index, 
                                     data.edge_attr, 
                                     relabel_nodes=True, 
                                     num_nodes=data.num_nodes)
    
    # Create an empty adjacency matrix with the same number of nodes
    adj_matrix = torch.zeros((mask.sum().item(), mask.sum().item()))
    adj_matrix[edge_index[0], edge_index[1]] = 1
    
    split_data = {}
    split_data['x'] = data.x[mask]
    split_data['y'] = data.y[mask]
    split_data['edge_index'] = edge_index
    split_data['edge_attr'] = data.edge_attr if data.edge_attr is not None else None
    split_data['adjacency_matrix'] = adj_matrix
    return split_data


    
    
def format_data(args, data):
    label_array = data['y'].detach().numpy()
    sort_indices = np.argsort(label_array)
    labels = [label_array[sort_indices].tolist()]
    feature_matrix = data['x'].detach().numpy()[sort_indices,:]
    X = torch.Tensor(feature_matrix)
    #in_adj = resort_graph(to_dense_adj(data['edge_index'])[0], sort_indices)
    in_adj = resort_graph(data['adjacency_matrix'], sort_indices)
    nodes = in_adj.shape[0]
    
    if args.use_true_graph:
        A = in_adj+torch.eye(nodes)
    else:    
        in_graph, in_adj = get_input_graph(X = feature_matrix, 
                                           method = 'Correlation', 
                                           r_cutoff = args.correlation_cutoff)
        if args.correlation_cutoff == 1:
            A = torch.Tensor(in_adj)
        else:
            A = torch.Tensor(in_adj)+torch.eye(nodes)
            
    if args.use_true_communities == True:
        comm_sizes = [len(np.unique(labels))]
    else:
        comm_sizes = args.community_sizes
    
    return X, A, labels
    
    








def safe_node_clust_eval(true_labels, pred_labels, verbose=False):
    """Safe evaluation wrapper that handles missing data"""
    if true_labels is None or pred_labels is None:
        return (None, None, None, None)
    try:
        return node_clust_eval(true_labels, pred_labels, verbose)
    except:
        return (None, None, None, None)
    

def generate_output_table(truth, louv_pred, kmeans_pred, thc_pred, hcd_preds, verbose=True):
    """Generate evaluation metrics table for clustering results."""
    
    # Initialize all metrics with None
    metrics = {
        'kmeans': {'middle': (None, None, None, None), 'top': (None, None, None, None)},
        'thc': {'middle': (None, None, None, None), 'top': (None, None, None, None)},
        'louvain': {'middle': (None, None, None, None), 'top': (None, None, None, None)},
        'hcd': {'middle': (None, None, None, None), 'top': (None, None, None, None)}
    }

    # Helper function for safe evaluation
    def safe_eval(true_labels, pred_labels):
        if true_labels is None or pred_labels is None:
            return (None, None, None, None)
        try:
            return node_clust_eval(true_labels, pred_labels, verbose=False)
        except:
            return (None, None, None, None)

    # KMeans evaluation
    if kmeans_pred and len(kmeans_pred) > 1:
        metrics['kmeans']['top'] = safe_eval(
            truth[0] if len(truth) > 0 else None,
            kmeans_pred[0]
        )
        metrics['kmeans']['middle'] = safe_eval(
            truth[1] if len(truth) > 1 else None,
            kmeans_pred[1]
        )

    # THC evaluation
    if thc_pred and len(thc_pred) > 1:
        metrics['thc']['top'] = safe_eval(
            truth[0] if len(truth) > 0 else None,
            thc_pred[1]
        )
        metrics['thc']['middle'] = safe_eval(
            truth[1] if len(truth) > 1 else None,
            thc_pred[0]
        )

    # Louvain evaluation - key change here to match original Git style
    if louv_pred is not None:
        # Single set of predictions compared against both layers
        metrics['louvain']['top'] = safe_eval(
            truth[0] if len(truth) > 0 else None,
            louv_pred
        )
        if len(truth) > 1:  # Only evaluate middle layer if truth has it
            metrics['louvain']['middle'] = safe_eval(
                truth[1],
                louv_pred
            )

    # HCD evaluation
    if hcd_preds and len(hcd_preds) > 0:
        metrics['hcd']['top'] = safe_eval(
            truth[0] if len(truth) > 0 else None,
            hcd_preds[0] if len(hcd_preds) > 0 else None
        )
        if len(hcd_preds) > 1:
            metrics['hcd']['middle'] = safe_eval(
                truth[1] if len(truth) > 1 else None,
                hcd_preds[1]
            )

    # Create dataframe
    data = {
        'Homogeneity': [
            metrics['louvain']['middle'][0], metrics['louvain']['top'][0],
            metrics['kmeans']['middle'][0], metrics['kmeans']['top'][0],
            metrics['thc']['middle'][0], metrics['thc']['top'][0],
            metrics['hcd']['middle'][0], metrics['hcd']['top'][0]
        ],
        'Completeness': [
            metrics['louvain']['middle'][1], metrics['louvain']['top'][1],
            metrics['kmeans']['middle'][1], metrics['kmeans']['top'][1],
            metrics['thc']['middle'][1], metrics['thc']['top'][1],
            metrics['hcd']['middle'][1], metrics['hcd']['top'][1]
        ],
        'NMI': [
            metrics['louvain']['middle'][2], metrics['louvain']['top'][2],
            metrics['kmeans']['middle'][2], metrics['kmeans']['top'][2],
            metrics['thc']['middle'][2], metrics['thc']['top'][2],
            metrics['hcd']['middle'][2], metrics['hcd']['top'][2]
        ],
        'ARI': [
            metrics['louvain']['middle'][3], metrics['louvain']['top'][3],
            metrics['kmeans']['middle'][3], metrics['kmeans']['top'][3],
            metrics['thc']['middle'][3], metrics['thc']['top'][3],
            metrics['hcd']['middle'][3], metrics['hcd']['top'][3]
        ]
    }

    df = pd.DataFrame(data)
    df.index = [
        'Louvain vs Middle Truth', 'Louvain vs Top Truth',
        'KMeans vs Middle Truth', 'KMeans vs Top Truth',
        'Ward Linkage vs Middle Truth', 'Ward Linkage vs Top Truth', 
        'HCD (middle) vs Middle Truth', 'HCD (top) vs Top Truth'
    ]
    
    # Create clean copy without color formatting for saving
    df2 = df.copy()
    
    # Apply color formatting only to display version
 
    if verbose:
        pd.set_option('display.max_columns', None)
        print(df.to_string())
    
    return df, df2





def post_hoc(args, output, k_layers, truth, bp, louv_pred,
             kmeans_pred, thc_pred, predicted, verbose=True):
    
    data = output.training_data['X_train']
    adjacency = output.training_data['A_train']
    I = torch.eye(data.shape[0]).cpu()
    
    # Initialize best_iter_metrics as None
    best_iter_metrics = None
    
    if truth:
        # Unpack the tuple returned by generate_output_table with all required arguments
        df, df2 = generate_output_table(truth=truth, 
                                      louv_pred=louv_pred,
                                      kmeans_pred=kmeans_pred,
                                      thc_pred=thc_pred,
                                      hcd_preds=predicted,
                                      verbose=verbose)
        
        best_iter_metrics = df  # Assign the primary dataframe
        
        if args.save_results:
            # Save the primary dataframe
            best_iter_metrics.to_csv(os.path.join(args.sp+f'best_iteration_metrics_{args.which_net}.csv'))
    
    # Rest of your processing code...
    X_hat, A_hat, X_all, A_all, P_all, S_relab, S_all, S_sub, S_sizes, AW = output.model_output_history[bp]
     
    if args.use_method == 'bottom_up':
        P_pred_bp = P_all
    else:
        P_pred_bp = [P_all[0].cpu(), torch.cat(P_all[1]).cpu()][::-1]

    post_hoc_embedding(graph=adjacency-I, #subtract out self loops
                            embed_X=output.latent_features,
                            data=data, 
                            probabilities=P_pred_bp,
                            size=150.0,
                            labels=predicted,
                            truth=truth,
                            fs=10,
                            node_size=25, 
                            cm='plasma',
                            font_size=10,
                            include_3d_plots=True,
                            save=args.save_results,
                            path=args.sp)

    plot_clust_heatmaps(A=adjacency, 
                        A_pred=output.reconstructed_adj-I, #subtract out self loops 
                        X=data,
                        X_pred=output.reconstructed_features,
                        true_labels=truth, 
                        pred_labels=predicted, 
                        layers=k_layers+1, 
                        epoch=bp, 
                        save_plot=args.save_results, 
                        sp=args.sp+'best_iteration_'+str(bp))

    # t-SNE and PCA visualizations
    TSNE_data = TSNE(n_components=3, 
                   learning_rate='auto',
                   init='random', 
                   perplexity=3).fit_transform(output.latent_features.detach().numpy())
    PCs = PCA(n_components=3).fit_transform(output.latent_features.detach().numpy())

    if not isinstance(truth, type(None)):
        fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12,10))
        # t-SNE plot
        ax1[0].scatter(TSNE_data[:,0], TSNE_data[:,1], s=25, c=truth[0], cmap='plasma')
        ax1[0].set_xlabel('Dimension 1')
        ax1[0].set_ylabel('Dimension 2')
        ax1[0].set_title('t-SNE Embedding Bottleneck (true labels)')
            
        # PCA plot
        ax1[1].scatter(PCs[:,0], PCs[:,1], s=25, c=truth[0], cmap='plasma')
        ax1[1].set_xlabel('Dimension 1')
        ax1[1].set_ylabel('Dimension 2')
        ax1[1].set_title('PCA Embedding Bottleneck (true labels)')

        if args.save_results:
            fig.savefig(args.sp+'topclusters_plotted_on_embeds.pdf')
            
    if truth:
        return best_iter_metrics
    return None





def plot_embeddings_heatmap(embeddings_list, use_correlations = False, verbose =False):
    """
    Plots a heatmap of model embeddings with a slider for epochs.

    Parameters:
    embeddings_list (list of np.ndarray): A list of 2D numpy arrays where each array
                                          represents the embeddings at a particular epoch.
    """
    # Ensure all embeddings have the same shape
    if use_correlations:
        mats = [np.corrcoef(i) for i in embeddings_list]
    else:
        mats = embeddings_list
    shape = mats[0].shape
    assert all(emb.shape == shape for emb in mats), "All embeddings must have the same shape."

    # Create initial heatmap for the first epoch
    fig = go.Figure(data=go.Heatmap(z=mats[0]))

    # Add frames for each epoch
    frames = [go.Frame(data=go.Heatmap(z=emb), name=f"Epoch {i}") for i, emb in enumerate(mats)]
    fig.frames = frames

    # Update layout with slider
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Epoch:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f"Epoch {i}"], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                    "label": str(i),
                    "method": "animate"
                } for i in range(len(embeddings_list))
            ]
        }]
    )

    if verbose:
        fig.show()
    
    return fig

    
    
    
def generate_attention_graph(args: object, A: torch.Tensor, AW: dict, gene_labels: list, S_all: list, 
                             cutoff: Optional[Union[str, float]] = 'mean'):
    
    """
    Generates a graph from attention weights of a fitted HCD model.

    Parameters
    ----------
    args : object
        An object containing arguments, including `sp` (save path) and `save_results` (a boolean flag).
    A : torch.Tensor
        The adjacency matrix of the original graph.
    AW : dict
        A dictionary containing attention weights. It should have an 'encoder' key with a value that is a list of tuples,
        where each tuple contains two tensors representing the source and target node indices, and another tensor representing
        the attention weights between these nodes.
    gene_labels : list
        A list of labels for the genes (nodes) in the graph.
    S_all : list
        A list of tensors representing top and middle labels for the genes.
    cutoff : str or float, optional
        The threshold value to filter attention weights. If 'mean', the mean of the attention weights is used as the threshold.
        If a float, this value is used directly as the threshold. Defaults to 'mean'.

    Returns
    -------
    new_G : nx.Graph
        The generated attention graph with weighted edges.
    weighted_adj : np.ndarray
        The adjacency matrix of the generated attention graph.

    Notes
    -----
    This function saves a histogram of attention weights and the generated graph (if `save_results` is True) to files named
    'histogram_attent_weights.pdf' and 'gene_network.gexf', respectively, in the directory specified by `args.sp`.
    """
    G = nx.from_numpy_array(A.cpu().detach().numpy())
    new_G = nx.Graph()

    edgelist = []
    nodelist = []

    attent_weights = AW['encoder'][0][1].mean(dim=1)
    atn_edges = [(i,j) for idx, (i,j) in enumerate(zip(AW['encoder'][0][0][0].cpu().detach().numpy(), AW['encoder'][0][0][1].cpu().detach().numpy()))]

    fig, ax = plt.subplots(1,1, figsize = (12, 10))
    ax.hist(attent_weights.cpu().detach().numpy())
    ax.set_title('Histogram of attention weights')
    fig.savefig(args.sp+'histogram_attent_weights.pdf')
    
    if cutoff:
        if cutoff == 'mean':
            ct = float(attent_weights.mean())
        else: 
            ct = cutoff
    else: 
        ct = float(attent_weights.min())


    for index, edge in enumerate(atn_edges):
        if edge[0] != edge[1]:
            attn_weight = attent_weights[index].cpu().detach().numpy()
            if attn_weight >= ct:
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

    if args.save_results:
        nx.write_gexf(new_G, args.sp+"gene_network.gexf")

    weighted_adj = nx.to_numpy_array(new_G)
    
    return new_G, weighted_adj