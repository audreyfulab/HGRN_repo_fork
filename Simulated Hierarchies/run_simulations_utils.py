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
from simulation_software.simulation_utilities import compute_beth_hess_comms, plot_nodes, post_hoc_embedding
import torch
import networkx as nx
from community import community_louvain as cl
import matplotlib.pyplot as plt
import seaborn as sbn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, subgraph
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from colorama import Fore, Style
from functools import partial
import os

def load_simulated_data(args):
    
    
    if args.read_from == 'local':
        readpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/'
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
            loadpath_main = os.path.join(readpath+'Toy_examples/Intermediate_examples_unique_dist/')
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



def set_up_model_for_simulated_data(args, loadpath_main, grid1, grid2, grid3, stats, device, **kwargs):
    
    #run simulations
    for idx, value in enumerate(zip(grid1, grid2, grid3)):
        
        if idx == args.which_net:
            
            #print network statistics
            print(f'summary of simulated network statistics for collection {args.dataset}:network {idx}')
            print(stats.loc[idx])
            print('='*55)
            
            #pdb.set_trace()
            layers = value[2][1]
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
    X = torch.Tensor(pe_sorted).requires_grad_()
    #generate input graphs
    if args.use_true_graph:
        A = torch.Tensor(true_adj_undi[:nodes,:nodes]).requires_grad_()+torch.eye(nodes)
    else:    
        in_graph, in_adj = get_input_graph(X = pe_sorted, 
                                           method = 'Correlation', 
                                           r_cutoff = args.correlation_cutoff)
        if args.correlation_cutoff == 1:
            A = torch.Tensor(in_adj).requires_grad_()
        else:
            A = torch.Tensor(in_adj).requires_grad_()+torch.eye(nodes)
            
    return X, A, target_labels, comm_sizes
    
    
    
    
    
    
    
def handle_output(args, output, A, comm_sizes):
    
    nodes = A.shape[0]
    #preallocate 
    comm_loss = []
    recon_A = []
    recon_X = []
    predicted_comms = []
    metrics = []
    
    
    #record best losses and best performances
    total_loss = np.array(output[-5])
    best_loss_idx = total_loss.tolist().index(min(total_loss))
                
    if args.return_result == 'best_perf_top':
        perf_mid = []
        temp_top = [i[0] for i in output[-1]]
        perf_top = np.array(temp_top)
        best_perf_idx = perf_top[:,2].tolist().index(max(perf_top[:,2]))
        print('Best Performance Top Layer: Epoch = {}, \nHomogeneity = {},\nCompleteness = {}, \nNMI = {}'.format(
            best_perf_idx,      
            perf_top[best_perf_idx, 0],
            perf_top[best_perf_idx, 1],
            perf_top[best_perf_idx, 2]
            ))
    else:
        perf_top = []
        temp_mid = [i[1] for i in output[-1]]
        perf_mid = np.array(temp_mid)
        best_perf_idx = perf_mid[:,2].tolist().index(max(perf_mid[:,2]))
        print('Best Performance Middle Layer: Epoch = {}, \nHomogeneity = {},\nCompleteness = {}, \nNMI = {}'.format(
            best_loss_idx,
            perf_mid[best_perf_idx, 0],
            perf_mid[best_perf_idx, 1],
            perf_mid[best_perf_idx, 2]
            ))
    
    #update lists
    comm_loss.append(np.round(output[-4][best_loss_idx], 4))
    recon_A.append(np.round(output[-3][best_loss_idx], 4))
    recon_X.append(np.round(output[-2][best_loss_idx], 4))
        
                
    #compute the upper limit of communities, the beth hessian, and max modularity
    upper_limit = torch.sqrt(torch.sum(A-torch.eye(nodes)))
    beth_hessian = compute_beth_hess_comms((A-torch.eye(nodes)).cpu().detach().numpy())
    max_mod = 1 - (2/upper_limit)
                
    #output assigned labels for all layers
    S_sub, S_layer, S_all = trace_comms(output[6], comm_sizes)
    predicted_comms.append(tuple([len(np.unique(i)) for i in S_layer]))
    
    
    if len(comm_sizes) == 2:
        metrics.append({'Top': tuple(np.round(output[-1][best_perf_idx][0], 4))})
        
    else:
        metrics.append({'Top': tuple(np.round(output[-1][best_perf_idx][0], 4)),
                        'Middle': tuple(np.round(output[-1][best_perf_idx][-1], 4))})
    
    return (beth_hessian, comm_loss, recon_A, recon_X, perf_mid, perf_top, upper_limit, max_mod, (best_perf_idx, best_loss_idx), metrics, predicted_comms, (S_sub, S_layer, S_all))
    
    
def run_louvain(args, output, A, layers, savepath, best_perf_idx, labels):
    
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
    fig, ax = plt.subplots()
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
        
        
        
    return [mid_labels, top_labels]
    
    
    
def read_benchmark_CORA(args, PATH, use_split = True, percent_train = 0.8, percent_test = 0.2):
    
    testing_set = []
    valid_set = []
    if args.read_from == 'local':
        readpath = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/DATA/'
    elif args.read_from == 'cluster':
        readpath = '/mnt/ceph/jarredk//HGRN_repo/Simulated_Hierarchies/DATA/'
    
    dataset = Planetoid(root=os.path.join(readpath+PATH, name='Cora'))
    data = dataset[0]
    
    if use_split:
        
        data = split_data(data, percent_train, percent_test)
        
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
    
    
    
    
    
    
def split_data(data, train_size: float = 0.8, val_size: float = 0.1):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_end = int(train_size * num_nodes)
    val_end = train_end + int(val_size * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

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
    X = torch.Tensor(feature_matrix).requires_grad_()
    #in_adj = resort_graph(to_dense_adj(data['edge_index'])[0], sort_indices)
    in_adj = resort_graph(data['adjacency_matrix'], sort_indices)
    nodes = in_adj.shape[0]
    
    if args.use_true_graph:
        A = in_adj.requires_grad_()+torch.eye(nodes)
    else:    
        in_graph, in_adj = get_input_graph(X = feature_matrix, 
                                           method = 'Correlation', 
                                           r_cutoff = args.correlation_cutoff)
        if args.correlation_cutoff == 1:
            A = torch.Tensor(in_adj).requires_grad_()
        else:
            A = torch.Tensor(in_adj).requires_grad_()+torch.eye(nodes)
            
    if args.use_true_communities == True:
        comm_sizes = [len(np.unique(labels))]
    else:
        comm_sizes = args.community_sizes
    
    return X, A, labels, comm_sizes
    
    








 
def generate_output_table(truth, louv_pred, kmeans_pred, hcd_preds, verbose = True):
    
    
    
    if kmeans_pred:
        if len(truth) > 1:
            homo_k2m, comp_k2m, nmi_k2m, ari_k2m = node_clust_eval(truth[1], 
                                                                   kmeans_pred[0],
                                                                   verbose = False)
        else:
            homo_k2m, comp_k2m, nmi_k2m, ari_k2m = (None,None,None,None)
            
        homo_k2t, comp_k2t, nmi_k2t, ari_k2t =node_clust_eval(truth[0], 
                                                              kmeans_pred[1],
                                                              verbose = False)
    else:
        homo_k2m, comp_k2m, nmi_k2m, ari_k2m = (None,None,None,None)
        homo_k2t, comp_k2t, nmi_k2t, ari_k2t = (None,None,None,None)
    
    

    if louv_pred:
        if len(truth) > 1:
            homo_l2m, comp_l2m, nmi_l2m, ari_l2m = node_clust_eval(truth[1], louv_pred,
                                                                   verbose = False)
        else:
            homo_l2m, comp_l2m, nmi_l2m, ari_l2m = (None,None,None)
            
        homo_l2t, comp_l2t, nmi_l2t, ari_l2t =node_clust_eval(truth[0], louv_pred,
                                                              verbose = False)
    else:
        homo_l2m, comp_l2m, nmi_l2m, ari_l2m = (None,None,None)
        homo_l2t, comp_l2t, nmi_l2t, ari_l2t = (None,None,None)


    homo_m2m, comp_m2m, nmi_m2m, ari_m2m = node_clust_eval(true_labels=truth[::-1][0], 
                                                           pred_labels = hcd_preds[0], 
                                                           verbose=False)

    homo_m2t, comp_m2t, nmi_m2t, ari_m2t = node_clust_eval(true_labels=truth[::-1][1], 
                                                           pred_labels = hcd_preds[0], 
                                                           verbose=False)

    homo_t2t, comp_t2t, nmi_t2t, ari_t2t = node_clust_eval(true_labels=truth[::-1][1], 
                                                           pred_labels = hcd_preds[1], 
                                                           verbose=False)
    
    data = {'Homogeneity': [homo_l2m, homo_l2t, homo_k2m, homo_k2t, homo_m2m, homo_m2t, homo_t2t],
            'Completeness': [comp_l2m, comp_l2t, comp_k2m, comp_k2t, comp_m2m, comp_m2t, comp_t2t],
            'NMI': [nmi_l2m, nmi_l2t, nmi_k2m, nmi_k2t, nmi_m2m, nmi_m2t, nmi_t2t],
            'ARI': [ari_l2m, ari_l2t, ari_k2m, ari_k2t, ari_m2m, ari_m2t, ari_t2t]}
    df = pd.DataFrame(data)
    df.index.name = 'Comparison'
    df.index = ['Louvain vs Middle Truth', 'Louvain vs Top Truth',
                'KMeans vs Middle Truth', 'KMeans vs Top Truth',
                'HCD (middle) vs Middle Truth','HCD (middle) vs Top Truth',
                'HCD (top) vs Top Truth']
    
    df2 = df.copy()
    
    
    def red_text(val):
        return Fore.RED+f"{str(val)}"+Style.RESET_ALL

    def green_text(val):
        return Fore.GREEN+f"{str(val)}"+Style.RESET_ALL
    
    def color_values(val, thresh):
        return green_text(val) if val >= thresh else red_text(val)

    
    df['Homogeneity'] = df['Homogeneity'].apply(lambda x: color_values(np.round(x,4), np.nanmax(np.round(df['Homogeneity'], 4))))
    df['Completeness'] = df['Completeness'].apply(lambda x: color_values(np.round(x,4), np.nanmax(np.round(df['Completeness'],4))))
    df['NMI'] = df['NMI'].apply(lambda x: color_values(np.round(x,4), np.nanmax(np.round(df['NMI'],4))))
    df['ARI'] = df['ARI'].apply(lambda x: color_values(np.round(x,4), np.nanmax(np.round(df['ARI'],4))))
    
    

    # Print the HTML table to the console
    if verbose:
        pd.set_option('display.max_columns', None)
        print(df)
    
    return df2











def post_hoc(args, output, data, adjacency, k_layers, truth, bp, louv_pred,
             kmeans_pred, predicted, verbose = True):
    
    best_iter_metrics = generate_output_table(truth, 
                                              louv_pred,
                                              kmeans_pred,
                                              predicted,
                                              verbose = verbose)
    
    
    A_pred_bp = output[0][bp][1]
    X_pred_bp = output[0][bp][0]
    embed_pred_bp = output[0][bp][2][0] 
    P_pred_bp = output[0][bp][4]
    Comm1_proj_bp = output[0][bp][2][1]
        
    if args.save_results:
        best_iter_metrics.to_csv(os.path.join(args.sp+f'best_iteration_metrics_{args.return_result}.csv'))



    post_hoc_embedding(graph=adjacency-torch.eye(data.shape[0]), 
                            embed_X = embed_pred_bp,
                            data = data, 
                            probabilities = P_pred_bp,
                            size = 150.0,
                            labels = predicted,
                            truth = truth[::-1],
                            fs=10,
                            node_size = 25, 
                            cm = 'plasma',
                            font_size = 10,
                            include_3d_plots = True,
                            save = args.save_results,
                            path = args.sp)


    plot_clust_heatmaps(A = adjacency, 
                        A_pred = A_pred_bp-torch.eye(data.shape[0]), 
                        X = data,
                        X_pred = X_pred_bp,
                        true_labels = truth, 
                        pred_labels = predicted, 
                        layers = k_layers+1, 
                        epoch = bp, 
                        save_plot = args.save_results, 
                        sp = args.sp+'best_iteration_'+str(bp))


    #tsne
    TSNE_data=TSNE(n_components=3, 
                   learning_rate='auto',
                   init='random', 
                   perplexity=3).fit_transform(embed_pred_bp.detach().numpy())
    #pca
    PCs = PCA(n_components=3).fit_transform(embed_pred_bp.detach().numpy())


    fig, (ax1, ax2) = plt.subplots(2,2, figsize = (12,10))
    #tsne plot
    ax1[0].scatter(TSNE_data[:,0], TSNE_data[:,1], s = 25, c = truth[0], cmap = 'plasma')
    ax1[0].set_xlabel('Dimension 1')
    ax1[0].set_ylabel('Dimension 2')
    ax1[0].set_title(' t-SNE Embedding Bottleneck (true labels)')
    #adding node labels
        
    #PCA plot
    ax1[1].scatter(PCs[:,0], PCs[:,1], s = 25, c = truth[0], cmap = 'plasma')
    ax1[1].set_xlabel('Dimension 1')
    ax1[1].set_ylabel('Dimension 2')
    ax1[1].set_title(' PCA Embedding Bottleneck (true labels)')

    if not args.add_output_layers:
        x1 = torch.mm(embed_pred_bp, Comm1_proj_bp.transpose(0,1))

        TSNE_data2=TSNE(n_components=3, 
                        learning_rate='auto',
                        init='random', 
                        perplexity=3).fit_transform(x1.detach().numpy())
        #pca
        PCs2 = PCA(n_components=3).fit_transform(x1.detach().numpy())

        #tsne plot
        ax2[0].scatter(TSNE_data2[:,0], TSNE_data2[:,1], s = 25, c = truth[0], cmap = 'plasma')
        ax2[0].set_xlabel('Dimension 1')
        ax2[0].set_ylabel('Dimension 2')
        ax2[0].set_title(' t-SNE Embedding Comm1-projection (true_labels)')
        #adding node labels
        
        #PCA plot
        ax2[1].scatter(PCs2[:,0], PCs2[:,1], s = 25, c = truth[0], cmap = 'plasma')
        ax2[1].set_xlabel('Dimension 1')
        ax2[1].set_ylabel('Dimension 2')
        ax2[1].set_title(' PCA Embedding Comm1-projection (true_labels)')

    if args.save_results == True:
        fig.savefig(args.sp+'topclusters_plotted_on_embeds.pdf')




    
    
    
    
    
    
    
    