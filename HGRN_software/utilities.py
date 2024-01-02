# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:59:45 2023

@author: Bruin
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, f1_score, normalized_mutual_info_score,homogeneity_score, completeness_score
from sklearn.neighbors import kneighbors_graph
import scipy as spy
import seaborn as sbn
import matplotlib.pyplot as plt

# A simple function which computes the modularity of a graph based on a set of 
#community assignments
def Modularity(A,P,res=1):
    r = A.sum(dim = 1)
    n = A.sum()
    B = A - res*(torch.outer(r,r) / n)
    modularity = torch.trace(torch.mm(P.T, torch.mm(B, P)))/(n)
    return modularity
    






#This function computes the within cluster sum of squares (WCSS)
def WCSS(X = None, clustlabs = None, num_clusters=None, norm_degree = 2,
         weight_by = ['kmeans','anova']):
    
    """
    
    """
    
    #X_tensor = torch.tensor(X, requires_grad=True)
    num_features = X.shape[1]
    num_nodes = X.shape[0]
    centroid_mat = torch.zeros(num_clusters, num_features)
    nodes = torch.arange(num_nodes)
    total_wcss = torch.zeros(1).float()
    clust_IDs = torch.unique(clustlabs)
    
    for i in np.arange(num_clusters):
        which = nodes[clustlabs == clust_IDs[i]]
        clust = torch.index_select(X, 0, which)
        centroid_mat[i,:] = torch.mean(clust, dim = 0)
        pdist = nn.PairwiseDistance(p=norm_degree)
        if weight_by == 'kmeans':
            total_wcss += torch.mean(pdist(clust, centroid_mat[i,:].reshape(1, num_features)))
        else:
            total_wcss += torch.sum(pdist(clust, centroid_mat[i,:].reshape(1, num_features)))
            
    if weight_by == 'kmeans':
        MSW = (1/num_clusters)*total_wcss
    else:
        MSW = (1/(X.shape[0]-num_clusters))*total_wcss
            
    return MSW, centroid_mat
    

    





#This function computes the between cluster sum of squares (BCSS) for the node
#attribute matrix X given a set of identified clusters 
def BCSS(X = None, cluster_centroids=None, numclusts = None, norm_degree = 2,
         weight_by = ['kmeans','anova']):
    """
    X: node attribute matrix
    cluster_centroids: the centroids corresponding to a set of identified clusters
                       in X
    numclusts: number of inferred clusters in X
    norm_degree: the norm used to compute the distance
    weight_by: weighting scheme
    """
    #X_tensor = torch.tensor(X, requires_grad=True)
    supreme_centroid = torch.mean(X, dim = 0)
    pdist = nn.PairwiseDistance(p=norm_degree)
    if weight_by == 'kmeans':
        BCSS_mean_distance = torch.mean(pdist(cluster_centroids, supreme_centroid))
    else:
        BCSS_mean_distance = (1/(numclusts - 1))*torch.sum(pdist(cluster_centroids, supreme_centroid))
    
    
    return BCSS_mean_distance









#this simple function resorts a graph adjacency
def resort_graph(A, sort_list):
    """
    A: adjacency matrix
    sort_list: a list of indices for sorting the adjacency A
    """
    A_temp1 = A[:,sort_list]
    A_sorted = A_temp1[sort_list, :]
    return A_sorted









#this function computes the homogeniety, completeness and NMI for a set of true and
#predicted cluster labels. 
def node_clust_eval(true_labels, pred_labels, verbose = True):
    homogeneity, completeness, nmi = homogeneity_score(true_labels, pred_labels),\
                                     completeness_score(true_labels, pred_labels),\
                                     normalized_mutual_info_score(true_labels, pred_labels)
    if verbose is True:
        print("\nHomogeneity = ",homogeneity,"\nCompleteness = ", completeness, "\nNMI = ", nmi)
    
    return np.array([homogeneity, completeness, nmi])
















#this function adjusts community level labels for all hierarchical layers
#to ensure the maximum label number is not greater the the number of allowable
#communities in heach layer
def trace_comms(comm_list, comm_sizes):
    
    """
    comm_list: the list of community labels given as output from HCD model
    comm_sizes: the list of community sizes passed to HCD during model set up
    
    """
    
    comm_copy = comm_list.copy()
    comm_relabeled = comm_list.copy()
    layer =[]
    layer.append(comm_list[0])
    for i in range(0, len(comm_list)):
        #make sure the maximum community label is not greater than the number
        #communities for said layer. This is so that one_hot encoding doesn't
        #misinterpret the number of predicted communities
        comm_copy[i][comm_copy[i] == torch.max(comm_copy[i])] = comm_sizes[i]-1            
    #convert labels into one_hot matrix and trace assignments from layer back 
    #to original node size N 
    for i in range(1, len(comm_list)):
        layer.append(torch.mm(F.one_hot(comm_copy[i-1]), 
                                 F.one_hot(comm_copy[i])).argmax(1))
    
    comm_relabeled = layer.copy()
    for i in range(0, len(comm_list)):
        clusts = np.unique(layer[i])
        newlabs = np.arange(len(clusts))
        for j in range(0, len(clusts)):
            comm_relabeled[i][layer[i] == clusts[j]] = newlabs[j]
            
    return comm_copy, comm_relabeled, layer
















# a simple function for quickly combining predicted and true community labels
# into a dataframe for plotting
def gen_labels_df(pred_comms_list, truth, sorting, sort = True):
    
    """
    pred_comms_list: the predicted communities as output by trace_comms
    truth: the true labels (assumes they have sorted according to 'sorting')
    sorting: a list of indices for sorting the nodes in a particular order
    """
    pred_comms_list_cp = pred_comms_list.copy()
    num_layers = np.arange(0, len(pred_comms_list_cp))
    layer_nm = ['Communities'+'_Layer'+str(i) for i in num_layers]
    
    for i in range(0, len(pred_comms_list)):
        if sort == True:
            pred_comms_list_cp[i] = pred_comms_list_cp[i].detach().numpy()[sorting]
        else:
            pred_comms_list_cp[i] = pred_comms_list_cp[i].detach().numpy()
    
    for j in range(0, len(truth)):
        if len(truth[j]) > 0:    
            pred_comms_list_cp.append(truth[j])
            layer_nm.append('Truth_layer_'+str(j))
    df2 = pd.DataFrame(pred_comms_list_cp).T
    df2.columns = layer_nm
    
    return df2
















#utility to build the whole hierarchy adjacency from h1,h2,and h3 graphs
def build_true_graph(file='filename'):
    
    """
    This function constructs the true graph of the hierarchical simulated data
    
    """
    
    obj = np.load(file)
    num_files = len(obj.files)
    num_layers = (num_files - 2)/2
    nodes_layer_i = []
    print('number of layers detected = {}, items in file = {}'.format(
        num_layers, obj.files
        ))
    for i in np.arange(num_layers)[::-1]:
        nodes_layer_i.append(len(obj['layer'+str(int(i+1))]))
    
    
    total_nodes = sum(nodes_layer_i)
    
    adj = np.zeros(shape = [total_nodes, total_nodes])
    
    for i, j in zip(np.arange(num_layers), np.arange(num_layers)[::-1]):
        if i == 0:
            adj[:nodes_layer_i[int(i)], :nodes_layer_i[int(i)]] = obj['adj_layer'+str(int(j+1))]
        else:
            splice_idx = [sum(nodes_layer_i[:int(i)]), sum(nodes_layer_i[:int(i+1)])] 
            print('layer = {}, Adj Splice = {}'.format(
                str(int(j+1)), splice_idx
                ))
            adj[splice_idx[0]:splice_idx[1],splice_idx[0]:splice_idx[1]] = obj['adj_layer'+str(int(j+1))]
            
    return adj







def sort_labels(labels):
    #pdb.set_trace()
    true_labels = pd.DataFrame(labels, columns = ['Nodes'])
    new_list = []
    l = len(true_labels['Nodes'])
    for i in np.arange(l):
        new_list.append(list(map(int, true_labels['Nodes'][i].split('_'))))
        
    new_list_array = np.array(new_list)
    new_true_labels = pd.DataFrame(new_list_array[:,:2])
    new_true_labels.columns = ['toplabs', 'nodelabs']
    clusts = np.unique(new_true_labels['toplabs'])
        
    indices_for_clusts = []
    #extract indices for nodes belonging to each cluster
    for i in clusts:
        indices_for_clusts.append(new_true_labels[new_true_labels['toplabs'] == i].index.tolist())
        

    #pe = np.load(path+'gexp.npy').transpose()
    #reorganize pe so that nodes are sorted according to clusters 0,1,2..etc
    flat_list_indices = []
    for i in clusts:
        flat_list_indices.extend(indices_for_clusts[i])
        
        
    #construct labels for middle layer nodes
    if new_list_array.shape[1]>2:
        #num_middle_nodes = new_list_array.shape[0]/len(np.unique(new_list_array[:,1]))
        #temp = [np.repeat(i,len(np.unique(new_list_array[:,1]))).tolist() for i in np.arange(num_middle_nodes)]
        #middle_labels = np.array([int(i[0]) for i in np.array(temp).reshape((l,1)).tolist()])
        temp = [int(str(i[0])+str(i[1])) for i in new_list]
        middle_labels = np.array(temp.copy())
        midclusts = np.unique(temp)
        newclusts = np.arange(len(midclusts))
        for i in range(0, len(midclusts)):
            middle_labels[middle_labels == midclusts[i]] = newclusts[i]
            
        new_true_labels['middlelabs'] = middle_labels
        indices_for_clusts2 = []
        flat_list_indices2 = []
        #extract indices for nodes belonging to each cluster
        for i in newclusts:
            indices_for_clusts2.append(new_true_labels[new_true_labels['middlelabs'] == i].index.tolist())
            flat_list_indices2.extend(indices_for_clusts2[i])
        
        sorted_true_labels_middle = middle_labels[flat_list_indices2]
    else:
        flat_list_indices2 = []
        sorted_true_labels_middle = []


    #the true labels sorted by cluster
    sorted_true_labels_top = np.array(new_true_labels['toplabs'][flat_list_indices])

    return flat_list_indices, flat_list_indices2, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle








def LoadData(filename):
    unparsed_labels = pd.read_csv(filename+'_gexp.csv', index_col=0).columns.tolist()
    flat_list_indices, flat_list_indices2, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle = sort_labels(unparsed_labels)
    pe = np.load(filename+'_gexp.npy').transpose()
    true_adj = build_true_graph(filename+'.npz')
    G = nx.from_numpy_array(true_adj)
    G = G.to_undirected()
    true_adj_undi = nx.adjacency_matrix(G).toarray()
    #the true labels sorted by cluster

    return pe, true_adj_undi, flat_list_indices, flat_list_indices2, new_true_labels, sorted_true_labels_top, sorted_true_labels_middle
    


   
def corr_dist(u, v, w=None, centered=True):
    """
    Compute the correlation distance between two 1-D arrays.
    The correlation distance between `u` and `v`, is
    defined as
    .. math::
        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
    where :math:`\\bar{u}` is the mean of the elements of `u`
    and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0
    centered : bool, optional
        If True, `u` and `v` will be centered. Default is True.
    Returns
    -------
    correlation : double
        The correlation distance between 1-D array `u` and `v`.
    """
    #u = _validate_vector(u)
    #v = _validate_vector(v)
    #if w is not None:
    #    w = _validate_weights(w)
    if centered:
        umu = np.average(u, weights=w)
        vmu = np.average(v, weights=w)
        u = u - umu
        v = v - vmu
    uv = np.average(u * v, weights=w)
    uu = np.average(np.square(u), weights=w)
    vv = np.average(np.square(v), weights=w)
    dist = 1.0 - np.square(uv / np.sqrt(uu * vv))
    # Return absolute value to avoid small negative value due to rounding
    return np.abs(dist) 


def get_pcor(data, cutoff = 0.05):
    
    """
        this function takes in a numpy array with features in the rows and observations
        in the columns and computes the precision and partial correlation matrices. It
        then preforms hypothesis testing on the partial correaltions to return the undirected 
        adjacency matrix
    """
    
    #get shapes
    n = data.shape[1]
    #calc precision matrix
    H = np.linalg.inv(np.cov(data))
    hdim = H.shape[0]
    S = hdim-2
    #preallocate
    pcor_mat = np.array([1.0]*np.square(hdim)).reshape((hdim, hdim))
    stat_mat = np.array([0.0]*np.square(hdim)).reshape((hdim, hdim))
    pval_mat = np.array([0.0]*np.square(hdim)).reshape((hdim, hdim))
    adj = np.array([0]*np.square(hdim)).reshape((hdim, hdim))
    # calculate partial correlation and stat, and pvalue
    print('Calculating partial correaltion matrix...')
    for i in np.arange(hdim):
        for j in np.arange(hdim):
            if i != j:
                pcor = -((H[i,j])/(np.sqrt(H[i,i])*np.sqrt(H[j,j])))
                pcor_mat[i,j] = pcor 
                stat = (np.sqrt(n - S - 3)/2)*np.log((1+pcor_mat[i,j])/(1-pcor_mat[i,j]))
                stat_mat[i,j] = stat
                pval = 2*(1-spy.stats.norm.cdf(np.absolute(stat_mat[i,j])))
                pval_mat[i,j] = pval
                
                
            
                if pval_mat[i,j]<cutoff:
                    adj[i,j] = 1
            
    return pcor_mat, stat_mat, pval_mat, adj





def get_input_graph(X = None, method = ['KNN','Presicion','Correlation', 'DAG-GNN'], 
                    K = 30, metric = '1-R^2', alpha = 0.05, r_cutoff = 0.2):
    
    
    '''
    This function computes the input graph to the GNN using the desired method
    
    '''
    
    
    
    if method == 'KNN':
        if metric == '1-R^2':
            A = kneighbors_graph(X, n_neighbors = K, metric = corr_dist)
        else:    
            #calculate the kneighbors graph
            A = kneighbors_graph(X, n_neighbors = K, metric = metric)
        
        #convert to adjacency (this matrix is directed)
        A_adj = A.toarray()
        
    if method == 'Precision':
        
        pcors, stats, pvals, A_adj = get_pcor(X, cutoff=alpha)
        
    if method == 'Correlation':
        #get the absolute correlations
        cormat = np.absolute(np.corrcoef(X))
        A_adj = np.copy(cormat)
        A_adj[A_adj>r_cutoff] = 1
        A_adj[A_adj<=r_cutoff] = 0
        np.fill_diagonal(A_adj, 0)
        
        
    #if method == 'DAG-GNN':
        
        
        
    #also get graph
    A_graph = convert_adj_to_graph(adj=A_adj)
    #get the undirected knn
    A_undi = nx.to_numpy_array(A_graph)
    return A_graph, A_undi



def convert_adj_to_graph(adj):
    graph = nx.from_numpy_array(adj, parallel_edges=False, create_using=nx.Graph)
    #set as undirected graph
    undi_h3 = graph.to_undirected()
    #extract original node labels
    gx = nx.Graph().to_undirected()
    #add in the nodes from undi_h3
    gx.add_nodes_from(undi_h3.nodes)
    #a small function to bind two lists into a list of tuples
    
    #extract the two componets of the edgelist from undi_h3 into separate lists
    list1 = list(nx.to_pandas_edgelist(undi_h3)['source'])
    list2 = list(nx.to_pandas_edgelist(undi_h3)['target'])
    #merge the lists into a list of 2-tuples i.e (source, target)
    edge_tuples = merge(list1, list2)
    #add the edges into the graph
    gx.add_edges_from(edge_tuples)
    return gx










    
def merge(list1, list2):
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list






















# a simple function to plot the loss curves during training
def plot_loss(epoch, loss_history, recon_A_loss_hist, recon_X_loss_hist, mod_loss_hist,
              path='path/to/file', loss_func = ['Modularity','Clustering'], save = True):
    
    layers = len(mod_loss_hist[0])
    fig, (ax1, ax2) = plt.subplots(2,2, figsize=(12,10))
    #total loss
    ax1[0].plot(range(0, epoch+1), loss_history, label = 'Total Loss')
    ax1[0].set_xlabel('Training Epochs')
    ax1[0].set_ylabel('Total Loss')
    #reconstruction of graph adjacency
    ax1[1].plot(range(0, epoch+1), recon_A_loss_hist, label = 'Graph Reconstruction Loss')
    ax1[1].set_xlabel('Training Epochs')
    ax1[1].set_ylabel('Graph Reconstruction Loss')
    #reconstruction of node attributes
    ax2[0].plot(range(0, epoch+1), recon_X_loss_hist, label = 'Attribute Reconstruction Loss')
    ax2[0].set_xlabel('Training Epochs')
    ax2[0].set_ylabel('Attribute Reconstruction Loss')
    #community loss
    ax2[1].plot(range(0, epoch+1), np.array(mod_loss_hist))
    ax2[1].set_xlabel('Training Epochs')
    if loss_func == 'Modularity':
        ax2[1].set_ylabel('Modularity Loss')
    else:
        ax2[1].set_ylabel('Clustering Loss')
    ax2[1].legend(labels = [i for i in ['middle','top'][:layers]], loc = 'lower right')
    
    if save == True:
        fig.savefig(path+'training_loss_curve_epoch_'+str(epoch+1)+'.pdf')
        
    




# a simple function for plotting the performance curves during training
def plot_perf(update_time, performance_hist, epoch, path='path/to/file', save = True):
    #evaluation metrics
    layers = len(performance_hist[0])
    titles = ['Top Layer', 'Middle Layer']
    fig, ax = plt.subplots(1, 2, figsize=(12,10))
    for i in range(0, layers):
        layer_hist = [j[i] for j in performance_hist]
        #homogeneity
        ax[i].plot(np.arange(update_time), np.array(layer_hist)[:,0], label = 'Homogeneity')
        ax[i].plot(np.arange(update_time), np.array(layer_hist)[:,1], label = 'Completeness')
        ax[i].plot(np.arange(update_time), np.array(layer_hist)[:,2], label = 'NMI')
        ax[i].set_xlabel('Training Epochs')
        ax[i].set_ylabel('Performance')
        ax[i].set_title(titles[i]+' Performance')
        ax[i].legend()

        if save == True:
            fig.savefig(path+'performance_curve_epoch_'+str(epoch+1)+'_layer_'+str(i)+'.pdf')
            
            
            
#A simple wrapper to plot and save the networkx graph
def plot_nodes(A, labels, path, node_size = 5, font_size = 10, add_labels = False,
               save = True, **kwargs):
    fig, ax = plt.subplots()
    G = nx.from_numpy_array(A)
    if add_labels == True:
        clust_labels = {list(G.nodes)[i]: labels.tolist()[i] for i in range(len(labels))}
        nx.draw_networkx(G, node_color = labels, 
                         pos = nx.spring_layout(G, seed = 123),
                         labels = clust_labels,
                         font_size = font_size,
                         node_size = node_size,
                         cmap = 'plasma', **kwargs)
    else:
        nx.draw_networkx(G, node_color = labels, 
                         ax = ax, 
                         pos = nx.spring_layout(G, seed = 123),
                         font_size = font_size,
                         node_size = node_size, 
                         with_labels = False,
                         cmap = 'plasma', **kwargs)
    if save == True:    
        fig.savefig(path+'.pdf')
    
  
    
  
    
  
    
#A simple wrapper to plot and save the adjacency heatmap
def plot_adj(A, path, **kwargs):
    fig, ax = plt.subplots()
    sbn.heatmap(A, ax = ax, **kwargs)
    fig.savefig(path+'.png', dpi = 300)
    
    
    
    
    
    
    
    
    
    
    
    
    
#a simple function to plot the clustering heatmaps 
def plot_clust_heatmaps(A, A_pred, true_labels, pred_labels, layers, epoch, save_plot = True, sp = ''):
    fig1, ax1 = plt.subplots(1,2, figsize=(12,10))
    sbn.heatmap(A_pred.detach().numpy(), ax = ax1[0])
    sbn.heatmap(A.detach().numpy(), ax = ax1[1])
    
    fig2, ax2 = plt.subplots(1,2, figsize=(12,10))
    TL = true_labels.copy()
    TL.reverse()   
    if layers == 3:

        first_layer = pd.DataFrame(np.vstack((pred_labels[0], TL[0])).T,
                                   columns = ['Predicted_Middle','Truth_middle'])
        second_layer = pd.DataFrame(np.vstack((pred_labels[1], TL[1])).T,
                                        columns = ['Predicted_Top','Truth_Top'])
        sbn.heatmap(first_layer, ax = ax2[0])
        sbn.heatmap(second_layer, ax = ax2[1])
            
            
    else:
        first_layer = pd.DataFrame(np.vstack((pred_labels[0], TL[1])).T,
                                       columns = ['Predicted_Top','Truth_Top'])
        sbn.heatmap(first_layer, ax = ax2[0])
        
        
    if save_plot == True:
        fig1.savefig(sp+'epoch_'+str(epoch)+'_Adjacency_maps.png', dpi = 300)
        fig2.savefig(sp+'epoch_'+str(epoch)+'_heatmaps.png', dpi = 300) 