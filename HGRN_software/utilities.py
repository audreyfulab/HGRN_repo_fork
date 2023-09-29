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
from sklearn.metrics import roc_auc_score, f1_score, normalized_mutual_info_score,homogeneity_score, completeness_score






def Modularity(A,P):
    r = A.sum(dim = 1)
    n = A.sum()
    B = A - torch.outer(r,r)
    modularity = (1/(4*n))*torch.trace(B)
    return modularity
    







def WCSS(X = None, clustlabs = None, num_clusters=None, norm_degree = 2,
         weight_by = ['kmeans','anova']):
    
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
    

    






def BCSS(X = None, cluster_centroids=None, numclusts = None, norm_degree = 2,
         weight_by = ['kmeans','anova']):
    
    #X_tensor = torch.tensor(X, requires_grad=True)
    supreme_centroid = torch.mean(X, dim = 0)
    pdist = nn.PairwiseDistance(p=norm_degree)
    if weight_by == 'kmeans':
        BCSS_mean_distance = torch.mean(pdist(cluster_centroids, supreme_centroid))
    else:
        BCSS_mean_distance = (1/(numclusts - 1))*torch.sum(pdist(cluster_centroids, supreme_centroid))
    
    
    return BCSS_mean_distance










def resort_graph(A, sort_list):
    A_temp1 = A[:,sort_list]
    A_sorted = A_temp1[sort_list, :]
    return A_sorted










def node_clust_eval(true_labels, pred_labels, verbose = True):
    homogeneity, completeness, nmi = homogeneity_score(true_labels, pred_labels),\
                                     completeness_score(true_labels, pred_labels),\
                                     normalized_mutual_info_score(true_labels, pred_labels)
    if verbose is True:
        print("\nhomogeneity = ",homogeneity,"\ncompleteness = ", completeness, "\nnmi = ", nmi)
    
    return np.array([homogeneity, completeness, nmi])