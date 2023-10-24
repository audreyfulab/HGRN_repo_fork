# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:19:48 2023

@author: Bruin
"""

from utilities import Modularity, build_true_graph, resort_graph, sort_labels
import networkx as nx
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F
import pdb


def compute_graph_STATs(A_all, comm_assign, layers):
    
    mod = []
    node_deg = []
    deg_within = []
    deg_between = []
    
    sort_indices, true_labels, sorted_labels, sorted_labels_mid = sort_labels(comm_assign)
    A_sorted = A_all[-1]
    #compute statistics for bottom layer
    mod.append(compute_modularity(Adj = A_sorted, sort_labels = sorted_labels))
    node_deg.append(compute_node_degree(A_sorted))
    deg_within.append(compute_node_degree_within(A=A_sorted, comm_labels = sorted_labels))
    deg_between.append(compute_node_degree_between(A=A_sorted, comm_labels = sorted_labels))
    
    #compute middle layer statistics 
    if layers > 2:
        mod.append(compute_modularity(Adj = A_sorted, sort_labels = sorted_labels_mid))
        node_deg.append(compute_node_degree(A_sorted))
        deg_within.append(compute_node_degree_within(A=A_sorted, comm_labels = sorted_labels_mid))
        deg_between.append(compute_node_degree_between(A=A_sorted, comm_labels = sorted_labels_mid))
        
    return mod, node_deg, deg_within, deg_between




#computes the modularity of an adjacency matrix based on a set of community 
#labels
def compute_modularity(Adj, sort_labels):
    A_tensor = torch.tensor(Adj).to(torch.float64)
    labels_tensor = torch.Tensor(sort_labels).to(torch.int64)
    P = F.one_hot(labels_tensor).to(torch.float64)
    mod = Modularity(A_tensor, P)
    
    return mod.detach().numpy()






#generic function compute node degree
def compute_node_degree(A):
    deg = ((1/2)*A.sum())/A.shape[0]
    return deg
    





#this function computes the average node degree within communities given by
#a set of supplied community assignment labels
def compute_node_degree_within(A, comm_labels):
    N = A.shape[0]
    deg_list = []
    comms = np.unique(comm_labels)
    for i in range(len(np.unique(comm_labels))):
        ix = np.arange(N)[comm_labels == comms[i]]
        A_temp = A[ix,:]
        A_temp2 = A_temp[:, ix]
        deg_list.append(A_temp2.sum()/2)
        
    return np.mean(deg_list)
 


#this function computes the average node degree between communities given by
#a set of supplied community assignment labels  
def compute_node_degree_between(A, comm_labels):
    N = A.shape[0]
    deg_list = []
    comms = np.unique(comm_labels)
    for i in range(0, len(comms)):
        for j in range(0, len(comms)):
            
            if i != j:
                ix1 = np.arange(N)[comm_labels == comms[i]]
                ix2 = np.arange(N)[comm_labels == comms[j]]
                A_temp = A[ix1,:]
                A_temp2 = A_temp[:, ix2]
                deg_list.append(A_temp2.sum()/2)
            
    return np.mean(deg_list)
                









# def sort_labels(labels):
#     #pdb.set_trace()
#     true_labels = pd.DataFrame(labels, columns = ['Nodes'])
#     new_list = []
#     l = len(true_labels['Nodes'])
#     for i in np.arange(l):
#         new_list.append(list(map(int, true_labels['Nodes'][i].split('_'))))
        
#     new_list_array = np.array(new_list)
#     new_true_labels = pd.DataFrame(new_list_array[:,:2])
#     new_true_labels.columns = ['clustlabs', 'nodelabs']
#     clusts = np.unique(new_true_labels['clustlabs'])
        
#     indices_for_clusts = []
#     #extract indices for nodes belonging to each cluster
#     for i in clusts:
#         indices_for_clusts.append(new_true_labels[new_true_labels['clustlabs'] == i].index.tolist())
        

#     #pe = np.load(path+'gexp.npy').transpose()
#     #reorganize pe so that nodes are sorted according to clusters 0,1,2..etc
#     flat_list_indices = []
#     for i in clusts:
#         flat_list_indices.extend(indices_for_clusts[i])
        
        
#     #construct labels for middle layer nodes
#     if new_list_array.shape[1]>2:
#         #num_middle_nodes = new_list_array.shape[0]/len(np.unique(new_list_array[:,1]))
#         #temp = [np.repeat(i,len(np.unique(new_list_array[:,1]))).tolist() for i in np.arange(num_middle_nodes)]
#         #middle_labels = np.array([int(i[0]) for i in np.array(temp).reshape((l,1)).tolist()])
#         temp = [str(i[0])+str(i[1]) for i in new_list]
#         middle_labels = np.array(temp.copy())
#         midclusts = np.unique(temp)
#         newclusts = np.arange(len(midclusts))
#         for i in range(0, len(midclusts)):
#             middle_labels[middle_labels == midclusts[i]] = newclusts[i]
            
#         middle_labels_final = middle_labels.astype(int)
#         sorted_true_labels_middle = middle_labels_final[flat_list_indices]
#         new_true_labels['middlelabs'] = sorted_true_labels_middle
#     else:
#         sorted_true_labels_middle = []


#     #the true labels sorted by cluster
#     sort_true_labels = np.array(new_true_labels['clustlabs'][flat_list_indices])

#     return flat_list_indices, new_true_labels, sort_true_labels, sorted_true_labels_middle




















