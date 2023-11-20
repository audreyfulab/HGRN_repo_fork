# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:29:50 2023

@author: Bruin
"""

import networkx as nx
import torch
import sys
import numpy as np
import pandas as pd
from networkx import community as comm
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
from utilities import Modularity
from simulation_utilities import compute_modularity
import matplotlib.pyplot as plt
from random import seed
path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Modularity/'
seed(111)


def create_clusts_graph(comms = 2, degree = [2]*2, connect_prob = [0.05]*2, 
                        nodes_per_comm = [10,10], sp=''):
    
    idx = [0]+np.cumsum(nodes_per_comm).tolist()
    graph_adj = np.zeros((sum(nodes_per_comm), sum(nodes_per_comm)))
    comm_assign = np.zeros(((sum(nodes_per_comm),comms)))
    onecomm = np.concatenate((np.ones((idx[-1],1)), np.zeros((idx[-1],1))), axis = 1)
    graphs = []
    adjs = []
    modlist = []
    for i in range(0, comms):
        
        graphs.append(nx.watts_strogatz_graph(nodes_per_comm[i], degree[i], connect_prob[i]))
        adjs.append(nx.to_numpy_array(graphs[i]))
        graph_adj[idx[i]:idx[i+1], idx[i]:idx[i+1]] = adjs[i]
        comm_assign[idx[i]:idx[i+1], i] = np.ones(nodes_per_comm[i])
        
    modlist.append(Modularity(torch.tensor(graph_adj), 
                     torch.tensor(comm_assign)))
    modlist.append(Modularity(torch.tensor(graph_adj), torch.tensor(onecomm)))
    print(modlist)
        
    fig, ax = plt.subplots()
    nx.draw_networkx(nx.from_numpy_array(graph_adj), ax = ax,
                     node_color = np.argmax(comm_assign, axis = 1)+1,
                     node_size = 150, cmap = 'cool')
    fig.savefig(sp)
    return graphs, adjs, graph_adj, comm_assign    





def create_singletons(num_singletons = 2, degree = 2, connect_prob = 0.05, 
                        nodes_in_full_comm = 10, sp=''):
    
    modlist = []
    graph_adj = np.zeros((nodes_in_full_comm+num_singletons, 
                          nodes_in_full_comm+num_singletons))
    comm_assign = np.zeros((nodes_in_full_comm+num_singletons, num_singletons+1))
    onecomm = np.concatenate((np.ones((nodes_in_full_comm+num_singletons,1)), 
                              np.zeros((nodes_in_full_comm+num_singletons,1))), axis = 1)
         
    comm_assign[:nodes_in_full_comm, 0] = 1
    maincom = nx.watts_strogatz_graph(nodes_in_full_comm, degree, connect_prob)
    graph_adj[:nodes_in_full_comm, :nodes_in_full_comm] = nx.to_numpy_array(maincom)
    for i in range(0, num_singletons):
        comm_assign[nodes_in_full_comm+i, i+1] = 1
        
    modlist.append(Modularity(torch.tensor(graph_adj), 
                     torch.tensor(comm_assign)))
    modlist.append(Modularity(torch.tensor(graph_adj), torch.tensor(onecomm)))
    print(modlist)
        
    fig, ax = plt.subplots()
    nx.draw_networkx(nx.from_numpy_array(graph_adj), ax = ax,
                     node_color = np.argmax(comm_assign, axis = 1)+1,
                     node_size = 150, cmap='cool')
    fig.savefig(sp)
    return graph_adj, comm_assign    





def one_net_multi_comms(comms = 2, degree = 2, connect_prob = 0.05, 
                        nodes_per_comm = [5,5], sp=''):
    
    idx = [0]+np.cumsum(nodes_per_comm).tolist()
    graph_adj = np.zeros((sum(nodes_per_comm), sum(nodes_per_comm)))
    comm_assign = np.zeros(((sum(nodes_per_comm),comms)))
    onecomm = np.concatenate((np.ones((idx[-1],1)), np.zeros((idx[-1],1))), axis = 1)
    modlist = []
        
    graphs = nx.watts_strogatz_graph(sum(nodes_per_comm), 
                                          degree, 
                                          connect_prob)
    graph_adj = nx.to_numpy_array(graphs)
    
    for i in range(0, comms):
        comm_assign[idx[i]:idx[i+1], i] = np.ones(nodes_per_comm[i])
        
    modlist.append(Modularity(torch.tensor(graph_adj), 
                     torch.tensor(comm_assign)))
    modlist.append(Modularity(torch.tensor(graph_adj), torch.tensor(onecomm)))
    print(modlist)
        
    fig, ax = plt.subplots()
    nx.draw_networkx(nx.from_numpy_array(graph_adj), ax = ax,
                     node_color = np.argmax(comm_assign, axis = 1)+1,
                     node_size = 150, cmap = 'cool')
    fig.savefig(sp)
    return graph_adj, comm_assign    










#simulate
graphs, adjs, graph_adj, comm_assign = create_clusts_graph(comms=2,
                                                           nodes_per_comm=[5,5],
                                                           sp = path+'case1_2clusts.pdf')

graph_adj, comm_assign = create_singletons(num_singletons = 1,
                                          nodes_in_full_comm = 5,
                                          sp = path+'case2_singleton.pdf')

graph_adj2 = graph_adj.copy()
graph_adj2[3,5] = 1
graph_adj2[5,3] = 1
fig, ax =plt.subplots()

nx.draw_networkx(nx.from_numpy_array(graph_adj2), 
                 node_color = comm_assign.argmax(1), 
                 cmap = 'cool', ax = ax)
fig.savefig(path+'singleton_with_edge.pdf')

graph_adj, comm_assign = one_net_multi_comms(comms = 2, 
                                             degree = 2, 
                                             connect_prob = 0.05, 
                                             nodes_per_comm = [5,5], 
                                             sp=path+'on_net_multi_comms.pdf')
    