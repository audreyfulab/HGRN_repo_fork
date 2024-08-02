# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:45:47 2024

@author: Bruin
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import randint as rd 
from random import shuffle, seed
import sys
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from HGRN_hierarchicalgraph import generate_pseudo_expression, gendata
import itertools
import seaborn as sbn

#seed(123)


def which(condition):
    """
    Returns the indices of elements in the list that satisfy the given condition.

    Parameters:
    condition (iterable): A boolean iterable where True indicates the position of elements that satisfy the condition.

    Returns:
    list: A list of indices where the condition is True.
    """
    return [index for index, value in enumerate(condition) if value]


def concat_list(_list):
    return list(itertools.chain.from_iterable(_list))


# Function to remove cycles and create a DAG
def make_dag(graph):
    try:
        # Find all simple cycles in the graph
        cycles = list(nx.simple_cycles(graph))
        shuffle(cycles)
        while cycles:
            print(len(cycles))
            for cycle in cycles:
                shuffle(cycle)
                try:
                    # Remove one edge from each cycle to break it
                    graph.remove_edge(cycle[0], cycle[1])
                    graph.add_edge(cycle[1], cycle[0])
                except:
                    pass
            cycles = list(nx.simple_cycles(graph))
    except nx.NetworkXNoCycle:
        pass
    return graph

N = 10
k = 5
p = 0.05/k
seeds = [123, 567]
G = nx.DiGraph()
L = 10

def get_subgraphs(N, k, p, L, seeds, _type = ['small_world', 'scale_free', 'random'], num_coms = 2):
    
    graphlist = []
    if _type == 'small_world':
        for i in range(0, num_coms):
            graphlist.append( make_dag(nx.DiGraph(nx.watts_strogatz_graph(n = N, 
                                                                          k = k, 
                                                                          p = p))))
        
    if _type == 'scale_free':
        for i in range(0, num_coms):
            graphlist.append(make_dag(nx.DiGraph(nx.barabasi_albert_graph(N,
                                                                          rd(2,(N-1))))))
        
    if _type == 'random':
        
        for i in range(0, num_coms):
            graphlist.append(nx.gnm_random_graph(N,
                                          rd(L, N),
                                          directed=True))
        

    return graphlist

subgraph_list = get_subgraphs(N = N, k = k, p = p, L = L, 
                              seeds = seeds, 
                              _type = 'scale_free',
                              num_coms= 2)


# fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15, 10))
# nx.draw_networkx(rg1, ax = ax1, 
#                  node_size = 300,
#                  arrowsize = 30)
# nx.draw_networkx(rg2, ax = ax2, 
#                  node_size = 300,
#                  arrowsize = 30)


def add_subgraph_to_graph(G, subgraphs, c_prob = 0.02, force_connections = True,
                          use_standard_weighting = False):
    color = ['red', 'blue', 'green', 'pink']
    comlist = []
    cl = []
    subgraph_nodes_list = []
    subgraph_edges_list = []
    
    for index, subgraph in enumerate(subgraphs):
        subgraph_nodes_list.append([str(index)+'_'+str(i) for i in list(subgraph.nodes)])
        subgraph_edges_list.append([tuple([str(index)+'_'+str(i) for i in x]) for x in list(subgraph.edges)])
        comlist += [index]*len(subgraph_nodes_list[index])
        cl += [color[index]]*len(subgraph_nodes_list[index])
        for node in subgraph_nodes_list[index]:
            G.add_node(node)
        for edge in subgraph_edges_list[index]:
            if use_standard_weighting:
                G.add_edge(edge[0], edge[1], weight = np.round(np.random.uniform(0.3, 1),2))
            else:
                G.add_edge(edge[0], edge[1])
    
    
    p_nodes, c_nodes = subgraph_nodes_list
    possible_edges = [(i, j) for i in p_nodes for j in c_nodes]+[(j, i) for i in p_nodes for j in c_nodes]
    num_possible = len(possible_edges)
    which_edges=[i<=c_prob for i in np.random.uniform(size = num_possible).tolist()]
    
    if sum(which_edges) == 0 and force_connections:
        which_edges_idx = np.array([np.random.randint(num_possible)])
    
    else:
        which_edges_idx = np.arange(num_possible)[which_edges]
    edges_to_add = [possible_edges[x] for x in which_edges_idx]
    
    for p_node, c_node in edges_to_add:
         # add based on probability
         if use_standard_weighting:
             G.add_edge(p_node, c_node, weight = np.round(np.random.uniform(0, 0.2),2))
         else:
             G.add_edge(p_node, c_node)
         
         
    if not use_standard_weighting:
        all_edges = sorted(list(G.edges))         
        all_nodes = list(G.nodes)
        targets = [edge[1] for edge in all_edges]
        tgt_idx = [which([i == j for i in targets]) for j in np.unique(targets)]
        weight_values, indices = concat_list([[(1/len(i))]*len(i) for i in tgt_idx]), concat_list(tgt_idx)
        weights = [0]*len(all_edges)
        for idx, i in enumerate(indices):
            weights[i] = np.round(weight_values[idx],2)
    
        new_DG = nx.DiGraph()
    
        for node in all_nodes:
            new_DG.add_node(node)
        for (edge, weight) in zip(all_edges, weights):
            new_DG.add_edge(edge[0], edge[1], weight = weight)
            
        topo_order = list(nx.topological_sort(new_DG))
        return new_DG.copy(), cl, topo_order
       
    else:
        Gnew = make_dag(G)
        topo_order = list(nx.topological_sort(G))
        return Gnew.copy(), cl, topo_order


use_standard_weighting = True
full_graph, cl, topo = add_subgraph_to_graph(G, subgraph_list, 
                                             use_standard_weighting=use_standard_weighting)


fig, ax = plt.subplots(figsize = (14,10))
#pos = nx.spring_layout(full_graph)
pos = nx.circular_layout(full_graph)
nx.draw_networkx(full_graph, ax = ax, 
                 node_size = 500, 
                 pos = pos,
                 with_labels = True,
                 arrowsize = 40,
                 node_color = cl)


nx.draw_networkx_edge_labels(full_graph,
                             pos,
                             edge_labels=nx.get_edge_attributes(full_graph,'weight'),
                             font_size=10)

sorted_indices = sorted(range(len(topo)), key=lambda x: topo[x])
adj_weighted = nx.adjacency_matrix(full_graph, topo).todense()
adj_full = adj_weighted.copy()
adj_full[adj_full > 0] = 1
cd = True
std = 0.5
if use_standard_weighting:
    pe, orn = gendata(topo, adj_weighted, 
                                         number_of_invididuals = 500,
                                         common_distribution=cd,
                                         std=std)
else:
    pe, orn = generate_pseudo_expression(topo, adj_full, 
                                         number_of_invididuals = 500,
                                         common_distribution=cd,
                                         std=std)

pe_sorted = pe[sorted_indices, :]

cormat = np.corrcoef(pe_sorted)

fig, ax = plt.subplots(figsize = (12, 10))
nl = np.array(topo)[sorted_indices]
sbn.heatmap(np.abs(cormat), ax = ax, yticklabels=nl, xticklabels=nl)






