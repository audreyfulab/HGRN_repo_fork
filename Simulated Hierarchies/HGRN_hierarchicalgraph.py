# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:18:48 2022

@author: Bruin
"""

#from IPython.core.display import HTML, Image, display
from IPython.display import HTML, Image, display
import networkx as nx
import numpy as np
from random import shuffle as sl  
from random import randint as rd    

from collections import defaultdict
from matplotlib import pyplot as plt

def hierachical_graph(top_graph, subgraph_node_number, subgraph_type, degree=3, 
                      connection_prob=0.05, sub_graph_prob=0.01, mixed = None):
    '''
    Input:
        top_graph: the base graph
        subgraph_node_number: the average node number of subgraph
        subgraph_type: the type of subgraph, now only small world
        sub_graph_prob: the connectivity inside subgraph
        sub_graph_prob: the probability of connection among subgraph
    Output:
        a directed graph that inplies hierarchical structure
    '''
    subgraphs, node_list = [], []
    full_graph = nx.DiGraph()
    full_graph_node_list = {}

    # generate a list of sub-graphs
    
        ##--------------------##
        ##   Mixed Subgraphs  ##
        ##--------------------##
    if (mixed =="True" and (subgraph_type == 'small world' or subgraph_type == 'random graph' or subgraph_type == 'scale free')):
        top_node_length =  len(list(top_graph.nodes())) ; g_index  = top_node_length // 3
        top_node_g1 = list(top_graph.nodes())[:g_index]
        top_node_g2 = list(top_graph.nodes())[g_index:(g_index + g_index)]
        top_node_g3 = list(top_graph.nodes())[(g_index + g_index):]
        
        for top_node_g1 in top_node_g1: ## sm = small world
            subgraph_sm = nx.watts_strogatz_graph(rd(subgraph_node_number[0], 
                                                     subgraph_node_number[1]), 
                                                  degree, 
                                                  sub_graph_prob/np.mean(subgraph_node_number))
            #subgraph_sme = nx.DiGraph([(u,v) for (u,v) in subgraph_sm.edges() if u!=v]
            subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph_sm.edges() if u!=v]))
            node_list.append(top_node_g1)
            print('small world',subgraph_sm.nodes())
            print(subgraph_sm.edges())
        
        
        for top_node_g2 in top_node_g2:
            subgraph_random = nx.gnm_random_graph(rd(subgraph_node_number[0], 
                                                     subgraph_node_number[1]),
                                              rd(subgraph_node_number[0], 
                                                 subgraph_node_number[1]),
                                              directed=True)
            subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph_random.edges() if u<v]))
            #print(node_list.append(top_node_g2))
            print('random',subgraph_random.nodes())
            print(subgraph_random.edges())
            
        for top_node_g3 in top_node_g3: # sf : scale free
            n = rd(subgraph_node_number[0], subgraph_node_number[1]); m = rd(2,(n-1))
            subgraph_sf = nx.barabasi_albert_graph(n,m)
            subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph_sf.edges() if u!=v]))
            node_list.append(top_node_g3)
            print('scale free',subgraph_sf.nodes())
            print(subgraph_sf.edges())
        
        for i in range(0,len(subgraphs)):
            print(i)
            print(subgraphs[i])
            print(list(nx.topological_sort(subgraphs[i])))
        
        print("Mixed subgraghs used")
        
    else :
        
        for topgraph_node in list(top_graph.nodes):
            ##--------------##
            ## Small world  ##
            ##--------------##
            if (subgraph_type == 'small world' and (mixed == 'False' or mixed is None)):
                subgraph = nx.watts_strogatz_graph(rd(subgraph_node_number[0], 
                                                      subgraph_node_number[1]), 
                                                   degree, 
                                                   sub_graph_prob/np.mean(subgraph_node_number))
                subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph.edges() if u!=v]))
            
            ##--------------##
            ## Random Graph ##
            ##--------------##
            elif (subgraph_type == 'random graph' and (mixed == 'False' or mixed is None)):
                subgraph = nx.gnm_random_graph(rd(subgraph_node_number[0], 
                                                  subgraph_node_number[1]),
                                               rd(subgraph_node_number[0], 
                                                  subgraph_node_number[1]),
                                               directed=True)
                subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph.edges() if u<v]))
            ##--------------##
            ##  Scale Free  ##
            ##--------------##
            elif (subgraph_type == 'scale free' and (mixed == 'False' or mixed is None)):
                n = rd(subgraph_node_number[0], subgraph_node_number[1]); m = rd(2,(n-1))
                subgraph = nx.barabasi_albert_graph(n,m)
                #subgraph = nx.scale_free_graph(rd(subgraph_node_number[0], subgraph_node_number[1]))
                subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph.edges() if u!=v]))
            
            


            node_list.append(topgraph_node)
        #print(subgraphs[1])
        print(subgraph_type, "subgraphs used")
        
    # generate full graph based on top graph
    #creates a directed DiGraph for each community corresponding nodes in the layer
    #above
    for index, topgraph_node in enumerate(node_list):
        subgraph_node_list = []
        # add nodes to full_graph
        for subgraph_node in list(subgraphs[index].nodes):
            full_graph.add_node(str(topgraph_node) + '_' + str(subgraph_node))
            subgraph_node_list.append(str(topgraph_node) + '_' + str(subgraph_node))
        # add the nodes to a dict for nex step
        full_graph_node_list[topgraph_node] = subgraph_node_list
        # add edges in sub-graphs to full graph 
        for subgraph_edge in subgraphs[index].edges:
            full_graph.add_edge(str(topgraph_node) + '_' + str(subgraph_edge[0]), 
                                str(topgraph_node) + '_' + str(subgraph_edge[1]))
    # add connections between sub-graphs
    for p_graph, c_graph in top_graph.edges:
        # get node lists in graph p and c, where edge p->c exists in top-graph
        p_list, c_list = full_graph_node_list[p_graph], full_graph_node_list[c_graph]
        # all possible connections between communities
        possible_edges = [(i, j) for i in p_list for j in c_list]
        num_possible = len(possible_edges)
        # unless specified, probability will be 1/(total cross connections)
        if connection_prob == 'use_baseline':
                 connection_prob = (num_possible/len(top_graph.edges))/num_possible
        which_edges=[i<=connection_prob for i in np.random.uniform(size = num_possible).tolist()]
        if sum(which_edges) == 0:
            which_edges_idx = np.array([np.random.randint(num_possible)])
        else:
            which_edges_idx = np.arange(num_possible)[which_edges]
        edges_to_add = [possible_edges[x] for x in which_edges_idx]
        for p_node, c_node in edges_to_add:
             # add based on probability
             full_graph.add_edge(p_node, c_node)
        
    return full_graph


def generate_pseudo_expression(topological_order, adjacency_matrix, 
                               number_of_invididuals, free_mean=0, std=0.5):
    pseudo_expression = np.zeros((len(topological_order), number_of_invididuals))
    for i in range(number_of_invididuals): 
        cur_sample = np.zeros((len(topological_order), ))
        for index, node in enumerate(topological_order):
            if np.sum(adjacency_matrix[:, index]) == 0:
                cur_sample[index] = np.random.normal(free_mean, std)
            else:
                parents_loc = [cur_sample[i] for i in range(len(cur_sample)) if adjacency_matrix[i, index]==1]
                cur_sample[index] = np.random.normal(np.mean(parents_loc), std)
        pseudo_expression[:, i] = cur_sample.reshape(-1,)
    return pseudo_expression



def same_cluster(s1, s2):
    l1 = s1.split('_')
    l2 = s2.split('_')
    if l1[0] == l2[0] and l1[1] == l2[1] and l1[2] != l2[2]:
        return True
    else:
        return False
