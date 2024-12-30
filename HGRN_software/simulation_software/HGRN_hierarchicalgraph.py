# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:18:48 2022

@author: Bruin
"""

#from IPython.core.display import HTML, Image, display
import networkx as nx
import numpy as np
from random import randint as rd    



def add_edges_to_subgraph(G, nodes_to_add, edges_to_add, weighted_graph = True, 
                       weighting = (0.1, 0.5)):
    
    low, high = weighting
    if nodes_to_add is not None:
        G.add_nodes_from(nodes_to_add)
    for edge in edges_to_add:
        p_node, c_node = edge
        if weighted_graph:
            G.add_edge(p_node, c_node,  weight = np.round(np.random.uniform(low, high),2))
        else:
            G.add_edge(p_node, c_node, weight = 1)
    return G.copy()




def add_edges_to_fullgraph(G, nodes_to_add, edges_to_add):
    
    if nodes_to_add is not None:
        G.add_nodes_from(nodes_to_add)
    for edge in edges_to_add:
        p_node, c_node, weight = edge
        G.add_edge(p_node, c_node, weight = weight)
    return G.copy()




def make_dag(graph):
    # Step 1: Replace bidirectional edges with a single directed edge
    edges = list(graph.edges)
    for u, v in edges:
        if graph.has_edge(v,u) and graph.has_edge(u,v):
            # Remove both edges to eliminate bidirectionality
            graph.remove_edge(u, v)
            graph.remove_edge(v, u)
            # Add a single directed edge (choose one direction, e.g., u -> v)
            graph.add_edge(u, v)

    # Step 2: Remove cycles by flipping edges
    print('\n resolving cycles...')
    while not nx.is_directed_acyclic_graph(graph):
        # Find a cycle
        try:
            cycle = nx.find_cycle(graph, orientation='original')
        except nx.NetworkXNoCycle:
            break
        
        # Flip an edge in the cycle
        u, v, _ = cycle[0]
        graph.remove_edge(u, v)
        graph.add_edge(v, u)
    
    return graph






def hierachical_graph(top_graph, subgraph_node_number, subgraph_type, as_weighted = True, 
                      degree=3, connection_prob_within=0.05, connection_prob_between = 0.01,
                      sub_graph_prob=0.01, weight_b = (0.1, 0.3), weight_w = (0.4, 0.8), mixed = None, 
                      force_connections = False, seed = None):
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

    if not seed:
        seed = rd(100, 500)
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
            subgraph_sm = nx.watts_strogatz_graph(n=rd(subgraph_node_number[0], 
                                                     subgraph_node_number[1]), 
                                                  k=degree, 
                                                  p=sub_graph_prob/np.mean(subgraph_node_number))
            #subgraph_sme = nx.DiGraph([(u,v) for (u,v) in subgraph_sm.edges() if u!=v]
            subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph_sm.edges() if u!=v]))
            node_list.append(top_node_g1)
            print('small world',subgraph_sm.nodes())
            print(subgraph_sm.edges())
        
        
        for top_node_g2 in top_node_g2:
            subgraph_random = nx.gnp_random_graph(n=rd(subgraph_node_number[0], 
                                                     subgraph_node_number[1]),
                                                  p=sub_graph_prob,
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
        
        for index, topgraph_node in enumerate(list(top_graph.nodes)):
            ##--------------##
            ## Small world  ##
            ##--------------##
            if (subgraph_type == 'small world' and (mixed == 'False' or mixed is None)):
                subgraph = nx.watts_strogatz_graph(n=rd(subgraph_node_number[0], 
                                                      subgraph_node_number[1]), 
                                                   k=degree, 
                                                   p=sub_graph_prob/np.mean(subgraph_node_number),
                                                   seed = seed)
                
                
                G = add_edges_to_subgraph(nx.DiGraph(), subgraph.nodes, subgraph.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
                subgraphs.append(G)
                #subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph.edges() if u!=v]))
            
            ##--------------##
            ## Random Graph ##
            ##--------------##
            elif (subgraph_type == 'random graph' and (mixed == 'False' or mixed is None)):
                # subgraph = generate_random_dag_with_gnp(n = rd(subgraph_node_number[0], 
                #                                         subgraph_node_number[1]),
                #                                         p = sub_graph_prob,
                #                                         make_directed=True)
                
                # subgraph = nx.gnm_random_graph(n = rd(subgraph_node_number[0], 
                #                                         subgraph_node_number[1]), 
                #                                m = rd(subgraph_node_number[0], 
                #                                         subgraph_node_number[1]),
                #                                directed = True)
                
                subgraph = nx.gnp_random_graph(n = rd(subgraph_node_number[0], 
                                                        subgraph_node_number[1]), 
                                                p = sub_graph_prob,
                                                directed = True)
                
                if not nx.is_directed_acyclic_graph(subgraph):
                    G = make_dag(subgraph)
                    
                G = add_edges_to_subgraph(nx.DiGraph(), subgraph.nodes, subgraph.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
                
                subgraphs.append(G)
                #subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph.edges() if u<v]))
            ##--------------##
            ##  Scale Free  ##
            ##--------------##
            elif (subgraph_type == 'scale free' and (mixed == 'False' or mixed is None)):
                n = rd(subgraph_node_number[0], subgraph_node_number[1]); m = rd(2,(n-1))
                subgraph = nx.barabasi_albert_graph(n,m)
                G = add_edges_to_subgraph(nx.DiGraph(), subgraph.nodes, subgraph.edges, 
                                       weighted_graph = as_weighted, weighting = weight_w)
                subgraphs.append(G)
                #subgraph = nx.scale_free_graph(rd(subgraph_node_number[0], subgraph_node_number[1]))
                #subgraphs.append(nx.DiGraph([(u,v) for (u,v) in subgraph.edges() if u!=v]))
            
            


            node_list.append(topgraph_node)
        #print(subgraphs[1])
        print(subgraph_type, "subgraphs used")
        
    # generate full graph based on top graph
    #creates a directed DiGraph for each community corresponding nodes in the layer
    #above
    for index, topgraph_node in enumerate(node_list):
        # add nodes and edges to full_graph
        nodes = [str(topgraph_node) + '_' + str(node) for node in list(subgraphs[index].nodes)]
        edges = [(str(topgraph_node) + '_' + str(u), 
                  str(topgraph_node) + '_' + str(v), 
                  subgraphs[index].get_edge_data(u, v)['weight']) for (u,v) in subgraphs[index].edges]
        full_graph = add_edges_to_fullgraph(full_graph, 
                                            nodes_to_add = nodes, 
                                            edges_to_add = edges)
        full_graph_node_list[topgraph_node] = nodes
        
        # subgraph_node_list = []
        # # add nodes to full_graph
        # for subgraph_node in list(subgraphs[index].nodes):
        #     full_graph.add_node(str(topgraph_node) + '_' + str(subgraph_node))
        #     subgraph_node_list.append(str(topgraph_node) + '_' + str(subgraph_node))
        # # add the nodes to a dict for nex step
        # full_graph_node_list[topgraph_node] = subgraph_node_list
        # # add edges in sub-graphs to full graph 
        # for subgraph_edge in subgraphs[index].edges:
        #     full_graph.add_edge(str(topgraph_node) + '_' + str(subgraph_edge[0]), 
        #                         str(topgraph_node) + '_' + str(subgraph_edge[1]))
    print(f'{top_graph.edges}')
    # add connections between sub-graphs
    for p_graph, c_graph in top_graph.edges:
        
        p_graph_name = str(p_graph).split('_')
        c_graph_name = str(c_graph).split('_')
        
        if len(p_graph_name) == 2:
            p_community, p_node_name = p_graph_name
            c_community, c_node_name = c_graph_name
        else:
            p_community = p_graph
            c_community = c_graph
        
        # get node lists in graph p and c, where edge p->c exists in top-graph
        p_list, c_list = full_graph_node_list[p_graph], full_graph_node_list[c_graph]
        # all possible connections between communities
        possible_edges = [(i, j) for i in p_list for j in c_list]
        np.random.shuffle(possible_edges)
        num_possible = len(possible_edges)
        # unless specified, probability will be 1/(total cross connections i.e equal probability)
        # if connection_prob == 'use_baseline':
        #          c_prob = (num_possible/len(top_graph.edges))/num_possible
        # else:
        #     c_prob = connection_prob
        if p_community == c_community:
            c_prob = connection_prob_within
        else: c_prob = connection_prob_between
            
        
        which_edges=[i<=c_prob for i in np.random.uniform(size = num_possible).tolist()]
        
        if sum(which_edges) == 0 and force_connections:
            which_edges_idx = np.array([np.random.randint(num_possible)])
        
        else:
            which_edges_idx = np.arange(num_possible)[which_edges]
        edges_to_add = [possible_edges[x] for x in which_edges_idx]
        print(f'Adding {len(edges_to_add)} edges between community {p_graph} and community {c_graph}: \n {edges_to_add}')
        
        full_graph = add_edges_to_subgraph(full_graph, 
                                           nodes_to_add = None, 
                                           edges_to_add = edges_to_add, 
                                           weighted_graph = as_weighted, 
                                           weighting = weight_b)
        
        # for p_node, c_node in edges_to_add:
        #       # add based on probability
        #       full_graph.add_edge(p_node, c_node)
    
    
    return full_graph, subgraphs


# def generate_pseudo_expression(topological_order, adjacency_matrix, 
#                                 number_of_invididuals, free_mean=0, std=0.5,
#                                 common_distribution = True):
#     pseudo_expression = np.zeros((len(topological_order), number_of_invididuals))
#     for i in range(number_of_invididuals): 
#         cur_sample = np.zeros((len(topological_order), ))
#         for index, node in enumerate(topological_order):
#             if np.sum(adjacency_matrix[:, index]) == 0:
#                 if common_distribution == True:
#                     cur_sample[index] = np.random.normal(free_mean, std)
#                 else:
#                     cur_sample[index] = np.random.normal(np.random.uniform(-10, 10), std)
#             else:
#                 parents_loc = [cur_sample[i] for i in range(len(cur_sample)) if adjacency_matrix[i, index]==1]
#                 cur_sample[index] = np.random.normal(np.mean(parents_loc), std)
#         pseudo_expression[:, i] = cur_sample.reshape(-1,)
#     return pseudo_expression




def generate_pseudo_expression(topological_order, adjacency_matrix, 
                               number_of_invididuals, free_mean=0, std=0.5,
                               common_distribution = True):
    
    N = len(topological_order)
    pseudo_expression = np.zeros((N, number_of_invididuals))
    origin_nodes = []
    for index, node in enumerate(topological_order):
        if np.sum(adjacency_matrix[:,index]) == 0:
            origin_nodes.append([index,node])
            if common_distribution == True:
                pseudo_expression[index,:] = np.random.normal(free_mean, std,
                                                          size = number_of_invididuals)
            else:
                pseudo_expression[index,:] = np.random.normal(index,
                                                              std,
                                                              size = number_of_invididuals)
        else:
            parents_idx = [i for i in np.arange(N) if adjacency_matrix[i, index]==1]
            parents_loc = pseudo_expression[parents_idx, :].mean(axis = 0)
            pseudo_expression[index, :] = np.random.normal(parents_loc, std) 
    return pseudo_expression, origin_nodes





def generate_pseudo_expression_weighted(topological_order, adjacency_matrix, 
                                        number_of_invididuals, free_mean=0, std=0.5,
                                        common_distribution = True):
    
    N = len(topological_order)
    pseudo_expression = np.zeros((N, number_of_invididuals))
    origin_nodes = []
    for index, node in enumerate(topological_order):
        if np.sum(adjacency_matrix[:,index]) == 0:
            origin_nodes.append([index,node])
            if common_distribution == True:
                pseudo_expression[index,:] = np.random.normal(free_mean, std,
                                                          size = number_of_invididuals)
            else:
                pseudo_expression[index,:] = np.random.normal(index,
                                                              std,
                                                              size = number_of_invididuals)
        else:
            parents_idx = [i for i in list(np.arange(N)) if adjacency_matrix[i, index] != 0]
            weights = adjacency_matrix[parents_idx, index].reshape(len(parents_idx), 1)
            #weights = adjacency_matrix[parents_idx, index]
            #parents_loc = np.multiply(pseudo_expression[parents_idx, :].transpose(),weights).sum(axis = 1).reshape(1, number_of_invididuals)
            parents_loc = np.matmul(pseudo_expression[parents_idx, :].transpose(), weights).reshape(1, number_of_invididuals)
            pseudo_expression[index, :] = np.random.normal(parents_loc, std) 
    return pseudo_expression, origin_nodes



def same_cluster(s1, s2):
    l1 = s1.split('_')
    l2 = s2.split('_')
    if l1[0] == l2[0] and l1[1] == l2[1] and l1[2] != l2[2]:
        return True
    else:
        return False
