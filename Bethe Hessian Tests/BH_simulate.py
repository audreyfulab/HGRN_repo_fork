# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 11:50:12 2023

@author: Bruin
"""
import random
import torch
import argparse
import numpy as np
import networkx as nx
import seaborn as sbn
import matplotlib.pyplot as plt
from hierarchicalgraph import hierachical_graph 
from hierarchicalgraph import generate_pseudo_expression
from hierarchicalgraph import same_cluster
from random import randint as rd   

def simulate_graph(args, method='Correlation', r_cutoff=0.5): 
    
    
    nodes_by_layer = []
    if args.connect == 'full':
        #randomly generate first layer of hierarchy
        h1_graph = nx.watts_strogatz_graph(args.top_layer_nodes, 3, 0.02)
        #h1_graph = nx.erdos_renyi_graph(args.top_layer_nodes, args.toplayer_connect_prob, directed=True)
        h1_graph = nx.DiGraph([(u,v) for (u,v) in h1_graph.edges() if u!=v])
        plt.figure(figsize=(10,7))
        plt.clf()
        nx.draw_networkx(h1_graph, arrows=True)
    
        h2_graph = hierachical_graph(top_graph=h1_graph, 
                                     subgraph_node_number=args.nodes_per_super2, 
                                     subgraph_type =args.subgraph_type, 
                                     sub_graph_prob=args.subgraph_prob, 
                                     connection_prob=args.connect_prob, 
                                     degree=args.node_degree)
    
        #sort toplayer
        ts_h1_graph = list(nx.topological_sort(h1_graph))
        adj_h1_graph = nx.adjacency_matrix(h1_graph, ts_h1_graph).todense()

        #draw top layer
        nx.draw_networkx(nx.from_numpy_matrix(adj_h1_graph),arrows=True)
    
        #print toplayer attributes
        print('-'*60)
        print("Number of edges:" ,h1_graph.number_of_edges())
        print("Number of nodes:",h1_graph.number_of_nodes())
        print("In degrees: ", h1_graph.in_degree())
        print("Out degrees: ", h1_graph.out_degree())
        print('-'*60)
        nodes_by_layer.append(h1_graph.number_of_nodes())
        if args.layers == 2:
            print("Bottom Layer")
            print("Number of edges:" ,h2_graph.number_of_edges())
            print("Number of nodes:",h2_graph.number_of_nodes())
            print('-'*60)
            nodes_by_layer.append(h2_graph.number_of_nodes())
    
        #for 3layer hierarchies
        if args.layers == 3:
            h3_graph = hierachical_graph(top_graph=h2_graph, 
                                         subgraph_node_number=args.nodes_per_super3, 
                                         subgraph_type =args.subgraph_type, 
                                         sub_graph_prob=args.subgraph_prob, 
                                         connection_prob=args.connect_prob, 
                                         degree=args.node_degree)
        
            print("Middle Layer")
            print("Number of edges:" ,h2_graph.number_of_edges())
            print("Number of nodes:",h2_graph.number_of_nodes())
            print('-'*60)
            print("Bottom Layer")
            print("Number of edges:" ,h3_graph.number_of_edges())
            print("Number of nodes:",h3_graph.number_of_nodes())
            print('-'*60)
            nodes_by_layer.append(h3_graph.number_of_nodes())
            
    
    else:
        h1_graph =  np.zeros((10,10))
        h1_graph = nx.from_numpy_array(h1_graph)
        plt.figure(figsize=(10,7))
        plt.clf()
        nx.draw_networkx(h1_graph, arrows=True)
    
        h2_graph = hierachical_graph(top_graph = h1_graph, 
                                     subgraph_node_number=args.nodes_per_super2, 
                                     subgraph_type =args.subgraph_type, 
                                     sub_graph_prob=args.subgraph_prob, 
                                     connection_prob=0, 
                                     degree=args.node_degree)
    
        #print toplayer attributes
        print('-'*60)
        print('Top Layer')
        print("Number of edges:" ,h1_graph.number_of_edges())
        print("Number of nodes:",h1_graph.number_of_nodes())
        print('-'*60)
        nodes_by_layer.append(h1_graph.number_of_nodes())
        if args.layers == 2:
            print("Bottom Layer")
            print("Number of edges:" ,h2_graph.number_of_edges())
            print("Number of nodes:",h2_graph.number_of_nodes())
            print('-'*60)
            nodes_by_layer.append(h2_graph.number_of_nodes())
    
        #for 3 layer hierarchies
        if args.layers == 3:
            h3_graph = hierachical_graph(top_graph=h2_graph, 
                                         subgraph_node_number=args.nodes_per_super3, 
                                         subgraph_type =args.subgraph_type, 
                                         sub_graph_prob=args.subgraph_prob, 
                                         connection_prob=0, 
                                         degree=args.node_degree)
        
            print("Middle Layer")
            print("Number of edges:" ,h2_graph.number_of_edges())
            print("Number of nodes:",h2_graph.number_of_nodes())
            print('-'*60)
            print("Bottom Layer")
            print("Number of edges:" ,h3_graph.number_of_edges())
            print("Number of nodes:",h3_graph.number_of_nodes())
            print('-'*60)
            nodes_by_layer.append(h3_graph.number_of_nodes())



    if args.layers == 2:
        ts_full = list(nx.topological_sort(h2_graph))
        adj_full = nx.adjacency_matrix(h2_graph, ts_full).todense()
        print(len(ts_full), adj_full.shape)
    
    else:
        ts_full = list(nx.topological_sort(h3_graph))
        adj_full = nx.adjacency_matrix(h3_graph, ts_full).todense()
        print(len(ts_full), adj_full.shape)
    
    print('Generating pseudoexpression...')
    pe = generate_pseudo_expression(ts_full, adj_full, args.sample_size)
    print('data dimension = {}'.format(pe.shape))

    #get the input graph 

    
    
    
    return pe, nodes_by_layer