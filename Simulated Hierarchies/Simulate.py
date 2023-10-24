# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 11:50:12 2023

@author: Bruin
"""
import random
import torch
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sbn
import matplotlib.pyplot as plt
from hierarchicalgraph import hierachical_graph 
from hierarchicalgraph import generate_pseudo_expression
from hierarchicalgraph import same_cluster
from random import randint as rd   
import pdb

def simulate_graph(args): 
    
    
    nodes_by_layer = []
    edges_by_layer = []
    if args.connect == 'full':
        #randomly generate first layer of hierarchy
        h1_graph = nx.watts_strogatz_graph(args.top_layer_nodes, args.node_degree, args.subgraph_prob)
        #h1_graph = nx.erdos_renyi_graph(args.top_layer_nodes, args.toplayer_connect_prob, directed=True)
        h1_graph = nx.DiGraph([(u,v) for (u,v) in h1_graph.edges() if u!=v])
        plt.figure(figsize=(10,7))
        plt.clf()
        nx.draw_networkx(h1_graph, arrows=True)
        
        #sort toplayer
        ts_h1_graph = list(nx.topological_sort(h1_graph))
        adj_h1_graph = nx.adjacency_matrix(h1_graph, ts_h1_graph).todense()
    
        h2_graph = hierachical_graph(top_graph=h1_graph, 
                                     subgraph_node_number=args.nodes_per_super2, 
                                     subgraph_type =args.subgraph_type, 
                                     sub_graph_prob=args.subgraph_prob, 
                                     connection_prob=args.connect_prob, 
                                     degree=args.node_degree)
        
        #sort middle layer
        ts_h2_graph = list(nx.topological_sort(h2_graph))
        adj_h2_graph = nx.adjacency_matrix(h2_graph, ts_h2_graph).todense()
        #draw top layer
        topfig, topax = plt.subplots(1,1, figsize = (10,12))
        nx.draw_networkx(nx.from_numpy_matrix(adj_h1_graph),arrows=True, ax = topax)
        topfig.savefig(args.savepath+'top_layer_graph.pdf')
        
        #print toplayer attributes
        print('-'*60)
        print("Number of edges:" ,h1_graph.number_of_edges())
        print("Number of nodes:",h1_graph.number_of_nodes())
        print("In degrees: ", h1_graph.in_degree())
        print("Out degrees: ", h1_graph.out_degree())
        print('-'*60)
        nodes_by_layer.append(h1_graph.number_of_nodes())
        edges_by_layer.append(h1_graph.number_of_edges())
        if args.layers == 2:
            
            #draw bottom layer
            botfig, botax = plt.subplots(1,1, figsize = (10,12))
            nx.draw_networkx(nx.from_numpy_matrix(adj_h2_graph),arrows=True, ax = botax)
            botfig.savefig(args.savepath+'bottom_layer_graph.pdf')
            #print and store network statistics
            print("Bottom Layer")
            print("Number of edges:" ,h2_graph.number_of_edges())
            print("Number of nodes:",h2_graph.number_of_nodes())
            print('-'*60)
            
            nodes_by_layer.append(h2_graph.number_of_nodes())
            edges_by_layer.append(h2_graph.number_of_edges())
    
        #for 3layer hierarchies
        if args.layers == 3:
            h3_graph = hierachical_graph(top_graph=h2_graph, 
                                         subgraph_node_number=args.nodes_per_super3, 
                                         subgraph_type =args.subgraph_type, 
                                         sub_graph_prob=args.subgraph_prob, 
                                         connection_prob=args.connect_prob, 
                                         degree=args.node_degree)
            
            
            ts_h3_graph = list(nx.topological_sort(h3_graph))
            adj_h3_graph = nx.adjacency_matrix(h3_graph, ts_h3_graph).todense()
            #draw middle layer
            midfig, midax = plt.subplots(1,1, figsize = (10,12))
            nx.draw_networkx(nx.from_numpy_matrix(adj_h2_graph),arrows=True, ax = midax)
            midfig.savefig(args.savepath+'middle_layer_graph.pdf')
            #draw bottom layer
            botfig, botax = plt.subplots(1,1, figsize = (10,12))
            nx.draw_networkx(nx.from_numpy_matrix(adj_h3_graph),arrows=True, ax = botax)
            botfig.savefig(args.savepath+'bottom_layer_graph.pdf')
            
            #print and store network statistics
            print("Middle Layer")
            print("Number of edges:" ,h2_graph.number_of_edges())
            print("Number of nodes:",h2_graph.number_of_nodes())
            print('-'*60)
            print("Bottom Layer")
            print("Number of edges:" ,h3_graph.number_of_edges())
            print("Number of nodes:",h3_graph.number_of_nodes())
            print('-'*60)
            nodes_by_layer.append(h2_graph.number_of_nodes())
            edges_by_layer.append(h2_graph.number_of_edges())
            nodes_by_layer.append(h3_graph.number_of_nodes())
            edges_by_layer.append(h3_graph.number_of_nodes())
    
    #---------------------disconnected networks-------------------------
    else:
        h1_graph =  np.zeros((args.top_layer_nodes,args.top_layer_nodes))
        h1_graph = nx.from_numpy_array(h1_graph)
        plt.figure(figsize=(10,7))
        plt.clf()
        nx.draw_networkx(h1_graph, arrows=True)
        #sort toplayer
        ts_h1_graph = list(h1_graph.nodes())
        adj_h1_graph = nx.adjacency_matrix(h1_graph, ts_h1_graph).todense()
        
        h2_graph = hierachical_graph(top_graph = h1_graph, 
                                     subgraph_node_number=args.nodes_per_super2, 
                                     subgraph_type =args.subgraph_type, 
                                     sub_graph_prob=args.subgraph_prob, 
                                     connection_prob=0, 
                                     degree=args.node_degree)
        
        #sort middle layer
        ts_h2_graph = list(nx.topological_sort(h2_graph))
        adj_h2_graph = nx.adjacency_matrix(h2_graph, ts_h2_graph).todense()
        #draw top layer
        topfig, topax = plt.subplots(1,1, figsize = (10,12))
        nx.draw_networkx(nx.from_numpy_matrix(adj_h1_graph),arrows=True, ax = topax)
        topfig.savefig(args.savepath+'top_layer_graph.pdf')
        #print toplayer attributes
        print('-'*60)
        print('Top Layer')
        print("Number of edges:" ,h1_graph.number_of_edges())
        print("Number of nodes:",h1_graph.number_of_nodes())
        print('-'*60)
        nodes_by_layer.append(h1_graph.number_of_nodes())
        edges_by_layer.append(h1_graph.number_of_edges())
        if args.layers == 2:
            
            #draw bottom layer
            botfig, botax = plt.subplots(1,1, figsize = (10,12))
            nx.draw_networkx(nx.from_numpy_matrix(adj_h2_graph),arrows=True, ax = botax)
            botfig.savefig(args.savepath+'bottom_layer_graph.pdf')
            #print network statistics
            print("Bottom Layer")
            print("Number of edges:" ,h2_graph.number_of_edges())
            print("Number of nodes:",h2_graph.number_of_nodes())
            print('-'*60)
            nodes_by_layer.append(h2_graph.number_of_nodes())
            edges_by_layer.append(h2_graph.number_of_edges())
    
        #for 3 layer hierarchies
        if args.layers == 3:
            h3_graph = hierachical_graph(top_graph=h2_graph, 
                                         subgraph_node_number=args.nodes_per_super3, 
                                         subgraph_type =args.subgraph_type, 
                                         sub_graph_prob=args.subgraph_prob, 
                                         connection_prob=0, 
                                         degree=args.node_degree)
            
            ts_h3_graph = list(nx.topological_sort(h3_graph))
            adj_h3_graph = nx.adjacency_matrix(h3_graph, ts_h3_graph).todense()
            #draw middle layer
            midfig, midax = plt.subplots(1,1, figsize = (10,12))
            nx.draw_networkx(nx.from_numpy_matrix(adj_h2_graph),arrows=True, ax = midax)
            midfig.savefig(args.savepath+'middle_layer_graph.pdf')
            #draw bottom layer
            botfig, botax = plt.subplots(1,1, figsize = (10,12))
            nx.draw_networkx(nx.from_numpy_matrix(adj_h3_graph),arrows=True, ax = botax)
            botfig.savefig(args.savepath+'bottom_layer_graph.pdf')
            
            #print and store network statistics
            print("Middle Layer")
            print("Number of edges:" ,h2_graph.number_of_edges())
            print("Number of nodes:",h2_graph.number_of_nodes())
            print('-'*60)
            print("Bottom Layer")
            print("Number of edges:" ,h3_graph.number_of_edges())
            print("Number of nodes:",h3_graph.number_of_nodes())
            print('-'*60)
            nodes_by_layer.append(h2_graph.number_of_nodes())
            edges_by_layer.append(h2_graph.number_of_edges())
            nodes_by_layer.append(h3_graph.number_of_nodes())
            edges_by_layer.append(h3_graph.number_of_edges())


    if args.layers == 2:
        h1_undi = h1_graph.to_undirected()
        h2_undi = h2_graph.to_undirected()
        h1_undi_adj = nx.to_numpy_array(h1_undi)
        h2_undi_adj = nx.to_numpy_array(h2_undi)
        
        ts_full = list(nx.topological_sort(h2_graph))
        adj_full = nx.adjacency_matrix(h2_graph, ts_full).todense()
        print(len(ts_full), adj_full.shape)
    
    else:
        h1_undi = h1_graph.to_undirected()
        h2_undi = h2_graph.to_undirected()
        h3_undi = h3_graph.to_undirected()
        
        h1_undi_adj = nx.to_numpy_array(h1_undi)
        h2_undi_adj = nx.to_numpy_array(h2_undi)
        h3_undi_adj = nx.to_numpy_array(h3_undi)
        
        ts_full = list(nx.topological_sort(h3_graph))
        adj_full = nx.adjacency_matrix(h3_graph, ts_full).todense()
        print(len(ts_full), adj_full.shape)
    
    print('Generating pseudoexpression...')
    pe = generate_pseudo_expression(ts_full, adj_full, args.sample_size)
    print('data dimension = {}'.format(pe.shape))
    #pdb.set_trace()
    #save as .npz
    if args.layers == 2:
        np.savez(args.savepath, layer1 = h1_undi, 
                 adj_layer1 = h1_undi_adj,
                 layer2 = h2_undi, 
                 adj_layer2 = h2_undi_adj,
                 gen_express= pe,
                 labels = ts_full)
        
        adj_all = [h1_undi_adj, h2_undi_adj]
        nx_all = [ts_h1_graph, ts_full]
    else:
        np.savez(args.savepath+'.npz', layer1 = h1_undi, 
                 adj_layer1 = h1_undi_adj,
                 layer2 = h2_undi, 
                 adj_layer2 = h2_undi_adj,
                 layer3 = h3_undi, 
                 adj_layer3= h3_undi_adj, 
                 gen_express= pe,
                 labels = ts_full)
        
        adj_all = [h1_undi_adj, h2_undi_adj, h3_undi_adj]
        nx_all = [ts_h1_graph, ts_h2_graph, ts_full]
        
    #EL = nx.to_pandas_edgelist(nx.DiGraph(adj_full)).head()
    gene_list, sample_list = ts_full, range(500)
    gexp = pd.DataFrame(data=np.transpose(pe), index=sample_list, columns=gene_list)
    print(gexp.shape)
    gexp.to_csv(args.savepath+'_gexp.csv')
    gexp_numpy = gexp.to_numpy()
    np.save(args.savepath+'_gexp.npy', gexp_numpy)
    gexp.head()
    
    
    return pe, gexp, nodes_by_layer, edges_by_layer, nx_all, adj_all, args.savepath, ts_full