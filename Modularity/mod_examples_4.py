# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:05:12 2023

@author: Bruin
"""

import numpy as np
import networkx as nx
#from networkx.community import modularity
import seaborn as sbn
import matplotlib.pyplot as plt
import itertools  
import sys
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
from utilities import Modularity
from simulation_utilities import compute_modularity
import torch.nn.functional as F
import random as rd



sp = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Modularity/'
def apply_edge(mat, node1, node2):
    
    mat[node1, node2] = 1
    mat[node2, node1] = 1
    
    return mat


def easy_modularity(adj, nd_list):
    nl = [0]+nd_list
    idx = np.cumsum(nl)
    deg_seq = adj.sum(axis = 1)
    sig_totals = []
    sig_in = []
    L = adj.sum()
    for i in range(0, len(nl)-1):
        sig_totals.append(deg_seq[idx[i]:idx[i+1]].sum())
        sig_in.append(adj[idx[i]:idx[i+1], idx[i]:idx[i+1]].sum())
    
    mod = (sig_in/L) - np.square(sig_totals/L)
    return mod, np.array(sig_in), np.array(sig_totals)
    

# def mod_sim_2g(nodes1, nodes2, gtype = ['wattz','random','scalef'], seeding = 555, 
#                 number_of_edges = 'all', **kwargs):
#     a = np.arange(nodes1).tolist()
#     b = np.arange(nodes1, nodes1+nodes2).tolist()
#     c = list(itertools.product(a, b))
#     rd.shuffle(c)
#     if number_of_edges != 'all':
#         c = c[:number_of_edges]
#     mat = np.zeros((nodes1+nodes2,nodes1+nodes2))
#     if gtype == 'wattz':    
#         g1 = nx.watts_strogatz_graph(nodes1, seed = seeding, **kwargs)
#         g2 = nx.watts_strogatz_graph(nodes2, seed = seeding, **kwargs)
#     elif gtype == 'random':
#         g1 = nx.gnm_random_graph(nodes1, seed = seeding, **kwargs)
#         g2 = nx.gnm_random_graph(nodes2, seed = seeding, **kwargs)
#     else:
#         g1 = nx.scale_free_graph(nodes1, seed = seeding, **kwargs)
#         g2 = nx.scale_free_graph(nodes2, seed = seeding, **kwargs)
        
#     mat[:nodes1, :nodes1] = nx.to_numpy_array(g1)
#     mat[nodes1:(nodes1+nodes2), nodes1:(nodes1+nodes2)] = nx.to_numpy_array(g2)
#     Gnew = nx.from_numpy_array(mat)
#     comms = [set(a),set(b)]
#     mod, sig_in, sig_tot = easy_modularity(mat, [nodes1, nodes2])
#     mod_by_comm = [mod.tolist()]
#     total_mod = [nx.community.modularity(Gnew, comms)]
#     edges_between_comms = [sig_tot.sum() - sig_in.sum()]
#     edges_within_comms = sig_in.tolist()
    
    
#     for idx, pair in enumerate(c):
#         mat = apply_edge(mat, pair[0], pair[1])
#         Gnew2 = nx.from_numpy_array(mat)
#         total_mod.append(nx.community.modularity(Gnew2, comms))
#         mod, sig_in, sig_tot = easy_modularity(mat, [nodes1, nodes2])
#         edges_between_comms.append(sig_tot.sum() - sig_in.sum())
#         mod_by_comm.append(mod.tolist())
        
        
#     fig, (ax1, ax2) = plt.subplots(2,2, figsize = (14,10))
#     ax1[0].plot(np.array(edges_between_comms)/2, total_mod, label = 'Total Modularity')
#     ax1[0].set_xlabel('Total Edges Between Communities')
#     ax1[0].set_ylabel('Total Modularity for Two Communities')
#     ax1[0].set_title('Modularity For Two Communities')
#     ax1[0].axhline(0, linestyle = '--', color = 'orange')
#     ax1[0].axvline(edges_within_comms[0], linestyle = '-.', color = 'green', label = 'Comm1 In Edges')
#     ax1[0].axvline(edges_within_comms[1], linestyle = '-.', color = 'orange', label = 'Comm2 In Edges')
#     ax1[0].legend()
#     ax1[0].plot(sum(edges_within_comms)/2, 0, 'ro')
#     nx.draw_networkx(Gnew, pos = nx.spring_layout(Gnew, seed = seeding), ax = ax1[1])
#     ax1[1].set_title('Starting Graph With Two Communities')
    
#     ax2[0].plot(np.array(edges_between_comms)/2, np.array(mod_by_comm)[:,0], label = 'Modularity Comm1')
#     ax2[0].plot(np.array(edges_between_comms)/2, np.array(mod_by_comm)[:,1], label = 'Modularity Comm2')
#     ax2[0].set_xlabel('Total Edges Between Communities')
#     ax2[0].set_ylabel('Modularity')
#     ax2[0].set_title('Modularity For Each Community')
    
#     nx.draw_networkx(Gnew2, pos = nx.spring_layout(Gnew2, seed = seeding+1), node_color = 'pink', ax = ax2[1])
#     ax2[1].set_title('Final Graph With Two Communities')
    
#     return mod_by_comm, total_mod
    
        

# mbc, tm = mod_sim_2g(3,3,'wattz', k=3, p = 0.05)





def mod_sim_3g(nds = [3,3,3], gtype = ['wattz','random','scalef'], seeding = 555, 
                connect2comms = True, savepath='', **kwargs):
    
    all_Gs = []
    color_labels = []
    seqs=[np.repeat(i,j).tolist() for i,j in zip([0,1,2], nds)]
    [color_labels.extend(x) for x in seqs]
    nl = np.cumsum([0]+nds)
    #get nodes in each community
    a = np.arange(nds[0]).tolist()
    b = np.arange(nds[0], nds[0]+nds[1]).tolist()
    c = np.arange(nds[0]+nds[1], sum(nds)).tolist()
    #if only connecting two communities
    if connect2comms == True:
        iterit = list(itertools.product(a, b))
    else:
        #allows edges to be placed between all 3 communities 
        iterit = list(itertools.product(a, b))+list(itertools.product(a, c))+list(itertools.product(b, c))
    #randomize order of edges
    rd.shuffle(iterit)
    mat = np.zeros((nl[-1],nl[-1]))
    Graphs = []
    #generate graphs for each community
    for g in range(0, len(nds)):
        if gtype == 'wattz':    
              Graphs.append(nx.watts_strogatz_graph(nds[g], seed = seeding, **kwargs))
        elif gtype == 'random':
            Graphs.append(nx.gnm_random_graph(nds[g], seed = seeding, **kwargs))
        else:
            Graphs.append(nx.scale_free_graph(nds[g], seed = seeding, **kwargs))
        
    #merge 3 subgraphs into single adjacency matrix (all disconnected)
    for i in range(0, len(nl)-1):    
        mat[nl[i]:nl[i+1], nl[i]:nl[i+1]] = nx.to_numpy_array(Graphs[i])
        
    #create and store graph
    Gnew = nx.from_numpy_array(mat)
    all_Gs = [Gnew]
    # get node sets for two communities
    comms2 = [set(a+b), set(c)]
    # get node sets for 3 communities
    comms3 = [set(a), set(b), set(c)]
    #compute initial modularity for each communitiy
    mod, sig_in, sig_tot = easy_modularity(mat, nds)
    #compute initial graph statistics
    mod_by_comm = [mod.tolist()]
    edges_in = [sig_in.tolist()]
    edges_all = [sig_tot.tolist()]
    #Total modularity for 2 and 3 communities
    total_mod2 = [nx.community.modularity(Gnew, comms2)]
    total_mod3 = [nx.community.modularity(Gnew, comms3)]
    # find between and withing edges
    if connect2comms == True:
        edges_between_comms = [(sig_tot.sum() - sig_in.sum())]
    else:
        edges_between_comms = [np.max(sig_tot - sig_in)]
    edges_between_comms = [sig_tot.sum() - sig_in.sum()]
    edges_within_comms = sig_in.tolist()
    
    #add in edges between communities iteratively and update graph and modularity
    #statistics
    for idx, pair in enumerate(iterit):
        mat = apply_edge(mat, pair[0], pair[1])
        Gnew2 = nx.from_numpy_array(mat)
        all_Gs.append(Gnew2)
        total_mod2.append(nx.community.modularity(Gnew2, comms2))
        total_mod3.append(nx.community.modularity(Gnew2, comms3))
        mod, sig_in, sig_tot = easy_modularity(mat, nds)
        edges_in.append(sig_in.tolist())
        edges_all.append(sig_tot.tolist())
        if connect2comms == True:
            edges_between_comms.append((sig_tot/2).sum() - (sig_in/2).sum())
        else:
            edges_between_comms.append(np.max(sig_tot-sig_in))
        mod_by_comm.append(mod.tolist())
        
    #plotting 
    fig, (ax1, ax2) = plt.subplots(2,2, figsize = (14,10))
    #plotting - modularity curves for 2 and 3 communities
    ax1[0].plot(np.array(edges_between_comms), total_mod2, label = 'Total Modularity 2 Comms')
    ax1[0].plot(np.array(edges_between_comms), total_mod3, label = 'Total Modularity 3 Comms')
    if connect2comms == True:
        ax1[0].set_xlabel('Total Edges Between Communities')
    else:
        ax1[0].set_xlabel('Max Edges Between Adjacenct Communities')
    ax1[0].set_ylabel('Modularity for Two Communities')
    ax1[0].set_title('Modularity For Two and Three Communities')
    ax1[0].axhline(0, linestyle = '--', color = 'orange')
    ax1[0].axvline(np.array(edges_within_comms[0])/2, linestyle = '-.', color = 'green', label = 'Comm1 In Edges')
    ax1[0].axvline(np.array(edges_within_comms[1])/2, linestyle = '-.', color = 'orange', label = 'Comm2 In Edges')
    ax1[0].axvline(np.array(edges_within_comms[2])/2, linestyle = '-.', color = 'blue', label = 'Comm3 In Edges')
    diffs = np.absolute(np.subtract(total_mod2, total_mod3))
    point_of_crossing = list(diffs).index(np.min(diffs))
    edges = (np.array(edges_between_comms))[point_of_crossing]
    ax1[0].plot(edges, np.array(total_mod3)[point_of_crossing], 'ro', label = 'Number of edges = '+ str(edges))
    ax1[0].legend()
    
    #graph plot
    nx.draw_networkx(Gnew, ax = ax1[1], node_color = color_labels, node_size = 100)
    ax1[1].set_title('Starting Graph With 3 Communities')
    
    #individual modularities
    ax2[0].plot(np.array(edges_between_comms), np.array(mod_by_comm)[:,0], label = 'Modularity For Comm1')
    ax2[0].plot(np.array(edges_between_comms), np.array(mod_by_comm)[:,1], label = 'Modularity For Comm2')
    ax2[0].plot(np.array(edges_between_comms), np.array(mod_by_comm)[:,2], label = 'Modularity For Comm3')
    ax2[0].axhline(0, linestyle = '--', color = 'black')
    if connect2comms == True:
        ax2[0].set_xlabel('Total Edges Between Communities')
    else:
        ax2[0].set_xlabel('Max Edges Between Adjacenct Communities')
    ax2[0].set_ylabel('Modularity')
    ax2[0].set_title('Modularity Of Each Community')
    ax2[0].legend()  
    
    
    #difference in modularity for 2 and 3 communities against max between edges
    ax2[1].plot(np.array(edges_between_comms), np.array(total_mod2) - np.array(total_mod3), 
                label = 'Diff. In. Modularity') 
    ax2[1].axhline(0, linestyle = '--', color = 'black')
    ax2[1].axvline(edges, linestyle = '-.', color = 'red', 
                label = 'Number of edges = '+ str(edges))
    ax2[1].set_ylabel('Difference in Modularity (2Comms - 3Comms)')
    if connect2comms == True:
        ax2[1].set_xlabel('Total Edges Between Communities')
    else:
        ax2[1].set_xlabel('Max Edges Between Adjacenct Communities')
    ax2[1].legend()
    
    
    fig.savefig(savepath+'_sim_plots.pdf')
    
    fig2, ax0 = plt.subplots(figsize = (14,10))
    ax0.plot(np.array(edges_between_comms), diffs, label = 'Absolute Difference In Modularity')
    if connect2comms == True:
        ax0.set_xlabel('Total Edges Between Communities')
    else:
        ax0.set_xlabel('Max Edges Between Adjacenct Communities')
    ax0.set_ylabel('Absolute Difference Modularity')
    ax0.set_title('Difference In Modularity For 2 and 3 communities')
    ax0.axhline(0, linestyle = '--', color = 'black')
    
    fig2.savefig(savepath+'_mod_diff.pdf')
    
    return mod_by_comm, total_mod2, total_mod3, all_Gs, color_labels

#mbc43, tm2, tm3, Gs, cl = mod_sim_3g([5,5,5],'scalef', connect2comms=True)
mbc43, tm2, tm3, Gs, cl = mod_sim_3g([3,3,3],'wattz', connect2comms=True, 
                             k=2, p = 0.05,
                             savepath = sp+'simplest_case_3community_iteradd')


mbc43, tm2, tm3, Gs, cl = mod_sim_3g([20,15,10],'wattz', connect2comms=True, 
                             k=2, p = 0.05,
                             savepath = sp+'complex_case1_3community_iteradd')



mbc43, tm2, tm3, Gs, cl = mod_sim_3g([25,10,5],'wattz', connect2comms=True, 
                             k=2, p = 0.05,
                             savepath = sp+'complex_case2_3community_iteradd')