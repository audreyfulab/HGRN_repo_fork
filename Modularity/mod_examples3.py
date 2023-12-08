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

mat = np.zeros((9,9))

def apply_edge(mat, node1, node2):
    
    mat[node1, node2] = 1
    mat[node2, node1] = 1
    
    return mat

mat = apply_edge(mat, 0, 1)
mat = apply_edge(mat, 0, 2)
mat = apply_edge(mat, 0, 3)
mat = apply_edge(mat, 1, 2)
#mat = apply_edge(mat, 1, 3)
mat = apply_edge(mat, 2, 3)
#mat = apply_edge(mat, 2, 4)
mat = apply_edge(mat, 3, 4)
mat = apply_edge(mat, 3, 5)
mat = apply_edge(mat, 4, 5)
mat = apply_edge(mat, 6, 7)
mat = apply_edge(mat, 6, 8)
mat = apply_edge(mat, 7, 8)

#sbn.heatmap(mat)
fig, ax = plt.subplots()
nx.draw_networkx(nx.from_numpy_array(mat), node_size = 15, ax = ax)
print('----------case 1-------------')
print(nx.community.modularity(nx.from_numpy_array(mat), [{0,1,2},{3,4,5},{6,7,8}]))
print(nx.community.modularity(nx.from_numpy_array(mat), [{0,1,2,3,4,5},{6,7,8}]))




mat2 = np.zeros((9,9))

mat2 = apply_edge(mat2, 0, 1)
mat2 = apply_edge(mat2, 0, 2)
mat2 = apply_edge(mat2, 0, 3)
mat2 = apply_edge(mat2, 1, 2)
#mat2 = apply_edge(mat2, 1, 3)
mat2 = apply_edge(mat2, 1, 4)
#mat2 = apply_edge(mat2, 2, 5)
mat2 = apply_edge(mat2, 3, 4)
mat2 = apply_edge(mat2, 3, 5)
mat2 = apply_edge(mat2, 4, 5)
mat2 = apply_edge(mat2, 6, 7)
mat2 = apply_edge(mat2, 6, 8)
mat2 = apply_edge(mat2, 7, 8)

#sbn.heatmap(mat)
fig2, ax2 = plt.subplots()
nx.draw_networkx(nx.from_numpy_array(mat2), node_size = 15, ax = ax2)
print('----------case 2-------------')
print(nx.community.modularity(nx.from_numpy_array(mat2), [{0,1,2},{3,4,5},{6,7,8}]))
print(nx.community.modularity(nx.from_numpy_array(mat2), [{0,1,2,3,4,5},{6,7,8}]))



