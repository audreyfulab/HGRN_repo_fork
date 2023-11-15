# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:39:39 2023

@author: Bruin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import sys
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')
from simulation_utilities import plot_nodes
#fig, ax = plt.subplots(1,2)
G = nx.erdos_renyi_graph(5, 0.5)
comms =  np.array([1,1,2,2,3])
A = nx.to_numpy_array(G)
plot_nodes(A, labels = comms, path = 'C:/Users/Bruin/Desktop/test.png', legend = True)
# handles, labels = ax.get_legend_handles_labels()
# cmap = plt.colormaps.get_cmap('plasma')
# rgba = [cmap(i) for i in [1,2,3]]
# patch = [mpatches.Patch(color=rgba[i], label='community_'+str(i)) for i in range(0, 3)]
# handles+patch
# ax[0].legend(handles = handles, loc ='upper right')
#fig.savefig('C:/Users/Bruin/Desktop/test.png',dpi = 300)
#nx.draw_networkx(G, node_color = comms, cmap = 'plasma')