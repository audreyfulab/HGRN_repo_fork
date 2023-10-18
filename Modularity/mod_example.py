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
from utilities import Modularity

source = np.array(['A','B',"C",'D','D','E','E','F','F','F','G','G','H'])
target = np.array(['D','C','B','A','C','F','G','E','G','H','E','F','F'])
weight = np.ones(len(source))

communities1 = [{'A','B','C','D'},{'E','F','G','H'}]
communities2 =[{'A','B','C','D','E','F','G','H'}]

P1 = torch.tensor(np.array([[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0]]).T)

P2 = torch.tensor(np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]).T)

EL = pd.DataFrame([source, target, weight]).T
EL.columns = ['source','target','weight']


graph = nx.from_pandas_edgelist(EL)
nx.draw_networkx(graph)

print(comm.modularity(graph, communities1))
print(comm.modularity(graph, communities2))

A = torch.tensor(nx.to_numpy_array(graph))


print(Modularity(A, P1))
print(Modularity(A, P2))
