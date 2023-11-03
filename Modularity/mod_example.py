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
import matplotlib.pyplot as plt

path = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Modularity/'

source = np.array(['A','B',"C",'D','D','E','E','F','F','F','G','G','H'])
target = np.array(['D','C','B','A','C','F','G','E','G','H','E','F','F'])
weight = np.ones(len(source))

communities1 = [{'A','B','C','D'},{'E','F','G','H'}]
communities2 =[{'A','B','C','D','E','F','G','H'}]

source = np.array(['E','E','F','F','G','G','H'])
target = np.array(['F','G','E','G','E','F','H'])
weight = np.ones(len(source))

#communities1 = [{'A','B','C','D'},{'E','F','G','H'}]
#communities2 =[{'A','B','C','D','E','F','G','H'}]

P1 = torch.tensor(np.array([[1.0,1.0,1.0,0.0],
                           [0.0,0.0,0.0,1.0]]).T)

P2 = torch.tensor(np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]).T)

EL = pd.DataFrame([source, target, weight]).T
EL.columns = ['source','target','weight']

fig, ax = plt.subplots()
graph = nx.from_pandas_edgelist(EL)
nx.draw_networkx(graph, ax = ax)
fig.savefig(path+'graph1_partial.pdf')

print('-----------------------')
A = torch.tensor(nx.to_numpy_array(graph))
print('case 1')
print('Modularity for two communities {:.4f}'.format(Modularity(A, P1)))
print('Modularity for one community {:.4f}'.format(Modularity(A, P2)))

print('-----------------------')



#case two fully connected

source1 = np.array(['A','A',"A",'B','B','B','C','C','C','D','D','D',
                   'E','E','E','F','F','F','G','G','G','H','H','H'])
target1 = np.array(['B','C','D','A','C','D','A','B','D','A','B','C',
                   'F','G','H','E','G','H','E','F','H','E','F','G'])
weight1 = np.ones(len(source1))

communities1 = [{'A','B','C','D'},{'E','F','G','H'}]
communities2 =[{'A','B','C','D','E','F','G','H'}]

P1 = torch.tensor(np.array([[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0],
                           [0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0],
                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]]).T)

P2 = torch.tensor(np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]).T)

EL = pd.DataFrame([source1, target1, weight1]).T
EL.columns = ['source','target','weight']

fig2, ax = plt.subplots()
graph = nx.from_pandas_edgelist(EL)
nx.draw_networkx(graph, ax = ax)
fig2.savefig(path+'graph1_fully.pdf')

print('case 2')
A = torch.tensor(nx.to_numpy_array(graph))
print('Modularity for two communities {:.4f}'.format(Modularity(A, P1)))
print('Modularity for one community {:.4f}'.format(Modularity(A, P2)))




#print(Modularity(A, P1))
#print(Modularity(A, P2))
