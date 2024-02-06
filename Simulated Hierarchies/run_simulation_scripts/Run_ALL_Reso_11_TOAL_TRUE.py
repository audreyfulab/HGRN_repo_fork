# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:51:04 2024

@author: Bruin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
from simulation_utilities import post_hoc_embedding
from run_simulations import run_simulations

print('*'*80)
print('*'*80)
print('*'*80)
print('Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], 0, True
    ))
output = run_simulations(save_results=True,
                         reso=[1,1],
                         hd=[256, 128, 64],
                         gam=0,
                         delt=1, 
                         learn_rate=1e-4,
                         epochs = 500,
                         updates = 100,
                         loss_fn='Modularity',
                         activation = 'LeakyReLU',
                         TOAL=True, 
                         sp = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/Simulation_Results/')

del output
gc.collect()

print('*'*80)
print('*'*80)
print('*'*80)
print('Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], 0.5, True
    ))
output = run_simulations(save_results=True,
                         reso=[1,1],
                         hd=[256, 128, 64],
                         gam=0.5,
                         delt=1, 
                         learn_rate=1e-4,
                         epochs = 500,
                         updates = 100,
                         loss_fn='Modularity',
                         activation = 'LeakyReLU',
                         TOAL=True, 
                         sp = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/Simulation_Results/')

del output
gc.collect()

print('*'*80)
print('*'*80)
print('*'*80)
print('Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], 1, True
    ))
output = run_simulations(save_results=True,
                         reso=[1,1],
                         hd=[256, 128, 64],
                         gam=1,
                         delt=1, 
                         learn_rate=1e-4,
                         epochs = 500,
                         updates = 100,
                         loss_fn='Modularity',
                         activation = 'LeakyReLU',
                         TOAL=True, 
                         sp = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/Simulation_Results/')

del output
gc.collect()