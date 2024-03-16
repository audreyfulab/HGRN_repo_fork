# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:39:34 2024

@author: Bruin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/')
sys.path.append('/mnt/ceph/jarredk/HGRN_repo/HGRN_software/')
from simulation_utilities import post_hoc_embedding
from run_simulations import run_simulations

savepath = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/DATA/Intermediate_examples/Results/'
#savepath = '/mnt/ceph/jarredk/HGRN_repo/Simulated_Hierarchies/test/'
epochs = 100

#TOAL = FALSE

print('*'*80)
print('*'*80)
print('*'*80)
print('Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [1,1], 0, False
    ))
run_simulations(dataset = 'intermediate',
                save_results = True, 
                gam = 0, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1,1], 
                lam = [1,1],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = False, 
                sp = savepath,
                true_comm_layers = True, 
                use_gpu = True)


print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [1,1], 0.5, False
    ))
run_simulations(save_results = True, 
                gam = 0.5, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1,1], 
                lam = [1,1],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = False, 
                true_comm_layers = True, 
                use_gpu = True)



print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [1,1], 1, False
    ))
run_simulations(save_results = True, 
                gam = 1, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1,1], 
                lam = [1,1],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = False, 
                true_comm_layers = True, 
                use_gpu = True)

#TOAL = TRUE

print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [1,1], 0, True
    ))
run_simulations(save_results = True, 
                gam = 0, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1,1], 
                lam = [1,1],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = True, 
                true_comm_layers = True, 
                use_gpu = True)


print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [1,1], 0.5, True
    ))
run_simulations(save_results = True, 
                gam = 0.5, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1,1], 
                lam = [1,1],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = True, 
                true_comm_layers = True, 
                use_gpu = True)



print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [1,1], 1, True
    ))
run_simulations(save_results = True, 
                gam = 1, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1,1], 
                lam = [1,1],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = True, 
                true_comm_layers = True, 
                use_gpu = True)




#Resolution 5, 1
print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [0.1, 0.0001], 0, False
    ))
run_simulations(save_results = True, 
                gam = 0, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1, 1], 
                lam = [0.1, 0.0001],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = False, 
                true_comm_layers = True, 
                use_gpu = True)


print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [0.1, 0.0001], 0.5, False
    ))
run_simulations(save_results = True, 
                gam = 0.5, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1, 1],
                lam = [0.1, 0.0001],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = False, 
                true_comm_layers = True, 
                use_gpu = True)



print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [0.1, 0.0001], 1, False
    ))
run_simulations(save_results = True, 
                gam = 1, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1, 1],
                lam = [0.1, 0.0001],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = False, 
                true_comm_layers = True, 
                use_gpu = True)

#TOAL = TRUE

print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [0.1, 0.0001], 0, True
    ))
run_simulations(save_results = True, 
                gam = 0, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1, 1], 
                lam = [0.1, 0.0001],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = True, 
                true_comm_layers = True, 
                use_gpu = True)


print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [0.1, 0.0001], 0.5, True
    ))
run_simulations(save_results = True, 
                gam = 0.5, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1, 1],
                lam = [0.1, 0.0001],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = True, 
                true_comm_layers = True, 
                use_gpu = True)



print('*'*80)
print('*'*80)
print('*'*80)
print(' Lambda = {}, Resolution = {}, Gamma = {}, TOAL = {}'.format(
    [1,1], [0.1, 0.0001], 1, True
    ))
run_simulations(save_results = True, 
                gam = 1, 
                delt = 1, 
                learn_rate = 1e-4, 
                epochs = epochs, 
                updates = epochs/2, 
                reso = [1, 1], 
                lam = [0.1, 0.0001],
                hd = [256, 128, 64], 
                loss_fn = 'Modularity', 
                activation = 'LeakyReLU',
                TOAL = True, 
                true_comm_layers = True, 
                use_gpu = True)