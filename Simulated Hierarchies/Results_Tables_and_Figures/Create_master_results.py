# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:44:30 2023

@author: Bruin
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import sys
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/')
sys.path.append('C:/Users/Bruin/Documents/GitHub/HGRN_repo/HGRN_software/')

from itertools import product, chain
import matplotlib.pyplot as plt 
import ast

nets = 22

resolu1=[1, 100]
resolu2=[1, 5]
gam = [1, 0.5, 0]
index = [0, 1]

case_nms = ['A_ingraph_true','A_corr_no_cutoff','A_ingraph02', 
           'A_ingraph05', 'A_ingraph07']
expansion = product(case_nms, index, gam)
lp_main = 'C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/Results_Tables_and_Figures/Simulation_Results'
tables = []

net_type = np.repeat('SM', 8).tolist()+np.repeat('SF', 2).tolist()  
connect = np.repeat('Disc', 4).tolist()+np.repeat('Full', 4).tolist()+['Disc','Disc']
layers = [2,2,3,3,2,2,3,3,2,2]
SD = [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5]

net_info = pd.DataFrame([net_type, connect, layers, SD]).T
net_info.columns = ['net_type','connect','layers','SD']

which_case = []
cnames = ['Network','Input_graph', 'Gamma', 'Delta', 
          'Resolution_lower',
          'Resolution_upper',
                                       'Beth_Hessian_Comms',
                                       'Communities_Upper_Limit',
                                       'Max_Modularity',
                                       'Top_Modularity',
                                       'Middle_Modularity',
                                       'Reconstruction_A',
                                       'Reconstruction_X', 
                                       'Top_homogeneity',
                                       'Middle_homogeneity',
                                       'Top_completeness',
                                       'Middle_completeness',
                                       'Top_NMI',
                                       'Middle_NMI',
                                       'Number_Predicted_Comms_Top',
                                       'Number_Predicted_Comms_Middle',
                                       'Louvain_Modularity',
                                       'Louvain_homogeneity',
                                       'Louvain_completeness',
                                       'Louvain_NMI',
                                       'Louvain_Predicted_comms']

stats = pd.read_csv('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/network_statistics.csv')

for idx, case in enumerate(expansion):
    print(case)
    i = case[1]
    case_nm = case[0]
    which_case.append(case_nm)
    gam_value = case[-1]
    nm = '_gam_'+str(gam_value)+'_delt_'+str(1)+'_reso_'+str(resolu1[i])+'_'+str(resolu2[i])
    tab = pd.read_csv(lp_main+'_'+case_nm+nm+'.csv')
    print(tab.shape)
    nets = tab.shape[0]
    key_val_pairs = [list(ast.literal_eval(i).values()) for i in tab['Metrics']]
    #tab_concat = pd.concat([net_info, tab], axis=1)
    #parse out performance top
    Top_Homogeneity = [i[0][0] for i in key_val_pairs]
    Top_Completeness = [i[0][1] for i in key_val_pairs]
    Top_NMI = [i[0][2] for i in key_val_pairs]
    #parse out performance middle
    Middle_Homogeneity = [i[1][0] if len(i) > 1 else 'NA' for i in key_val_pairs]
    Middle_Completeness = [i[1][1] if len(i) > 1 else 'NA' for i in key_val_pairs]
    Middle_NMI = [i[1][2] if len(i) > 1 else 'NA' for i in key_val_pairs]
    
    #parameter settings
    Resolution_middle = np.repeat(resolu1[i], nets)
    Resolution_top = np.repeat(resolu2[i], nets)
    Gamma_value = np.repeat(gam_value, nets)
    input_graph = np.repeat(case_nm, nets)
    network = stats.subgraph_type[:nets]
    connection = stats.connection_prob[:nets]
    layers = stats.layers[:nets]
    standard_dev = stats.StDev[:nets]
    
    literal = [ast.literal_eval(i) for i in tab['Louvain_Metrics']]
    key_value_pairs2 = [list(i.values()) if len(i) <2 else list(i[0].values()) for i in literal]
    key_value_pairs3 = [list(i[1].values()) if len(i) >1 else 'NA' for i in literal]
    Louvain_Modularity = tab['Louvain_Modularity']
    Louvain_homogeneity_top = [i[0][0] for i in key_value_pairs2]
    Louvain_completeness_top = [i[0][1] for i in key_value_pairs2] 
    Louvain_NMI_top = [i[0][2] for i in key_value_pairs2] 
    
    Louvain_homogeneity_middle = [list(i[1].values())[0][0] if len(i) >1 else 'NA' for i in literal]
    Louvain_completeness_middle = [list(i[1].values())[0][1] if len(i) >1 else 'NA' for i in literal]
    Louvain_NMI_middle = [list(i[1].values())[0][2] if len(i) >1 else 'NA' for i in literal] 
    Louvain_pred_comms = tab['Louvain_Predicted_comms']
    
    
    literal2 = [ast.literal_eval(i) for i in tab['Number_Predicted_Comms']]
    predicted_top = [ i[0] if len(i) < 2 else i[1] for i in literal2]
    predicted_middle = [i[0] if len(i) >1 else 'NA' for i in literal2]
    
    
    literal3 = [ast.literal_eval(i) for i in tab['Modularity']]
    modularity_top = [ i[0] if len(i) < 2 else i[1] for i in literal3]
    modularity_middle = [i[0] if len(i) >1 else 'NA' for i in literal3]
    true_modularity_top = stats.modularity_top[:nets]
    true_modularity_middle = stats.modularity_middle[:nets]
    true_avg_edges_within_top = stats.avg_connect_within_top[:nets]
    true_avg_edges_between_top = stats.avg_connect_between_top[:nets]
    true_avg_edges_within_middle = stats.avg_connect_within_middle[:nets]
    true_avg_edges_between_middle = stats.avg_connect_between_middle[:nets]

    temp = pd.DataFrame([Gamma_value,
                         input_graph,
                         network,
                         connection,
                         layers,
                         standard_dev,
                         Top_Homogeneity, 
                         Top_Completeness, 
                         Top_NMI,
                         Middle_Homogeneity, 
                         Middle_Completeness, 
                         Middle_NMI,
                         Resolution_top, 
                         Resolution_middle, 
                         Louvain_Modularity,
                         Louvain_homogeneity_top, 
                         Louvain_completeness_top,
                         Louvain_NMI_top, 
                         Louvain_homogeneity_middle, 
                         Louvain_completeness_middle, 
                         Louvain_NMI_middle,
                         predicted_top, 
                         predicted_middle, 
                         Louvain_pred_comms,
                         modularity_top,
                         modularity_middle,
                         true_modularity_top,
                         true_modularity_middle,
                         true_avg_edges_between_top,
                         true_avg_edges_between_middle,
                         true_avg_edges_within_top,
                         true_avg_edges_within_middle]).T
    temp.columns=['Gamma', 'input_graph', 
                  'Network_type',
                  'Connection_prob',
                  'Layers',
                  'Dtandard_dev',
                  'Top_homogeneity',
                                  'Top_completeness',
                                  'Top_NMI',
                                  'Middle_homogeneity',
                                  'Middle_Completeness',
                                  'Middle_NMI',
                                  'Resolution_top',
                                  'Resolution_middle',
                                  'Louvain_modularity',
                                  'Louvain_homogenity_top',
                                  'Louvain_completeness_top',
                                  'Louvain_NMI_top',
                                  'Louvain_homogenity_middle',
                                  'Louvain_completeness_middle',
                                  'Louvain_NMI_middle',
                                  'Comms_predicted_top',
                                  'Comms_predicted_middle',
                                  'Louvain_predicted',
                                  'Modularity_top',
                                  'Modularity_middle',
                                  'True_mod_top',
                                  'True_mod_middle',
                                  'True_aeb_top',
                                  'True_aeb_middle',
                                  'True_aew_top',
                                  'True_aew_middle']
    #tab_concat_final = pd.concat([net_info, temp], axis=1)

    tables.append(temp)
    
master_table = pd.concat(tables)

master_table.to_csv('C:/Users/Bruin/Documents/GitHub/HGRN_repo/Simulated Hierarchies/MASTER_results.csv')


# from plotnine import ggplot, aes, geom_bar, geom_boxplot, facet_wrap, geom_point

# #barplots
# (ggplot(data = MT[(MT['Resolution_lower'] == 1) & (MT['Network'] == networks[0])], 
#         mapping = aes(x = 'Input_graph', y = 'Top_NMI',fill = 'Input_graph'))
#  +geom_bar(stat = 'identity', position = 'stack',color = 'black')
#  )

# (ggplot(data = MT[(MT['Resolution_lower'] == 1) & (MT['Network'] == networks[0])], 
#         mapping = aes(x = 'Input_graph', y = 'Top_completeness',fill = 'Input_graph'))
#  +geom_bar(stat = 'identity', position = 'stack',color = 'black')
#  )

# (ggplot(data = MT[(MT['Resolution_lower'] == 1) & (MT['Network'] == networks[0])], 
#         mapping = aes(x = 'Input_graph', y = 'Top_homogeneity',fill = 'Input_graph'))
#  +geom_bar(stat = 'identity', position = 'stack',color = 'black')
#  )
    

# #boxplots
# (ggplot(data = MT, 
#         mapping = aes(x = 'Network', y = 'Top_homogeneity', fill = 'Input_graph'))
#  +geom_point(shape = 'o', size = 3)+facet_wrap(facets = 'Resolution_lower')
#  )