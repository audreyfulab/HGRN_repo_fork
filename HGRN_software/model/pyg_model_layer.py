# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:59:45 2023

@author: Bruin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv as GATgeo, GATv2Conv
import torch_geometric.utils as pyg_utils

        
        
# class GAT_layer(nn.Module):
#     """
#     GAT layer described in https://arxiv.org/pdf/1905.10715.pdf
#     Gain Setting Recommendations for Xavier Initialization
#         activation       recommended
#         function         gain
#         =======================================
#         sigmoid()        1.0
#         tanh()           5/3 = 1.6667
#         relu()           sqrt(2) = 1.4142
#         Identity         1.0
#         Convolution      1.0
#         LeakyReLU        sqrt(2 / (1 + (-m)^2)
#     """

#     def __init__(self, nodes, in_features, out_features, attention_act = ['LeakyReLU','Sigmoid'], 
#                  act = nn.Identity(), norm = True, heads = 1, dropout = 0.2):
#         super(GAT_layer, self).__init__()
#         #store in and features for layer
#         self.nodes = nodes
#         self.in_features = in_features
#         self.out_features = out_features
#         self.heads = heads
#         self.norm = norm
#         self.GAT = GATv2Conv(in_channels=in_features,
#                           out_channels=out_features,
#                           heads=heads,
#                           dropout = dropout)
        
        
#         if self.norm == True:
#             self.act_norm = nn.LayerNorm(out_features)
#         else:
#             self.act_norm = nn.Identity()
        
#         self.attention_weights = []
        
        
#     def forward(self, inputs):
#         """

#         """
#         X, E, attr = inputs
#         H = self.GAT(x = X, edge_index=E)
        
#         H_out = H.reshape(self.nodes, self.out_features, self.heads).sum(dim = 2)
#         #self.attention_weights+pyg_utils.to_dense_adj(edge_index = alpha[0], edge_attr = alpha[1])
        
#         return (H_out, E, attr)
            

#layer after updating using llama3
class GAT_layer(nn.Module):
    """
    GAT layer described in https://arxiv.org/pdf/1905.10715.pdf
    Gain Setting Recommendations for Xavier Initialization
        activation       recommended
        function         gain
        =======================================
        sigmoid()        1.0
        tanh()           5/3 = 1.6667
        relu()           sqrt(2) = 1.4142
        Identity         1.0
        Convolution      1.0
        LeakyReLU        sqrt(2 / (1 + (-m)^2)
    """

    def __init__(self, nodes, in_features, out_features, attention_act=['LeakyReLU', 'Sigmoid'], 
                 act=nn.Identity(), norm=True, heads=1, dropout=0.2, use_bias = True):
        super(GAT_layer, self).__init__()
        # Store in and features for layer
        self.nodes = nodes
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.norm = norm

        # Define the GAT layer
        self.GAT = GATv2Conv(in_channels=in_features,
                             out_channels=out_features,
                             heads=heads,
                             dropout=dropout)
        
        # Define normalization layer
        if self.norm:
            self.act_norm = nn.LayerNorm(out_features)
        else:
            self.act_norm = nn.Identity()
        
        self.final_linear = nn.Linear(out_features * heads, out_features, bias = use_bias)
        
    def forward(self, inputs):
        """
        Forward pass for the GAT layer.
        """
        X, E, attr = inputs

        # Compute the GAT layer output
        H = self.GAT(x=X, edge_index=E)

        # Reshape and concatenate heads
        H_out = H.view(self.nodes, -1)  # (num_nodes, out_features * heads)

        # Apply the final linear transformation
        H_out = self.final_linear(H_out)

        # Apply normalization
        H_out = self.act_norm(H_out)

        return (H_out, E, attr)



            
     
