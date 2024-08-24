# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:59:45 2023

@author: Bruin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv
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

    def __init__(self, nodes, in_features, out_features, 
                 operator = ['GATConv', 'GATv2Conv', 'SAGEConv'],
                 act=nn.Identity(), norm=True, heads=1, dropout=0.2):
        super(GAT_layer, self).__init__()
        # Store in and features for layer
        self.nodes = nodes
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.norm = norm

        # Define the GAT layer
        if operator ==  'GATConv':
            
            self.layer = GATConv(in_channels = in_features, 
                                 out_channels = out_features,
                                 heads=heads,
                                 dropout=dropout,
                                 concat = False)
            
        if operator == 'GATv2Conv':
            
            self.layer = GATv2Conv(in_channels=in_features,
                                   out_channels=out_features,
                                   heads=heads,
                                   dropout=dropout,
                                   concat = False)
            
        if operator == 'SAGEConv':
            
            self.layer = SAGEConv(in_channels = in_features, 
                                  out_channels = out_features)
        
        # Define normalization layer
        if self.norm:
            self.act_norm = nn.LayerNorm(out_features)
        else:
            self.act_norm = nn.Identity()
        
        #self.final_linear = nn.Linear(out_features * heads, out_features, bias = use_bias)
        
    def forward(self, inputs):
        """
        Forward pass for the GAT layer.
        """
        X, E, attr, attention_list = inputs
        
        # Compute the GAT layer output
        H, (edge_index, attention_weights) = self.GAT(x=X, edge_index=E, return_attention_weights=True)
        
        # append attention weights to existing list
        attention_list.append((edge_index, attention_weights))
        
        # Apply the final linear transformation
        # H_linear = self.final_linear(H)
        
        # Reshape and concatenate heads
        #H_out = H.view(nodes, -1)  # (num_nodes, out_features * heads)
        #H_out = H.reshape(nodes, self.out_features, self.heads).sum(-1)

        # Apply normalization
        H_norm = self.act_norm(H)

        return (H_norm, E, attr, attention_list)







class Fully_ConnectedLayer(nn.Module):
    """
    
    """

    def __init__(self, in_dim, out_dim, norm = True, alpha=0.01, use_bias = True, 
                 layer_gain = 'automatic', dropout = 0.2):

        super(Fully_ConnectedLayer, self).__init__()
        #store community info
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = norm
        #set parameter initilization gain
        if layer_gain == 'automatic':
            self.gain = nn.init.calculate_gain('leaky_relu', alpha)
        else:
            self.gain = layer_gain
            
        self.Linearlayer = nn.Linear(in_features = in_dim, 
                                     out_features = out_dim, 
                                     bias = use_bias)
        
        self.Dropout_layer = nn.Dropout1d(p = dropout)
        #set layer activation
        self.act = nn.LeakyReLU(negative_slope=alpha)
        #normalize inputs
        if self.norm == True:
            self.act_norm = nn.LayerNorm(out_dim)
        else:
            self.act_norm = nn.Identity()
            
            
    def forward(self, X):
        
        #apply linear layer and dropout
        H = self.Dropout_layer(self.Linearlayer(X))
        
        #apply LeakyReLU activation
        H_act = self.act(H)
        
        #normalize and return
        H_out = self.act_norm(H_act)
        
        return H_out













class Comm_DenseLayer2(nn.Module):
    """
    Basic Dense Layer For Community Detection on Node Representations
    
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

    def __init__(self, in_feats, out_comms, norm = True, alpha=0.01, use_bias = True, 
                 init_bias = 0, layer_gain = 'automatic', dropout = 0.2):
        super(Comm_DenseLayer2, self).__init__()
        #store community info
        self.in_feats = in_feats
        self.out_comms = out_comms
        self.norm = norm
        #set parameter initilization gain
        if layer_gain == 'automatic':
            self.gain = nn.init.calculate_gain('leaky_relu', alpha)
        else:
            self.gain = layer_gain
            
        self.Linearlayer = nn.Linear(in_features = in_feats, 
                                     out_features = out_comms, 
                                     bias = use_bias)
        
        self.Dropout_layer = nn.Dropout1d(p = dropout)
        #set layer activation
        self.act = nn.LeakyReLU(negative_slope=alpha)
        #normalize inputs
        if self.norm == True:
            self.act_norm = nn.LayerNorm(out_comms)
        else:
            self.act_norm = nn.Identity()

    
    def forward(self, inputs):
        """
        inputs: a list [Z, A, A_tilde, S] where 
                Z: Node representations   N x q
                A: Adjacency matrix   N x N
                A_tilde: graph for connected communities
                S: predicted class labels
        """
        Z=inputs[0]
        A=inputs[1]
        #compute assignment probabilities
        # P = F.softmax(torch.mm(Z, self.W)+self.b, dim = 1)
        # P = F.softmax(self.Linearlayer(Z), dim = 1)
        #linear layer with dropout
        H = self.Dropout_layer(self.Linearlayer(Z))
        # class prediction probabilities
        P = F.softmax(H, dim = 1)
        #get the centroids and layer adjacency matrix
        X_tilde = self.act(self.act_norm(torch.mm(Z.transpose(0,1), P))).transpose(0,1)
        #X_tilde = self.act(torch.mm(Z.transpose(0,1), P))
        A_tilde = torch.mm(P.transpose(0,1), torch.mm(A, P))
        #store
        inputs[0] = X_tilde
        inputs[1] = A_tilde
        #get assignment labels
        S = torch.argmax(P, dim = 1)
        #store subgraphs, assignment probs, and node labels
        inputs[2].append(X_tilde)
        inputs[3].append(A_tilde)
        inputs[4].append(P)
        inputs[5].append(S)
        
        return inputs
     
