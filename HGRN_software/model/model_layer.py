# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:59:45 2023

@author: Bruin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv, GraphNorm
import torch_geometric.utils as pyg_utils
            

#layer after updating using llama3
class AE_layer(nn.Module):
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
        super(AE_layer, self).__init__()
        # Store in and features for layer
        self.nodes = nodes
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.norm = norm
        self.operator = operator

        # Define the GAT layer
        if self.operator ==  'GATConv':
            
            self.layer = GATConv(in_channels = in_features, 
                                 out_channels = out_features,
                                 heads=heads,
                                 dropout=dropout,
                                 concat = False)
            
        if self.operator == 'GATv2Conv':
            
            self.layer = GATv2Conv(in_channels=in_features,
                                   out_channels=out_features,
                                   heads=heads,
                                   dropout=dropout,
                                   concat = False)
            
        if self.operator == 'SAGEConv':
            
            self.layer = SAGEConv(in_channels = in_features, 
                                  out_channels = out_features,
                                  aggr='mean',
                                  normalize=norm)
        
        # Define normalization layer
        if self.norm:
            #self.act_norm = nn.LayerNorm(out_features)
            self.act_norm = GraphNorm(out_features)
        else:
            self.act_norm = nn.Identity()
        
        #self.final_linear = nn.Linear(out_features * heads, out_features, bias = use_bias)
        
    def forward(self, inputs):
        """
        Forward pass for the GAT layer.
        """
        X, E, attr, attention_list = inputs
        
        
        if self.operator == 'SAGEConv':
            M = self.layer(x=X, edge_index=E)
            attention_list = []
        else:
            # Compute the GAT layer output
            H, (edge_index, attention_weights) = self.layer(x=X, edge_index=E, return_attention_weights=True)
        
            # append attention weights to existing list
            attention_list.append((edge_index, attention_weights))
            # Apply normalization
            M = self.act_norm(H)
        
        # Apply the final linear transformation
        # H_linear = self.final_linear(H)
        
        # Reshape and concatenate heads
        #H_out = H.view(nodes, -1)  # (num_nodes, out_features * heads)
        #H_out = H.reshape(nodes, self.out_features, self.heads).sum(-1)

        
        

        return (M, E, attr, attention_list)







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
            self.act_norm = nn.BatchNorm1d(out_dim)
        else:
            self.act_norm = nn.Identity()
            
            
    def forward(self, X):
        
        #apply linear layer and dropout
        H = self.Linearlayer(X)
        
        #normalize and return
        H_norm = self.act_norm(H)
        
        #apply LeakyReLU activation
        H_act = self.act(H_norm)
        
        #apply dropout
        H_out = self.Dropout_layer(H_act)
        
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

    def __init__(self, in_features, out_comms, norm = True, alpha=0.01, use_bias = True, 
                 init_bias = 0, layer_gain = 'automatic', dropout = 0.2, heads = 1,
                 operator = ['Linear', 'GATConv', 'GATv2Conv', 'SAGEConv']):
        super(Comm_DenseLayer2, self).__init__()
        #store community info
        self.in_features = in_features
        self.out_comms = out_comms
        self.norm = norm
        self.operator = operator
        #set parameter initilization gain
        
        if self.operator == 'Linear':
            
            self.layer = nn.Linear(in_features = in_features, 
                                   out_features = out_comms, 
                                   bias = use_bias)
        
            #self.Dropout_layer = nn.Dropout1d(p = dropout)
            
            
        if self.operator == 'SAGEConv':
            
            self.layer = SAGEConv(in_channels = in_features, 
                                  out_channels = out_comms,
                                  aggr='mean',
                                  normalize=norm)
            
        if self.operator == 'GATConv':
            
            self.layer = GATConv(in_channels = in_features, 
                                 out_channels = out_comms,
                                 heads=heads,
                                 dropout=dropout,
                                 concat = False)
        if self.operator == 'GATv2Conv':
            
            self.layer = GATv2Conv(in_channels=in_features,
                                   out_channels=out_comms,
                                   heads=heads,
                                   dropout=dropout,
                                   concat = False)
        
        #normalize layer output
        if self.norm == True:
            self.out_norm = nn.LayerNorm(out_comms)
        else:
            self.out_norm = nn.Identity()
            
        #output activation 
        if layer_gain == 'automatic':
            self.gain = nn.init.calculate_gain('leaky_relu', alpha)
        else:
            self.gain = layer_gain
        #set layer activation
        self.act = nn.LeakyReLU(negative_slope=alpha)

    
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
        
        if self.operator == 'Linear':
            
            #linear layer and activation
            M = self.layer(Z)
            M_act = self.act(M)
            H = self.out_norm(M_act)
            
        if self.operator == 'SAGEConv':
            
            ei, ea = pyg_utils.dense_to_sparse(A)
            H = self.layer(x=Z, edge_index=ei)
            
        if self.operator in ['GATConv', 'GATv2Conv']:
            ei, ea = pyg_utils.dense_to_sparse(A)
            M = self.layer(x=Z, edge_index=ei)
            
            H = self.out_norm(M)
        
        # class prediction probabilities
        P = F.softmax(H, dim = 1)
        
        #get the centroids and layer adjacency matrix
        X_tilde = self.act(torch.mm(Z.transpose(0,1), P)).transpose(0,1)
        
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
     
