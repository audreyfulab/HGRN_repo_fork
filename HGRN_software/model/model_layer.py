# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:59:45 2023

@author: Bruin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



# class graph_attention(nn.Module):
#     """
    
#     """
    
#     def __init__(self, in_features, out_features, activation, gain, alpha):
#         super(graph_attention, self).__init__()
#         #attention applied to self
#         self.a_s = nn.Parameter(torch.zeros(size=(out_features, 1)))
#         nn.init.xavier_uniform_(self.a_s.data, gain=gain)
        
#         #attention applied to neighbors
#         self.a_r = nn.Parameter(torch.zeros(size=(out_features, 1)))
#         nn.init.xavier_uniform_(self.a_r.data, gain=gain)
        
#         #set attention weight activations
#         if activation == 'LeakyReLU':
#             self.attn_act = nn.LeakyReLU(negative_slope=alpha)
#         elif activation == 'Sigmoid':
#             self.attn_act = nn.Sigmoid()
        
        
#     def forward(self, inputs, act):
        
#         X, A, H, W = inputs
#         #compute the attention for self
#         M_s = torch.mul(A, torch.mm(H, self.a_s))
#         #compute the attendtion for neighbors
#         M_r = torch.mul(A, torch.mm(H, self.a_r).transpose(0,1))
#         #concatenated into attention weight matrix
#         concat_atten = self.attn_act(M_s + M_r)
#         #ensure that non-edges are not given attention weights
#         zero_vec = -9e15 * torch.ones_like(A)
#         #this function replaces zero values with -9e15 and non-zero values
#         #with their corresponding attention coefficient
#         temp_atten = torch.where(A > 0, concat_atten, zero_vec)
#         C_atten = F.softmax(temp_atten, dim=1)
#         #compute final embeddings
#         H_out = act(torch.mm(C_atten, torch.mm(X, W)))
        
        
#         return H_out

# class gaeGAT_layer(nn.Module):
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

#     def __init__(self, in_features, out_features, heads = 1, 
#                  attention_act = ['LeakyReLU','Sigmoid'], act = nn.Identity(), 
#                  norm = True, alpha=0.2, gain = 1.414, concat = 'sum'):
#         super(gaeGAT_layer, self).__init__()
#         #store in and features for layer
#         self.in_features = in_features
#         self.out_features = out_features
#         self.norm = norm
#         self.alpha = alpha
#         self.gain = gain
#         self.attn_act = attention_act
#         self.attn_heads = heads
#         self.concat = concat
#         #set dense linear layer parameters and initialize
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=gain)
        
#         #activation normalization layer
#         if self.norm == True:
#             self.Norm_layer = nn.LayerNorm(in_features)
            
#         #set dense layer activation
#         self.act = act
        
#         #generate attention heads
#         self.attention_heads = []
#         for i in range(0, self.attn_heads):
#             self.attention_heads.append(graph_attention(in_features = self.in_features, 
#                                         out_features = self.out_features, 
#                                         activation = self.attn_act, 
#                                         gain = self.gain, 
#                                         alpha = self.alpha))
        
        
        
        
#     def forward(self, inputs):
#         """
#         inputs: a tuple (X,A) where 
#                 X: Node features
#                 A: Adjacency matrix
#         """
#         X, A = inputs
#         if self.norm == True:
#             X = self.Norm_layer(X)
#         #compute dense layer embeddings - default activation is identity function
#         H_init = self.act(torch.mm(X, self.W))
        
#         attn_out = []
#         for i in range(0, self.attn_heads):
#             attn_out.append(self.attention_heads[i]((X, A, H_init, self.W), 
#                                                     self.act))
#         attn_stacked = torch.stack(attn_out, dim=0)
#         if self.concat == 'sum':
#             H_final = attn_stacked.sum(dim = 0)
#         if self.concat == 'product':
#             H_final = attn_stacked.prod(dim = 0)
            
        
        
#         return (H_final, A)
        
        
class gaeGAT_layer(nn.Module):
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

    def __init__(self, nodes, in_features, out_features, attention_act = ['LeakyReLU','Sigmoid'], 
                 act = nn.Identity(), norm = True, alpha=0.2, init_bias = 0, gain = 1,
                 dropout = 0.2, use_bias = True):
        super(gaeGAT_layer, self).__init__()
        #store in and features for layer
        self.in_features = in_features
        self.out_features = out_features
        self.norm = norm
        #set dense linear layer parameters and initialize
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=gain)
        # #Set bias parameters
        # self.b = nn.Parameter(torch.zeros(size=(1,out_features)))
        # if use_bias:
        #     torch.nn.init.constant_(self.b.data, init_bias)
        
        self.LinearLayer = nn.Linear(self.in_features, self.out_features,
                                     bias = use_bias)
        #attention applied to self
        self.a_s = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_s.data, gain=gain)
        
        #attention applied to neighbors
        self.a_r = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_r.data, gain=gain)
        
        #set attention weight activations
        if attention_act == 'LeakyReLU':
            self.attn_act = nn.LeakyReLU(negative_slope=alpha)
        elif attention_act == 'Sigmoid':
            self.attn_act = nn.Sigmoid()
            
            
        if self.norm == True:
            self.act_norm = nn.LayerNorm(out_features)
        else:
            self.act_norm = nn.Identity()
            
        #set dense layer activation
        self.act = act
        # layer dropout
        self.dropout = nn.Dropout(dropout)
        
        
        
    def forward(self, inputs):
        """
        inputs: a tuple (X,A) where 
                X: Node features
                A: Adjacency matrix
        """
        X, A = inputs
        #compute dense layer embeddings - default activation is identity function
        #H_in = self.act(torch.mm(X, self.W)+self.b)
        H_in = self.LinearLayer(X)
        #compute the attention for self
        M_s = torch.mul(A, torch.mm(H_in, self.a_s))
        #compute the attendtion for neighbors
        M_r = torch.mul(A, torch.mm(H_in, self.a_r).transpose(0,1))
        #M_r = torch.mul(A, torch.mm(H_in, self.a_r))
        #concatenated into attention weight matrix
        concat_atten = self.attn_act(M_s + M_r)
        #ensure that non-edges are not given attention weights
        zero_vec = -9e15 * torch.ones_like(A)
        #this function replaces zero values with -9e15 and non-zero values
        #with their corresponding attention coefficient
        temp_atten = torch.where(A > 0, concat_atten, zero_vec)
        C_atten = F.softmax(temp_atten, dim=1)
        #compute final embeddings
        #H_out = self.act(torch.mm(self.dropout(C_atten), torch.mm(X, self.W)+self.b))
        H_out = self.act(torch.mm(self.dropout(C_atten), H_in))
        X = self.act_norm(H_out)
        return (H_out, A, [])
        






#multi headed attention layers
class  multi_head_GAT(nn.Module):
    """
    Multi head graph attention layer
    
    - This function wraps the gaeGAT_layer module
    - applies GAT layer k times where k is the specified number of heads
    - concat: specify the concatenation function (in this case average)
    """

    def __init__(self, nodes, in_features, out_features, heads = 1, 
                  attention_act = ['LeakyReLU','Sigmoid'], act = nn.Identity(), 
                  norm = True, alpha=0.2, gain = 1, dropout = 0.2, concat = 'sum'):
        super(multi_head_GAT, self).__init__()
        
        self.nodes = nodes
        self.in_features = in_features
        self.out_features = out_features
        self.attn_act = attention_act
        self.act = act
        self.normalize = norm
        self.concat = concat
        self.num_heads = heads
        self.attention_layers = []

        self.attention_layer = gaeGAT_layer(nodes = self.nodes,
                                            in_features = self.in_features, 
                                            out_features = self.out_features * self.num_heads, 
                                            attention_act = self.attn_act, 
                                            act = self.act, 
                                            norm = self.normalize, 
                                            alpha = alpha, 
                                            gain = gain,
                                            dropout = dropout)
            
        
        
        
    def forward(self, inputs):
        """
        inputs: a tuple (X,A) where 
                X: Node features
                A: Adjacency matrix
        """
        
        H, A, attn = self.attention_layer(inputs[:2])
        
        
        attn_stacked = H.view(self.nodes, self.num_heads, self.out_features)
        if self.concat == 'mean':
            H_final = attn_stacked.mean(dim = 1)
        if self.concat == 'sum':
            H_final = attn_stacked.sum(dim = 1)
        if self.concat == 'product':
            H_final = attn_stacked.prod(dim = 1)
            
        return (H_final, A, attn_stacked)
            
            
            
            
            
            
            
     
class Comm_DenseLayer(nn.Module):
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
                 init_bias = 0, layer_gain = 'automatic'):
        super(Comm_DenseLayer, self).__init__()
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
        #P = F.softmax(torch.mm(Z, self.W)+self.b, dim = 1)
        P = F.softmax(self.Linearlayer(Z), dim = 1)
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
    
        

        
        
        