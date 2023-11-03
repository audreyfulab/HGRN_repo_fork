# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:59:45 2023

@author: Bruin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, in_features, out_features, attention_act = ['LeakyReLU','Sigmoid'], 
                 act = nn.Identity(), alpha=0.2, gain = 1):
        super(gaeGAT_layer, self).__init__()
        #store in and features for layer
        self.in_features = in_features
        self.out_features = out_features
        
        #set dense linear layer parameters and initialize
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        
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
            
        #set dense layer activation
        self.act = act
        
        
        
        
    def forward(self, inputs):
        """
        inputs: a tuple (X,A) where 
                X: Node features
                A: Adjacency matrix
        """
        X = inputs[0]
        A = inputs[1]
        #compute dense layer embeddings - default activation is identity function
        H_init = self.act(torch.mm(X, self.W))
        #compute the attention for self
        M_s = torch.mul(A, torch.mm(H_init, self.a_s))
        #compute the attendtion for neighbors
        M_r = torch.mul(A, torch.mm(H_init, self.a_r).transpose(0,1))
        #concatenated into attention weight matrix
        concat_atten = self.attn_act(M_s + M_r)
        #ensure that non-edges are not given attention weights
        zero_vec = -9e15 * torch.ones_like(A)
        #this function replaces zero values with -9e15 and non-zero values
        #with their corresponding attention coefficient
        temp_atten = torch.where(A > 0, concat_atten, zero_vec)
        C_atten = F.softmax(temp_atten, dim=1)
        #compute final embeddings
        H_final = self.act(torch.mm(C_atten, torch.mm(X, self.W)))
        
        return (H_final, A)
        
        
        
        
     
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

    def __init__(self, in_feats, out_comms, alpha=0.2, gain = 1.414):
        super(Comm_DenseLayer, self).__init__()
        #store community info
        self.in_feats = in_feats
        self.out_comms = out_comms
        #set dense linear layer parameters and initialize
        self.W = nn.Parameter(torch.zeros(size=(in_feats, out_comms)))
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        #set layer activation
        self.act = nn.LeakyReLU(alpha)
        
    # def compute_centroids(self, P, Z):
    #     """
    #     Z: node representations N x q    or    layer centroids  k_l x q 
    #     P: super node assignment probabilities 
    #     """
    #     X_tilde = self.act(torch.mm(Z, P))
        
    #     return X_tilde
    
    # def compute_adjacency(self, P, A):
    #     """
    #     P: super node assignment probabilities
    #     A: input adjacency matrix of size N x N
    #     """
    #     A_tilde = torch.mm(P.transpose(0,1), torch.mm(A, P))
        
    #     return A_tilde
    
    def forward(self, inputs):
        """
        inputs: a list [Z,A,A_tilde, S] where 
                Z: Node representations   N x q
                A: Adjacency matrix   N x N
                A_tilde:
                S:
        """
        Z=inputs[0]
        A=inputs[1]
        #compute assignment probabilities
        P = F.softmax(torch.mm(Z, self.W), dim = 1)
        #get the centroids and layer adjacency matrix
        X_tilde = self.act(torch.mm(Z.transpose(0,1), P)).transpose(0,1)
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
    
        

        
        
        