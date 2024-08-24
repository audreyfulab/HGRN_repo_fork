# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:59:42 2023

@author: Bruin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_layer import gaeGAT_layer as GAT 
from model.model_layer import multi_head_GAT
from model.model_layer import Comm_DenseLayer as CDL
from collections import OrderedDict
from torchinfo import summary


class GATE(nn.Module):
    """
    GATE model described in https://arxiv.org/pdf/1905.10715.pdf
    
    """

    def __init__(self, in_nodes, in_attrib, normalize = True, hid_sizes=[256, 128, 64], 
                 use_multi_head = False, attn_heads = 1, layer_act = nn.Identity(), **kwargs):
        super(GATE, self).__init__()
        #store size
        self.in_nodes = in_nodes
        self.in_attrib = in_attrib
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        for idx, out in enumerate(hid_sizes):
            
            if use_multi_head == True:
                # add multi head attendtion layers to dictionary
                module_dict.update({'GAT_'+str(out): multi_head_GAT(nodes = in_nodes,
                                                                    in_features = in_attrib, 
                                                                    out_features = out,
                                                                    heads = attn_heads,
                                                                    norm = normalize,
                                                                    **kwargs)})
            else:
                #add GAT layers to dictionary
                module_dict.update({'GAT_'+str(out): GAT(in_features = in_attrib, 
                                                         out_features = out,
                                                         norm = normalize,
                                                         **kwargs)})
            
            module_dict.update({'act'+str(out): layer_act})
            
            in_attrib = out
        
        #build model by pytorch sequential
        self.model = nn.Sequential(module_dict)
        
        
    def forward(self, X, A):
        out = self.model((X, A))
        
        return out
    
    
class CommClassifer(nn.Module):
    """
    Community Detection Module
    
    """

    def __init__(self, in_nodes, in_attrib, comm_sizes=[60, 10], 
                 normalize = True, **kwargs):
        super(CommClassifer, self).__init__()
        #store size
        self.in_nodes = in_nodes
        self.in_attrib = in_attrib
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        for idx, comms in enumerate(comm_sizes):
            #add GAT layers to dictionary
            module_dict.update({'Comm_Linear_'+str(comms): CDL(in_feats = in_attrib, 
                                                         out_comms = comms,
                                                         norm = normalize,
                                                         **kwargs)})
            
        
        #build model by pytorch sequential
        self.model = nn.Sequential(module_dict)
        
        
    def forward(self, Z, A):
        H_layers = self.model([Z, A, [], [], [], []])
        
        return H_layers
        


class HCD(nn.Module):
    """
    Hierarchical Graph Representation Network for genes
    nodes: (integer) number of nodes in attributed graph
    attrib: (integer) number of node-attributes (i.e features)
    hidden_dims: (list) of integers giving the size of the hidden layers
    comm_sizes: (list) giving the number of super nodes/communities in 
                hierarchcial layers
    attn_act: one of 'Sigmoid' or 'LeakyReLU' - sets the activation function
              used when computing the attention weights (default = 'Sigmoid')
    **kwargs: Keyword arguments passed to GATE module
    """

    def __init__(self, nodes, attrib, hidden_dims = [256, 128, 64], comm_sizes = [60, 10],
                 attn_act='Sigmoid', normalize_inputs = False, **kwargs):
        super(HCD, self).__init__()
        #copy and reverse decoder layer dims
        decode_dims = hidden_dims.copy()
        decode_dims.reverse()
        decode_dims.append(attrib)
        self.comm_sizes = comm_sizes
        self.hidden_dims = hidden_dims
        #set up encoder
        self.encoder = GATE(in_nodes = nodes, in_attrib = attrib, hid_sizes=hidden_dims,
                            attention_act = attn_act, normalize = normalize_inputs, **kwargs)
        #set up decoder
        self.decoder = GATE(in_nodes = nodes, in_attrib = hidden_dims[-1], hid_sizes=decode_dims[1:],
                            attention_act = attn_act, normalize = normalize_inputs, **kwargs)
        
        #set up community detection module
        self.commModule = CommClassifer(in_nodes = nodes, in_attrib = hidden_dims[-1], 
                                        normalize = normalize_inputs, comm_sizes=comm_sizes)
        

        # set layer normalization
        if normalize_inputs == True:
            self.act_norm = nn.LayerNorm(attrib)
        else:
            self.act_norm = nn.Identity()
            
        #normalization for dpd_activation
        self.dpd_norm = nn.LayerNorm(nodes)
        #set dot product decoder activation to sigmoid
        self.dpd_act = nn.Sigmoid()
        #self.dpd_act = nn.Identity()
        
        
    def forward(self, X, A):
        
        #normalize inputs
        H = self.act_norm(X)
        #get representation
        Z, A, encoder_attention = self.encoder(H,A)
        #reconstruct adjacency matrix using simple dot-product decoder
        #A_hat = self.dpd_act(torch.mm(Z, Z.transpose(0,1)))
        A_hat = self.dpd_act(self.dpd_norm(torch.mm(Z, Z.transpose(0,1))))
        #reconstruct node features
        X_hat, A, decoder_attention = self.decoder(Z, A)
        #fit hierarchy
        X_top, A_top, X_all, A_all, P_all, S = self.commModule(Z, A)
        
        A_all_final = [A]+A_all
        X_all_final = [Z]+X_all
        #return 
        return X_hat, A_hat, X_all_final, A_all_final, P_all, S, [encoder_attention, decoder_attention]
        #return X_hat
        
    def summarize(self):
        print('-----------------GATE-Encoder-------------------')
        print(summary(self.encoder))
        print('-----------------GATE-Decoder-------------------')
        print(summary(self.decoder))
        print('----------Community-Detection-Module------------')
        print(summary(self.commModule))
        