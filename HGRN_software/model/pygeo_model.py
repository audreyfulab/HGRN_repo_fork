# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:38:18 2024

@author: Bruin
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from model_layer import Comm_DenseLayer as CDL
from collections import OrderedDict
from pyg_model_layer import GAT_layer
import torch_geometric.utils as pyg_utils




class GATEgeo(nn.Module):
    """
    GATE model described in https://arxiv.org/pdf/1905.10715.pdf
    
    """

    def __init__(self, in_nodes, in_attrib, normalize = True, hid_sizes=[256, 128, 64], 
                 use_multi_head = False, attn_heads = 1, layer_act = nn.Identity(), **kwargs):
        super(GATEgeo, self).__init__()
        #store size
        self.in_nodes = in_nodes
        self.in_attrib = in_attrib
        
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        for idx, out in enumerate(hid_sizes):
            
            if use_multi_head == True:
                # add multi head attendtion layers to dictionary
                module_dict.update({'GATpyg_multi_'+str(out): GAT_layer(nodes = in_nodes, 
                                                                        in_features=in_attrib,
                                                                        out_features=out,
                                                                        heads=attn_heads,
                                                                        norm = normalize,
                                                                        **kwargs)})
            else:
                #add GAT layers to dictionary
                module_dict.update({'GATpyg_'+str(out): GAT_layer(nodes = in_nodes, 
                                                                  in_features=in_attrib,
                                                                  out_features=out,
                                                                  heads=1,
                                                                  norm=normalize,
                                                                  **kwargs)}) 
                 
            module_dict.update({'act'+str(out): layer_act})
            
            in_attrib = out
        
        #build model by pytorch sequential
        self.seqmodel = nn.Sequential(module_dict)
        
        
    def forward(self, X, A):
        
        ei, ea = pyg_utils.dense_to_sparse(A)
        #A_SPT = pyg_utils.to_torch_csr_tensor(Asparse[0], Asparse[1])
        #ei, ea = pyg_utils.to_edge_index(A_SPT)
        H, E, attr = self.seqmodel((X, ei, ea))
        
        return (H, A) 
    
    
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
    **kwargs: Keyword arguments passed to GATE/GAT module
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
        self.encoder = GATEgeo(in_nodes = nodes, 
                               in_attrib = attrib, 
                               hid_sizes=hidden_dims, 
                               normalize = normalize_inputs, 
                               **kwargs)
        #set up decoder
        self.decoder = GATEgeo(in_nodes = nodes, 
                               in_attrib = hidden_dims[-1], 
                               hid_sizes=decode_dims[1:], 
                               normalize = normalize_inputs, 
                               **kwargs)
        
        #set up community detection module
        self.commModule = CommClassifer(in_nodes = nodes, 
                                        in_attrib = hidden_dims[-1], 
                                        normalize = normalize_inputs, 
                                        comm_sizes=comm_sizes)
        

        # set layer normalization
        if normalize_inputs == True:
            self.act_norm = nn.LayerNorm(nodes)
        else:
            self.act_norm = nn.Identity()
            
        #set dot product decoder activation to sigmoid
        self.dpd_act = nn.Sigmoid()
        #self.dpd_act = nn.Identity()
        
        
    def forward(self, X, A):
        #get representation
        Z, A = self.encoder(X,A)
        #reconstruct adjacency matrix using simple dot-product decoder
        #A_hat = self.dpd_act(torch.mm(Z, Z.transpose(0,1)))
        A_hat = self.dpd_act(self.act_norm(torch.mm(Z, Z.transpose(0,1))))
        #reconstruct node features
        X_hat, A = self.decoder(Z, A)
        #fit hierarchy
        X_top, A_top, X_all, A_all, P_all, S = self.commModule(Z, A)
        
        A_all_final = [A]+A_all
        X_all_final = [Z]+X_all
        #return 
        return X_hat, A_hat, X_all_final, A_all_final, P_all, S
        #return X_hat





































