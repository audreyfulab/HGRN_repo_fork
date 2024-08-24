# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:38:18 2024

@author: Bruin
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
#from model.model_layer import Comm_DenseLayer as CDL
from model.pyg_model_layer import Comm_DenseLayer2 as CDL2
from model.pyg_model_layer import Fully_ConnectedLayer as FCL
from collections import OrderedDict
from model.pyg_model_layer import GAT_layer
import torch_geometric.utils as pyg_utils
from torchinfo import summary



class GATEgeo(nn.Module):
    """
    GATE model described in https://arxiv.org/pdf/1905.10715.pdf
    
    """

    def __init__(self, in_nodes, in_attrib, normalize = True, hid_sizes=[256, 128, 64], 
                 use_multi_head = False, attn_heads = 1, layer_act = nn.Identity(), 
                 dropout = 0.2, **kwargs):
        super(GATEgeo, self).__init__()
        #store size
        self.in_nodes = in_nodes
        self.in_attrib = in_attrib
        
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        for idx, out in enumerate(hid_sizes):
            
            if use_multi_head == True:
                # add multi head attendtion layers to dictionary
                layer_name = 'GATpyg_multi_'+str(idx)+'-'+str(out)
                module_dict.update({layer_name: GAT_layer(nodes = in_nodes, 
                                                          in_features=in_attrib,
                                                          out_features=out,
                                                          heads=attn_heads,
                                                          norm = normalize,
                                                          dropout = dropout,
                                                          **kwargs)})
            
            else:
                #add GAT layers to dictionary
                layer_name = 'GATpyg_'+str(idx)+'-'+str(out)
                module_dict.update({layer_name: GAT_layer(nodes = in_nodes, 
                                                          in_features=in_attrib,
                                                          out_features=out,
                                                          heads=1,
                                                          norm=normalize,
                                                          dropout = dropout,
                                                          **kwargs)
                                    }) 
            
            module_dict.update({'act'+str(out): layer_act})
            
            in_attrib = out
        
        #build model by pytorch sequential
        self.seqmodel = nn.Sequential(module_dict)
        
        
    def forward(self, X, A):
        weights_list = []
        ei, ea = pyg_utils.dense_to_sparse(A)
        #A_SPT = pyg_utils.to_torch_csr_tensor(Asparse[0], Asparse[1])
        #ei, ea = pyg_utils.to_edge_index(A_SPT)
        H, E, attr, weights_list = self.seqmodel((X, ei, ea, weights_list))
        
        return (H, A, weights_list) 
    

    


class AddLearningLayers(nn.Module):
    """
    extra layers between embedding and comm prediction layers that aim
    to improve class learning
    """
    def __init__(self, in_nodes, in_attrib, sizes = [64, 32], 
                 normalize = True, dropout = 0.2):
        super(AddLearningLayers, self).__init__()
        self.nodes = in_nodes
        self.attrib = in_attrib
        
        
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        
        for idx, size in enumerate(sizes):
            #add output layers to dictionary
            module_dict.update({'LinearLayer'+str(size): FCL(in_dim = in_attrib, 
                                                             out_dim = size,
                                                             norm = normalize,
                                                             dropout = dropout)
                                })
            
            in_attrib = size

        
        #build model by pytorch sequential
        self.model = nn.Sequential(module_dict)
        
    def forward(self, Z):
        
        H = self.model(Z)
        
        return H



class CommClassifer(nn.Module):
    """
    Community Detection Module
    
    """

    def __init__(self, in_nodes, in_attrib, comm_sizes=[60, 10], 
                 dropout = 0.2, normalize = True, **kwargs):
        super(CommClassifer, self).__init__()
        #store size
        self.in_nodes = in_nodes
        self.in_attrib = in_attrib
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        
        for idx, comms in enumerate(comm_sizes):
            #add output layers to dictionary
            module_dict.update({'Comm_Linear_'+str(comms): CDL2(in_feats = in_attrib, 
                                                                out_comms = comms,
                                                                norm = normalize,
                                                                dropout = dropout,
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

    def __init__(self, nodes, attrib, ae_hidden_dims = [256, 128, 64], 
                 ll_hidden_dims = [64, 64], comm_sizes = [60, 10],
                 normalize_inputs = False, dropout = 0.2, use_output_layers = False, 
                 **kwargs):
        super(HCD, self).__init__()
        #copy and reverse decoder layer dims
        decode_dims = ae_hidden_dims.copy()
        decode_dims.reverse()
        decode_dims.append(attrib)
        self.use_output_layers = use_output_layers
        self.comm_sizes = comm_sizes
        self.ae_hidden_dims = ae_hidden_dims
        self.ll_hidden_dims = ll_hidden_dims
        self.normalize_inputs = normalize_inputs
        #set up encoder
        self.encoder = GATEgeo(in_nodes = nodes, 
                               in_attrib = attrib, 
                               hid_sizes=ae_hidden_dims, 
                               normalize = normalize_inputs, 
                               dropout = dropout,
                               **kwargs)
        #set up decoder
        self.decoder = GATEgeo(in_nodes = nodes, 
                               in_attrib = ae_hidden_dims[-1], 
                               hid_sizes=decode_dims[1:], 
                               normalize = normalize_inputs, 
                               dropout = dropout,
                               **kwargs)
        
        if use_output_layers:
            self.fully_connected_layers = AddLearningLayers(in_nodes=nodes, 
                                                            in_attrib=ae_hidden_dims[-1],
                                                            sizes=ll_hidden_dims,
                                                            dropout = dropout)
        
        
            #set up community detection module
            self.commModule = CommClassifer(in_nodes = nodes, 
                                            in_attrib = ll_hidden_dims[-1], 
                                            normalize = normalize_inputs, 
                                            comm_sizes=comm_sizes,
                                            dropout = dropout)
        else:
            #set up community detection module
            self.commModule = CommClassifer(in_nodes = nodes, 
                                            in_attrib = ae_hidden_dims[-1], 
                                            normalize = normalize_inputs, 
                                            comm_sizes=comm_sizes,
                                            dropout = dropout)
        

        # set layer normalization
        if normalize_inputs == True:
            self.act_norm = nn.LayerNorm(attrib)
        else:
            self.act_norm = nn.Identity()
            
        #set dot product decoder activation to sigmoid
        self.dpd_act = nn.Sigmoid()
        #normalization for dpd_activation
        #self.dpd_norm = nn.LayerNorm(nodes)
        #self.dpd_norm = nn.Identity()
        self.dpd_norm = nn.LayerNorm(nodes)
        
    def forward(self, X, A):
        
        #normalize input
        H = self.act_norm(X)
        
        #get representation
        Z, A, encoder_attention_weights = self.encoder(H,A)
        
        #reconstruct adjacency matrix using simple dot-product decoder
        #A_hat = self.dpd_act(torch.mm(Z, Z.transpose(0,1)))
        A_hat = self.dpd_act(self.dpd_norm(torch.mm(Z, Z.transpose(0,1))))
            
        #get reconstructed adjacency
        X_hat, A, decoder_attention_weights = self.decoder(Z, A)
        
        if self.use_output_layers:
            #Output learning layers:
            W = self.fully_connected_layers(Z)
        
            #fit hierarchy
            X_top, A_top, X_all, A_all, P_all, S = self.commModule(W, A)
        else:
            X_top, A_top, X_all, A_all, P_all, S = self.commModule(Z, A)
        
        A_all_final = [A]+A_all
        X_all_final = [Z]+X_all
        #return 
        return X_hat, A_hat, X_all_final, A_all_final, P_all, S, {'encoder': encoder_attention_weights, 'decoder':decoder_attention_weights}
        #return X_hat
        
    def summarize(self):
        print('-----------------GATE-Encoder-------------------')
        print(summary(self.encoder))
        print('-----------------GATE-Decoder-------------------')
        print(summary(self.decoder))
        if self.use_output_layers:
            print('------------Fully-Connected-Layers--------------')
            print(summary(self.fully_connected_layers))
        print('----------Community-Detection-Module------------')
        print(summary(self.commModule))
        
    





































