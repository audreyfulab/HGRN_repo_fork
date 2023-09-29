# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 00:22:43 2023

@author: Bruin
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import torch.optim as optimizers 
from utilities import Modularity, BCSS, WCSS, node_clust_eval

#------------------------------------------------------
#custom pytorch dataset
class CustomDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

#------------------------------------------------------
#funciton which generates batches from data
def batch_data(input, batch_size, shuffle_data = True):
    """
    This function generates batches from node attribute matrix
    """
    dataset = CustomDataset(input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

#-----------------------------------------------------
#function which splits training and testing data
def train_test(dataset, prop_train):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train, test, train_size, test_size

#-----------------------------------------------------
#custom modularity loss function
class ModularityLoss(nn.Module):
    def __init__(self):
        super(ModularityLoss, self).__init__()
        
    def forward(self, all_A, all_P):
        loss = torch.Tensor([0])
        for index, (A, P) in enumerate(zip(all_A, all_P)):
            loss+= Modularity(A, P)
        return loss.sum()
    
#------------------------------------------------------  
class ClusterLoss(nn.Module):
    def __init__(self, weighting = ['kmeans','anova'], norm_degree=2):
        super(ClusterLoss, self).__init__()
        self.norm_deg = norm_degree
        self.weighting = weighting


    
    
    def forward(self, Attributes, cluster_labels):
        
        """
        computes forward loss
        """
        loss = torch.Tensor([0])
        for idx, labels in enumerate(cluster_labels):
            #compute total number of clusters
            number_of_clusters = torch.tensor(torch.unique(labels).shape[0])
            #within cluster sum of squares
            within_ss, centroids = WCSS(X = Attributes, 
                                        clustlabs=labels, 
                                        num_clusters = number_of_clusters,
                                        norm_degree=self.norm_deg,
                                        weight_by = self.weighting)
            #between cluster sum of squares
            between_ss = BCSS(X = Attributes,
                              cluster_centroids=centroids,
                              numclusts = number_of_clusters,
                              norm_degree = self.norm_deg, 
                              weight_by = self.weighting)
            #add loss
            loss += between_ss/within_ss

        return loss

    
   


#------------------------------------------------------
#this function fits the HRGNgene model to data
def fit(model, X, A, optimizer='Adam', batch = 128, epochs = 100, update_interval=10, 
        lr = 1e-4, prop_train = 0.8, gamma = 1, delta = 1, 
        comm_loss = ['Modularity', 'Clustering'], **kwargs):
    """
    
    """
    train_results = []
    test_results = []
    
    #train_data, test_data, train_size, test_size = train_test(X.transpose(0,1), 
    #                                                          prop_train) 
    #batched_train_data = batch_data(train_data, batch_size=batch)
    
    
    print(model)
    
    #set optimizer Adam
    optimizer = optimizers.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4
    )
    
    #set loss functions
    A_recon_loss = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
    #A_recon_loss = torch.nn.NLLLoss()
    X_recon_loss = torch.nn.MSELoss(reduction = 'mean')
    if comm_loss == 'Modularity':    
        community_loss_fn = ModularityLoss()
    elif comm_loss == 'Clustering':
        community_loss_fn = ClusterLoss(weighting='kmeans')
    
    #begin training epochs
    for idx, epoch in enumerate(range(epochs)):
        #epoch printing
        start_epoch = time.time()
        if epoch % update_interval == 0:
            print('Epoch {} starts !'.format(epoch))
            print('=' * 80)
        total_loss = 0
        model.train()

        #zero out gradient
        optimizer.zero_grad()
        #batch = data.transpose(0,1)
        #compute forward output
        X_hat, A_hat, A_all, P_all, S = model.forward(X, A)

        
        #compute community detection loss
        if comm_loss == 'Modularity':    
            community_loss = community_loss_fn(A_all, P_all)
        elif comm_loss == 'Clustering':
            community_loss = community_loss_fn(X, S)
        
        #sum up losses
        loss = X_recon_loss(X_hat, X)+gamma*A_recon_loss(A_hat, A)-delta*community_loss

        # update
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()
        #evaluate epoch
        if epoch % update_interval == 0:
            
            model.eval()
            #test_X = test_data.__getitem__(torch.arange(0, test_size).tolist())
            X_pred, A_pred, A_list, P_list, S_pred = model.forward(X, A)
            pred_labels = S_pred[0].detach().numpy()
            
            print('Epoch {} total_loss = {:.3f}'.format(
                epoch, total_loss
                ))
            
            print('Community Detection Loss = {:.3f}, X Recontrstuction = {:.3f}, A Recontructions = {:.3f}'.format(
                int(community_loss.detach().numpy()), 
                X_recon_loss(X_hat, X).detach().numpy(), 
                A_recon_loss(A_hat, A).detach().numpy()
                ))
            eval_metrics = node_clust_eval(pred_labels=pred_labels, **kwargs)
            print('Evaluations: homogeniety = {:.3f}, completeness = {:.3f}, NMI = {:.3f}'.format(
                eval_metrics[0], eval_metrics[1], eval_metrics[2]
                ))
            print('-' * 80)
            
    
    X_final, A_final, A_all_final, P_all_final, S_final = model.forward(X, A)
    return X_final, A_final, A_all_final, P_all_final, S_final
    #X_final = model.forward(X, A)
    #return X_final