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
import matplotlib.pyplot as plt
import seaborn as sbn

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
        for index, (A,P) in enumerate(zip(all_A, all_P)):
            loss+= Modularity(A, P)
        return loss
    
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
    #train_results = []
    #test_results = []
    
    #train_data, test_data, train_size, test_size = train_test(X.transpose(0,1), 
    #                                                          prop_train) 
    #batched_train_data = batch_data(train_data, batch_size=batch)
    
    loss_history=[]
    recon_A_loss_hist=[]
    recon_X_loss_hist=[]
    mod_loss_hist=[]
    
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
        
        
        X_loss = X_recon_loss(X_hat, X)
        A_loss = A_recon_loss(A_hat, A)
        #sum up losses
        loss = X_loss+gamma*A_loss-delta*community_loss

        # update
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()
        #evaluate epoch
        loss_history.append(total_loss)
        recon_A_loss_hist.append(float(A_loss.detach().numpy()))
        recon_X_loss_hist.append(float(X_loss.detach().numpy()))
        mod_loss_hist.append(float(community_loss.detach().numpy()))
        if epoch % update_interval == 0:
            
            model.eval()
            #test_X = test_data.__getitem__(torch.arange(0, test_size).tolist())
            X_pred, A_pred, A_list, P_list, S_pred = model.forward(X, A)
            pred_labels = S_pred[0].detach().numpy()
            
            #loss printing
            print('Epoch {} total_loss = {:.3f}'.format(
                epoch, total_loss
                ))
            
            print('Community Detection Loss = {:.4f}, X Recontrstuction = {:.4f}, A Recontructions = {:.4f}'.format(
                mod_loss_hist[-1], 
                recon_X_loss_hist[-1], 
                recon_A_loss_hist[-1]))
            
            #evaluating performance homogenity, completeness and NMI
            eval_metrics = node_clust_eval(pred_labels=pred_labels, **kwargs)
            print('Evaluations: homogeniety = {:.3f}, completeness = {:.3f}, NMI = {:.3f}'.format(
                eval_metrics[0], eval_metrics[1], eval_metrics[2]
                ))
            print('-' * 80)
            
            #plotting training curves
            if epoch >= 10:
                fig, (ax1, ax2) = plt.subplots(2,2, figsize=(12,10))
                #total loss
                ax1[0].plot(range(0, epoch+1), loss_history, label = 'Total Loss')
                ax1[0].set_xlabel('Training Epochs')
                ax1[0].set_ylabel('Total Loss')
                #reconstruction of graph adjacency
                ax1[1].plot(range(0, epoch+1), recon_A_loss_hist, label = 'Graph Reconstruction Loss')
                ax1[1].set_xlabel('Training Epochs')
                ax1[1].set_ylabel('Graph Reconstruction Loss')
                #reconstruction of node attributes
                ax2[0].plot(range(0, epoch+1), recon_X_loss_hist, label = 'Attribute Reconstruction Loss')
                ax2[0].set_xlabel('Training Epochs')
                ax2[0].set_ylabel('Attribute Reconstruction Loss')
                #modularity
                ax2[1].plot(range(0, epoch+1), mod_loss_hist, label = 'Modularity Loss')
                ax2[1].set_xlabel('Training Epochs')
                ax2[1].set_ylabel('Modularity Loss')
    
    
    #return 
    X_final, A_final, A_all_final, P_all_final, S_final = model.forward(X, A)
    return X_final, A_final, A_all_final, P_all_final, S_final