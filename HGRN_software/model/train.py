# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 00:22:43 2023

@author: Bruin
"""


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import torch.optim as optimizers 
from model.utilities import Modularity, WCSS, node_clust_eval
from tqdm import tqdm
from model.utilities import trace_comms, get_layered_performance
from model.utilities import plot_loss, plot_perf, plot_nodes, plot_clust_heatmaps
import copy

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
        
    def forward(self, all_A, all_P, resolutions):
        loss = torch.Tensor([0])
        loss_list = []
        for index, (A,P) in enumerate(zip(all_A, all_P)):
            mod = Modularity(A, P, resolutions[index])
            loss+= mod
            loss_list.append(float(mod.cpu().detach().numpy()))
        return loss, loss_list
 
    
 

 
#------------------------------------------------------  
class ClusterLoss(nn.Module):
    
    def __init__(self):
        """
        Hierarchical Clustering Loss
        """
        super(ClusterLoss, self).__init__()

    
    # forward method for loss computed using input feature matrix
    def forward(self, Lamb, Attributes, Probabilities, Cluster_labels):
        
        """
        Computes forward loss for hierarchical within-cluster sum of squares loss
        Lamb: list of lenght l corresponding to the tuning loss for l hierarchical layers
        Attributes: Node feature matrix
        Probabilities: a list of length l corresponding the assignment probabilities for 
                        assigning nodes to communities in l hierarchical layers
        Cluster_labels: list of length l containing cluster assignment labels 
        """
        N = Attributes.shape[0]
        loss = torch.Tensor([0])
        loss_list = []
        ptensor_list = [torch.eye(N)]+Probabilities
        for idx, labels in enumerate(Cluster_labels):
            #compute total number of clusters
            number_of_clusters = len(torch.unique(labels))
            #within cluster sum of squares
            within_ss, centroids = WCSS(X = Attributes,
                                        Plist = ptensor_list[:(idx+2)],
                                        k = number_of_clusters)
            
            #update loss list
            loss_list.append(Lamb[idx]*float(within_ss.cpu().detach().numpy()))
            #update loss
            loss += Lamb[idx]*within_ss

        return loss, loss_list



    # forward method for loss computed using GAE model embedding
    # def forward(self, Lamb, Attributes, Probabilities, cluster_labels):

    #     """
    #     computes forward loss
    #     Computes forward loss for hierarchical within-cluster sum of squares loss
    #     Lamb: list of lenght l corresponding to the tuning loss for l hierarchical layers
    #     Attributes: Node feature matrix
    #     Probabilities: a list of length l corresponding the assignment probabilities for 
    #                     assigning nodes to communities in l hierarchical layers
    #     Cluster_labels: list of length l containing cluster assignment labels 
    #     """
    
    #     #N = Attributes[0].shape[0]
    #     loss = torch.Tensor([0])
    #     loss_list = []
    #     #problist = [torch.eye(N)]+Probabilities
    #     #onehots = [torch.eye(N)]+[F.one_hot(i).type(torch.float32) for i in cluster_labels]
    #     for idx, (features, probs, labels) in enumerate(zip(Attributes, Probabilities, cluster_labels)):
    #         #compute total number of clusters
    #         number_of_clusters = len(torch.unique(labels))
    #         #within cluster sum of squares
    #         within_ss, centroids = WCSS(X = features,
    #                                     P = probs,
    #                                     k = number_of_clusters)


    #         #update loss list
    #         loss_list.append(Lamb[idx]*float(within_ss.cpu().detach().numpy()))
    #         #update loss
    #         loss += Lamb[idx]*within_ss

    #     return loss, loss_list







def evaluate(model, X, A, k, true_labels):
    
    model.eval()
    X_pred, A_pred, X_list, A_list, P_list, S_pred, AW_pred = model.forward(X, A)
    S_trace_eval = trace_comms([i.cpu().clone() for i in S_pred], model.comm_sizes)
    
    S_all, S_relab, S_out = S_trace_eval
    perf_layers = get_layered_performance(k, S_trace_eval[1], true_labels)
        
    return perf_layers, (X_pred, A_pred, S_relab)



def print_performance(history, comm_layers, k):
    lnm = ['top']+['middle_'+str(i) for i in np.arange(comm_layers-1)[::-1]]
    for i in range(0, k):
        print('-' * 36 + '{} layer'.format(lnm[i]) + '-' * 36)
        print('\nHomogeneity = {:.4f}, \nCompleteness = {:.4f}, \nNMI = {:.4f}'.format(
            history[-1][i][0], 
            history[-1][i][1], 
            history[-1][i][2]))
        print('-' * 80)
        
        
        
def print_losses(epoch, total_loss, mod_loss_hist, clust_loss_hist, X_loss_hist, A_loss_hist):
    #------------------------------
    print('\nEpoch {} \nTotal Loss = {:.4f}'.format(
        epoch+1, total_loss
        ))
    
    print('\nModularity = {}, \nClustering = {}, \nX Recontrstuction = {:.4f}, \nA Recontructions = {:.4f}'.format(
        np.round(mod_loss_hist[-1],4),
        np.round(clust_loss_hist[-1],4),
        X_loss_hist[-1], 
        A_loss_hist[-1]))
    
   


#------------------------------------------------------
#this function fits the HRGNgene model to data
def fit(model, X, A, optimizer='Adam', epochs = 100, update_interval=10, lr = 1e-4, 
        gamma = 1, delta = 1, lamb = 1, layer_resolutions = [1,1], k = 2,
        true_labels = [], turn_off_A_loss = False, validation_data = None, 
        save_output = False, output_path = 'path/to/output', fs = 10, ns = 10, 
        verbose = True, **kwargs):
    
    """
    
    """
    
    #preallocate storage
    loss_history=[]
    A_loss_hist=[]
    X_loss_hist=[]
    mod_loss_hist=[]
    clust_loss_hist=[]
    perf_hist = []
    valid_perf_hist = []
    updates = []
    time_hist = []
    comm_layers = len(model.comm_sizes)
    print(model)
    pred_list = []
    
    #set optimizer Adam
    optimizer = optimizers.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4
    )
    
    #set loss functions
    #A_recon_loss = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
    A_recon_loss = torch.nn.BCELoss(reduction = 'mean')
    #A_recon_loss = torch.nn.NLLLoss()
    X_recon_loss = torch.nn.MSELoss(reduction = 'mean')

    #initiate loss functions
    modularity_loss_fn = ModularityLoss()
    clustering_loss_fn = ClusterLoss()
    #pre-allocate storage space for training info
    all_out = []
    
    #------------------begin training epochs----------------------------
    for idx, epoch in enumerate(tqdm(range(epochs), desc="Fitting...", ascii=False, ncols=75)):
        #epoch printing
        start_epoch = time.time()
        if epoch % update_interval == 0:
            print('Epoch {} starts !'.format(epoch))
            print('=' * 55)
            print('-' * 55)
            print('=' * 55+'\n')
        total_loss = 0
        model.train()
        
        #zero out gradient
        optimizer.zero_grad()
        
        #compute forward output 
        X_hat, A_hat, X_all, A_all, P_all, S, AW = model.forward(X, A)
        
        S_sub, S_relab, S_all = trace_comms([i.cpu().clone() for i in S], model.comm_sizes)
        
        #update all output list
        all_out.append([X_hat, A_hat, X_all, A_all, P_all, S_relab, S_all, S_sub, [len(np.unique(i.cpu())) for i in S_all]])
        
        
        
        #compute reconstruction losses for graph and attributes
        X_loss = X_recon_loss(X_hat, X)
        A_loss = A_recon_loss(A_hat, A)
        #compute community detection loss components
        #modularity loss (only computed over the last k layers of community model)
        Mod_loss, Modloss_values = modularity_loss_fn(A_all, P_all, layer_resolutions)
        #Compute clustering loss
        #Clust_loss, Clustloss_values = clustering_loss_fn(lamb, X_all, P_all, S)
        Clust_loss, Clustloss_values = clustering_loss_fn(lamb, X, P_all, S_relab)
        
        if(turn_off_A_loss == True):
            loss = gamma*X_loss+Clust_loss-delta*Mod_loss
        else:
            loss = A_loss+gamma*X_loss+Clust_loss-delta*Mod_loss
        

        #compute backward pass
        loss.backward()
        #update gradients
        optimizer.step()
        #update total loss function
        total_loss += loss.cpu().item()
        
        #store loss component information
        loss_history.append(total_loss)
        A_loss_hist.append(float(A_loss.cpu().detach().numpy()))
        X_loss_hist.append(gamma*float(X_loss.cpu().detach().numpy()))
        mod_loss_hist.append(delta*np.array(Modloss_values))
        clust_loss_hist.append(Clustloss_values)
        
        
        # if use_graph_updating:
        #     if (epoch+1) > burn_in:
        #         A = A_hat.detach()
            
        #--------------------------------------------------------------------
        #evaluating performance homogenity, completeness and NMI
        train_perf, output = evaluate(model, X, A, k, true_labels)
        #update history
        perf_hist.append(train_perf)
        X_pred, A_pred, S_pred = output
        pred_list.append(S_pred)
        #check for and apply validation 
        if validation_data:
            X_val, A_val, val_labels, val_size = validation_data
            valid_perf, output = evaluate(model, X_val, A_val, k, val_labels)
            valid_perf_hist.append(valid_perf)
        
        #evaluate epoch
        if (epoch+1) % update_interval == 0:
            
            #store update interval
            updates.append(epoch+1)

            #print performance 
            print('MODEL PEFORMANCE\n')
            print_performance(perf_hist, comm_layers, k)
            
            #print validation performance
            if validation_data:
                print('VALIDATION PERFORMANCE\n')
                print_performance(valid_perf_hist, comm_layers, k)
            
            print('MODEL LOSS\n')
            #loss printing
            print_losses(epoch, total_loss, mod_loss_hist, clust_loss_hist, 
                         X_loss_hist, A_loss_hist)
           
            
            
            #------------------------------
            #plotting training curves
            if ((epoch+1) >= 10):
                #loss plot
                print('plotting loss curve ...')
                plot_loss(epoch = epoch, 
                          loss_history = loss_history, 
                          recon_A_loss_hist = A_loss_hist, 
                          recon_X_loss_hist = X_loss_hist, 
                          mod_loss_hist = mod_loss_hist,
                          clust_loss_hist = clust_loss_hist,
                          path=output_path, 
                          save = save_output)
                if verbose == True:
                    #plotting graphs in networkx 
                    print('plotting nx graphs ...')
                    plot_nodes(A = (A-torch.eye(A.shape[0])).cpu().detach().numpy(), 
                               labels=S_relab[-k:][-1], 
                               path = output_path+'Top_Clusters_result_'+str(epoch+1),
                               node_size=ns, 
                               font_size=fs,
                               save = save_output,
                               add_labels = True)
                    if k == 2:
                        plot_nodes(A = (A-torch.eye(A.shape[0])).cpu().detach().numpy(), 
                                   labels=S_relab[-k:][0], 
                                   add_labels = True,
                                   node_size=ns,
                                   font_size=fs,
                                   save=save_output,
                                   path = output_path+'midde_Clusters_result_'+str(epoch+1))
                    
                
                print('plotting heatmaps ...')
                plot_clust_heatmaps(A = A, 
                                    A_pred = A_pred, 
                                    X = X,
                                    X_pred = X_pred,
                                    true_labels = true_labels, 
                                    pred_labels = S_relab[-k:], 
                                    layers = k+1, 
                                    epoch = epoch+1, 
                                    save_plot = save_output, 
                                    sp = output_path)
                
                
                #plot the performance history
                if len(perf_hist)>1:
                    print('plotting performance curves ...')
                    #performance plot
                    plot_perf(update_time = updates[-1], 
                              performance_hist = perf_hist, 
                              valid_hist = valid_perf_hist,
                              epoch = epoch, 
                              path= output_path, 
                              save = save_output)
                
                
                
                    
                
            
            print(".... Average epoch time = %.2f seconds ---" % (np.mean(time_hist)))
        time_hist.append(time.time() - start_epoch)
            
    #return 
    X_final, A_final, X_all_final, A_all_final, P_all_final, S_final, AW_final = model.forward(X, A)
    return (all_out, X_final, A_final, X_all_final, A_all_final, P_all_final, S_final, mod_loss_hist, loss_history, clust_loss_hist, A_loss_hist, X_loss_hist, perf_hist), pred_list