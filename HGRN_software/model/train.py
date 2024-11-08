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
import numpy as np
import copy
import gc

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

#an alternative batching approach to above
def get_batched_data(X, A, batch_size = 64):
    num_nodes = X.size(0)
    indices = torch.randperm(num_nodes)
    X_batches = []
    A_batches = []
    index_batches = []
    
    for start_idx in range(0, num_nodes, batch_size):
        end_idx = min(start_idx + batch_size, num_nodes)
        batch_indices = indices[start_idx:end_idx]
        X_batch = torch.index_select(X, 0, batch_indices)
        A_batch = torch.index_select(torch.index_select(A, 0, batch_indices), 1, batch_indices)
        X_batches.append(X_batch)
        A_batches.append(A_batch)
        index_batches.append(batch_indices)
    
    return X_batches, A_batches, index_batches


#for train, test splitting
def split_dataset(X, A, labels, split=[0.8, 0.2]):
    #an alternative batching approach to above
    num_nodes = X.size(0)
    train_size = int(np.round(0.8*num_nodes))
    
    indices = torch.randperm(num_nodes)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    sort_train, sort_test = np.argsort(train_indices), np.argsort(test_indices)
    
    train_X, test_X = torch.index_select(X, 0, train_indices[sort_train]), torch.index_select(X, 0, test_indices[sort_test])
    train_A = torch.index_select(torch.index_select(A, 0, train_indices[sort_train]), 1, train_indices[sort_train])
    test_A = torch.index_select(torch.index_select(A, 0, test_indices[sort_test]), 1, test_indices[sort_test])
    labels_train = [lab[train_indices[sort_train]] for lab in labels]
    labels_test = [lab[test_indices[sort_test]] for lab in labels]
    
    train_set = [train_X, train_A, labels_train]
    test_set = [test_X, test_A, labels_test]
    
    return train_set, test_set


#-----------------------------------------------------
#custom modularity loss function
class ModularityLoss(nn.Module):
    def __init__(self):
        super(ModularityLoss, self).__init__()
        
    def forward(self, all_A, all_P, resolutions = None):
        loss = torch.Tensor([0])
        loss_list = []
        for index, (A,P) in enumerate(zip(all_A, all_P)):
            if resolutions:
                mod = Modularity(A, P, resolutions[index])
            else:
                mod = Modularity(A, P, res= 1)
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
    def forward(self, Lamb, Attributes, Probabilities, method):
        
        """
        Computes forward loss for hierarchical within-cluster sum of squares loss
        Lamb: list of lenght l corresponding to the tuning loss for l hierarchical layers
        Attributes: Node feature matrix
        Probabilities: a list of length l corresponding the assignment probabilities for 
                        assigning nodes to communities in l hierarchical layers
        Cluster_labels: list of length l containing cluster assignment labels 
        """
        loss = torch.Tensor([0])
        loss_list = []
        if not isinstance(Attributes, list):
            N = Attributes.shape[0]
            ptensor_list = [torch.eye(N)]
            
        for idx, P in enumerate(Probabilities):
            #within cluster sum of squares
            if isinstance(Attributes, list):
                Attr = Attributes[idx]
            else:
                Attr = Attributes
            
            
            #handle for bottom up vs top down learning
            if method == 'bottom_up':
                ptensor_list+=[P]
            else:
                ptensor_list = P
                
            within_ss, centroids = WCSS(X = Attr,
                                        Plist = ptensor_list,
                                        method = method)
            
            if isinstance(Lamb, list):
                weight = Lamb[idx]
            else:
                weight = Lamb
            #update loss list
            loss_list.append(weight*float(within_ss.cpu().detach().numpy()))
            #update loss
            loss += weight*within_ss

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
    
    #set model to evaluation mode
    model.eval()
    X_pred, A_pred, X_list, A_list, P_list, S_pred, AW_pred = model.forward(X, A)
    perf_layers = []
    
    if model.method == 'bottom_up':
        S_trace_eval = trace_comms([i.cpu().clone() for i in S_pred], model.comm_sizes)
        S_all, S_temp, S_out = S_trace_eval
        S_relab = [i.detach().numpy() for i in S_temp]
    else:
        if any([True if max(i) > len(np.unique(i)) else False for i in S_pred]):
            gp = [torch.unique(i, sorted=True, return_inverse=True) for i in S_pred]
            
        S_relab = [i[1] for i in gp]
        
    if true_labels:
        perf_layers = get_layered_performance(k, S_relab, true_labels)
        
    return perf_layers, (X_pred, A_pred, X_list, A_list, P_list, S_pred, AW_pred), S_relab



def print_performance(history, comm_layers, k):
    lnm = ['top']+['middle_'+str(i) for i in np.arange(comm_layers-1)[::-1]]
    for i in range(0, k):
        print('-' * 36 + '{} layer'.format(lnm[i]) + '-' * 36)
        print('\nHomogeneity = {:.4f}, \nCompleteness = {:.4f}, \nNMI = {:.4f}, \nARI = {:.4f}'.format(
            history[-1][i][0], 
            history[-1][i][1], 
            history[-1][i][2],
            history[-1][i][3]))
        print('-' * 80)
        
        
        
def print_losses(epoch, loss_history):
    #------------------------------
    print('\nEpoch {} \nTotal Loss = {:.4f}'.format(
        epoch+1, loss_history[-1]['Total Loss']
        ))
    
    print('\nModularity = {}, \nClustering = {}, \nX Recontrstuction = {:.4f}, \nA Recontructions = {:.4f} \n'.format(
        np.round(loss_history[-1]['Modularity'],4),
        np.round(loss_history[-1]['Clustering'],4),
        loss_history[-1]['X Reconstruction'], 
        loss_history[-1]['A Reconstruction']))
    
    
    
    
def get_mod_clust_losses(model, Xbatch, Abatch, output, lamb, resolution, modlossfn, clustlossfn):
    
    X_hat, A_hat, X_all, A_all, P_all, S_all, AW = output
    
    if model.method == 'bottom_up':
        S_sub, S_relab, S_all = trace_comms([i.cpu().clone() for i in S_all], model.comm_sizes)
        #compute community detection loss components
        #modularity loss (only computed over the last k layers of community model)
        Mod_loss, Modloss_values = modlossfn([Abatch]+A_all[1], P_all, resolution)
        #Compute clustering loss
        #Clust_loss, Clustloss_values = clustering_loss_fn(lamb, X_all, P_all, S)
        Clust_loss, Clustloss_values = clustlossfn(lamb, Xbatch, P_all, model.method)
    elif model.method == "top_down":
        S_sub, S_relab = [], []
        #Modularity
        top_mod_loss, values_top = modlossfn([A_all[0]], [P_all[0]], resolution)
        middle_mod_loss, values_mid = modlossfn(A_all[-1], P_all[1], resolution)
        Mod_loss = top_mod_loss+middle_mod_loss
        Modloss_values = values_top+[torch.mean(torch.tensor(values_mid)).detach().tolist()]
        #Compute clustering loss
        #Clust_loss, Clustloss_values = clustering_loss_fn(lamb, X_all, P_all, S)
        Clust_loss_top, Clustloss_values_top = clustlossfn(lamb[0], Xbatch, [P_all[0]], model.method)
        Clust_loss_mid, Clustloss_values_mid = clustlossfn(lamb[1], X_all[-1], P_all[1], model.method)
        Clust_loss = Clust_loss_top+Clust_loss_mid
        Clustloss_values = Clustloss_values_top+[torch.mean(torch.tensor(Clustloss_values_mid)).detach().tolist()]
        
    return Mod_loss, Modloss_values, Clust_loss, Clustloss_values, S_sub, S_relab
    
   


#------------------------------------------------------
#this function fits the HRGNgene model to data
def fit(model, X, A, optimizer='Adam', epochs = 100, update_interval=10, lr = 1e-4, 
        gamma = 1, delta = 1, lamb = 1, layer_resolutions = [1,1], k = 2, 
        use_batch_learning = True, batch_size = 64, true_labels = None, turn_off_A_loss = False, 
        validation_data = None, test_data = None, save_output = False, output_path = 'path/to/output', 
        fs = 10, ns = 10, verbose = True, **kwargs):
    
    """
    
    """
    
    #preallocate storage
    train_loss_history=[]
    perf_hist = []
    valid_perf_hist = []
    updates = []
    time_hist = []
    comm_layers = len(model.comm_sizes)
    print(model)
    pred_list = []
    all_out = []
    test_loss_history = []
    test_loss = 0
    max_epochs = epochs
    
    
    #set optimizer Adam
    optimizer = optimizers.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4
    )
    
    #initialize loss functions
    #A_recon_loss = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
    A_recon_loss = torch.nn.BCELoss(reduction = 'mean')
    #A_recon_loss = torch.nn.NLLLoss()
    X_recon_loss = torch.nn.MSELoss(reduction = 'mean')

    #initiate custom loss functions
    modularity_loss_fn = ModularityLoss()
    clustering_loss_fn = ClusterLoss()
    
    #get batches
    if use_batch_learning:
        X_batches, A_batches, index_batches = get_batched_data(X, A, batch_size = batch_size)
    else:
        X_batches, A_batches = [X], [A]
    
    #------------------begin training epochs----------------------------
    for idx, epoch in enumerate(range(epochs)):
        #epoch printing
        epoch_start = time.time()
        if idx == 0:
            print('Epoch {} starts !'.format(epoch))
            print('=' * 55)
            print('-' * 55)
            print('=' * 55+'\n')
        
        total_loss = 0
        
        
        batch_iterable = tqdm(zip(X_batches, A_batches), ascii=False, ncols=75)
        for index, (Xbatch, Abatch) in enumerate(batch_iterable):
            
            #zero out gradient
            optimizer.zero_grad()

            #compute forward output 
            forward_output = model.forward(Xbatch, Abatch)
            
            
            get_output = get_mod_clust_losses(model, 
                                              Xbatch, 
                                              Abatch, 
                                              forward_output, 
                                              lamb, 
                                              layer_resolutions, 
                                              modularity_loss_fn, 
                                              clustering_loss_fn)
            
            Mod_loss, Modloss_values, Clust_loss, Clustloss_values, S_sub, S_relab = get_output
            X_hat, A_hat, X_all, A_all, P_all, S_all, AW = forward_output
            #update output list
            all_out.append([X_hat, A_hat, X_all, A_all, P_all, S_relab, S_all, S_sub, [len(np.unique(i.cpu())) for i in S_all]])
            
            #compute reconstruction losses for graph and attributes
            X_loss = X_recon_loss(X_hat, Xbatch)
            A_loss = A_recon_loss(A_hat, Abatch)
            #compute the total loss function
            loss = A_loss+gamma*X_loss+Clust_loss-delta*Mod_loss
            
            #compute backward pass
            loss.backward()
            #update gradients
            optimizer.step()
            #update total loss function
            total_loss += loss.cpu().item()
            
            # #--------------------------------------------------------------------
            #evaluationg test performance
            if test_data:
                eval_X, eval_A, eval_labels = test_data
                test_perf, test_output, S_replab_test = evaluate(model, eval_X, eval_A, k, eval_labels)
                X_hat_test, A_hat_test = test_output[0], test_output[1]
                
                
                get_test_output = get_mod_clust_losses(model, 
                                                       eval_X, 
                                                       eval_A, 
                                                       test_output, 
                                                       lamb, 
                                                       layer_resolutions, 
                                                       modularity_loss_fn, 
                                                       clustering_loss_fn)
                
                Mod_loss_test, Modloss_values_test, Clust_loss_test, Clustloss_values_test = get_test_output[:4]
                #compute loss
                X_loss_test = gamma*(X_recon_loss(X_hat_test, eval_X)).cpu().detach().numpy()
                A_loss_test = (A_recon_loss(A_hat_test, eval_A)).cpu().detach().numpy()
                mod_weighted = delta*Mod_loss_test
                
                test_loss = (A_loss_test+X_loss_test+Clust_loss_test-mod_weighted).cpu().item()
                print_loss_test = f'{test_loss:.2f}'
            else:
                print_loss_test = 'No test set provided'
                test_loss = 0
                Modloss_values_test = [0,0]
                Clustloss_values_test = [0,0]
                X_loss_test = 0
                A_loss_test = 0
                
            batch_iterable.set_description(f'Epoch {idx} Processing batch {"-"*15} batch loss: {total_loss:.2f} test loss: {print_loss_test}')
           
        train_loss = total_loss
        epoch_end = time.time()
        #store loss component information
        train_loss_history.append({'Total Loss': total_loss,
                                   'A Reconstruction': A_loss.cpu().detach().numpy(),
                                   'X Reconstruction': gamma*float(X_loss.cpu().detach().numpy()),
                                   'Modularity': delta*np.array(Modloss_values),
                                   'Clustering': Clustloss_values})
        
        test_loss_history.append({'Total Loss': test_loss,
                                   'A Reconstruction': A_loss_test,
                                   'X Reconstruction': X_loss_test,
                                   'Modularity': Modloss_values_test,
                                   'Clustering': Clustloss_values_test})

            
        print(f'Epoch Time: {epoch_end - epoch_start}')
                
        train_perf, eval_output, S_eval= evaluate(model, X, A, k, true_labels)
        X_eval, A_eval = eval_output[0], eval_output[1]
        #update history
        perf_hist.append(train_perf)
        pred_list.append(S_eval)
                     
        
        #check for and apply validation 
        if validation_data:
            X_val, A_val, val_labels = validation_data
            valid_perf, output, Sval = evaluate(model, X_val, A_val, k, val_labels)
            valid_perf_hist.append(valid_perf)
        
        #evaluate epoch
        if (epoch+1) % update_interval == 0:
            
            #store update interval
            updates.append(epoch+1)

            #print performance 
            if true_labels:
                print('\nMODEL PEFORMANCE\n')
                print_performance(perf_hist, comm_layers, k)
            
            #print validation performance
            if validation_data:
                print('VALIDATION PERFORMANCE\n')
                print_performance(valid_perf_hist, comm_layers, k)
            
            print('MODEL LOSS\n')
            #loss printing
            print_losses(epoch, train_loss_history)
           
            
            
            #------------------------------
            #plotting training curves
            if ((epoch+1) >= 10):
                #loss plot
                print('plotting loss curve ...')
                plot_loss(epoch = epoch, 
                          layers = comm_layers,
                          train_loss_history = train_loss_history,
                          test_loss_history = test_loss_history,
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
                                    A_pred = A_eval, 
                                    X = X,
                                    X_pred = X_eval,
                                    true_labels = true_labels, 
                                    pred_labels = S_eval, 
                                    layers = k+1, 
                                    epoch = epoch+1, 
                                    save_plot = save_output, 
                                    sp = output_path)
                
                
                #plot the performance history
                if len(perf_hist)>1 and true_labels is not None:
                    print('plotting performance curves ...')
                    #performance plot
                    plot_perf(update_time = updates[-1], 
                              performance_hist = perf_hist, 
                              valid_hist = valid_perf_hist,
                              epoch = epoch, 
                              path= output_path, 
                              save = save_output)
                
                
                
                    
                
            
            print(".... Average epoch time = %.2f seconds ---" % (np.mean(time_hist)))
        time_hist.append(time.time() - epoch_start)
        if verbose:
            print(f'Total Epoch Time: {(epoch_end - epoch_start):.2f}')
            
        if test_loss > total_loss:
            print(f"Stopping early at epoch {epoch+1} due to high test loss: {test_loss:.4f}")
            break
    
    #return 
    X_final, A_final, X_all_final, A_all_final, P_all_final, S_final, AW_final = model.forward(X, A)
    
    return (all_out, X_final, A_final, X_all_final, A_all_final, P_all_final, S_final, train_loss_history, test_loss_history, perf_hist), pred_list