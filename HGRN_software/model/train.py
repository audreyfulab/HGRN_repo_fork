# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 00:22:43 2023

@author: Bruin
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import torch.optim as optimizers 
from model.utilities import Modularity, WCSS
from tqdm import tqdm
from model.utilities import trace_comms, get_layered_performance
from model.utilities import plot_loss, plot_perf, plot_nodes, plot_clust_heatmaps
import os
from typing import Optional, Union, List,  Literal

# model early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = float('inf')
        self.delta = delta
        
        if not path:
            self.path = os.getcwd()
        else:
            self.path = path
        

    def __call__(self, loss, model, _type = ['test', 'total']):
        score = loss
        self._type = _type
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        
        if self.verbose:
            print(f'\n {self._type} loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ... \n')
        torch.save(model.cpu(), os.path.join(self.path, 'checkpoint.pth'))
        self.loss_min = loss


#model output class
class HCD_output():
    
    """
    A class to store the results and outputs of the HRGNgene model after training.

    This class captures the model outputs, loss histories, performance metrics, 
    clustering results, and adjacency structures for further analysis.

    Attributes:
    -----------
    model_output_history : list
        List containing all model outputs over training epochs.
    attention_weights : torch.Tensor
        Final attention weights computed by the model.
    reconstructed_features : torch.Tensor
        The reconstructed node features after training.
    reconstructed_adj : torch.Tensor
        The reconstructed adjacency matrix after training.
    train_loss_history : list
        Training loss history over epochs.
    test_loss_history : list
        Test loss history over epochs (if test data is provided).
    performance_history : list
        Performance metrics tracked during training.
    latent_features : torch.Tensor
        Extracted latent feature representations of the input data.
    partitioned_data : torch.Tensor
        Data partitioned into hierarchical clusters.
    partitioned_latent_features : torch.Tensor
        Latent feature representations of partitioned clusters.
    training_data : dict
        Dictionary containing:
        - `'X_train'`: Training feature matrix.
        - `'A_train'`: Training adjacency matrix.
        - `'labels_train'`: Training labels (if available).
    test_data : dict
        Dictionary containing:
        - `'X_test'`: Test feature matrix.
        - `'A_test'`: Test adjacency matrix.
        - `'labels_test'`: Test labels (if available).
    probabilities : dict
        Cluster membership probabilities at different levels:
        - `'top'`: Probabilities at the top hierarchical level.
        - `'middle'`: Probabilities at the middle hierarchical level.
    pred_history : list
        History of predicted cluster assignments over epochs.
    adjacency : dict
        Graph adjacency structures at different clustering levels:
        - `'community_graphs'`: Middle layer within community graph for each top-layer community.
        - `'partitioned_graphs'`: The partitions of the input graph for each top-layer community.
    predicted_train : dict
        Predicted hierarchical clustering assignments:
        - `'top'`: Predictions at the top hierarchical level.
        - `'middle'`: Predictions at the middle hierarchical level.
    best_loss_index : int or None
        Index corresponding to the epoch with the lowest recorded loss.
    hierarchical_clustering_preds : dict
        Cluster assignments using hierarchical clustering.
    louvain_preds : dict
        Cluster assignments using the Louvain community detection algorithm.
    kmeans_preds : ldict
        Cluster assignments using the K-means clustering algorithm.
    table : pandas.DataFrame
        A structured summary table of summary results and statistics
    perf_table : pandas.DataFrame
        A structured table containing performance evaluation metrics.

    Methods:
    --------
    to_dict():
        Converts the object into a dictionary format for easier access and storage.
    
    show_results():
        Displays the performance table (`perf_table`) if available.
    """
    
    def __init__(self, X, A, test_set, labels, all_output, model_output, test_history, train_history, perf_history, pred_history, batch_indices):
        
        X_final, A_final, X_all_final, A_all_final, P_all_final, S_final, AW_final = model_output
        
        eval_X, eval_A, eval_labels = test_set
        
        self.model_output_history = all_output
        self.attention_weights = AW_final
        self.reconstructed_features = X_final.cpu()
        self.reconstructed_adj = A_final.cpu()
        self.train_loss_history = train_history
        self.test_loss_history = test_history
        self.performance_history = perf_history
        self.latent_features = X_all_final[0].cpu()
        self.partitioned_data = [i.cpu() for i in X_all_final[1]]
        self.partitioned_latent_features = [i.cpu() for i in X_all_final[2]]
        self.training_data = {'X_train': X.cpu(), 'A_train': A.cpu(), 'labels_train': labels}
        self.test_data = {'X_test': eval_X.cpu(), 'A_test': eval_A.cpu(), 'labels_test': eval_labels}
        self.probabiltiies = {'top': P_all_final[0].cpu(), 'middle': [i.cpu() for i in P_all_final[1]]}
        self.pred_history = pred_history
        self.adjacency = {'community_graphs': [i.cpu() for i in A_all_final[1]],'partitioned_graphs': [i.cpu() for i in A_all_final[2]]}
        self.predicted_train = {'top': S_final[0].cpu(), 'middle': S_final[1].cpu()}
        self.best_loss_index = None
        self.hierarchical_clustering_preds = None
        self.louvain_preds = None
        self.kmeans_preds = None
        self.table = None
        self.perf_table = None
        self.batch_indices = [i.cpu() for i in batch_indices]
        
    def to_dict(self):
        
        """
        Converts the `HCD_output` object into a dictionary format for easier access, storage, and analysis.

        This method creates a structured dictionary containing all key attributes of the `HCD_output` class, 
        making it easier to store or further process the results.

        Returns:
        --------
        dict
        """
        
        return {'model_output_history': self.model_output_history,
                'attention_weights': self.attention_weights,
                'reconstructed_features': self.reconstructed_features,
                'reconstructed_adj': self.reconstructed_adj,
                'train_loss_history': self.train_loss_history,
                'test_loss_history': self.test_loss_history,
                'performance_history': self.performance_history,
                'latent_features': self.latent_features,
                'partitioned_data': self.partitioned_data,
                'paritioned_latent_features': self.partitioned_latent_features,
                'test_data': self.test_data,
                'training_date': self.training_data,
                'probabilities': self.probabiltiies,
                'adjacency': self.adjacency,
                'predicted_train': self.predicted_train,
                'best_loss_index': self.best_loss_index,
                'hierarchical_clustering_preds': self.hierarchical_clustering_preds,
                'louvain_preds': self.louvain_preds,
                'kmeans_preds': self.kmeans_preds,
                'stat_table': self.table,
                'perf_table': self.perf_table,
                'batch_indices': self.batch_indices}
        
    def show_results(self):
        """
        Displays the performance table (`perf_table`) if available.

        This method prints the `perf_table` attribute, which contains the model’s performance evaluation metrics. 
        If `perf_table` is not available (None), the method does nothing.
        """
        print(self.perf_table)


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
def get_batched_data(X, A, batch_size=64, min_batch_size=11):
    num_nodes = X.size(0)
    device = X.device
    indices = torch.randperm(num_nodes, device=device)
    
    # Calculate all indices upfront
    start_indices = range(0, num_nodes, batch_size)
    end_indices = range(batch_size, num_nodes + batch_size, batch_size)
    end_indices = [min(e, num_nodes) for e in end_indices]
    
    # Process all batches in one go
    X_batches = []
    A_batches = []
    index_batches = []
    
    for start, end in zip(start_indices, end_indices):
        batch_indices = indices[start:end]
        X_batch = X[batch_indices]
        A_batch = A[batch_indices][:, batch_indices]
        X_batches.append(X_batch)
        A_batches.append(A_batch)
        index_batches.append(batch_indices)
    
    return X_batches, A_batches, index_batches



# def get_batched_data(X, A, batch_size=64, min_samples_per_batch=11):
#     num_nodes = X.size(0)
#     device = X.device
#     indices = torch.randperm(num_nodes, device=device)
#     X_batches = []
#     A_batches = []
#     index_batches = []

#     for start_idx in range(0, num_nodes, batch_size):
#         end_idx = min(start_idx + batch_size, num_nodes)
#         batch_indices = indices[start_idx:end_idx]
#         X_batch = torch.index_select(X, 0, batch_indices)
#         A_batch = torch.index_select(torch.index_select(A, 0, batch_indices), 1, batch_indices)
#         X_batches.append(X_batch)
#         A_batches.append(A_batch)
#         index_batches.append(batch_indices)

#     # Handle the case where the last batch has fewer than min_samples_per_batch
#     if len(index_batches) > 1 and len(index_batches[-1]) < min_samples_per_batch:
#         # Take some samples from the second last batch
#         deficit = min_samples_per_batch - len(index_batches[-1])
#         index_batches[-2] = torch.cat((index_batches[-2], index_batches[-1][:deficit]))
#         X_batches[-2] = torch.index_select(X, 0, index_batches[-2])
#         A_batches[-2] = torch.index_select(torch.index_select(A, 0, index_batches[-2]), 1, index_batches[-2])

#         # Update the last batch
#         index_batches[-1] = index_batches[-1][deficit:]
#         X_batches[-1] = torch.index_select(X, 0, index_batches[-1])
#         A_batches[-1] = torch.index_select(torch.index_select(A, 0, index_batches[-1]), 1, index_batches[-1])
        
#         # Remove the last batch if it's empty
#         if len(index_batches[-1]) == 0:
#             del X_batches[-1]
#             del A_batches[-1]
#             del index_batches[-1]

#     return X_batches, A_batches, index_batches




#for train, test splitting
def split_dataset(X: torch.Tensor, A: torch.Tensor, labels: List[torch.Tensor], split: List[int] =[0.8, 0.2]):
    """
        Splits the dataset into training and testing sets based on the given split ratio.

        Parameters:
        -----------
        X : torch.Tensor
            Feature matrix of shape (num_nodes, num_features).
        A : torch.Tensor
            Adjacency matrix of shape (num_nodes, num_nodes), representing graph connectivity.
        labels : list of torch.Tensor
            List of label tensors corresponding to each layer, with shape (num_nodes,).
        split : list of float, optional (default = [0.8, 0.2])
            The proportion of data to be split into training and testing sets. The first value represents the training fraction,
            and the second value represents the testing fraction.

        Returns:
        --------
        train_set : list
            A list containing:
            - train_X (torch.Tensor): Feature matrix for training nodes.
            - train_A (torch.Tensor): Adjacency matrix for training nodes.
            - labels_train (list of torch.Tensor): Labels for training nodes.

        test_set : list
            A list containing:
            - test_X (torch.Tensor): Feature matrix for test nodes.
            - test_A (torch.Tensor): Adjacency matrix for test nodes.
            - labels_test (list of torch.Tensor): Labels for test nodes.

        Notes:
        ------
        - This function randomly shuffles the node indices before splitting.
        - Sorting the indices ensures consistent selection of training and testing sets.

        Example:
        --------
        >>> X = torch.randn(100, 16)  # 100 nodes, 16 features
        >>> A = torch.randint(0, 2, (100, 100))  # Random adjacency matrix
        >>> labels = [torch.randint(0, 2, (100,))]  # Binary labels
        >>> train_set, test_set = split_dataset(X, A, labels, layers=1)
    """
    #an alternative batching approach to above
    num_nodes = X.size(0)
    train_size = int(np.round(split[0]*num_nodes))
    
    indices = torch.randperm(num_nodes)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    sort_train, sort_test = torch.argsort(train_indices), torch.argsort(test_indices)
    
    train_X, test_X = torch.index_select(X, 0, train_indices[sort_train]), torch.index_select(X, 0, test_indices[sort_test])
    train_A = torch.index_select(torch.index_select(A, 0, train_indices[sort_train]), 1, train_indices[sort_train])
    test_A = torch.index_select(torch.index_select(A, 0, test_indices[sort_test]), 1, test_indices[sort_test])
    
    labels_train = [lab[train_indices.cpu()[sort_train.cpu()]] for lab in labels]
    labels_test = [lab[test_indices.cpu()[sort_test.cpu()]] for lab in labels]
    
    train_set = [train_X, train_A, labels_train]
    test_set = [test_X, test_A, labels_test]
    
    return train_set, test_set


#-----------------------------------------------------
#custom modularity loss function
class ModularityLoss(nn.Module):
    """
        Custom Hierarchical Modularity Loss Function.

        This loss function computes modularity-based clustering loss for multiple adjacency matrices
        and assignment probability matrices. It helps optimize cluster assignments by maximizing 
        modularity scores across hierarchical layers.

        Methods:
        --------
        forward(all_A, all_P, resolutions=None):
            Computes the modularity loss for given adjacency matrices and assignment probabilities.
    """
    def __init__(self):
        """
            Initializes the ModularityLoss module.
            
            This class extends `torch.nn.Module` and implements a loss function based on modularity,
            which is commonly used in graph-based clustering tasks.
        """
        super(ModularityLoss, self).__init__()
        
    def forward(self, all_A, all_P, resolutions = None):
        """
            Computes the modularity loss for hierarchical clustering.

            This function calculates the modularity score for each hierarchical level 
            based on the provided adjacency matrices (`all_A`) and assignment probabilities (`all_P`).
            The modularity score measures the quality of clustering, aiming to optimize
            community detection in graphs.

            Parameters:
            -----------
            all_A : list of torch.Tensor
                A list of adjacency matrices, where each tensor `A` has shape `(N, N)`, 
                representing graph connectivity at a given hierarchical level.
            all_P : list of torch.Tensor
                A list of assignment probability matrices, where each tensor `P` has shape `(N, k)`,
                representing the probability of `N` nodes belonging to `k` clusters at each level.
            resolutions : list of float, optional (default = None)
                A list of resolution parameters for modularity computation, controlling the scale 
                at which communities are detected. If `None`, a default resolution of `1` is used.

            Returns:
            --------
            loss : torch.Tensor
                The total modularity loss across all hierarchical levels.
            loss_list : list of float
                A list containing individual modularity loss values for each level.

            Notes:
            ------
            - The `Modularity` function (assumed to be defined elsewhere) computes the modularity 
            score given an adjacency matrix `A` and an assignment probability matrix `P`.
            - Higher modularity scores indicate better clustering, but in the context of loss optimization, 
            this function minimizes modularity-based deviation.

            Example:
            --------
            >>> N, k, l = 100, 4, 3
            >>> all_A = [torch.randint(0, 2, (N, N)) for _ in range(l)]
            >>> all_P = [torch.rand(N, k) for _ in range(l)]
            >>> resolutions = [1.0, 0.8, 0.5]
            >>> loss_fn = ModularityLoss()
            >>> loss, loss_list = loss_fn(all_A, all_P, resolutions)
            >>> print(loss.item(), loss_list)
        """
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
    """
        Hierarchical Clustering Loss Module.

        This loss function calculates the hierarchical Within-Cluster Sum of Squares (WCSS) loss
        to measure the quality of hierarchical clustering. It supports both bottom-up and 
        top-down clustering approaches.

        Methods:
        --------
        forward(Lamb, Attributes, Probabilities, method):
            Computes the clustering loss based on node feature assignments across hierarchical levels.
    """
    
    def __init__(self):
        """
            Initializes the ClusterLoss module.
            
            This class extends `torch.nn.Module` and implements a clustering loss function for
            hierarchical clustering methods.
        """
        super(ClusterLoss, self).__init__()

    
    # forward method for loss computed using input feature matrix
    def forward(self, Lamb, Attributes, Probabilities, method):
        
        """
            Computes the hierarchical clustering loss.

            This function calculates the Within-Cluster Sum of Squares (WCSS) loss across multiple 
            hierarchical layers based on node feature matrices and cluster assignment probabilities.

            Parameters:
            -----------
            Lamb : list or float
                A list of length `l`, where each element corresponds to a weight controlling 
                the contribution of each hierarchical layer to the total loss. If a single value 
                is provided, it is used for all layers.
            Attributes : torch.Tensor or list of torch.Tensor
                The node feature matrix of shape `(N, d)`, where `N` is the number of nodes 
                and `d` is the feature dimension. If hierarchical attributes are provided, 
                it should be a list of `l` feature matrices.
            Probabilities : list of torch.Tensor
                A list of length `l`, where each tensor of shape `(N, k)` represents the 
                assignment probabilities of `N` nodes to `k` communities at a given hierarchical layer.
            method : str
                Clustering method to be used:
                - `'bottom_up'`: Aggregates assignment probabilities in a bottom-up manner.
                - `'top_down'`: Uses independent cluster assignments at each level.

            Returns:
            --------
            loss : torch.Tensor
                The computed hierarchical clustering loss.
            loss_list : list of float
                A list containing individual WCSS loss values for each hierarchical layer.

            Notes:
            ------
            - The function supports both single-level and multi-level hierarchical clustering.
            - The `WCSS` function (assumed to be defined elsewhere) is used to compute the 
            within-cluster sum of squares loss.

            Example:
            --------
            >>> N, d, k, l = 100, 16, 4, 3
            >>> Attributes = torch.randn(N, d)
            >>> Probabilities = [torch.rand(N, k) for _ in range(l)]
            >>> Lamb = [0.5, 0.3, 0.2]
            >>> loss_fn = ClusterLoss()
            >>> loss, loss_list = loss_fn(Lamb, Attributes, Probabilities, method='bottom_up')
            >>> print(loss.item(), loss_list)
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
    #torch no grad


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







def evaluate(model, X, A, k, true_labels ,run_eval = True):
    
    #set model to evaluation mode
    if run_eval == False:
        return None, (None, None, None, None, None, None, None), None
    with torch.no_grad():
        model.eval()
        X_pred, A_pred, X_list, A_list, P_list, S_pred, AW_pred = model.forward(X, A)
    perf_layers = []
    
    if model.method == 'bottom_up':
        S_trace_eval = trace_comms([i.cpu().clone() for i in S_pred], model.comm_sizes)
        S_all, S_temp, S_out = S_trace_eval
        S_relab = [i.cpu().detach().numpy() for i in S_temp][::-1]
    else:
        #if any([True if max(i) > len(np.unique(i)) else False for i in S_pred]):
        gp = [torch.unique(i, sorted=True, return_inverse=True) for i in S_pred]
        
        S_relab = [i[1] for i in gp]
        
    if true_labels:
        perf_layers = get_layered_performance(k, S_relab, true_labels)
        
    return perf_layers, (X_pred, A_pred, X_list, A_list, P_list, S_pred, AW_pred), S_relab



def print_performance(history, comm_layers, k):
    """Print performance metrics with safe handling of missing data"""
    if not history or all(h is None for h in history):
        print("No performance history available")
        return

    # Filter out None entries and get last valid performance
    valid_history = [h for h in history if h is not None]
    if not valid_history:
        print("No valid performance data available")
        return

    last_perf = valid_history[-1]
    lnm = ['top'] + ['middle_'+str(i) for i in np.arange(comm_layers-1)[::-1]]
    
    for i in range(min(k, len(last_perf))):  # Ensure we don't exceed available layers
        # Skip if layer data is missing
        if i >= len(last_perf) or last_perf[i] is None:
            print(f"----- No data available for {lnm[i]} layer -----")
            continue
            
        print('-' * 36 + f' {lnm[i]} layer ' + '-' * 36)
        
        # Check if all metrics exist
        metrics = last_perf[i]
        if len(metrics) >= 4:  # Ensure we have all 4 metrics
            print(f'\nHomogeneity = {metrics[0]:.4f},'
                  f'\nCompleteness = {metrics[1]:.4f},'
                  f'\nNMI = {metrics[2]:.4f},'
                  f'\nARI = {metrics[3]:.4f}')
        else:
            print("\nIncomplete metrics available:")
            for j, name in enumerate(['Homogeneity', 'Completeness', 'NMI', 'ARI']):
                if j < len(metrics):
                    print(f'{name} = {metrics[j]:.4f}', end=', ')
            print()
            
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
        S_sub, S_relab, S = trace_comms([i.cpu().clone() for i in S_all], model.comm_sizes)
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
        #Clust_loss_mid, Clustloss_values_mid = clustlossfn(lamb[1], torch.concat(X_all[-1]), [torch.concat(P_all[1])], model.method)
        Clust_loss_mid, Clustloss_values_mid = clustlossfn(lamb[1], X_all[-1], P_all[1], model.method)
        Clust_loss = Clust_loss_top+Clust_loss_mid
        Clustloss_values = Clustloss_values_top+[torch.sum(torch.tensor(Clustloss_values_mid)).detach().tolist()]
        
    return Mod_loss, Modloss_values, Clust_loss, Clustloss_values, S_sub, S_relab
    
   


#------------------------------------------------------
#this function fits the HRGNgene model to data
def fit(model, X, A, optimizer='Adam', epochs = 100, update_interval=10, lr = 1e-4, 
        gamma = 1, delta = 1, lamb = 1, layer_resolutions = [1,1], k = 2, use_batch_learning = True, 
        batch_size = 64, early_stopping = False, patience = 5, true_labels = None, validation_data = None, 
        test_data = None, save_output = False, output_path = '', fs = 10, ns = 10, verbose = True, 
        device = 'cpu', **kwargs):
    
    """
    Trains the HRGNgene model on the given dataset.

    This function optimizes the HRGNgene model using modularity-based and clustering-based loss terms 
    while performing hierarchical clustering on gene regulatory networks (GRNs). It supports batch learning, 
    early stopping, and modularity-based clustering optimization.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to be trained.
    X : array-like (torch.Tensor)
        Feature matrix of shape (N, F), where N is the number of samples and F is the number of features.
    A : array-like (torch.Tensor)
        Adjacency matrix of the input graph, representing connections between samples.
    optimizer : str, optional (default='Adam')
        The optimization algorithm to use for training (e.g., 'Adam', 'SGD').
    epochs : int, optional (default=100)
        Number of training epochs.
    update_interval : int, optional (default=10)
        Frequency (in epochs) at which performance metrics are updated and logged.
    lr : float, optional (default=1e-4)
        Learning rate for the optimizer.
    gamma : float, optional (default=1)
        Weighting factor for the attribute reconstruction loss term.
    delta : float, optional (default=1)
        Weighting factor for the modularity loss term.
    lamb : float, optional (default=1)
        Weighting factor for the clustering loss term.
    layer_resolutions : list of float, optional (default=[1,1])
        Resolution parameters for modularity calculation at different hierarchical layers.
    k : int, optional (default=2)
        Number of hierarchical clustering levels.
    use_batch_learning : bool, optional (default=True)
        Whether to use mini-batch training or full-batch training.
    batch_size : int, optional (default=64)
        Size of each batch if `use_batch_learning` is enabled.
    early_stopping : bool, optional (default=False)
        Whether to stop training early if no improvement is observed.
    patience : int, optional (default=5)
        Number of epochs to wait for improvement before stopping (if early stopping is enabled).
    true_labels : array-like, optional (default=None)
        Ground truth labels used for evaluation.
    turn_off_A_loss : bool, optional (default=False)
        Whether to disable the loss term related to adjacency matrix reconstruction.
    validation_data : tuple, optional (default=None)
        Validation dataset provided as (X_val, A_val, val_labels).
    test_data : tuple, optional (default=None)
        Test dataset provided as (X_test, A_test, test_labels).
    save_output : bool, optional (default=False)
        Whether to save the model's outputs and training history.
    output_path : str, optional (default='')
        Directory path for saving model output, if enabled.
    fs : int, optional (default=10)
        Font size for plots and visualizations.
    ns : int, optional (default=10)
        Node size for graph visualizations.
    verbose : bool, optional (default=True)
        Whether to print detailed logs and progress updates during training.
    **kwargs : dict
        Additional keyword arguments for customizing training behavior.

    Returns:
    --------
    output : HCD_output
        An instance of `HCD_output`, which contains:
        - `all_model_output`: List of all model outputs over training.
        - `attention_weights`: Attention weights from the final model.
        - `train_loss_history`: History of training losses.
        - `test_loss_history`: History of test losses.
        - `performance_history`: Performance metrics over epochs.
        - `latent_features`: Extracted latent feature representations.
        - `partitioned_data`: Data after partitioning into hierarchical clusters.
        - `partitioned_latent_features`: Partitioned latent features at different levels.
        - `training_data`: Dictionary containing training feature matrix and adjacency matrix.
        - `test_data`: Dictionary containing test feature matrix and adjacency matrix.
        - `probabilities`: Cluster membership probabilities at different hierarchy levels.
        - `pred_history`: Predicted cluster assignments over epochs.
        - `adjacency`: Graph adjacency structures at different clustering levels.



    Notes:
    ------
    - Supports early stopping based on total loss or test loss.
    - Generates plots for loss curves, network clustering, and performance (if labels are provided)
    - Uses the `evaluate` function to compute performance metrics at different stages.
    - Batch learning is enabled by default but can be disabled for full-batch training.

    Example:
    --------
    >>> from model.model import HCD
    >>> from model.train import fit
    >>> model = HCD()  # Initialize your model
    >>> X = torch.rand(100, 20)  # Example feature matrix
    >>> A = torch.randint(0, 2, (100, 100))  # Example adjacency matrix
    >>> fit(model, X, A, epochs=50, lr=1e-3, batch_size=32, early_stopping=True, patience=5)

    """
    total_training_start = time.time()
    prelim_time = time.time()
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
    update_interval = 10
    batch_time_hist = []
    run_eval_true_hist =[]
    plot_time_hist = []
        
    if early_stopping:
        early_stop = EarlyStopping(patience=patience, verbose=True,  path = output_path)
    
    
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
    prelim_end = time.time()

    print(f"...fit set up time: {prelim_end-prelim_time}")
    prelim_setup_time = str(prelim_end-prelim_time)
    #get batches
    batch_start = time.time()
    if use_batch_learning:
        if batch_size > X.shape[0]:
            raise ValueError(f'ERROR! Batch size is larger than number of items to split features.shape[0] = {X.shape[0]}')
        X_batches, A_batches, index_batches = get_batched_data(X, A, batch_size = batch_size)
    else:
        X_batches, A_batches, index_batches = [X], [A], None
    batch_end = time.time()
    print(f'...get batched data time: {batch_end - batch_start}')
    fetching_batch_data = str(batch_end - batch_start)
    last_valid_A_eval = None
    last_valid_X_eval = None
    last_valid_S_eval = None
    #------------------begin training epochs----------------------------
    for idx, epoch in enumerate(range(epochs)):
        #allocate storage for train and test total epoch losses (sum of all batches)
        train_epoch_loss_A, train_epoch_loss_X, train_epoch_loss_clust, train_epoch_loss_mod = 0,0,[0,0],[0,0]
        test_epoch_loss_A, test_epoch_loss_X, test_epoch_loss_clust, test_epoch_loss_mod = 0,0,[0,0],[0,0]
    
        #epoch printing
        epoch_start = time.time()
        if idx == 0:
            print('Epoch {} starts !'.format(epoch))
            print('=' * 55)
            print('-' * 55)
            print('=' * 55+'\n')
        
        total_loss = 0
        
        
        batch_iterable = zip(X_batches, A_batches)
        for index, (Xbatch, Abatch) in enumerate(batch_iterable):
            print(f'batch {index}')
            #zero out gradient
            batch_compute_start = time.time()
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
            all_out.append([X_hat, A_hat, X_all, A_all, P_all, S_relab, S_all, S_sub, [len(np.unique(i.cpu())) for i in S_all], AW])
            
            #compute reconstruction losses for graph and attributes
            X_loss = X_recon_loss(X_hat, Xbatch)
            print(A_hat)
            A_loss = A_recon_loss(A_hat, Abatch)
            #compute the total loss function
            loss = A_loss+gamma*X_loss+Clust_loss-delta*Mod_loss
            #vanishing gradients in back prop, grabbing from matrices
            
            #compute backward pass
            loss.backward()
            #update gradients
            optimizer.step()
            #update total loss function
            total_loss += loss.cpu().item()
            
            #update batch losses
            train_epoch_loss_A += float(A_loss.cpu().detach().numpy())
            train_epoch_loss_X += float(X_loss.cpu().detach().numpy())
            train_epoch_loss_clust = [float(i+j) for (i,j) in zip(train_epoch_loss_clust, Clustloss_values)]
            train_epoch_loss_mod = [float(i+j) for (i,j) in zip(train_epoch_loss_mod, Modloss_values)]
            batch_compute_end = time.time()
           
            batch_time_hist.append(batch_compute_end-batch_compute_start)
            

            #batch_iterable.set_description(f'Epoch {idx} Processing batch {"-"*15} batch loss: {total_loss:.2f} test loss: {print_loss_test}')

            
            # #--------------------------------------------------------------------
            #evaluationg test performance
        test_perf_hist = []
        test_perf_time_start = time.time()
        if test_data:
            eval_X, eval_A, eval_labels = test_data
            with torch.no_grad():
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
            X_loss_test = gamma*(X_recon_loss(X_hat_test, eval_X)).cpu().detach().numpy().item()
            A_loss_test = (A_recon_loss(A_hat_test, eval_A)).cpu().detach().numpy().item()
            mod_weighted = delta*Mod_loss_test
            
            test_loss = (A_loss_test+X_loss_test+Clust_loss_test-mod_weighted).cpu().item()
            print_loss_test = f'{test_loss:.2f}'
            #update batch losses
            test_epoch_loss_A += float(A_loss_test)
            test_epoch_loss_X += float(X_loss_test)
            test_epoch_loss_clust = [float(i+j) for (i,j) in zip(test_epoch_loss_clust, Clustloss_values_test)]
            test_epoch_loss_mod = [float(i+j) for (i,j) in zip(test_epoch_loss_mod, Modloss_values_test)]
            
        else:
            print_loss_test = 'No test set provided'
            test_loss = 0
            Modloss_values_test = [0,0]
            Clustloss_values_test = [0,0]
            X_loss_test = 0
            A_loss_test = 0
        test_perf_time_end = time.time()
        test_perf_hist.append(test_perf_time_end - test_perf_time_start)
        print(f'...evaluate test performance time: {test_perf_time_end - test_perf_time_start}')
        test_perf_compute_time = str(test_perf_time_end - test_perf_time_start)

                        
           
        # P_truth_top = F.one_hot(torch.tensor(true_labels[0], dtype=torch.int64), num_classes = 5)
        # P_truth_mid = F.one_hot(torch.tensor(true_labels[1], dtype=torch.int64), num_classes = 15)
        # ptlt = clustering_loss_fn(lamb[0], X, [P_truth_top.to(dtype = torch.float32)], model.method)
        # ptlm = clustering_loss_fn(lamb[0], X, [P_truth_mid.to(dtype = torch.float32)], model.method)
        # print(f'Top loss (Truth) {float(ptlt[0])}')
        # print(f'Middle loss (Truth) {float(ptlm[0])}')
        
        #store loss component information
        epoch_end = time.time()
        append_time_start = time.time()
        train_loss_history.append({'Total Loss': total_loss,
                                   'A Reconstruction': train_epoch_loss_A,
                                   'X Reconstruction': gamma*train_epoch_loss_X,
                                   'Modularity': delta*np.array(train_epoch_loss_mod),
                                   'Clustering': np.array(train_epoch_loss_clust)})
        
        test_loss_history.append({'Total Loss': test_loss,
                                   'A Reconstruction': test_epoch_loss_A,
                                   'X Reconstruction': test_epoch_loss_X,
                                   'Modularity': test_epoch_loss_mod,
                                   'Clustering': test_epoch_loss_clust})

        append_time_end = time.time()
        print(f'...append loss history time: {append_time_end - append_time_start}')
        loss_hist_update = str(append_time_end - append_time_start)
        
        #evaluation on whole data

        eval_time_start = time.time()
        if epoch%update_interval == 0:
            run_eval_true_start =time.time()
            train_perf, eval_output, S_eval= evaluate(model, X, A, k, true_labels,run_eval=True)
            X_eval, A_eval = eval_output[0], eval_output[1]
            last_valid_A_eval = A_eval  # Store the last valid evaluation
            last_valid_X_eval = X_eval
            last_valid_S_eval = S_eval
            run_eval_true_end = time.time()
            run_eval_true_hist.append(run_eval_true_end-run_eval_true_start)

        else:
            train_perf, eval_output, S_eval= evaluate(model, X, A, k, true_labels,run_eval=False)
            X_eval, A_eval = eval_output[0], eval_output[1]
            # Use last valid evaluation if current is None
            A_eval = last_valid_A_eval if A_eval is None else A_eval
            X_eval = last_valid_X_eval if X_eval is None else X_eval
            S_eval = last_valid_S_eval if S_eval is None else S_eval
        
        
        eval_time_end = time.time()
        print(f'...evaluation whole data time: {eval_time_end - eval_time_start}')
        eval_whole_data = str(eval_time_end - eval_time_start)
        #update history
        append_perf_start = time.time()
        perf_hist.append(train_perf)
        pred_list.append(S_eval)
        print("updated history")             
        append_perf_end = time.time()
        print(f'...append perf hist time: {append_perf_end - append_perf_start}')
        append_perf_hist = str(append_perf_end - append_perf_start)
        #check for and apply validation 
        if validation_data:
            with torch.no_grad():
                X_val, A_val, val_labels = validation_data
                valid_perf, output, Sval = evaluate(model, X_val, A_val, k, val_labels)
                valid_perf_hist.append(valid_perf)
            print("finished validate")
        #evaluate epoch
        plot_time_start = time.time()
        
        if epoch % update_interval == 0:
            
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
                          save = save_output,)
                          #true_losses = [ptlt[0], ptlm[0]])
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
                
                plot_clust_heatmaps(A = A.cpu(), 
                                    A_pred = A_eval.cpu() if A_eval is not None else None, 
                                    X = X.cpu(),
                                    X_pred = X_eval.cpu() if X_eval is not None else None,
                                    true_labels = true_labels, 
                                    pred_labels = [i.cpu() for i in S_eval], 
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
        plot_time_end = time.time()
        print(f'...plot time: {plot_time_end-plot_time_start}')
        plot_time_hist.append(plot_time_end-plot_time_start)
        plotting_time = str(plot_time_end-plot_time_start)
        if early_stopping:
            if test_data:
                # Check for early stopping
                early_stop(test_loss, model, _type = 'test')
            else:
                # Check for early stopping
                early_stop(total_loss, model, _type = 'total')
                
            if early_stop.early_stop == True:
                break
            
        
        if epoch > 0:
            print(".... Average epoch time = %.2f seconds ---" % (np.mean(time_hist)))
            print(f"...Average batch computation time = {np.mean(batch_time_hist)}\n\n")
        time_hist.append(time.time() - epoch_start)
        if verbose:
            print(f'Total Epoch Time: {(epoch_end - epoch_start):.2f}')

    
    #return 
    forward_start = time.time()
    print("model forward")
    final_out = model.forward(X, A)
    forward_end = time.time()
    print(f'...forward whole data time: {forward_end-forward_start}')
    forward_time = str(forward_end-forward_start)
    print("testing HCD out")
    output_start = time.time()
    output = HCD_output(X = X, 
                        A = A, 
                        test_set= test_data,
                        labels=true_labels,
                        all_output=all_out, 
                        model_output=final_out, 
                        train_history=train_loss_history, 
                        test_history=test_loss_history, 
                        perf_history=perf_hist, 
                        pred_history=pred_list, 
                        batch_indices=index_batches)
    output_end =time.time()
    print(f'...output time: {output_end - output_start}')
    model_output = str(output_end - output_start)

    total_training_end = time.time()
    total_training_time = str(total_training_end-total_training_start)
    file_path = os.path.join(output_path, 'time_data.txt')
    with open(file_path, 'w') as f:
        f.write(f"Preliminary Set up: {prelim_setup_time} \n")
        f.write(f'fetching batched data: {fetching_batch_data}\n')
        f.write(f'test performance compute time: {test_perf_compute_time}\n')
        f.write(f'loss history updating: {loss_hist_update}\n')
        f.write(f'evaluation on whole data:{eval_whole_data}\n')
        f.write(f'appending performance history: {append_perf_hist}\n')
        f.write(f'model output: {model_output}\n')
        f.write(f'Average plotting time: {np.mean(plot_time_hist)}\n')
        f.write(f'Average time for evaluate when run eval is true: {np.mean(run_eval_true_hist)}\n')
        f.write(f'Average test performance evaluation: {np.mean(test_perf_hist)}\n')
        f.write(f'Average batch time: {np.mean(batch_time_hist)}\n')
        f.write(f'Average epoch time: {np.mean(time_hist)}\n')
        f.write(f'total training time: {total_training_time}\n')

    print(f"File saved to: {file_path}")

    return output