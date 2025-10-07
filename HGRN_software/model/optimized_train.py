# -*- coding: utf-8 -*-
"""
Optimized training code for HRGNgene model with memory efficiency improvements
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
from typing import Optional, Union, List, Literal
import gc
from contextlib import contextmanager

# Optimized early stopping with memory management
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = float('inf')
        self.delta = delta
        self.path = path if path else os.getcwd()

    def __call__(self, loss, model, _type=['test', 'total']):
        score = loss
        self._type = _type
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score >= self.best_score + self.delta:
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
            print(f'\n{self._type} loss decreased ({self.loss_min:.6f} --> {loss:.6f}). Saving model...\n')
        # Save to CPU to avoid GPU memory issues
        torch.save(model.cpu().state_dict(), os.path.join(self.path, 'checkpoint.pth'))
        model.to(next(model.parameters()).device)  # Move back to original device
        self.loss_min = loss

# Memory-efficient context manager
@contextmanager
def memory_efficient_context():
    """Context manager for aggressive memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Optimized HCD output class with lazy loading
class HCD_output:
    def __init__(self, X, A, test_set, labels, model_output, test_history, train_history, 
                 perf_history, pred_history, batch_indices, device='cpu'):
        
        # Store only essential data, move to CPU immediately
        X_final, A_final, _, X_all_final, A_all_final, P_all_final, S_final, AW_final = model_output
        eval_X, eval_A, eval_labels = test_set if test_set else (None, None, None)

        self.model_output_history = [
            (X_final, A_final, X_all_final, A_all_final, P_all_final, S_final, AW_final)
        ]
        '''
        for k, v in AW_final.items():
            print("Key:", k)
            for i, item in enumerate(v):
                print(f"  Index {i}: type={type(item)}")
                if isinstance(item, list):
                    for j, subitem in enumerate(item):
                        print(f"    Subindex {j}: type={type(subitem)}")
        '''
        # Move tensors to CPU and detach to free GPU memory
        self.attention_weights = {
            k: [[t.detach().cpu() for t in head_list] for head_list in v]
            for k, v in AW_final.items()
        } if AW_final is not None else None


        self.reconstructed_features = X_final.detach().cpu()
        self.reconstructed_adj = A_final.detach().cpu()
        self.latent_features = X_all_final[0].detach().cpu()
        
        # Handle partitioned data efficiently
        self.partitioned_data = [x.detach().cpu() for x in X_all_final[1]]
        self.partitioned_latent_features = [x.detach().cpu() for x in X_all_final[2]]
        
        # Store histories (these are small)
        self.train_loss_history = train_history
        self.test_loss_history = test_history
        self.performance_history = perf_history
        self.pred_history = pred_history
        
        # Store data references
        self.training_data = {
            'X_train': X.detach().cpu(), 
            'A_train': A.detach().cpu(), 
            'labels_train': labels
        }
        
        if test_set:
            self.test_data = {
                'X_test': eval_X.detach().cpu() if eval_X is not None else None,
                'A_test': eval_A.detach().cpu() if eval_A is not None else None,
                'labels_test': eval_labels
            }
        else:
            self.test_data = {'X_test': None, 'A_test': None, 'labels_test': None}
            
        self.probabilities = {
            'top': P_all_final[0].detach().cpu(),
            'middle': [p.detach().cpu() for p in P_all_final[1]]
        }
        
        self.adjacency = {
            'community_graphs': [a.detach().cpu() for a in A_all_final[1]],
            'partitioned_graphs': [a.detach().cpu() for a in A_all_final[2]]
        }
        
        self.predicted_train = {
            'top': S_final[0].detach().cpu(),
            'middle': S_final[1].detach().cpu()
        }
        
        self.batch_indices = [idx.detach().cpu() for idx in batch_indices] if batch_indices else None
        
        # Initialize additional attributes
        self.best_loss_index = None
        self.hierarchical_clustering_preds = None
        self.louvain_preds = None
        self.kmeans_preds = None
        self.table = None
        self.perf_table = None

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in self.__dict__}
    
    def load_history_item(self, idx, map_location="cpu"):
        idx = min(idx, len(self.model_output_history) - 1)
        ref = self.model_output_history[idx]
        if isinstance(ref, dict) and "file" in ref:
            return load_batch_output(ref["file"], map_location=map_location)
        return ref
    def show_results(self):
        if self.perf_table is not None:
            print(self.perf_table)

# Optimized batching functions
def get_efficient_batches(X, A, batch_size=64, device='cpu'):
    """Memory-efficient batch generation"""
    num_nodes = X.size(0)
    indices = torch.randperm(num_nodes, device=device)
    
    batches = []
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        batch_indices = indices[start:end]
        batches.append(batch_indices)
    
    return batches

def get_batch_data(X, A, batch_indices, device):
    """Get batch data on demand to save memory"""
    X_batch = X[batch_indices].to(device)
    A_batch = A[batch_indices][:, batch_indices].to(device)
    return X_batch, A_batch

def load_batch_output(path, map_location='cpu'):
    """Load a saved batch output file (torch.save)"""
    return torch.load(path, map_location=map_location)

# Optimized loss functions
class OptimizedModularityLoss(nn.Module):
    def __init__(self):
        super(OptimizedModularityLoss, self).__init__()
        
    def forward(self, all_A, all_P, resolutions=None):
        loss = 0.0
        loss_list = []
        
        for index, (A, P) in enumerate(zip(all_A, all_P)):
            resolution = resolutions[index] if resolutions else 1.0
            
            with memory_efficient_context():
                mod = Modularity(A, P, resolution)
                loss += mod
                loss_list.append(float(mod.detach().cpu().numpy()))
                
        return loss, loss_list

class OptimizedClusterLoss(nn.Module):
    def __init__(self):
        super(OptimizedClusterLoss, self).__init__()

    def forward(self, Lamb, Attributes, Probabilities, method):
        loss = 0.0
        loss_list = []
        
        if not isinstance(Attributes, list):
            N = Attributes.shape[0]
            ptensor_list = [torch.eye(N, device=Attributes.device)]
            
        for idx, P in enumerate(Probabilities):
            Attr = Attributes[idx] if isinstance(Attributes, list) else Attributes
            
            if method == 'bottom_up':
                ptensor_list.append(P)
            else:
                ptensor_list = P
                
            with memory_efficient_context():
                within_ss, centroids = WCSS(X=Attr, Plist=ptensor_list, method=method)
                
                weight = Lamb[idx] if isinstance(Lamb, list) else Lamb
                weighted_loss = weight * within_ss
                
                loss_list.append(float(weighted_loss.detach().cpu().numpy()))
                loss += weighted_loss

        return loss, loss_list

# Optimized evaluation function
def evaluate_efficient(model, X, A, k, true_labels, run_eval=True, device='cpu'):
    """Memory-efficient evaluation"""
    if not run_eval:
        return None, (None, None, None, None, None, None, None), None
    
    with torch.no_grad():
        with memory_efficient_context():
            model.eval()
            # Move data to device only during forward pass
            X_device = X.to(device)
            A_device = A.to(device)
            
            output = model.forward(X_device, A_device)
            
            # Move results back to CPU immediately
            output_cpu = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    output_cpu.append(item.detach().cpu())
                elif isinstance(item, list):
                    output_cpu.append([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in item])
                else:
                    output_cpu.append(item)
            
            X_pred, A_pred, A_logit, X_list, A_list, P_list, S_pred, AW_pred = output_cpu
    
    perf_layers = []
    
    # Process predictions efficiently
    if model.method == 'bottom_up':
        S_trace_eval = trace_comms([s.clone() for s in S_pred], model.comm_sizes)
        S_all, S_temp, S_out = S_trace_eval
        S_relab = [s.detach().numpy() for s in S_temp][::-1]
    else:
        gp = [torch.unique(s, sorted=True, return_inverse=True) for s in S_pred]
        S_relab = [g[1] for g in gp]
        
    if true_labels:
        perf_layers = get_layered_performance(k, S_relab, true_labels)
        
    return perf_layers, (X_pred, A_pred, X_list, A_list, P_list, S_pred, AW_pred), S_relab

# Optimized training function
def fit_optimized(model, X, A, optimizer='Adam', epochs=100, update_interval=10, lr=1e-4, 
                 gamma=1, delta=1, lamb=1, layer_resolutions=[1,1], k=2, use_batch_learning=True, 
                 batch_size=64, early_stopping=False, patience=5, true_labels=None, 
                 validation_data=None, test_data=None, save_output=False, output_path='', 
                 fs=10, ns=10, verbose=True, device='cpu', **kwargs):
    """
    Optimized training function with memory efficiency improvements
    """
    
    # Move model to device
    model = model.to(device)
    
    # Initialize storage with minimal memory footprint
    train_loss_history = []
    perf_hist = []
    pred_list = []
    test_loss_history = []
    
    comm_layers = len(model.comm_sizes)
    
    # Early stopping
    if early_stopping:
        early_stop = EarlyStopping(patience=patience, verbose=True, path=output_path)
    
    # Optimizer
    optimizer = optimizers.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Loss functions
    A_recon_loss = nn.BCELoss(reduction='mean')
    X_recon_loss = nn.MSELoss(reduction='mean')
    modularity_loss_fn = OptimizedModularityLoss()
    clustering_loss_fn = OptimizedClusterLoss()
    
    # Generate batch indices once (memory efficient)
    if use_batch_learning:
        if batch_size > X.shape[0]:
            raise ValueError(f'Batch size ({batch_size}) larger than dataset size ({X.shape[0]})')
        batch_indices_list = get_efficient_batches(X, A, batch_size, device='cpu')
    else:
        batch_indices_list = [torch.arange(X.shape[0])]
    
    print(f"Training on {len(batch_indices_list)} batches")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        
        # Initialize epoch losses
        total_loss = 0.0
        train_epoch_losses = {
            'A': 0.0, 'X': 0.0, 
            'clust': [0.0] * len(model.comm_sizes), 
            'mod': [0.0] * len(model.comm_sizes)
        }
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print('=' * 50)
        
        # Batch processing with memory management
        for batch_idx, batch_indices in enumerate(batch_indices_list):
            with memory_efficient_context():
                # Get batch data on device
                A = A / A.max()
                X_batch, A_batch = get_batch_data(X, A, batch_indices, device)
                
                optimizer.zero_grad()
                
                # Forward pass
                forward_output = model.forward(X_batch, A_batch)
                X_hat, A_hat, A_logit, X_all, A_all, P_all, S_all, AW = forward_output
                
                # Compute losses efficiently
                mod_clust_output = get_optimized_losses(
                    model, X_batch, A_batch, forward_output, lamb, 
                    layer_resolutions, modularity_loss_fn, clustering_loss_fn
                )
                Mod_loss, Modloss_values, Clust_loss, Clustloss_values = mod_clust_output
                
                
                # Reconstruction losses
                A_hat = torch.clamp(A_hat, min=1e-7, max=1 - 1e-7)
                X_loss = X_recon_loss(X_hat, X_batch)
                A_loss = A_recon_loss(A_hat, A_batch)
                
                # Total loss
                batch_loss = A_loss + gamma * X_loss + Clust_loss - delta * Mod_loss
                print(f'A_loss: ',A_loss,', X_loss: ',X_loss,', Clust_loss',Clust_loss,', Mod_loss: ',Mod_loss)
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update epoch losses
                total_loss += batch_loss.item()
                print(f'batch loss: ',batch_loss.item())
                train_epoch_losses['A'] += A_loss.item()
                train_epoch_losses['X'] += X_loss.item()
                
                for i, (c, m) in enumerate(zip(Clustloss_values, Modloss_values)):
                    if i < len(train_epoch_losses['clust']):
                        train_epoch_losses['clust'][i] += c
                        train_epoch_losses['mod'][i] += m
                
                # Clear batch data from GPU
                del X_batch, A_batch, forward_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Store training history
        train_loss_history.append({
            'Total Loss': total_loss,
            'A Reconstruction': train_epoch_losses['A'],
            'X Reconstruction': gamma * train_epoch_losses['X'],
            'Modularity': delta * np.array(train_epoch_losses['mod']),
            'Clustering': np.array(train_epoch_losses['clust'])
        })
        
        # Evaluation (less frequent to save memory)
        test_loss = 0.0
        if test_data:
            eval_X, eval_A, eval_labels = test_data
            with memory_efficient_context():
                test_perf, test_output, S_replab_test = evaluate_efficient(
                    model, eval_X, eval_A, k, eval_labels, device=device
                )
                
                if test_output[0] is not None:
                    X_hat_test, A_hat_test = test_output[0], test_output[1]
                    eval_X_dev = eval_X.to(device)
                    eval_A_dev = eval_A.to(device)
                    X_hat_dev = X_hat_test.to(device)
                    A_hat_dev = A_hat_test.to(device)
                    
                    X_loss_test = X_recon_loss(X_hat_dev, eval_X_dev).item()
                    A_loss_test = A_recon_loss(A_hat_dev, eval_A_dev).item()
                    print(A_hat)
                    test_loss = A_loss_test + gamma * X_loss_test
                    
        
        test_loss_history.append({'Total Loss': test_loss})
        
        # Performance evaluation (periodic)
        if epoch % update_interval == 0:
            with memory_efficient_context():
                train_perf, eval_output, S_eval = evaluate_efficient(
                    model, X, A, k, true_labels, device=device
                )
                perf_hist.append(train_perf)
                pred_list.append(S_eval)
                
                if true_labels:
                    print('\nMODEL PERFORMANCE')
                    print_performance_efficient(perf_hist, comm_layers, k)
        
        # Early stopping check
        if early_stopping:
            print('Early Stopping Start\n')
            print(f'A_loss_test: ',A_loss_test, ', X_loss_test: ', X_loss_test)
            print(f'Total Loss: ',  total_loss)
            print(f'Test Loss: ', test_loss)
         
            early_stop(test_loss if test_data else total_loss, model)
            if early_stop.early_stop:
                print("Early stopping triggered")
                break
        
        epoch_time = time.time() - epoch_start
        if verbose:
            print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s')
            print(f'Total Loss: {total_loss:.4f}')
            print('-' * 50)
    
    # Final model output
    print("Generating final output...")
    with memory_efficient_context():
        model.eval()
        with torch.no_grad():
            final_out = model.forward(X.to(device), A.to(device))
            # Move to CPU immediately
            final_out_cpu = []
            for item in final_out:
                if isinstance(item, torch.Tensor):
                    final_out_cpu.append(item.detach().cpu())
                elif isinstance(item, list):
                    final_out_cpu.append([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in item])
                else:
                    final_out_cpu.append(item)
    
    # Create optimized output object
    output = HCD_output(
        X=X, A=A, test_set=test_data, labels=true_labels,
        model_output=final_out_cpu, train_history=train_loss_history,
        test_history=test_loss_history, perf_history=perf_hist,
        pred_history=pred_list, batch_indices=batch_indices_list, device='cpu'
    )
    
    return output

def get_optimized_losses(model, Xbatch, Abatch, output, lamb, resolution, modlossfn, clustlossfn):
    """Optimized loss computation with memory management"""
    X_hat, A_hat, A_logit, X_all, A_all, P_all, S_all, AW = output
    
    if model.method == 'bottom_up':
        S_sub, S_relab, S = trace_comms([s.clone() for s in S_all], model.comm_sizes)
        Mod_loss, Modloss_values = modlossfn([Abatch] + A_all[1], P_all, resolution)
        Clust_loss, Clustloss_values = clustlossfn(lamb, Xbatch, P_all, model.method)
    elif model.method == "top_down":
        # Top-down processing
        top_mod_loss, values_top = modlossfn([A_all[0]], [P_all[0]], resolution)
        middle_mod_loss, values_mid = modlossfn(A_all[-1], P_all[1], resolution)
        Mod_loss = top_mod_loss + middle_mod_loss
        Modloss_values = values_top + [torch.mean(torch.tensor(values_mid)).item()]
        
        Clust_loss_top, Clustloss_values_top = clustlossfn(lamb[0], Xbatch, [P_all[0]], model.method)
        Clust_loss_mid, Clustloss_values_mid = clustlossfn(lamb[1], X_all[-1], P_all[1], model.method)
        Clust_loss = Clust_loss_top + Clust_loss_mid
        Clustloss_values = Clustloss_values_top + [torch.sum(torch.tensor(Clustloss_values_mid)).item()]
    
    return Mod_loss, Modloss_values, Clust_loss, Clustloss_values

def print_performance_efficient(history, comm_layers, k):
    """Efficient performance printing with error handling"""
    if not history or all(h is None for h in history):
        print("No performance history available")
        return

    valid_history = [h for h in history if h is not None]
    if not valid_history:
        print("No valid performance data available")
        return

    last_perf = valid_history[-1]
    layer_names = ['top'] + [f'middle_{i}' for i in range(comm_layers-1)]
    
    for i in range(min(k, len(last_perf))):
        if i >= len(last_perf) or last_perf[i] is None:
            print(f"No data available for {layer_names[i]} layer")
            continue
            
        print(f'{"-"*20} {layer_names[i]} layer {"-"*20}')
        
        metrics = last_perf[i]
        metric_names = ['Homogeneity', 'Completeness', 'NMI', 'ARI']
        
        for j, (name, value) in enumerate(zip(metric_names, metrics[:4])):
            print(f'{name}: {value:.4f}')
        print('-' * 50)
