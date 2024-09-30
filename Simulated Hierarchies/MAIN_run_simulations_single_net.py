# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:46:39 2023

@author: Bruin
"""


#preamble
import torch
import numpy as np
import pandas as pd
from model.model import HCD 
from model.train import fit
from run_simulations_utils import load_simulated_data, set_up_model_for_simulated_data, handle_output, run_louvain, read_benchmark_CORA, post_hoc, run_kmeans
import pickle
import random as rd
import os


def run_single_simulation(args, **kwargs):
    
    if args.save_results:
        if args.make_directories:
            print(f'Creating new directory at {os.path.join(args.sp)}')
            os.makedirs(args.sp, exist_ok=True)
    
    if args.set_seed:
        rd.seed(123)
        torch.manual_seed(123)
    
    print(f'***** GPU AVAILABLE: {torch.cuda.is_available()} ******')
    device = 'cuda:'+str(0) if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print('***** Using device {} ********'.format(device))
    
    
    savepath_main = args.sp
    
    if args.dataset in ['complex', 'intermediate', 'toy']:
        
        loadpath_main, grid1, grid2, grid3, stats = load_simulated_data(args)        
            
        X, A, target_labels, comm_sizes = set_up_model_for_simulated_data(args, loadpath_main, grid1, grid2, grid3, stats, device, **kwargs)
                    
        valid = None       
                
    else:
        
        if args.dataset == 'cora':
            train_size, test_size = args.train_test_size
            train, test, valid = read_benchmark_CORA(args, 
                                                     PATH = '/benchmarks/',
                                                     use_split=args.split_data,
                                                     percent_train=train_size,
                                                     percent_test=test_size)
            X, A, target_labels, comm_sizes = train
            
    layers = len(comm_sizes)
    nodes, attrib = X.shape
    
    print('-'*25+'setting up and fitting models'+'-'*25)
    model = HCD(
                nodes, attrib, 
                method=args.use_method,
                use_kmeans=args.use_softKMeans,
                ae_hidden_dims=args.AE_hidden_size,
                ll_hidden_dims = args.LL_hidden_size,
                comm_sizes=comm_sizes, 
                normalize_input = args.normalize_input,
                normalize_outputs = args.normalize_layers,
                ae_operator = args.AE_operator,
                comm_operator = args.COMM_operator,
                ae_attn_heads = args.attn_heads, 
                dropout = args.dropout_rate,
                use_output_layers = args.add_output_layers,
                **kwargs
                ).to(device)
        
            
    print('summary of model architecture:')
    model.summarize()
        
    #preallocate results table
    res_table = pd.DataFrame(columns = ['Beth_Hessian_Comms',
                                        'Communities_Upper_Limit',
                                        'Max_Modularity',
                                        'Comm_Loss',
                                        'Reconstruction_A',
                                        'Reconstruction_X', 
                                        'Metrics',
                                        'Number_Predicted_Comms',
                                        'Louvain_Modularity',
                                        'Louvain_Metrics',
                                        'Louvain_Predicted_comms'])
    
    
    print('finished set up stage ...')
    #fit the models
    print("*"*80)
    
    #train the model
    out, pred_list = fit(model, X, A, 
                         k = layers,
                         optimizer='Adam', 
                         epochs = args.training_epochs, 
                         update_interval=args.steps_between_updates, 
                         layer_resolutions=args.resolution,
                         lr = args.learning_rate, 
                         gamma = args.gamma, 
                         delta = args.delta, 
                         lamb = args.lambda_, 
                         true_labels = target_labels, 
                         validation_data = valid,
                         verbose=args.verbose, 
                         save_output=args.save_results, 
                         turn_off_A_loss= args.remove_graph_loss,
                         output_path=savepath_main, 
                         ns = args.plotting_node_size, 
                         fs = args.fs,
                         update_graph = args.use_graph_updating)
         
    #handle output and return relevant values               
    results = handle_output(args, out, A, comm_sizes)
    beth_hessian, comm_loss, recon_A, recon_X, perf_mid, perf_top, upper_limit, max_mod, indices, metrics, preds, trace = results
    
    bpi, bli = indices
    S_sub, S_layer, S_all = trace
    
    #run louvain method on same dataset
    if args.run_louvain:
        louv_metrics, louv_mod, louv_num_comms, louv_preds = run_louvain(args, out, A, len(comm_sizes)+1, savepath_main, bpi, target_labels)          
    else:
        louv_metrics, louv_mod, louv_num_comms, louv_preds = (None, None, None, None)
        
    #run Kmeans on same dataset
    if args.run_kmeans:
        kmeans_preds = run_kmeans(args, X=X.detach().numpy(), 
                                  labels=target_labels, 
                                  layers=len(comm_sizes)+1,
                                  sizes=comm_sizes)
    else:
        kmeans_preds = None
    
    
    if args.post_hoc_plots:
        post_hoc(args, 
                 output = out, 
                 data = X, 
                 adjacency = A, 
                 predicted = pred_list[bpi],
                 truth = target_labels,
                 louv_pred = louv_preds,
                 kmeans_pred = kmeans_preds,
                 bp = bpi,
                 k_layers = layers,
                 verbose = True)
    
    
    #update performance table
    row_add = [beth_hessian,
               np.round(upper_limit.cpu().detach().numpy()),
               np.round(max_mod.cpu().detach().numpy(),4),
               tuple(comm_loss[-1].tolist()), 
               recon_A[-1], 
               recon_X[-1],
               metrics[-1], 
               preds[-1],
               louv_mod,
               louv_metrics,
               louv_num_comms]
    print(row_add)
    print('updating performance statistics...')
    res_table.loc[0] = row_add
    print('*'*80)
    print(res_table.loc[0])
    print('*'*80)
    if args.save_results == True:
        res_table.to_csv(savepath_main+'Simulation_Results'+'.csv')
        with open(savepath_main+'Simulation_Results_'+'OUTPUT'+'.pkl', 'wb') as f:
            pickle.dump(out, f)

    del model
    
    print('done')
    return out, res_table, A, X, target_labels, S_all, S_sub, louv_preds, indices


