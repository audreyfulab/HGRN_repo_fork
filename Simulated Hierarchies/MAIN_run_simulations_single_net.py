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
from model.train import fit, split_dataset
from run_simulations_utils import load_simulated_data, set_up_model_for_simulated_data, handle_output, run_louvain, run_trad_hc, read_benchmark_CORA, post_hoc, run_kmeans, load_application_data
import pickle
import random as rd
import os


def run_single_simulation(args, **kwargs):
    
    if args.save_results:
        if args.make_directories:
            print(f'Creating new directory at {os.path.join(args.sp)}')
            os.makedirs(args.sp, exist_ok=True)
    
    if args.set_seed:
        seed = 123
        rd.seed(seed)
        torch.manual_seed(seed)
    
    print(f'***** GPU AVAILABLE: {torch.cuda.is_available()} ******')
    device = 'cuda:'+str(0) if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print('***** Using device {} ********'.format(device))
    
    
    savepath_main = args.sp
    comm_sizes = args.community_sizes
    
    if args.dataset in ['complex', 'intermediate', 'toy']:
        
        loadpath_main, grid1, grid2, grid3, stats = load_simulated_data(args)        
            
        X, A, target_labels, comm_sizes = set_up_model_for_simulated_data(args, loadpath_main, grid1, grid2, grid3, stats, device, **kwargs)
           
        if args.split_data:
            train, test = split_dataset(X, A, target_labels, args.train_test_size)         
            
            X, A, target_labels = train
        else:
            test = None
        
        valid = None       
                
    elif args.dataset in ['cora', 'pubmed']:
        
        if args.dataset == 'cora':
            train_size, test_size = args.train_test_size
            train, test, valid = read_benchmark_CORA(args, 
                                                     PATH = '/benchmarks/',
                                                     use_split=args.split_data,
                                                     percent_train=train_size,
                                                     percent_test=test_size)
            X, A, target_labels = train
            target_labels = None
            
    elif args.dataset in ['regulon.EM', 'regulon.DM']:
        
        train, test, gene_names = load_application_data(args)
        target_labels = None
        valid = None
        
        X, A, [] = train
            
    layers = len(args.community_sizes)
    nodes, attrib = X.shape
    
    #move inputs to device
    #X.to(device)
    #A.to(device)
    
    print('-'*25+'setting up and fitting models'+'-'*25)
    model = HCD(
                nodes, attrib, 
                method=args.use_method,
                use_kmeans_top=args.use_softKMeans_top,
                use_kmeans_middle=args.use_softKMeans_middle,
                ae_hidden_dims=args.AE_hidden_size,
                ll_hidden_dims = args.LL_hidden_size,
                comm_sizes=args.community_sizes, 
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
    out, pred_list = fit(
                         model, X, A, 
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
                         test_data = test,
                         verbose=args.verbose, 
                         save_output=args.save_results, 
                         output_path=savepath_main, 
                         ns = args.plotting_node_size, 
                         fs = args.fs,
                         update_graph = args.use_graph_updating,
                         use_batch_learning = args.use_batch_learning,
                         batch_size = args.batch_size
                         )
         
    #handle output and return relevant values               
    results = handle_output(args, out, A, comm_sizes, args.use_method, target_labels)
    beth_hessian, comm_loss, recon_A, recon_X, perf_mid, perf_top, upper_limit, max_mod, indices, metrics, preds, trace = results
    
    #choose results to return (best perf mid, top or best loss)
    bpi, bli = indices
    if args.return_result == 'best_loss':
        best_result_index = bli
    else:
        best_result_index = bpi
    
    S_sub, S_layer, S_all = trace
    
    #run louvain method on same dataset
    if args.run_louvain:
        louv_metrics, louv_mod, louv_num_comms, louv_preds = run_louvain(args, out, A, len(comm_sizes)+1, savepath_main, best_result_index, target_labels)          
    else:
        louv_metrics, louv_mod, louv_num_comms, louv_preds = (None, None, None, None)
        
    #run Kmeans on same dataset
    if args.run_kmeans:
        kmeans_preds = run_kmeans(args, X=X.detach().numpy(), 
                                  labels=target_labels, 
                                  layers=len(comm_sizes)+1,
                                  sizes=[len(np.unique(i)) for i in target_labels])
    else:
        kmeans_preds = None
        
        
    if args.run_hc:
        hc_preds = run_trad_hc(args, X=X.detach().numpy(), 
                                  labels=target_labels, 
                                  layers=len(comm_sizes)+1,
                                  sizes=[len(np.unique(i)) for i in target_labels])
    else:
        hc_preds = None
    
    
    if args.use_method == 'bottom_up':
        best_preds = pred_list[best_result_index][::-1]
    else:
        best_preds = pred_list[best_result_index]
        
    #post fit plots and results
    if args.post_hoc_plots:
        post_hoc(args, 
                 output = out, 
                 data = X, 
                 adjacency = A, 
                 predicted = best_preds,
                 truth = target_labels,
                 louv_pred = louv_preds,
                 kmeans_pred = kmeans_preds,
                 thc_pred = hc_preds,
                 bp = best_result_index,
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
    
    if args.save_model:
        print(f'saving model to {savepath_main+"MODEL.pth"}')
        torch.save(model, savepath_main+'MODEL.pth')

    del model
    
    print('done')
    return out, res_table, A, X, target_labels, S_all, S_sub, louv_preds, indices


