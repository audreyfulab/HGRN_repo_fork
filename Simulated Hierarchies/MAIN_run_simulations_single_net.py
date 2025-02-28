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
from run_simulations_utils import load_simulated_data, set_up_model_for_simulated_data, handle_output, run_louvain, run_trad_hc, read_benchmark_CORA, post_hoc, run_kmeans, load_application_data_regulon, load_application_data_Dream5, set_up_model_for_simulation_inplace
from model.utilities import compute_kappa
import pickle
import random as rd
import os


def run_single_simulation(args, simulation_args = None, return_model = False, **kwargs):
    
    """Run a single simulation with the specified parameters.

    Args:
        args: Parsed argument namespace containing primary model parameters.
        simulation_args (optional): Additional parsed argument namespace for 
            data simulation-specific settings. Defaults to None.
        return_model (bool, optional): If True, returns the simulation model along 
            with results. Defaults to False.
        **kwargs: Additional keyword arguments to passed to model.HCD

    Returns:
        If return_model is False:
            dict: Results of the simulation
        If return_model is True:
            tuple: (dict, SimulationModel) containing both the results and the model instance

    Raises:
        ValueError: If required parameters in args are missing or invalid.
        
    Example:
        args = parser.parse_args(['--param1', 'value1'])
        results = run_single_simulation(args)
        results, model = run_single_simulation(args, return_model=True) #save model in output directory as checkpoint.pth file
    """
    
    #set up output directories
    if args.save_results:
        if args.make_directories:
            print(f'Creating new directory at {os.path.join(args.sp)}')
            os.makedirs(args.sp, exist_ok=True)
    
    #set seed
    if args.set_seed:
        seed = 123
        rd.seed(seed)
        torch.manual_seed(seed)
    
    #check for local GPU
    print(f'***** GPU AVAILABLE: {torch.cuda.is_available()} ******')
    device = 'cuda:'+str(0) if args.use_gpu and torch.cuda.is_available() else 'cpu'
    if 'cuda' in device:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print('***** Using device {} ********'.format(device))
    
    
    savepath_main = args.sp
    comm_sizes = args.community_sizes
    layers = len(args.community_sizes)
    
    
    
    #read in pre-generated data 
    if args.dataset in ['complex', 'intermediate', 'toy']:
        
        loadpath_main, grid1, grid2, grid3, stats = load_simulated_data(args)        
            
        X, A, target_labels, comm_sizes = set_up_model_for_simulated_data(args, loadpath_main, grid1, grid2, grid3, stats, **kwargs)
           
        if args.split_data:
            train, test = split_dataset(X, A, target_labels, args.train_test_size)         
            
            X, A, target_labels = train
        else:
            test = None
        
        valid = None       
                
    # read in benchmark
    elif args.dataset in ['cora', 'pubmed']:
        
        if args.dataset == 'cora':
            train_size, test_size = args.train_test_size
            train, test, valid = read_benchmark_CORA(args, 
                                                     PATH = '/benchmarks/',
                                                     use_split=args.split_data,
                                                     percent_train=train_size,
                                                     percent_test=test_size)
            X, A, target_labels = train
            
    # read in regulon data
    elif args.dataset in ['regulon.EM', 'regulon.DM']:
        
        train, test, gene_names = load_application_data_regulon(args)
        target_labels = None
        valid = None
        
        X, A, [] = train
        
    # read in Dream5 data
    elif args.dataset in ['Dream5.'+i for i in ['E_coli', 'in_silico', 'S_aureus', 'S_cerevisiae']]:
        
        train, test, gene_names = load_application_data_Dream5(args)
        target_labels = None
        valid = None
        
        X, A, [] = train
        
    #randomly generate a new dataset according to argments passed from simulation_args
    elif args.dataset == 'generated':
        
        X, A, target_labels = set_up_model_for_simulation_inplace(args, simulation_args, load_from_existing=args.load_from_existing)
        
        if args.split_data:
            train, test = split_dataset(X, A, target_labels, args.train_test_size)         
            
            X, A, target_labels = train
        else:
            test = None
        
        valid = None
        
    else:
        raise ValueError(f'Unknown value args.dataset == {args.dataset}')
      
    nodes, attrib = X.shape
    
    
    # estimate number of communities k
    if args.compute_optimal_clusters:
        comm_sizes = compute_kappa(X, A, method = args.kappa_method, save = args.save_results, PATH = args.sp, verbose = True)
    
    #initiate model 
    print('-'*25+'setting up and fitting models'+'-'*25)
    model = HCD(
                nodes, attrib, 
                method=args.use_method,
                use_kmeans_top=args.use_softKMeans_top,
                use_kmeans_middle=args.use_softKMeans_middle,
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
    
    X = X.to(device)
    A = A.to(device)
            
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
    model_output = fit(model, X, A, 
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
                       early_stopping = args.early_stopping,
                       patience = args.patience,
                       verbose=args.verbose, 
                       save_output=args.save_results, 
                       output_path=savepath_main, 
                       ns = args.plotting_node_size, 
                       fs = args.fs,
                       use_batch_learning = args.use_batch_learning,
                       batch_size = args.batch_size,
                       device = device
                       )
         
    #handle output and return relevant values               
    results = handle_output(args = args, 
                            output = model_output, 
                            comm_sizes=comm_sizes)
    beth_hessian, comm_loss, recon_A, recon_X, perf_mid, perf_top, upper_limit, max_mod, indices, metrics, preds, trace = results
    
    #choose results to return (best perf mid, top or best loss)
    bpi, bli = indices
    if args.return_result == 'best_loss':
        best_result_index = bli
    else:
        best_result_index = bpi
    
    model_output.best_loss_index = best_result_index
    S_sub, S_layer, S_all = trace
    
    #run louvain method on same dataset
    if args.run_louvain:
        louv_metrics, louv_mod, louv_num_comms, louv_preds = run_louvain(args, A, target_labels, len(comm_sizes)+1, savepath_main) 
        model_output.louvain_preds = louv_preds         
    else:
        louv_metrics, louv_mod, louv_num_comms, louv_preds = (None, None, None, None)
        model_output.louvain_preds = None
        
    #run Kmeans on same dataset
    if args.run_kmeans:
        
        if args.compute_optimal_clusters:
            km_sizes = comm_sizes[::-1]
        else: 
            km_sizes = [len(np.unique(i)) for i in target_labels]
            
        kmeans_preds = run_kmeans(args, X=X.detach().numpy(), 
                                  labels=target_labels, 
                                  layers=len(comm_sizes)+1,
                                  sizes=km_sizes)
        
        model_output.kmeans_preds = {'top': kmeans_preds[1], 'middle': kmeans_preds[0]}
    else:
        kmeans_preds = None
        model_output.kmeans_preds = {'top': None, 'middle': None}
        
    # runs hierarchical clustering using Ward's metric on data 
    if args.run_hc:
        
        if args.compute_optimal_clusters:
            hc_sizes = comm_sizes[::-1]
        else: 
            hc_sizes = [len(np.unique(i)) for i in target_labels]
            
        hc_preds = run_trad_hc(args, X=X.cpu().detach().numpy(), 
                                  labels=target_labels, 
                                  layers=len(comm_sizes)+1,
                                  sizes=hc_sizes)
        
        model_output.hierarchical_clustering_preds = {'top': hc_preds[1], 'middle': hc_preds[0]}
    else:
        hc_preds = None
        model_output.hierarchical_clustering_preds = {'top': None, 'middle': None}

    best_preds = [i.cpu() for i in model_output.pred_history[best_result_index]]

        
    #post fit plots and results
    if args.post_hoc_plots:
        pbmt=post_hoc(args, 
                      output = model_output, 
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
    
    model_output.table = res_table
    model_output.perf_table = pbmt
    
    if args.save_results == True:
        res_table.to_csv(savepath_main+'Simulation_Results'+'.csv')
        with open(savepath_main+'Simulation_Results_'+'OUTPUT'+'.pkl', 'wb') as f:
            pickle.dump(model_output, f)
    
    if args.save_model:
        print(f'saving model to {savepath_main+"MODEL.pth"}')
        torch.save(model.cpu(), savepath_main+'MODEL.pth')
        
    print('done')
    
    if return_model:
        return model_output, model
    else:
        del model
        return model_output
    
    
    
    


