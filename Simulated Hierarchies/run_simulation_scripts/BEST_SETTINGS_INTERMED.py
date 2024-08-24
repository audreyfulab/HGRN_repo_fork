# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:44:08 2024

@author: Bruin
"""

ep = 500
wn = 0
wg = 0
out, res, graphs, data, truth, preds, preds_sub, louv_pred, idx = run_simulations(
    save_results=False,
    which_net=wn,
    which_ingraph=wg,
    reso=[1,1],
    hd=[256, 128, 64, 32],
    gam = 1,
    delt = 1,
    lam = [0.5,0.001],
    learn_rate=1e-5,
    use_true_comms=True,
    cms=[50, 10],
    epochs = ep,
    updates = ep,
    activation = 'LeakyReLU',
    TOAL=False,
    use_multi_head=False,
    attn_heads=5,
    verbose = True,
    return_result = 'best_perf_top',
    normalize_inputs=True)