import pickle
import pandas as pd
import numpy as np
from data_analysis.save_mask import load_and_save, save_homo_masks
from data_analysis.loader import load_data, load_out, load_out_old, generate_homo_mask
# in this code page, we aims to learn the further result analysis
import torch
import math
import json
from data_analysis.utils import *


def main_output_analysis():
    datasets = ["PubMed", "arxiv", "Chameleon", "Squirrel"]
    models = ["GCN", "MLP", "graphless"]  
    
    data_is_old = False
    model_is_old = False
    train_masks_list, test_masks_list, homo_ratios_list, adjust_homo_ratios_list, labels = load_data(datasets, data_is_old)
    # num_dataset, num_split, num_nodes
    interval_value = 0.2
    homo_mask_list, intervals = generate_homo_mask(homo_ratios_list, interval_value=interval_value) 
    adjust_homo_mask_list, adjust_intervals = generate_homo_mask(adjust_homo_ratios_list, interval_value=interval_value)
    
    if not model_is_old:
        outs_list = load_out(models, datasets)
    else:
        outs_list = load_out_old(models, datasets)

    is_old = True
    MLP_models, GNN_models, MLP_models_index, GNN_models_index = match_model(models)

    for dataset, train_masks, test_masks, homo_masks, adjust_homo_masks, outs, label in zip(datasets, train_masks_list, test_masks_list, homo_mask_list, adjust_homo_mask_list, outs_list, labels):
        masks, intervals = adjust_homo_masks, adjust_intervals
        
        hete_index = find_hete_mask_index(masks, intervals)
        homo_nums, homo_ratios = homo_count(masks)

        dataset = f"{dataset}_old" if is_old else f"{dataset}_new"        
        acc_means, acc_stds, results = find_average_performance_with_std(models, outs, test_masks, masks, label)
        compare_with_GCN(models, dataset, results)
        

    
        
    

