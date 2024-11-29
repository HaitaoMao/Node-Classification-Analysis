import pickle
import pandas as pd
import numpy as np
import torch
import math
from utils.data_utils import *
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from output_analyze.fair_analyze_func import *
from output_analyze.fairness_exp import *
from output_analyze.plot_exp import *

# this experiment revolve on the relationship between feature distance and accuracy, whether connect to the 
def main_fair_analysis6():
    # "Cora", "arxiv", "CiteSeer", "PubMed", "Actor", "Chameleon", "Squirrel", "genius", "twitch-gamer"
    homo_datasets = ["Cora", "CiteSeer", "PubMed", "arxiv", "IGB_tiny"]
    hete_datasets = ["Actor", "Chameleon", "Squirrel", "amazon_ratings", "twitch-gamer"]
    datasets = ["PubMed", "arxiv", "Chameleon", "Squirrel"]  # , "IGB_tiny", "Actor"
    datasets = ["Chameleon", "Squirrel"] #  , "Chameleon", "Squirrel", "genius", "twitch-gamer", "IGB_tiny" 
    #  , "arxiv"
    # datasets = ["arxiv"] #  , "Chameleon", "Squirrel", "genius", "twitch-gamer", "IGB_tiny" 
    plot_name = "hete"
    args = generate_args()
    is_old = True
    num_hop = 5
    adj = "asy" # "asy" "non" "sym"
    is_self_loop = False
    interval = 0.2
    normalize_features = 1
    model = "GCN"
    args.normalize_features = normalize_features
    

    
    arg_dict = {"is_old": is_old, "num_hop": num_hop ,"model": model, "adj": adj, "is_self_loop": is_self_loop, "interval": interval, "normalize_features": normalize_features}

    records = {}
    for dataset_name in datasets:
        if dataset_name in ["Chameleon", "Squirrel"]: 
            args.is_new = 0
        else:
            args.is_new = 1
        
        homo_mask_name = f"output_analyze/homo_mask_old/{dataset_name}_{interval}.txt" if is_old else f"output_analyze/homo_mask/adjust_{dataset_name}_{interval}.txt"

        arg_dict["name"] = dataset_name
        args.dataset = dataset_name
        dataset, data = get_dataset(args)
        with open(homo_mask_name, "rb") as f:
            homo_masks = pickle.load(f)
            homo_masks = preprocess_mask(homo_masks)
            # num_homo_region, num_node
        record = []
        
        if dataset_name in ["Cora", "CiteSeer", "PubMed", "Actor", "Chameleon", "Squirrel", "amazon_ratings"]:
            num_split = 10
        elif dataset_name in ["twitch-gamer"]:
            num_split = 5
        else:
            num_split = 1
        for split_idx in range(num_split):
            data, _ = get_split(args, dataset, data, split_idx)  # , index=split_idx
            arg_dict["num_node"] = np.max(data.y.shape)
            edge_index, edge_value = prepare_adj(data.edge_index, adj, is_self_loop, arg_dict["num_node"])
            record_data = class_distance_to_train_node_one(data, edge_index, edge_value, homo_masks, arg_dict)
            record.append(record_data)

        record = np.array(record)
        mean, std = np.mean(record, axis=0), np.std(record, axis=0)


        
        records[dataset_name] = [mean, std]
        
    plot(records, num_hop + 1, plot_name=plot_name)
    
    
    
    
    
    
    
            
            

    
    
