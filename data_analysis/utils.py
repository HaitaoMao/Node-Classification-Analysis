# in this file, we majorly focus on the analysis without the mask, just rough understand the difference
import pickle
import math
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from output_analyze.plot import *
from scipy.stats import ttest_rel
from output_analyze.plot_exp import *
# in this figure, we aims to see 
# node distribution (finish)
# the performance of different model on different homo ratio (finish)
# the performance gap between two selected model,  
# the performance gap between two different group of model (finish)
# the performance gap within the same group of model: on the most high and tail homo part, check those four matrix

# The next step, we are going to study on what GNN prediction condition that lead to the failure, and may need significant test
# or we can have similar test on the test node

def fine_grain_analysis(datasets, models, outs, masks, labels):
    MLP_models, GNN_models, MLP_models_index, GNN_models_index = match_model(models)
    for idx, dataset in enumerate(datasets):
        '''
        This part we majorly focus on the difference between similar models
        '''
        homo_results = per_dataset_similar(MLP_models, outs[idx][MLP_models_index], masks[idx], labels[idx], -1)
        # The index should select via the population, now it is just the last one
        per_dataset_draw(homo_results, MLP_models, filename)
        hete_results = per_dataset_similar(MLP_models, outs[idx][GNN_models_index], masks[idx], labels[idx], 0)
        per_dataset_draw(hete_results, MLP_models, filename)

        homo_results = per_dataset_similar(GNN_models, outs[idx][GNN_models_index], masks[idx], labels[idx], -1)
        # The index should select via the population, now it is just the last one
        per_dataset_draw(homo_results, GNN_models, filename)
        hete_results = per_dataset_similar(GNN_models, outs[idx][GNN_models_index], masks[idx], labels[idx], 0)
        per_dataset_draw(hete_results, GNN_models, filename)


        '''
        This part we majorly focus on the avergae gap between difference models
        '''
        results = find_average_performance(models, outs[idx], masks[idx], labels[idx])
        
        

def per_dataset_similar_idx(models, outs, masks, homo_masks, labels, idx):
    # outs: [num_model, num_split, num_nodes]
    # masks: [num_split, num_nodes]
    # models [num_model]
    # labels: [num_node]
    # homo_masks: [num_homo, num_split, num_nodes]
    # import ipdb; ipdb.set_trace()
    
    homo_masks = homo_masks[idx]
    new_masks = []
    for mask in masks:
        new_masks.append(mask & homo_masks)
    masks = new_masks

    results = {}
    num_model =  len(models) 
    num_split = len(masks)
    
    names = ["cc", "cw", "wc", "ww"] # four keys including correct correct, correct wrong for both two model

    for name in names:
        results[name] = np.zeros([num_model, num_model])

    # import ipdb; ipdb.set_trace()
    num_node_sum = 0
    for i, (model1, outs1) in enumerate(zip(models, outs)):
        for j, (model2, outs2) in enumerate(zip(models, outs)):
            # outs1: num_split num_node

            # print(f"{model2}: {len(outs2)}")
            for split_id in range(num_split):
                mask = masks[split_id]
                num_node = torch.sum(mask).item()
                if i == 0 and j == 0: num_node_sum += num_node

                # import ipdb; ipdb.set_trace()
                # print(split_id)
                out1_new, out2_new = outs1[split_id][mask], outs2[split_id][mask]
                label = labels[mask]
                                
                predict1, predict2 = out1_new.argmax(dim=-1), out2_new.argmax(dim=-1)
                correct1, correct2 = (predict1 == label), (predict2 == label)

                cc = torch.sum(correct1 & correct2).item()
                cw = torch.sum(correct1 & ~correct2).item()
                wc = torch.sum(~correct1 & correct2).item()
                ww = torch.sum(~correct1 & ~correct2).item()

                tmp_results = [cc, cw, wc, ww]

                for name, result in zip(names, tmp_results):
                    results[name][i][j] += result  
    
    # import ipdb; ipdb.set_trace()
    if num_node_sum == 0: num_node_sum = 1
    for key in results.keys():
        results[key] = results[key] / num_node_sum

    return results


def match_model(models):
    MLP_model_names, GNN_model_names = ["MLP", "MLP_reg", "MLP_dist"], ["APPNP", "GCN", "SGC", "GAT", "SAGE" ]
    MLP_models, GNN_models, MLP_models_index, GNN_models_index = [], [], [], []

    for name in MLP_model_names:
        try:
            index = models.index(name)
            MLP_models.append(name)
            MLP_models_index.append(index)
        except:
            continue
            
    for name in GNN_model_names:
        try:
            index = models.index(name)
            GNN_models.append(name)
            GNN_models_index.append(index)
        except:
            continue
    
    return MLP_models, GNN_models, torch.tensor(MLP_models_index), torch.tensor(GNN_models_index)


def find_average_performance(models, outs_list, masks, homo_masks, labels):
    homo_num = len(homo_masks)
    num_model = len(models) 
    num_split = len(masks)
    
    results = np.zeros([num_model, homo_num])
    num_nodes = np.zeros([homo_num])
    for model_idx, (model, outs)  in enumerate(zip(models, outs_list)):
        for homo_idx, homo_mask in enumerate(homo_masks):
            for mask_idx, (mask, out) in enumerate(zip(masks, outs)):
                mask = homo_mask & mask
                num_node = torch.sum(mask).item()
                if model_idx == 0: num_nodes[homo_idx] += num_node
                out1, label1 = out[mask], labels[mask]
                correct = torch.sum((out1.argmax(dim=-1) == label1)).item()
                results[model_idx][homo_idx] += correct
    
    for i in range(len(num_nodes)):
        if num_nodes[i] == 0:
            num_nodes[i] = 1
    num_nodes = np.expand_dims(num_nodes, axis=0)
    
    results = results / num_nodes

    return results


def find_average_performance_with_std(models, outs_list, masks, homo_masks, labels):
    homo_num = len(homo_masks)
    num_model = len(models) 
    num_split = len(masks)
    

    results = np.zeros([num_model, homo_num, num_split])
    final_results_mean, final_results_std = np.zeros([num_model, homo_num]), np.zeros([num_model, homo_num])

    
    for model_idx, (model, outs) in enumerate(zip(models, outs_list)):
        for homo_idx, homo_mask in enumerate(homo_masks):
            for split_idx, (mask, out) in enumerate(zip(masks, outs)):

                mask = homo_mask & mask
                num_node = torch.sum(mask).item()
                if num_node == 0: continue
                out1, label1 = out[mask], labels[mask]
                correct = torch.sum((out1.argmax(dim=-1) == label1)).item()
                acc = correct / num_node
                results[model_idx][homo_idx][split_idx] = acc
    
    for model_idx in range(len(results)):
        for homo_idx in range(len(results[0])):
            datas = results[model_idx][homo_idx].tolist() 
            datas = [data for data in datas if data > 0]
            # print(len(datas))
            final_results_mean[model_idx][homo_idx] = np.mean(datas)
            final_results_std[model_idx][homo_idx] = np.std(datas)

    return final_results_mean, final_results_std, results



def average_on_group(models, outs, masks, homo_masks, labels, model_index):
    results = find_average_performance(models, outs, masks, homo_masks, labels)
    # num_model, homo_ratio
    mean, std = np.mean(results, 0), np.std(results, 0)

    return mean, std

def difference(graph_mean, mlp_mean):
    mean_gap = graph_mean - mlp_mean

    return mean_gap

def model_distribution(masks, homo_masks):
    homo_num = len(homo_mask)
    
    homo_num_nodes = np.zeros([homo_num])
    for homo_idx, homo_mask in enumerate(homo_masks):
        for split_idx, mask in enumerate(masks):
            mask = homo_mask & mask
            num_node = torch.sum(mask).item()
            homo_num_nodes[homo_idx] += num_node
    
    return homo_num_nodes / np.sum(homo_num_nodes) 


def find_hete_mask_index(homo_masks, intervals):
    # for homo_idx, homo_mask in enumerate(homo_masks):
    intervals = intervals[1:]
    homo_nums, interval_ids = [], []
    for idx, (homo_mask, interval) in enumerate(zip(homo_masks, intervals)):
        if interval <= 0.4:
            homo_nums.append(torch.sum(homo_mask).item())
            interval_ids.append(idx)
    
    max_idx = np.argmax(homo_nums)
    idx = interval_ids[max_idx]
    return idx


def homo_count(homo_masks):
    homo_nums = []
    num_homo = len(homo_masks)
    for idx, homo_mask in enumerate(homo_masks):
        homo_nums.append(torch.sum(homo_mask).item())
    homo_nums = np.array(homo_nums)
    
    homo_ratios = homo_nums / np.sum(homo_nums)
    return homo_nums, homo_ratios


def compare_with_GCN(models, dataset, results, base_model = "GCN"):
    base_model_index = models.index(base_model)
    mean_dict, std_dict = {}, {}
    base_model_accs = np.array(results[base_model_index])
    # num_homo, num_split
    p_values = {}
    for model_index, model in enumerate(models):
        if model_index == base_model_index:
            continue
        # import ipdb; ipdb.set_trace()
        
        comp_model_accs = np.array(results[model_index])
        dis_model_accs = comp_model_accs - base_model_accs
        # num_homo, num_split
        acc_mean = np.mean(dis_model_accs, axis=-1)
        acc_std = np.std(dis_model_accs, axis=-1)
        mean_dict[model] = acc_mean
        std_dict[model] = acc_std
        
        # std_dict[model] = np.abs(np.array(acc_stds[model_index]) - base_model_std) 
    # if dataset not in ["arxiv_old", "IGB_tiny_old"]: 
    difference_plot_single([0.0, 0.2, 0.4, 0.6, 0.8], mean_dict, std_dict, dataset)


def stat_real(base_results, comp_results):
    p_values = []

    for i in range(base_results.shape[0]):
        base_result = base_results[i]
        comp_result = comp_results[i]
        t_statistic, p_value = ttest_rel(base_result, comp_result)
        p_value = round(p_value, 2)
        p_values.append(p_value)


    return p_values

def stat(mean1, std1, num_samples):
    condition1 = np.random.normal(mean1, std1, num_samples).tolist()
    condition2 = np.random.normal(mean1, std[1], num_samples).tolist()
    t_statistic, p_value = ttest_rel(condition1, condition2)
    print(f"t_statistic: {t_statistic:4f}, p_value: {p_value:4f}")
    
    return p_value



def bar_draw(accs, models, homo_ratios, intervals, filename):
    intervals = np.array(intervals[:-1])
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    num_models, num_homos = len(models), len(homo_ratios)
    bar_width = (intervals[1] - intervals[0]) / (num_models + 1)
    for model_idx, model in enumerate(models):
        plt.bar(intervals+ model_idx * bar_width, accs[model_idx], bar_width, color=colors[model_idx], label=models[model_idx]) # ,marker=markers[i]
    plt.legend(frameon=False,loc=4,prop = {'size':12})
    
    plt.savefig(filename)
    plt.clf()



def get_model_performance(outs_list, test_masks, label):
    final_performance = np.zeros([len(outs_list)])
    for model_idx, outs in enumerate(outs_list):
        for idx, test_mask in enumerate(test_masks):
            out = torch.squeeze(outs[idx])
            acc = torch.sum(out[test_mask].argmax(dim=-1) == label[test_mask]) / torch.sum(test_mask)
            final_performance[model_idx] += acc.item()
        
    
    final_performance /= len(outs_list[0])
    
    return final_performance.tolist()
            



def analyze_worse(results_list, best_model_idx):
    num_homo = len(results_list)
    num_model = len(results_list[0]['cc'][0])
    
    final_results = np.zeros([num_homo, num_model])

    for homo_idx, results in enumerate(results_list):
        results = results['cw']

        for model_idx in range(num_model):
            if model_idx != best_model_idx:
                final_results[homo_idx][model_idx] = results[model_idx][best_model_idx]

    return final_results                
                





