from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, to_undirected, index_to_mask, remove_self_loops, homophily, k_hop_subgraph, to_dense_adj
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_sparse
import sklearn
import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, WikiCS, Coauthor, Amazon, CoraFull, Actor, WikipediaNetwork
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import math
from scipy.stats import entropy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_graph_hete(data):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    num_edges = np.max(edge_index.shape)

    num_homo = torch.sum(data.y[edge_index[0]] - data.y[edge_index[1]] == 0)

    return (num_homo / num_edges).item()

def compute_homo_mask(data, args, is_hete):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    label = F.one_hot(data.y, num_classes=num_labels)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    deg = torch.unsqueeze(deg, dim=-1)
    ego_matrix = deg * label

    neighor_matrix = torch_sparse.spmm(edge_index, edge_value, num_nodes, num_nodes, label)
    results = ego_matrix - neighor_matrix
    results = results[torch.arange(num_nodes), data.y]
    results = 1 - torch.div( results, torch.squeeze(deg))
    mask_begin = results >= 0
    
    ends = [0.2, 0.4, 0.6, 0.8, 1.0]
    if is_hete:
        ends = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    masks = []
    for end in ends:
        mask_end = (results <= end)

        mask = mask_begin * mask_end
        masks.append(mask)
        mask_begin = (results >= end)
    
    return results, masks, ends 




def compute_homo_mask_new(data, args, is_hete):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    
    edge_homo_value = (data.y[row] == data.y[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    results = homo_ratio / deg

    mask_begin = (results >= 0)
    
    ends = [0.2, 0.4, 0.6, 0.8, 1.0]
    if is_hete:
        ends = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    masks = []
    for end in ends:
        mask_end = (results <= end)

        mask = mask_begin * mask_end
        masks.append(mask)
        mask_begin = (results >= end)
    
    return results, masks, ends 


def compute_node_adjust_homo_new(data, mask=None):
    data.y = torch.squeeze(data.y)
    edge_index, _ = remove_self_loops(data.edge_index)
    num_edges = edge_index.shape[1]
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    
    edge_homo_value = (data.y[row] == data.y[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    deg = torch.where(deg == 0, 1, deg)
    original_ratio = homo_ratio / deg    
    
    # the next part, we focus on the random ratio then
    # first, we compute the probability on the edge end with each class
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    if mask != None: 
        deg = deg[mask]
        label = data.y[mask]
        num_edges = torch.sum(deg) / 2
        original_ratio = original_ratio[mask]    

    else:   
        label = data.y    
    marginal_prob = scatter_add(deg, label, dim=0) / num_edges

    random_ratio = marginal_prob[label]
    result = (original_ratio - random_ratio) / (1 - random_ratio)

    return result




def compute_homo_ratio(data):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    
    edge_homo_value = (data.y[row] == data.y[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    deg = torch.where(deg == 0, 1, deg)
    results = homo_ratio / deg
    

    return results


def compute_homo_ratio_new(data, num_hop=1, is_self_loop=False):
    num_node = data.x.shape[0]
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    if is_self_loop:
        edge_index, edge_value = add_self_loops(edge_index, edge_value)
        
    for i in range(num_hop - 1):
        edge_index, edge_value = torch_sparse.spspmm(edge_index, edge_value, edge_index, edge_value, num_node, num_node, num_node)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    
    edge_homo_value = (data.y[row] == data.y[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    deg = torch.where(deg == 0, 1, deg)
    results = homo_ratio / deg

    return results


def compute_higher_order_homo_ratio(data, num_order):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    
    origin_edge_index = edge_index.clone()
    origin_edge_value = edge_value.clone()
    for i in range(num_order - 1):
        edge_index, edge_value = torch_sparse.spspmm(edge_index, edge_value, origin_edge_index, origin_edge_value, num_nodes, num_nodes, num_nodes, coalesced=True)
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    
    edge_homo_value = (data.y[row] == data.y[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    deg = torch.where(deg == 0, 1, deg)
    results = homo_ratio / deg
    
    return results



def to_direct(edge_index, edge_value=None):
    reverse_edge_index = torch.stack([edge_index[1], edge_index[0]])
    mask = torch.stack([torch.eq(edge_index, reverse_edge_index[:, i]).all(dim=0)
                        for i in range(edge_index.shape[1])])
    reverse_mask = torch.cat([mask, mask])

    directed_edge_index = torch.where(reverse_mask, reverse_edge_index, edge_index)

    return directed_edge_index



def compute_edge_label_inform(data, args):
    edge_index = data.edge_index
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    num_edges = np.max(edge_index.shape)
    row, col = edge_index[0], edge_index[1]
    
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    marginal_prob = scatter_add(deg, data.y, dim=0) / num_edges

    labels_out_fake = data.y[edge_index[0]]  * num_labels
    # for each labels, I can have a unique labeling space
    labels_together = data.y[edge_index[1]] + labels_out_fake
    joint_count = scatter_add(torch.ones(labels_together.shape[0]), labels_together, dim=0 )
    joint_prob = joint_count / num_edges 
    joint_prob += 1e-7
    marginal_prob += 1e-7

    entropy = torch.sum(marginal_prob * torch.log(marginal_prob))
    condition_entropy = torch.sum(joint_prob * torch.log(joint_prob))        
    

    return (2 - condition_entropy / entropy).item()



def adjust_homo_ratio(data, args):
    # https://arxiv.org/pdf/2209.06177v1.pdf   page 4
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]

    num_edges = np.max(data.edge_index.shape)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    
    homo_edge_ratio = compute_hete_edge(data, args)

    class_degree = scatter_add(deg, data.y, dim=0)
    class_degree_square = torch.square(class_degree)
    degree_stats = torch.sum(class_degree_square) / (num_edges * num_edges) # 4 * 

    unnormalized_ratio = homo_edge_ratio - degree_stats

    normalized_ratio = unnormalized_ratio / (1 - degree_stats)
    

    return normalized_ratio.item()


def compute_node_adjust_homo(data, args):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    num_edges = np.max(edge_index.shape)

    label = F.one_hot(data.y, num_classes=num_labels)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    deg = torch.unsqueeze(deg, dim=-1)
    ego_matrix = deg * label

    neighor_matrix = torch_sparse.spmm(edge_index, edge_value, num_nodes, num_nodes, label)
    results = ego_matrix - neighor_matrix
    results = results[torch.arange(num_nodes), data.y]
    original_ratio = 1 - torch.div(results, torch.squeeze(deg))

    
    # the next part, we focus on the random ratio then
    # first, we compute the probability on the edge end with each class

    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    marginal_prob = scatter_add(deg, data.y, dim=0) / num_edges

    random_ratio = marginal_prob[data.y]
    
    result = (original_ratio - random_ratio) / (1 - random_ratio)

    return result
    

def compute_hete_edge(data, args):
    edge_index, _ = remove_self_loops(data.edge_index)
    num_edges = np.max(data.edge_index.shape)
    row, col = edge_index[0], edge_index[1]
    in_label = data.y[row]
    out_label = data.y[col]
    
    homo_ratio = torch.sum(in_label == out_label) / in_label.shape[0]
    
    return homo_ratio.item()

def category_node_by_label(data):
    labels = data.y.tolist()
    dict_with_idx = defaultdict(list)
    for pos, ele in enumerate(labels):
        dict_with_idx[ele].append(pos)
    return dict_with_idx

def find_neighbor_hist(edge_index, node, labels):
    row, _ = edge_index[0], edge_index[1]
    edge_index = remove_self_loops(edge_index)[0]
    adj = to_dense_adj(edge_index).squeeze(dim = 0)
    node_neighbor = adj[node]
    neigh_pos = torch.where(node_neighbor != 0)
    row = neigh_pos[0]
    corr_l = labels[neigh_pos[1]]
    num_of_labels = int(labels.max()) + 1
    hist = torch.zeros((len(node), num_of_labels))
    for i, _ in enumerate(neigh_pos[0]):
        hist[row[i]][corr_l[i]] += 1
    return hist


def CCNS(data, args):
    def cross_class_neighborhood_similarity(data, c, c_prime):
        """
            CCNS, def 2 of Yao's paper
        """
        node_label_cat = category_node_by_label(data)
        score = 0
        class_c_nodes = node_label_cat[c]
        class_c_prime_nodes = node_label_cat[c_prime]
        edge_index = data.edge_index
        cc = find_neighbor_hist(edge_index, class_c_nodes, data.y)
        # ipdb.set_trace()
        ccp = find_neighbor_hist(edge_index, class_c_prime_nodes, data.y)

        cc = F.normalize(cc, p = 2., dim = 1)
        ccp = F.normalize(ccp, p = 2, dim = 1)

        score = torch.matmul(cc, ccp.T)
        return score.sum() / len(class_c_nodes) / len(class_c_prime_nodes)
    
    num_class = torch.max(data.y).item() + 1
    results = np.zeros([num_class, num_class])
    for i in tqdm(range(num_class)):
        for j in range(num_class):
            results[i][j] = cross_class_neighborhood_similarity(data, i, j)
    
    return results.tolist() 


def compare_distribution(datas, interval_value = 0.1):
    max_value = -1000
    min_value = 1000
    for key in datas.keys():
        datas[key] = list(filter(lambda x: not math.isnan(x), datas[key]))
        max_value = max(np.max(datas[key]), max_value)
        min_value = min(np.min(datas[key]), min_value)
    
    num_max_interval, num_min_interval = (max_value // interval_value), (min_value // interval_value)
    if max_value % interval_value != 0: num_max_interval += 1
    if min_value % interval_value != 0: num_min_interval += 1
    new_max_value, new_min_value = num_max_interval * interval_value, num_min_interval * interval_value

    max_value = new_max_value if max_value > 0 else -new_max_value
    min_value = new_min_value if min_value > 0 else -new_min_value

    num_interval = int((max_value - min_value) // interval_value) + 1 
    bins = [min_value + i * interval_value for i in range(num_interval)]
    bins_indices = {}
    for key in datas.keys():
        index = np.digitize(datas[key], bins)
        max_index = np.max(index)
        index = np.where(index == max_index, max_index-1, index)
        bins_indices[key] = index
    
    probs = {}
    for key in bins_indices.keys():
        data = torch.tensor(bins_indices[key])
        num = data.shape[0]
        value = torch.ones([num])
        prob = scatter_add(value, data, dim=0) / num
        probs[key] = prob.numpy().tolist()

    
    return bins, probs


def probs_correlation(probs, name):
    keys = list(probs.keys())
    num = len(keys)
    datas = np.zeros([num, num]).astype(float)

    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            prob1, prob2 = probs[key1], probs[key2]
            prob1 = np.array(prob1) + 1e-9
            prob2 += np.array(prob2) + 1e-9
            KL_score = entropy(prob1, prob2)
            datas[i][j] = KL_score
    for i in range(num):
        datas[i][i] = 1
    df = pd.DataFrame(columns=['data1', 'data2', 'sim'])

    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):            
            df.loc[len(df.index)] = [key1, key2, datas[i][j]]
    df.to_csv(f'data_statistic/results/{name}_class_corr.csv')
    img = df.pivot("data1", "data2", "sim")
    sns.heatmap(img, annot=True)
    plt.savefig(f"data_statistic/results/{name}_class_corr.png")
    plt.clf()


def compat_matrix(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0,:], edge_index[1,:]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max()+1
    H = torch.zeros((c,c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k,:], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H


def class_homo(edge_index, label):
    label = label.squeeze()
    c = label.max()+1
    H = compat_matrix(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k,k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1

    return val

