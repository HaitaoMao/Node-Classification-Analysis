
import sklearn
import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, WikiCS, Coauthor, Amazon, CoraFull, Actor, WikipediaNetwork
import torch_geometric.transforms as T
import torch_geometric
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import scipy.sparse as sp
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph import utils
from data_statistic.calculate import compute_homo_ratio_new
from deeprobust.graph.utils import get_train_val_test_gcn, get_train_val_test
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import add_remaining_self_loops, to_undirected, index_to_mask, remove_self_loops, mask_to_index
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch_sparse
import pickle

def get_split(args, dataset, data, seed, index=None, sparse=False):
    # index is  for multiple fixed split
    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None
        
    if args.dataset == 'arxiv' or args.dataset == 'product' :
        # only fix split
        split_idx = dataset.get_idx_split()
        # print(split_idx)
        data.y = torch.squeeze(data.y) ## add this for make y [num, 1] to [num]
        data.train_mask = index_to_mask(split_idx['train'], data.x.shape[0])
        data.test_mask = index_to_mask(split_idx['test'], data.x.shape[0]) ## add this for convenience
        data.val_mask = index_to_mask(split_idx['valid'], data.x.shape[0]) ## add this for convenience
        # Toread,
        # compute_node_adjust_homo(data, args)

        # return data, split_idx

    if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
        # fixed split is utilized by default
        if args.is_fix:
            data = data
        # elif args.num_split:
        #     data = random_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed, num=args.num_fix)
        else:
            data = proportion_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed, proportion=args.ratio_fix)
        # import ipdb; ipdb.set_trace()        
    elif args.dataset == "cs" or args.dataset == "physics":
        # data = random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        # print(f'random split {args.dataset} split {split}')
        print()
        
    elif args.dataset == "computers" or args.dataset == "photo":
        # data = random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        # print(f'random split {args.dataset} split {split}')
        print()

    elif args.dataset in ["Chameleon", "Squirrel"]:
        dataset = get_wiki_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        if args.is_fix:
            data = load_wiki_fix_split(data, args.dataset, seed)
        else:
            data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=int(seed))

    elif args.dataset in ["Cornell", "Texas", "Wisconsin"]:
        dataset = get_WebKB_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=seed)

    elif args.dataset in ["Actor"]:
        dataset = get_Actor_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        data.edge_index = data.edge_index
        if args.is_fix:
            data = load_actor_fix_split(data, args.dataset, seed)
        else:
            data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=int(seed))

    split_idx = {}
    

    # load the new dataset and new data split here
    if args.is_new and args.dataset in ["Cora", "CiteSeer", "PubMed", "photo", "cs"]:
        data = torch.load(f"CPF/dataset/{args.dataset}.pt")
        if args.normalize_features:
            data.x /= torch.norm(data.x, dim=-1, keepdim=True)
        data.edge_index = data.edge_index.to(torch.int64)
        data.train_mask = data.train_masks[seed]
        data.val_mask = data.val_masks[seed]
        data.test_mask = data.test_masks[seed]  
        
    if args.is_new and args.dataset in ["roman_empire", "amazon_ratings", "Chameleon", "Squirrel"]:
        data = torch.load(f"heterophilous_graphs/dataset/{args.dataset}.pt").cpu()
        data.x = torch.where(torch.isnan(data.x), 0, data.x)
        norm = torch.norm(data.x, dim=-1, keepdim=False)
        if args.normalize_features:
            data.x /= torch.unsqueeze(norm, -1)
        norm_mask = (norm != 0)
        
        data.edge_index = data.edge_index.to(torch.int64)
        data.edge_index, _ = torch_geometric.utils.subgraph(norm_mask, data.edge_index, relabel_nodes=True)
        data.x = data.x[norm_mask]
        data.y = data.y[norm_mask]
        data.train_mask = data.train_masks[seed][norm_mask]
        data.val_mask = data.val_masks[seed][norm_mask]
        data.test_mask = data.test_masks[seed][norm_mask]

        

    if args.dataset in ["Penn94", "twitch-gamer", "genius"]:
        data.edge_index = data.edge_index.to(torch.int64)
        data.train_mask = data.train_masks[seed]
        data.val_mask = data.val_masks[seed]
        data.test_mask = data.test_masks[seed]
    
    if args.is_homo:
        with open(f"data_statistic/homo_data/{args.dataset}_homo_ratio.txt", 'rb') as f:
            homo_ratio = pickle.load(f)
            data.homo_ratio = homo_ratio.cuda()
        with open(f"data_statistic/homo_data/{args.dataset}_adjust_homo_ratio.txt", 'rb') as f:
            adjust_homo_ratio = pickle.load(f)
            data.adjust_homo_ratio = adjust_homo_ratio.cuda()
    
    if args.is_ood:
        data = generate_ood_mask(args, data)
        torch.save(data, f"data/ood_data_new/{args.dataset}_new.pt")
 
    if args.is_iid:
        data = equal_random_splits(args, data)

    return data, split_idx


def generate_ood_mask(args, data):
    data = data.cuda()
    homo_datasets = ["Cora", "CiteSeer", "PubMed", "arxiv", "IGB_tiny"]
    hete_datasets = ["twitch-gamer", "Chameleon", "Squirrel", "Actor", "roman_empire", "amazon_ratings"]
    is_homo = True if args.dataset in homo_datasets else False
    num_node = data.x.shape[0]
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
    homo_ratio = homo_ratio / deg

    homo_ratio = compute_homo_ratio_new(data, 2)
    if is_homo:
        mask1 = (homo_ratio <= 0.5)
        mask2 = (homo_ratio >= 0.0)
    else:
        if args.dataset in ["Squirrel"]:
            mask1 = (homo_ratio <= 1.0)
            mask2 = (homo_ratio > 0.4)
        else:
            mask1 = (homo_ratio <= 1.0)
            mask2 = (homo_ratio > 0.5)
    
    test_mask = (mask1 & mask2)
    num_test = torch.sum(test_mask).item()
    train_val_mask = ~test_mask
    train_val_index = mask_to_index(train_val_mask.cpu(), num_nodes)
    num_train_val = train_val_index.shape[0]
    train_val_index =  train_val_index[torch.randperm(num_train_val)]
    num_train = int(num_train_val * 0.8)
    num_val = int(num_train_val - num_train)
    train_mask = index_to_mask(torch.tensor(train_val_index[:num_train]).cuda(), size=num_nodes).cuda()
    val_mask = index_to_mask(torch.tensor(train_val_index[num_train:]).cuda(), size=num_nodes).cuda()

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(f"num_train: {num_train}, num_val: {num_val}, num_test: {num_test}")

    return data

def extre_mask(args, data):
    with open(f"data/extreme_mask/{args.dataset}.txt", "rb") as f:
        data = pickle.load(f)
    data.train_mask = data["train_mask"]    
    data.val_mask = data["val_mask"]    
    data.test_mask = data["test_mask"]    

    return data

def equal_random_splits(args, data):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    # homo_datasets = ["Cora", "CiteSeer", "PubMed", "arxiv", "IGB_tiny"]
    # hete_datasets = ["twitch-gamer", "Chameleon", "Squirrel", "Actor", "roman_empire", "amazon_ratings"]
    # "Chameleon":523  "Squirrel": 1363 
    # "Chameleon":133 , "Squirrel": 341
    # , "roman_empire":17271, "amazon_ratings": 13647  
    # , "roman_empire":4318, "amazon_ratings": 3412
    num_train_dict = {"Cora": 1599, "CiteSeer":1160,  "PubMed":12466, "Chameleon": 1642, "Squirrel": 3709, "arxiv": 85788}
    num_val_dict = {"Cora": 400, "CiteSeer":290,  "PubMed":3117, "Chameleon": 411, "Squirrel": 928, "arxiv": 21447}
    # num_train_dict = {"Cora": 1766, "CiteSeer":1160,  "PubMed":12466, "Chameleon":3709 , "Squirrel": 1363, "arxiv": 85788, "roman_empire":17271, "amazon_ratings": 13647}
    # num_val_dict = {"Cora": 442, "CiteSeer":290,  "PubMed":3117, "Chameleon":928 , "Squirrel": 341, "arxiv": 21447, "roman_empire":4318, "amazon_ratings": 3412}
    g = None
    seed = args.random_seed
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed) #train_test_split_edges
    
    num_train = num_train_dict[args.dataset]
    num_val = num_val_dict[args.dataset]
    num_nodes = data.x.shape[0]
    index = torch.arange(num_nodes).cuda()
    index = index[torch.randperm(index.size(0), generator=g)]

    train_index = index[:num_train]
    val_index = index[num_train:num_train+num_val]
    test_index = index[num_train+num_val:]
    
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data



def get_dataset(args, sparse=False, is_large_com=False, **kwargs):
    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None
    
    if is_large_com:
        transform = T.LargestConnectedComponents()
    

    if args.dataset == "arxiv":
        # import ipdb; ipdb.set_trace()
        dataset = get_ogbn_dataset("ogbn-arxiv", False, transform=transform)
        data = dataset[0]
    elif args.dataset == "product":
        # import ipdb; ipdb.set_trace()
        dataset = get_ogbn_dataset("ogbn-products", False, transform=transform)
        data = dataset[0]

    elif args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
        dataset = get_planetoid_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset == "cs" or args.dataset == "physics":
        dataset = get_coauthor_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset == "computers" or args.dataset == "photo":
        dataset = get_amazon_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset in ["Cornell", "Texas", "Wisconsin"]:
        dataset = get_WebKB_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset in ["Actor"]:
        dataset = get_Actor_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
    
    elif args.dataset in ["Chameleon", "Squirrel"]:
        dataset = get_wiki_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
    elif args.dataset in ["roman_empire", "amazon_ratings"]:
        # here is just a place holder, we re upload the file in another file
        dataset = get_wiki_dataset("Chameleon", args.normalize_features, transform=transform)
        data = dataset[0]

    elif args.dataset in ["Penn94", "twitch-gamer", "genius"]:
        data = torch.load(f"Non_Homophily_Large_Scale/new_data/{args.dataset}.pt")
        if args.normalize_features:
            data.x /= torch.norm(data.x, dim=-1, keepdim=True)
        dataset = None
    # IGB_tiny
    elif args.dataset in ["IGB_tiny"]:
        data = torch.load(f"data/{args.dataset}.pt")
        if args.normalize_features:
            data.x /= torch.norm(data.x, dim=-1, keepdim=True)
        dataset = None    

    else:
        print("wrong")
        exit()

    return dataset, data




def get_transform(normalize_features, transform):
    # import ipdb; ipdb.set_trace()
    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform

    return transform



def largest_connected_components(data, connection='weak'):
    import numpy as np
    import scipy.sparse as sp

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

    num_components, component = sp.csgraph.connected_components(
        adj, connection=connection)

    if num_components <= 1:
        return data

    _, count = np.unique(component, return_counts=True)
    subset = np.in1d(component, count.argsort()[-1:])

    return data.subgraph(torch.from_numpy(subset).to(torch.bool))

def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_coauthor_dataset(name, normalize_features=False, transform=None):
    name_dict = {"cs": "CS"}
    name = name_dict[name]
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Coauthor(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_amazon_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Amazon(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_WebKB_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WebKB(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_Actor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Actor(path)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset


def get_wiki_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WikipediaNetwork(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_ogbn_dataset(name, normalize_features=True, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = PygNodePropPredDataset(name, path)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset


def load_actor_fix_split(data, name, seed):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)

    masks = np.load(f"{path}/raw/film_split_0.6_0.2_{seed}.npz")
    data.train_mask = torch.tensor(masks['train_mask']) > 0
    data.val_mask = torch.tensor(masks['val_mask']) > 0
    data.test_mask = torch.tensor(masks['test_mask']) > 0
    
    return data


def load_wiki_fix_split(data, name, seed):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    name_dict = {"Squirrel": "squirrel", "Chameleon": "chameleon"}
    name2 = name_dict[name]
    masks = np.load(f"{path}/{name2}/geom_gcn/raw/{name2}_split_0.6_0.2_{seed}.npz")
    data.train_mask = torch.tensor(masks['train_mask']) > 0
    data.val_mask = torch.tensor(masks['val_mask']) > 0
    data.test_mask = torch.tensor(masks['test_mask']) > 0
    
    return data


def random_planetoid_splits(data, num_classes, seed, num):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(train_test_split_edges)
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:num] for i in indices], dim=0)
    # print('len(train)', len(train_index))
    rest_index = torch.cat([i[num:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
    return data

def proportion_planetoid_splits(data, num_classes, seed, proportion):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(1)
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:int(proportion*len(i))] for i in indices], dim=0)
    rest_index = torch.cat([i[int(proportion*len(i)):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    print('len(train)', len(train_index))

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:int(0.5*len(rest_index))], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[int(0.5*len(rest_index)):], size=data.num_nodes)
    return data



def random_coauthor_amazon_splits(data, num_classes, seed):
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def random_WebKB_splits(data, num_classes, seed):
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)
    train_index = torch.cat([i[:int(0.48*len(i))] for i in indices], dim=0)
    val_index = torch.cat([i[int(0.48*len(i)):int(0.8*len(i))] for i in indices], dim=0)
    rest_index = torch.cat([i[int(0.8*len(i)):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    
    return data

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]
