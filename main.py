"""
    To be merged later;
"""
# import wandb
import optuna
from helper.args import *
from helper.data_utils import *
from helper.hete_utils import generate_graph as generate_graph_syn
from helper.hete_utils import generate_base_features
from helper.hete_utils import new_edge_addition, save_pyg_data, local_hetero_edge_addition, hetero_edge_addition
from backward.main_backward_new import main_backward
from forward.main_forward_new import main_forward
from transfer.main_distill_backup import main_transfer
import torch_geometric
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import itertools
import os.path as osp
import logging 
import torch


logging.basicConfig(level=logging.INFO)

def test_acc(res, data):
    return (res[0].reshape(-1).cuda()[data.test_mask] == data.y.cuda()[data.test_mask]).sum() / torch.sum(data.test_mask)

def test_acc_res(res, data):
    return (res[0].reshape(-1).cuda()[data.test_mask] == data.y.cuda()[data.test_mask]).sum() / torch.sum(data.test_mask)


def main():
    seeds = [1, 3, 5, 7, 11] # ,  13, 17, 19, 23, 29
    args = generate_args()
    if args.generate_graph:
        generate_graph(args)
    elif args.expmode == 'inductive':
        inductive_run(args, seeds)
    elif args.expmode == 'transductive':
        fix_param_run(args, seeds)




def generate_graph(args):
    if args.generate_mode == 'sitao':
        with open(args.generation_config, 'r') as f:
                exec(f.read())
        generation_config = locals()['generation']
        generate_graph_syn(
            generation_config['num_class'],
            generation_config['num_node_total'],
            generation_config['degree_intra'],
            generation_config['num_graph'],
            generation_config['graph_type'],
            generation_config['edge_homos'],
            generation_config['base_dir']
        )
        _, data = get_dataset(args)
        for i in range(generation_config['num_graph']):
            base_features = generate_base_features(data, generation_config['num_node_total'], 
                                'feature',
                                generation_config['num_node_total'] // generation_config['num_class'],
                                generation_config['num_class'])
            data_dir = f"{generation_config['base_dir']}/features/{args.dataset}"
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            torch.save(base_features, f"{data_dir}/{args.dataset}_{i}.pt")
    else:
        _, data = get_dataset(args)
        if args.is_new and args.dataset in ["Cora", "CiteSeer", "PubMed", "photo", "cs"]:
            data = torch.load(f"/mnt/home/haitaoma/Graph-smooth/code/CPF/dataset/{args.dataset}.pt")
            data.edge_index = data.edge_index.to(torch.int64)
        if args.dataset == 'Cora':
            orig_data = deepcopy(data)
            for i in range(10):
                num_classes = orig_data.y.max().item() + 1
                mask = 'train'
                if mask == 'train':
                    ratio = 1.0
                    unit = 200
                else:
                    ratio = 0.05
                    unit = 100
                mode = 'tohete' 
                orig_mask = mask
                if i == 0:
                    orig_data = local_hetero_edge_addition(orig_data, ratio = ratio, edges=0, dataset = args.dataset, mask=mask, mode = mode)
                else:
                    orig_data = local_hetero_edge_addition(orig_data, ratio = ratio, edges=unit, node_label_cat=orig_data.original_index, dataset = args.dataset, mask=mask, mode = mode)
                save_pyg_data(orig_data, i * unit, ratio, args.dataset, orig_mask)
        elif args.dataset == 'PubMed':
            orig_data = deepcopy(data)
            for i in range(10):
                mask = 'train'
                if mask == 'train':
                    ratio = 1.0
                    unit = 300
                else:
                    ratio = 0.05
                    unit = 300
                mode = 'tohete' 
                orig_mask = mask
                if i == 0:
                    orig_data = local_hetero_edge_addition(orig_data, ratio = ratio, edges=0, dataset = args.dataset, mask=mask, mode = mode)
                else:
                    orig_data = local_hetero_edge_addition(orig_data, ratio = ratio, edges=unit, node_label_cat=orig_data.original_index, dataset = args.dataset, mask=mask, mode = mode)
                save_pyg_data(orig_data, i * unit, ratio, args.dataset, orig_mask)
        else:
            orig_data = deepcopy(data)
            for i in range(10):
                mask = 'test'
                mode = 'tohomo'
                orig_mask = mask
                if mask == 'train':
                    ratio = 1.0
                    edge_unit = 1500
                else:
                    ratio = 0.15
                    edge_unit = 100
                if i == 0:
                    orig_data = local_hetero_edge_addition(orig_data, ratio = ratio, edges=0, dataset = args.dataset, mask=mask, mode = mode)
                else:
                    orig_data = local_hetero_edge_addition(orig_data, ratio=ratio, edges=edge_unit, node_label_cat=orig_data.original_index, dataset = args.dataset, mask = mask, mode=mode)
                mask = torch.zeros(orig_data.x.shape[0], dtype=torch.bool)
                total_sampled_node_idxs = torch.LongTensor(list(itertools.chain(*orig_data.original_index.values())))
                mask[total_sampled_node_idxs] = True
                save_pyg_data(orig_data, i * edge_unit, ratio, args.dataset, orig_mask)





def fix_param_run(args, seeds):
    dataset, odata = get_dataset(args)
    acc_list = []
    odata.edge_index, _ = torch_geometric.utils.remove_self_loops(torch_geometric.utils.to_undirected(odata.edge_index))
    split_seeds = np.array(range(10))
    best_acc_across_split = -1
    all_preds = []
    all_res = []
    # ipdb.set_trace()
    for split_idx in range(args.num_split):
        split_seed = split_seeds[split_idx].item()
        data, _ = get_split(args, dataset, odata, split_seed)  
        data.edge_index, _ = torch_geometric.utils.remove_self_loops(torch_geometric.utils.to_undirected(data.edge_index))
        data = data.cuda()
        vars(args)["num_node"] = data.x.shape[0]
        vars(args)["num_feat"] = data.x.shape[1]
        vars(args)["num_class"] = max(data.y).item() + 1
        vars(args)["split_seed"] = split_seed
        for random_seed in seeds:
            args.random_seed = random_seed
            set_seed_config(random_seed)
            vars(args)["split_seed"] = split_seed
            vars(args)["num_node"] = data.x.shape[0]
            vars(args)["num_feat"] = data.x.shape[1]
            vars(args)["num_class"] = max(data.y).item() + 1
            if args.teacher != 'No':
                acc_teacher, acc_student, out_student, pred_student, _ = main_transfer(data, args)
                acc = acc_student if args.is_distill else acc_teacher
                all_preds.append(pred_student)    
                acc_list.append(acc)
                #acc_list.append(acc)
                #all_preds.append(pred)
                all_res.append(out_student)
                res = out_student
                pred = pred_student
            else:
                if args.train_schema == "forward":
                    acc, res, pred, _, best_model = main_forward(data, args)
                elif args.train_schema == "backward":
                    acc, res, pred, _ = main_backward(data, args)
                    best_model = None
                acc_list.append(acc)
                all_preds.append(pred)
                all_res.append(res)
            if best_acc_across_split < acc:
                best_acc_across_split = acc
    acc = torch.sum(all_res[-1][data.test_mask].argmax(-1) == data.y[data.test_mask]).item() / torch.sum(data.test_mask).item()
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    logging.info(f'acc: {acc_mean}')
    logging.info(f'acc_std: {acc_std}')
    logging.info(f'acc_log: {acc_list}')
    if args.best_run:
        if not args.is_synthetic:
            logging.info(f"Best run: {args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-{args.expmode}")
            save_best(args, osp.join(args.best_path, f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-args.pkl"))
            save_best(all_preds, osp.join(args.best_path, f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-pred.pkl"))
            save_best(all_res, osp.join(args.best_path, f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-res.pkl"))
        else:
            logging.info(f"Best run: {args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-{args.expmode}")
            save_best(args, osp.join(args.best_path, f"{args.algo_name}-{args.change}-args.pkl"))
            save_best(all_preds, osp.join(args.best_path, f"{args.algo_name}-{args.change}-pred.pkl"))
            save_best(all_res, osp.join(args.best_path, f"{args.algo_name}-{args.change}-res.pkl"))
    else:
        save_best(args, osp.join(args.save_path, f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-args.pkl"))
        save_best(all_preds, osp.join(args.save_path, f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-pred.pkl"))
        save_best(all_res, osp.join(args.save_path, f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-res.pkl"))
    # ipdb.set_trace()
    print(args)
    return acc_mean
    # return best_model

def inductive_run(args, seeds):
    dataset, odata = get_dataset(args)
    acc_list = []
    odata.edge_index, _ = torch_geometric.utils.remove_self_loops(torch_geometric.utils.to_undirected(odata.edge_index))
    split_seeds = np.array(range(10))
    all_preds = []
    all_res = []
    ind_list = []
    for split_idx in range(args.num_split):
        split_seed = split_seeds[split_idx].item()
        data, _ = get_split(args, dataset, odata, split_seed)  
        vars(args)["num_node"] = data.x.shape[0]
        vars(args)["num_feat"] = data.x.shape[1]
        vars(args)["num_class"] = max(data.y).item() + 1
        data.edge_index, _ = torch_geometric.utils.remove_self_loops(torch_geometric.utils.to_undirected(data.edge_index))
        data = data.cuda()
        vars(args)["split_seed"] = split_seed
        indices = graph_split(data.train_idx, data.val_idx, data.test_idx, args.inductive_rate, split_seed)
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        data.obs_idx_train = obs_idx_train
        data.obs_idx_val = obs_idx_val
        data.obs_idx_test = obs_idx_test
        data.idx_obs = idx_obs.sort().values
        data.idx_test_ind = idx_test_ind

        for random_seed in seeds:
            args.random_seed = random_seed
            set_seed_config(random_seed)
            vars(args)["split_seed"] = split_seed
            vars(args)["num_node"] = data.x.shape[0]
            vars(args)["num_feat"] = data.x.shape[1]
            vars(args)["num_class"] = max(data.y).item() + 1
            if args.teacher != 'No':
                acc_teacher, acc_student, out_student, pred_student, ind_test_acc = main_transfer(data, args)
                acc = acc_student if args.is_distill else acc_teacher
                all_preds.append(pred_student)    
                acc_list.append(acc)
                all_res.append(out_student)
                ind_list.append(ind_test_acc)
                continue
            if args.train_schema == "forward":
                acc, res, pred, ind_acc, best_model = main_forward(data, args)
            elif args.train_schema == "backward":
                acc, res, pred, ind_acc = main_backward(data, args)
            acc_list.append(acc)
            all_res.append(res)
            all_preds.append(pred)
            ind_list.append(ind_acc)
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    logging.info(f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}")
    logging.info(f"test_acc: {acc_mean}, test_acc_std: {acc_std}")
    # ipdb.set_trace()
    ind_acc_mean, ind_acc_std = np.mean(ind_list), np.std(ind_list)
    logging.info(f"ind_test_acc: {ind_acc_mean}, ind_test_acc_std: {ind_acc_std}")
    idx = {'obs_idx_train':data.obs_idx_train, 'obs_idx_val':data.obs_idx_val, 'obs_idx_test':data.obs_idx_test, 'idx_test_ind':data.idx_test_ind}
    if args.best_run:
        logging.info(f"Best run: {args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-{args.expmode}")
        save_best(args, osp.join(args.best_path, "inductive", f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-args.pkl"))
        save_best(all_preds, osp.join(args.best_path, "inductive", f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-pred.pkl"))
        save_best(all_res, osp.join(args.best_path, "inductive", f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-res.pkl"))
        save_best(idx, osp.join(args.best_path, "inductive", f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-mask.pkl"))
    
    else:
        save_best(args, osp.join(args.save_path, "inductive", f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-args.pkl"))
        save_best(all_preds, osp.join(args.save_path, "inductive", f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-pred.pkl"))
        save_best(all_res, osp.join(args.save_path, "inductive", f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-res.pkl"))
        save_best(idx, osp.join(args.save_path, "inductive", f"{args.exp_name}-{args.algo_name}-{args.dataset}-{args.num_split}-{args.is_new}-mask.pkl"))
    print(args)

def sweep_run(trial, args, seeds, param_f):
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed'] or 'Cora' in args.dataset:
        mode = 'normal'
    elif args.dataset == 'arxiv' or args.dataset == 'IGB_tiny':
        mode = 'arxiv'
    else:
        mode = 'hete'
    params = param_f(trial, args.algo_name, mode)    
    dataset, odata = get_dataset(args)
    args = update_parameter(args, params)
    acc_list = []
    odata.edge_index, _ = torch_geometric.utils.remove_self_loops(torch_geometric.utils.to_undirected(odata.edge_index))
    split_seeds = np.array(range(10))
    best_acc_across_split = -1
    ind_list = []
    seeds = [1]
    for split_idx in range(args.sweep_split):
        split_seed = split_seeds[split_idx].item()
        data, _ = get_split(args, dataset, odata, split_seed)  
        data.edge_index, _ = torch_geometric.utils.remove_self_loops(torch_geometric.utils.to_undirected(data.edge_index))
        data = data.cuda()
        vars(args)["split_seed"] = split_seed
        vars(args)["num_node"] = data.x.shape[0]
        vars(args)["num_feat"] = data.x.shape[1]
        vars(args)["num_class"] = max(data.y).item() + 1
        if args.expmode == 'inductive':
            indices = graph_split(data.train_idx, data.val_idx, data.test_idx, args.inductive_rate, split_seed)
            obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
            data.obs_idx_train = obs_idx_train
            data.obs_idx_val = obs_idx_val
            data.obs_idx_test = obs_idx_test
            data.idx_obs = idx_obs.sort().values
            data.idx_test_ind = idx_test_ind
        for random_seed in seeds:
            set_seed_config(random_seed)
            args.random_seed = random_seed
            set_seed_config(random_seed)
            vars(args)["split_seed"] = split_seed
            vars(args)["num_node"] = data.x.shape[0]
            vars(args)["num_feat"] = data.x.shape[1]
            vars(args)["num_class"] = max(data.y).item() + 1
            if args.teacher != 'No':
                acc_teacher, acc_student, out_student, pred_student, ind_acc = main_transfer(data, args)
                acc = acc_student if args.is_distill else acc_teacher
                acc_list.append(acc)
                ind_list.append(ind_acc)
                continue
            if args.train_schema == "forward":
                acc, res, pred, ind_acc, best_model = main_forward(data, args)
            elif args.train_schema == "backward":
                acc, res, pred, ind_acc = main_backward(data, args)
            acc_list.append(acc)
            ind_list.append(ind_acc)
            if best_acc_across_split < acc:
                best_acc_across_split = acc    
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    logging.info(f"test_acc: {acc_mean}, test_acc_std: {acc_std}")
    ind_acc_mean, ind_acc_std = np.mean(ind_list), np.std(ind_list)
    logging.info(f"ind_test_acc: {ind_acc_mean}, ind_test_acc_std: {ind_acc_std}")
    if args.expmode == 'transductive':
        return acc_mean
    else:
        return ind_acc_mean


def gen(dataset):
    params = {}
    params["single"] = False
    args = generate_args()
    args = update_parameter(args, params)
    args = check_parameter(args)
    args = generate_prefix(args)
    vars(args)['dataset'] = dataset
    vars(args)['is_new'] = 1
    vars(args)['is_fix'] = 1
    vars(args)['expmode'] = 'transductive'
    _, data = get_dataset(args)
    if args.is_new and args.dataset in ["Cora", "CiteSeer", "PubMed", "photo", "cs"]:
        data = torch.load(f"/mnt/home/haitaoma/Graph-smooth/code/CPF/dataset/{args.dataset}.pt")
        data.edge_index = data.edge_index.to(torch.int64)
    orig_data = deepcopy(data)
    total_edges = data.edge_index.shape[1] // 2
    for i in range(11):
        mask = torch.ones(orig_data.x.shape[0], dtype=torch.bool)
        if i == 0:
            K = 0
            orig_data = hetero_edge_addition(orig_data, K = 0, mode = 'tohete')
        else:
            K = int(i * 2 / 10 * total_edges)
            if dataset == 'Cora':
                orig_data = hetero_edge_addition(orig_data, K = int(2 / 10 * total_edges), mode = 'tohete')
            elif dataset == 'Squirrel':
                orig_data = hetero_edge_addition(orig_data, K = int(2 / 10 * total_edges), mode = 'tohomo')
        filename = osp.join("/mnt/home/haitaoma/Graph-smooth/code/synthetic_data/full", f"full-syn-{dataset}-{K}.pt")
        torch.save(orig_data, filename)



    



if __name__ == '__main__':
    main()
    #gen('Cora')
    # gen('Squirrel')