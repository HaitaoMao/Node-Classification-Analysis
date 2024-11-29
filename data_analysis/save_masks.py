import torch
import pickle
from utils.data_utils import * 
from utils.args import *

def load_and_save():
    args = generate_args()
    dataset, data = get_dataset(args)
    data, _ = get_split(args, dataset, data, 0)
    vars(args)["num_node"] = data.x.shape[0]
    vars(args)["num_feat"] = data.x.shape[1]
    vars(args)["num_class"] = max(data.y).item() + 1
    
    print(args.dataset)
    one_split_name = ["arxiv", "product", "IGB_tiny"]
    five_split_name = ["Penn94", "twitch-gamer", "genius"] 
    split_num = 10
    if args.dataset in one_split_name: split_num = 1
    if args.dataset in five_split_name: split_num = 5
    train_masks, val_masks, test_masks, homo_ratios, adjust_homo_ratios = [], [], [], [], []
    for split_seed in range(split_num):
        data, _ = get_split(args, dataset, data, split_seed)  # , index=split_idx
        
        train_masks.append(data.train_mask.cpu().numpy())
        val_masks.append(data.val_mask.cpu().numpy())
        test_masks.append(data.test_mask.cpu().numpy())
    
    mask_dict = {"train": train_masks, "test": test_masks, "val": val_masks, "label": data.y.cpu().numpy() }
    with open(f"output_analyze/mask/{args.dataset}.txt", "wb") as f:
        pickle.dump(mask_dict, f)
    # import ipdb; ipdb.set_trace()
    print()


def save_homo_masks(datasets, homo_mask_list, adjust_homo_mask_list, interval, is_old):
    if not is_old:
        for dataset, homo_mask, adjust_homo_mask in zip(datasets, homo_mask_list, adjust_homo_mask_list):
            with open(f"output_analyze/homo_mask/{dataset}_{interval}.txt", "wb") as f:
                pickle.dump(homo_mask, f)

            with open(f"output_analyze/homo_mask/adjust_{dataset}_{interval}.txt", "wb") as f:
                pickle.dump(adjust_homo_mask, f)
    else:
        for dataset, homo_mask, adjust_homo_mask in zip(datasets, homo_mask_list, adjust_homo_mask_list):
            with open(f"output_analyze/homo_mask_old/{dataset}_{interval}.txt", "wb") as f:
                pickle.dump(homo_mask, f)

            with open(f"output_analyze/homo_mask_old/adjust_{dataset}_{interval}.txt", "wb") as f:
                pickle.dump(adjust_homo_mask, f)