import pickle
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import os
import numpy as np
import scipy.sparse as sp
from pylatexenc.latexencode import utf8tolatex
import math




def dist_plot_single(datas, xname, save_name='check'):
    num_x_tick = 5
    x_tick_interval = 1 / num_x_tick
    x_ticks =  [0 + i * x_tick_interval for i in range(num_x_tick + 1)]    

    dataset_dict = {"IGB_tiny": "IGB", "amazon_ratings": "Amazon"}
    fig, ax1 = plt.subplots(figsize=(6,5))
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    for idx, key in  enumerate(datas.keys()):
        data = datas[key]
        data = [x for x in data if not math.isnan(x)]  
                     
        sns.kdeplot(data, fill=True, label=key, color=colors[idx],cbar_ax=ax1, linewidth=2, alpha=0.3)
    
    ax1.grid(False) # ,loc=(0.02, 0.6)
    ax1.set_xlim([0,1])
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    plt.ylabel('Density', fontsize=26, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    plt.xlabel("$h_{node}$", fontsize=26,  fontfamily='serif') # fontweight='bold',
    plt.xticks(np.array(x_ticks), [f"{x_tick:.1f}" for x_tick in x_ticks], fontsize=18, fontfamily='serif')
    plt.yticks(fontsize=18, fontfamily='serif')
    
    plt.savefig(f'results/distribution/{key}_distribution.png', bbox_inches='tight' ) # 
    plt.savefig(f'results/distribution/{key}_distribution.pdf', bbox_inches='tight' ) # 
    plt.close()



datasets = ["Cora", "CiteSeer", "PubMed", "arxiv", "Chameleon", "amazon_ratings", "Squirrel", "Actor", "twitch-gamer", "IGB_tiny"] 
node_keys = ["node_homo_ratios"]
node_homo_dict, node_adjust_homo_dict = {}, {}
record_datasets = ["Squirrel", "Chameleon"]

node_homo_dict = {}
for index, dataset in enumerate(record_datasets):
    with open(f"homo_data_old/{dataset}_homo_ratio.txt", "rb") as f:
        node_homo_dict[dataset] = pickle.load(f).numpy()

for key in node_homo_dict.keys():
    homo_dict = {key: node_homo_dict[key]}
    dist_plot_single(homo_dict, xname="check", is_adjust=False)

