This is the official implementation for [Demystifying structural disparity in graph neural networks: Can one size fit all?](https://proceedings.neurips.cc/paper_files/paper/2023/file/74f1edadbdf495e7258ee8db7b1d3acd-Paper-Conference.pdf) accepted by NeurIPS'23. A summary can be found at [here](https://haitaomao.github.io/categories/ResearchSummary/#section134).


## Abstract

Recent studies on Graph Neural Networks(GNNs) provide both empirical and theoretical evidence supporting their effectiveness in capturing structural patterns on both homophilic and certain heterophilic graphs. Notably, most real-world homophilic and heterophilic graphs are comprised of a mixture of nodes in both homophilic and heterophilic structural patterns, exhibiting a structural disparity. However, the analysis of GNN performance with respect to nodes exhibiting different structural patterns, e.g., homophilic nodes in heterophilic graphs, remains rather limited. In the present study, we provide evidence that Graph Neural Networks(GNNs) on node classification typically perform admirably on homophilic nodes within homophilic graphs and heterophilic nodes within heterophilic graphs while struggling on the opposite node set, exhibiting a performance disparity. We theoretically and empirically identify effects of GNNs on testing nodes exhibiting distinct structural patterns. We then propose a rigorous, non-i.i.d PAC-Bayesian generalization bound for GNNs, revealing reasons for the performance disparity, namely the aggregated feature distance and homophily ratio difference between training and testing nodes. Furthermore, we demonstrate the practical implications of our new findings via (1) elucidating the effectiveness of deeper GNNs; and (2) revealing an over-looked distribution shift factor on graph out-of-distribution problem and proposing a new scenario accordingly.


- Backward contains all the codes related to the MLP-based models
- Forward contains all the codes related to the GNN-based models
- Tranfer contains all the codes for GLNN

Run the code with: `python3 main_backup.py --dataset Cora --algo_name SGCNet --expmode transductive`.

### Requirements

Dependencies (with python >= 3.9):
Main dependencies are

pytorch==1.13

torch_geometric==2.2.0

torch-scatter==2.1.1+pt113cpu

torch-sparse==0.6.17+pt113cpu

torch-spline-conv==1.2.2+pt113cpu


Example commands to install the dependencies in a new conda environment (tested on a Linux machine without GPU).

```
conda create --name ss python=3.9
conda activate ss
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch_geometric
```


For GPU installation (assuming CUDA 11.8): 

```
conda create --name ss python=3.9
conda activate ss
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-sparse -c pyg
conda install pyg -c pyg
```

### Analsysis experiment 

To run experiments

```
cd data_analysis
conda activate ss
python runners/run_analysis.py --mode 1
```

## Cite us

If you found this work useful, please cite our paper

```
@article{mao2024demystifying,
  title={Demystifying structural disparity in graph neural networks: Can one size fit all?},
  author={Mao, Haitao and Chen, Zhikai and Jin, Wei and Han, Haoyu and Ma, Yao and Zhao, Tong and Shah, Neil and Tang, Jiliang},
  journal={Advances in neural information processing systems},
  volume={36},
  year={2023}
}
```
