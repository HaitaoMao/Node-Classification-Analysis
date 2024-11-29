
import torch
import torch.nn as nn
# from torch_sparse import SparseTensor, matmul
import torch_sparse
import torch_geometric
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import mask_to_index, to_undirected, to_dense_adj, add_remaining_self_loops, dense_to_sparse, to_scipy_sparse_matrix, remove_self_loops, add_self_loops, k_hop_subgraph, contains_self_loops
from torch_scatter import scatter_add
from scipy.sparse import coo_matrix
import numpy as np
import torch.nn.functional as F


class smooth_loss(torch.nn.Module):
    def __init__(self, args, predefine_group=None, a=None, m1=None, m2=None, m3=None, e1=None, e2=None, e3=None):
        super(smooth_loss, self).__init__()
        self.args = args
        

        predefine_dict = {
            "A": {"a":0 , "m1":1, "m2":0, "m3":0, "e1":0, "e2":0, "e3":0},
            "D-A": {"a":1 , "m1":-1, "m2":0, "m3":1, "e1":0, "e2":0, "e3":0},
            "D+A": {"a":1, "m1":1, "m2":0, "m3":1, "e1":0, "e2":0, "e3":0},
            "I-D-1A": {"a":0, "m1":-1, "m2":1, "m3":0, "e1":-1, "e2":0, "e3":0},
            "I-D2AD2": {"a":1, "m1":0, "m2":-1, "m3":1, "e1":0, "e2":-0.5, "e3":-0.5},
            "D2AD2": {"a":0 , "m1":1, "m2":0, "m3":0, "e1":-0.5, "e2":-0.5, "e3":1},
            "D-1A": {"a":0, "m1":1, "m2":0, "m3":0, "e1":-1, "e2":0, "e3":0},
        }
        if predefine_group:
            predefine_params = predefine_dict[predefine_group]
        self.a, self.m1, self.m2, self.m3, self.e1, self.e2, self.e3 = predefine_params["a"], predefine_params["m1"], predefine_params["m2"], predefine_params["m3"], predefine_params["e1"], predefine_params["e2"], predefine_params["e3"]
        
        if a: self.a = a 
        if m1: self.m1 = m1 
        if m2: self.m2 = m2 
        if m3: self.m3 = m3 
        if e1: self.e1 = e1 
        if e2: self.e2 = e2 
        if e3: self.e3 = e3 
        
    def forward_check(self, data, hidden, is_analyze_norm, mask=None, edge_weight=None, improved=False,dtype=None, order=1):
        
        edge_index = data.edge_index
        edge_index, _ = remove_self_loops(edge_index)
        num_nodes = hidden.shape[0]
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)
        
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=self.a, num_nodes=num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        

        left_deg = self.m2 * deg.pow(self.e2)
        left_deg[left_deg == float('inf')] = 0
        right_deg = deg.pow(self.e3)
        right_deg[right_deg == float('inf')] = 0
        self_deg = deg.pow(self.e1)
        self_deg[self_deg == float('inf')] = 0
        

        diag_edge_index = torch.stack([torch.arange(num_nodes).long(),torch.arange(num_nodes).long()]).reshape([2, -1]).cuda()
        identify_edge_value = torch.ones(num_nodes).cuda()
        normalized_edge_value = left_deg[row] * edge_weight * right_deg[col]

        final_edge_index = torch.cat((edge_index, diag_edge_index, diag_edge_index), dim=-1)
        final_edge_value = torch.cat((normalized_edge_value, identify_edge_value * self.m3, self_deg * self.m1), dim=-1)
        
        final_edge_index, final_edge_value = torch_sparse.coalesce(final_edge_index, final_edge_value, m=num_nodes, n=num_nodes, op='sum')
                
        if mask != None:
            mask_idx = mask_to_index(mask)
            _, final_edge_index, _, new_edge_mask = k_hop_subgraph(mask_idx, 1, final_edge_index)
            final_edge_value = final_edge_value[new_edge_mask]
        
        num_node = self.args.num_node if mask == None else torch.sum(mask).item()
        
        if is_analyze_norm:
            new_hidden = hidden / torch.norm(hidden, dim=-1, keepdim=True)
        else: 
            new_hidden = hidden

        for i in range(order - 1):
            final_edge_index_record, final_edge_value_record = final_edge_index.clone(), final_edge_value.clone()
            final_edge_index, final_edge_value = torch_sparse.spspmm(final_edge_index, final_edge_value, final_edge_index_record, final_edge_value_record, num_nodes, num_nodes, num_nodes)
        
        reg = torch.matmul(new_hidden.T, torch_sparse.spmm(final_edge_index, final_edge_value, num_nodes, num_nodes, new_hidden))
    
        reg = torch.trace(reg) / num_node
        
        return reg
        
    def feature_preserve(self, origin_feature, hidden_feature, is_analyze_norm):
        if is_analyze_norm:
            norm_origin_feature = origin_feature / torch.norm(origin_feature, keepdim=True)
            norm_hidden_feature = hidden_feature / torch.norm(hidden_feature, keepdim=True)
            res = torch.mean(torch.square(torch.norm(norm_origin_feature - norm_hidden_feature, p="fro", dim=-1)))


        else:
            res = torch.mean(torch.square(torch.norm(hidden_feature - origin_feature, p="fro", dim=-1)))
        
        return res
        
        

    def forward_dense(self, data, hidden):
        edge_index = gcn_norm(data.edge_index, add_self_loops=True)
        # edge_index = dense_to_sparse(edge_index)
        edge_index = torch_sparse.SparseTensor(row=edge_index[0][1], col=edge_index[0][0],
                    value=edge_index[1], sparse_sizes=(self.args.num_node, self.args.num_node)).cuda()
        Ax = torch_sparse.matmul(edge_index, hidden)
        # A2x = matmul(edge_index, Ax)
        # xtAx = torch.matmul(x.T, A2x)
        xtAx = torch.matmul(hidden.T, Ax)
        xtx = torch.matmul(hidden.T, hidden)
        xtLx = xtx - xtAx
        reg = torch.trace(xtLx) / 2708

        return reg


    def forward(self, data, hidden):
        # https://programtalk.com/python-more-examples/torch_sparse.SparseTensor/
        # adj = torch_geometric.transforms.to_sparse_tensor(data.edge_index)
        sparse_adj = torch_sparse.SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                    value=torch.ones(data.edge_index.shape[1]).cuda(), sparse_sizes=(self.args.num_node, self.args.num_node)).cuda()
        # index = torch.range(0,self.args.num_node-1)
        # index = torch.stack([torch.range(0,self.args.num_node-1), torch.range(0,self.args.num_node-1)])
        # value = torch.ones(self.args.num_node)
        index = torch.arange(0,self.args.num_node).long().cuda()
        # sparse_diag = torch_sparse.SparseTensor(row=index, col=index,
                    # value=torch.ones(self.args.num_node).cuda(), sparse_sizes=(self.args.num_node, self.args.num_node)).cuda()
        
        # sparse_diag.mul_(self.a)
        # import ipdb; ipdb.set_trace()
        tmp_diag = torch_sparse.SparseTensor(row=index, col=index,
                    value=torch.ones(self.args.num_node).cuda() * self.a, sparse_sizes=(self.args.num_node, self.args.num_node)).cuda()
        adj = sparse_adj + tmp_diag
        diags = torch_sparse.sum(adj, dim=1)        
        # no relationship with graph
        identity_matrix = torch_sparse.SparseTensor(row=index, col=index,
                    value=self.m3 * torch.ones(self.args.num_node).cuda(), sparse_sizes=(self.args.num_node, self.args.num_node)).cuda()
        
        left_diagnoal_matrix = torch_sparse.SparseTensor(row=index, col=index,
                    value= self.m2 * (diags ** self.e2), sparse_sizes=(self.args.num_node, self.args.num_node)).cuda()
        right_diagnoal_matrix = torch_sparse.SparseTensor(row=index, col=index,
                    value= diags ** self.e3, sparse_sizes=(self.args.num_node, self.args.num_node)).cuda()
        
        normalized_adjacent_matrix = torch_sparse.matmul(torch_sparse.matmul(left_diagnoal_matrix, adj), right_diagnoal_matrix)  
        # (q_diags.mm(temp_adj)).mm(r_diags)
        
        diagnoal_matrix = torch_sparse.SparseTensor(row=index, col=index,
                    value= self.m1 * diags ** self.e1, sparse_sizes=(self.args.num_node, self.args.num_node)).cuda()
        
        # TODO: higher order implementation

        structrure_matrix = identity_matrix + normalized_adjacent_matrix + diagnoal_matrix

        reg = torch.matmul(hidden.T, torch_sparse.matmul(structrure_matrix, hidden))
        
        reg = torch.trace(reg) / self.args.num_node

        return reg
        

        
        
        
    
     
        