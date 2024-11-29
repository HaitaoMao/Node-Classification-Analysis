import torch
import torch.nn as nn
import torch_sparse
import torch_geometric
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import mask_to_index, to_undirected, to_dense_adj, add_remaining_self_loops, dense_to_sparse, to_scipy_sparse_matrix, remove_self_loops, add_self_loops, k_hop_subgraph, contains_self_loops
from torch_scatter import scatter_add
from scipy.sparse import coo_matrix
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std, scatter
from scipy.sparse.linalg import svds

class Collapser(nn.Module):
    # implmentation of four NC laws, and make some observations
    def __init__(self, args):
        super(Collapser, self).__init__()
        self.args = args
        self.num_class = args.num_class
    
    def sample_per_class(self, label):        
        return [torch.sum(label == i).item() for i in range(self.num_class)]
        
        
    def class_mean(self, hidden, label):
        # import ipdb; ipdb.set_trace()

        # label = torch.unsqueeze(label, dim=-1)
        class_mean = scatter_mean(hidden, label, dim=0)
        return class_mean

    def mean(self, hidden):    # , mask=None        
        return torch.mean(hidden, dim=0)

    def class_inner_std1(self, hidden, label):
        label = torch.unsqueeze(label, dim=-1)
        class_inner_std = scatter_std(hidden, label)
        # num_hidden, num_class

        return class_inner_std


    def class_outer_std1(self, hidden, label, mask=None):
        class_mean = self.class_mean(hidden, label)
        
        class_outer_std = class_mean.std(dim=-1)
        # num_hidden, num_class

        return class_outer_std

        
    def class_inner_std2(self, hidden, label):
        # print(hidden.shape)
        num_sample, num_hidden = hidden.shape
        class_mean = self.class_mean(hidden, label)
        # num_class, num_hidden
        class_inner_std = 0
        for class_idx in range(self.args.num_class):
            idxs = (label == class_idx).nonzero(as_tuple=True)[0]
            hidden_class = hidden[idxs, :]
            # num_sample, num_hidden
            mean_class = class_mean[class_idx, :].reshape([1, -1])
            # 1, num_hidden
            hidden_gap = hidden_class - mean_class
            # num_sample, num_hidden
            # import ipdb; ipdb.set_trace()

            conv = torch.bmm(hidden_gap.unsqueeze(-1), hidden_gap.unsqueeze(1))
            # num_sample, num_hidden, num_hidden
            class_inner_std += torch.sum(conv, dim=0)
            
            # num_hidden, num_hidden

        class_inner_std /= num_sample # * num_hidden * num_hidden 
        return class_inner_std

    def class_outer_std2(self, hidden, label, mask=None):
        num_sample, num_hidden = hidden.shape
        class_mean = self.class_mean(hidden, label)
        # num_class, num_hidden

        global_mean = torch.mean(hidden, dim=0).reshape([1, -1])
        # 1, num_hidden
        class_gap = class_mean - global_mean
        # num_class, num_hidden

        class_outer_std = torch.mean(torch.bmm(class_gap.unsqueeze(-1), class_gap.unsqueeze(1)), dim=0) # / self.args.num_class
        # class_outer_std = torch.matmul(class_gap.T, class_gap) / (self.args.num_class) #  * num_hidden * num_hidden
        # num_hidden, num_hidden

        return class_outer_std
        
    # TODO: still not know whether std or the variance, maybe need to double
    def within_class_variability_collapse(self, hidden, label):
        class_inner_std = self.class_inner_std2(hidden, label).cpu().numpy()
        class_out_std = self.class_outer_std2(hidden, label).cpu().numpy()
        

        eigvec, eigval, _ = svds(class_out_std, k=self.args.num_class-1)
        inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
        return np.trace(class_inner_std @ inv_Sb)
        
        within_class_variability_collapse = torch.matmul(class_inner_std, torch.pinverse(class_out_std)) / self.args.num_class
                
        return torch.trace(within_class_variability_collapse).item() # / self.args.num_class


    def class_inner_value(self, hidden, label):
        class_inner_std = self.class_inner_std2(hidden, label)
        class_inner_std = torch.trace(class_inner_std).item() / class_inner_std.shape[0]

        return class_inner_std

    def class_outer_value(self, hidden, label):
        class_outer_std = self.class_outer_std2(hidden, label)
        class_outer_std = torch.trace(class_outer_std).item() / class_outer_std.shape[0]

        return class_outer_std

    def checkSelfDuality(self, hidden, label, weight):
        # ||W-M_||
        if weight.size(0) > weight.size(1):
            weight = weight.permute([1, 0])
        # weight: num_class, num_hidden
        
        class_mean = self.class_mean(hidden, label)
        # num_class, num_hidden
        mean = self.mean(hidden).reshape([1, -1])
        # 1, num_hidden
        normed_weight = weight / torch.norm(weight, 'fro')
        # num_class, num_hidden
        mean_gap = class_mean - mean
        # num_class, num_hidden
        normed_mean_gap = mean_gap / torch.norm(normed_weight, 'fro')
        # num_class, num_hidden
        
        # import ipdb; ipdb.set_trace()
        return torch.square(torch.norm(normed_weight - normed_mean_gap)).item()   

    def coherence(self, V, C): 
        G = V.T @ V
        G += torch.ones([C,C]).cuda() / (C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (C*(C-1))
    

    def check_simplex_data(self, hidden, label):
        
        class_mean = self.class_mean(hidden, label)
        # num_class, num_hidden
        mean = self.mean(hidden).reshape([1, -1])
        # 1, num_hidden
        mean_gap = class_mean - mean
        # num_class, num_hidden
        mean_gap = mean_gap.T
        
        coherence_hidden = self.coherence(mean_gap / torch.norm(mean_gap, dim=0, p=2, keepdim=True), self.args.num_class)

        return coherence_hidden


    def check_simplex_weight(self, weight):
        if weight.size(0) > weight.size(1):
            weight = weight.permute([1, 0])
        # weight: num_class, num_hidden
        weight = weight.T
        # num_hidden, num_class
        coherence_weight = self.coherence(weight / torch.norm(weight, dim=0, p=2, keepdim=True), self.args.num_class)
         
        return coherence_weight


    def NNC(self, hidden_train, hidden_test, label, predict):
        net_pred = self.net_pred(predict, label)   
        
        class_mean = self.class_mean(hidden_train, label)
        # num_class, num_hidden 
        
        NCC_scores = torch.stack([torch.norm(hidden_test - torch.unsqueeze(class_mean[i, :], dim=0), dim=-1)  for i in range(self.args.num_class)])
        
        NCC_pred = torch.argmin(NCC_scores.T, dim=-1)
        
        return torch.sum(NCC_pred==net_pred).item()


    def data_equinorm(self, hidden, label):
        
        class_mean = self.class_mean(hidden, label)
        # num_class, num_hidden
        mean = self.mean(hidden).reshape([1, -1])
        class_gap = class_mean - mean
        # num_class, num_hidden
        class_norm = torch.norm(class_gap, dim=-1, p=2).cpu().numpy()

        data_equinorm = np.std(class_norm) / np.mean(class_norm)

        return data_equinorm
                
    def weight_equinorm(self, hidden, weight):
        if weight.size(0) > weight.size(1):
            weight = weight.permute([1, 0])
        weight_norm = torch.norm(weight, dim=-1, p=2).cpu().numpy()
        weight_equinorm = np.std(weight_norm) / np.mean(weight_norm)
    
        return weight_equinorm
    

    def net_pred(self, predict, label):
        net_pred = torch.argmax(predict)

        return net_pred

    def net_correct(self, net_pred, data, mask):
        if mask: label = label[mask]
        
        net_correct = torch.sum(net_pred ==  label).item()

        return net_correct


    def smv(self, hidden):
        num_samples = hidden.shape(0)
        hidden = hidden / torch.norm(hidden, dim=-1, keepdims=True)


    def dimension_singular_value(self, hidden, epsilon=1e-4):
        c = self.dimension_covariance(hidden, is_norm=True)
        
        '''
        eigvec, eigval, _ = svds(class_out_std, k=self.args.num_class-1)
        inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
        return np.trace(class_inner_std @ inv_Sb)
        '''
        
        _, singular_value, _ = torch.svd(c, compute_uv=False)

        log_singular_value = torch.log(singular_value)
        
        log_singular_value[torch.isinf(log_singular_value)] = 0
        
        rank = torch.sum(singular_value > epsilon).item()

        nesum = torch.max(singular_value) / torch.sum(singular_value)

        return log_singular_value, rank, nesum.item()
        
    
    
    def dimension_covariance(self, hidden, is_norm=True):
        num_hidden = hidden.size(1)

        hidden = hidden.T
        mean_hidden = torch.mean(hidden, 1)
        hiddenm = hidden - mean_hidden.reshape([-1, 1])
        # num_hidden, num_sample
        # x_i - \bar{x}
        c = torch.matmul(hiddenm, hiddenm.T)
        # num_hidden, num_hidden
        # \sum_{i=1}^N (x_i-\bar{x})(y_i-\bar{y})
        
        if is_norm: 
            normed_factor = torch.sum(torch.square(hiddenm), dim=-1)
            # num_hidden  \sum_{i=1}^N (x_i-\bar{x})^2
            normed_factor = torch.sqrt(torch.unsqueeze(normed_factor, dim=-1) * torch.unsqueeze(normed_factor, dim=0))
            c /= (normed_factor + 1e-5)


        return c        

        #singular value recovery rate can be good
        

    def corr(self, hidden, is_norm=True):
        num_hidden = hidden.size(1)
        c = self.dimension_covariance(hidden, is_norm=is_norm)
        # import ipdb; ipdb.set_trace()
        c *= (1-torch.eye(num_hidden).cuda())
        corr = torch.sum(c).item() / (num_hidden * (num_hidden - 1))

        corr = np.clip(corr, -1.0, 1.0)

        return corr
    
        
    def oversmooth_metric(self, hidden):
        num_nodes, num_hidden = hidden.shape
        # will normalized and unnoamrlize really matter in our exp result
        normed_hidden = hidden / torch.norm(hidden, dim=-1, keepdim=True)
        # import ipdb; ipdb.set_trace()
        smooth_matrix = torch.matmul(normed_hidden, normed_hidden.T) * (1-torch.eye(num_nodes).cuda())
        node_smooth = 1 / (num_nodes - 1) *  torch.sum(smooth_matrix, dim=-1)
        graph_smooth = torch.mean(node_smooth)

        return node_smooth, graph_smooth
        
        # distance to oversmooth matrix
        # too desnes, we will not discuss here.
        # edge_index, edge_value = add_self_loops(edge_index, edge_value, fill_value=self.a, num_nodes=num_nodes)
        # row, col = edge_index
        # num_edges = row.shape[0]
        # deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)

        # deg_new = torch.sqrt(deg + 1)
        

    def get_pairwise_sim(self, x):
        try:
            x = x.detach().cpu().numpy()
        except:
            pass

        if sp.issparse(x):
            x = x.todense()
            x = x / (np.sqrt(np.square(x).sum(1))).reshape(-1,1)
            x = sp.csr_matrix(x)
        else:
            x = x / (np.sqrt(np.square(x).sum(1))+1e-10).reshape(-1,1)
        # x = x / x.sum(1).reshape(-1,1)
        try:
            dis = euclidean_distances(x)
            return 0.5 * (dis.sum(1)/(dis.shape[1]-1)).mean()
        except:
            return -1


    def loss_corr(self, x, nnodes=None):
        if nnodes is None:
            nnodes = x.shape[0]
        idx = np.random.choice(x.shape[0], int(np.sqrt(nnodes)))
        x = x[idx]
        x = x - x.mean(0)
        cov = x.t() @ x
        I_k = torch.eye(x.shape[1]).cuda() / np.sqrt(x.shape[1])
        loss = torch.norm(cov/torch.norm(cov) - I_k)
        return loss
