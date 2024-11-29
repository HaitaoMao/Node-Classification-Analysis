


import torch
import torch.linalg as linalg
from torch.distributions import Bernoulli, MultivariateNormal
import torch_geometric
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from torch_scatter import scatter_add

class synthetic_dataset(object):
    def __init__(self, n_points, n_features, dist_sd_ratio, sig_sq=None, nonlinear=False, p=1., q=0.):
        # n_points: number of nodes
        # n_features: feature dimensions
        # sig_sq: the std of feature 
        self.n_points = n_points
        self.n_features = n_features
        self.K = dist_sd_ratio
        self.sig_sq = 1./n_features if sig_sq is None else sig_sq
        # self.sig_sq = 1 if sig_sq is None else sig_sq
        self.p = p
        self.q = q
        self.nonlinear = nonlinear
        # Fix a random pair of orthogonal means.
        u = torch.rand(n_features) # First mean.
        v = torch.rand(n_features) # Second mean.
        if nonlinear:
            v[-1] = -(u[:-1] @ v[:-1]) / u[-1] # Make v orthogonal to u.
        u /= linalg.norm(u)
        v /= linalg.norm(v)
        self.normed_uv = torch.stack((u, v))
        

    def gmm(self, seed):
        K_ = self.K/np.sqrt(2.0)
        u = K_*self.normed_uv[0]
        v = K_*self.normed_uv[1]
        X = torch.zeros((self.n_points, self.n_features))
        y = torch.zeros(self.n_points, dtype=torch.long)
        # Decide class and cluster based on two independent Bernoullis.
        eps = Bernoulli(torch.tensor([0.5]))
        for i in range(self.n_points):
            y[i] = eps.sample()
            # Mean is -mu, mu, -nu or nu based on eps_i and eta_i.
            mean = ((1-y[i])*u + y[i]*v)
            if self.sig_sq > 0:
                cov = torch.eye(self.n_features) * self.sig_sq
                distr = MultivariateNormal(mean, cov)
                X[i] = distr.sample()
            else:
                X[i] = mean
        return Data(x=X, y=y, edge_index=None)

    # The XGMM synthetic data model definition.
    def xgmm(self, seed):
        # Make K = norm(u-v) = sqrt(2)*norm(u). 
        # 调整norm的大小 
        
        K_ = self.K/np.sqrt(2.0)
        u = K_*self.normed_uv[0]
        v = K_*self.normed_uv[1]
        X = torch.zeros((self.n_points, self.n_features))
        y = torch.zeros(self.n_points, dtype=torch.long)
        # Decide class and cluster based on two independent Bernoullis.
        eps = Bernoulli(torch.tensor([0.5]))
        eta = Bernoulli(torch.tensor([0.5]))
        for i in range(self.n_points):
            y[i] = eps.sample()
            cluster = eta.sample()
            # Mean is -mu, mu, -nu or nu based on eps_i and eta_i.
            mean = (2*cluster - 1)*((1-y[i])*u + y[i]*v)
            if self.sig_sq > 0:
                cov = torch.eye(self.n_features) * self.sig_sq
                distr = MultivariateNormal(mean, cov)
                X[i] = distr.sample()
            else:
                X[i] = mean
        return Data(x=X, y=y, edge_index=None)

    def generate_graph(self, seed):
        # The inbuilt function stochastic_blockmodel_graph does not support
        # random permutations of the nodes, hence, design it manually.
        # Use with_replacement=True to include self-loops.
        if self.nonlinear:
            data = self.xgmm(seed)
        else:
            data = self.gmm(seed)
        
        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)


        probs = torch.tensor([[self.p, self.q], [self.q, self.p]], dtype=torch.float)
        row, col = torch.combinations(torch.arange(self.n_points), r=2, with_replacement=True).t()
        mask = torch.bernoulli(probs[data.y[row], data.y[col]], generator=g).to(torch.bool)
        data.edge_index = torch.stack([row[mask], col[mask]], dim=0)
        data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, num_nodes=self.n_points)
        data.edge_value = torch.ones((data.edge_index.size(1),))

        data.edge_index, data.edge_value = torch_geometric.utils.add_remaining_self_loops(data.edge_index, data.edge_value)

        return data
    
    # Generate a dataset from the XCSBM synthetic data model.
    def generate_data(self, seed=None):
        data = self.generate_graph(seed)
        data = self.set_stat(data)

        return data

    def set_stat(self, data):
        data.num_nodes, data.num_hidden = data.x.shape
        data.num_edges = np.max(data.edge_index.shape)

        return data
        
    def generate_split(self, data, seed, ratios):
        # ratios: [train_ratio, validation_ratio], the remaining are test
        num_node, num_hidden = data.x.shape
        num_classes = 2

        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)

        indices = []
        for i in range(num_classes):
            index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
            index = index[torch.randperm(index.size(0), generator=g)]
            indices.append(index)

        train_index = torch.cat([i[:int(ratios[0]*len(i))] for i in indices], dim=0)
        val_index = torch.cat([i[int(ratios[0]*len(i)):int((ratios[0]+ratios[1])*len(i))] for i in indices], dim=0)
        test_index = torch.cat([i[int((ratios[0]+ratios[1])*len(i)):] for i in indices], dim=0)
        test_index = test_index[torch.randperm(test_index.size(0), generator=g)]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes).cuda()
        data.val_mask = index_to_mask(val_index, size=data.num_nodes).cuda()
        data.test_mask = index_to_mask(test_index, size=data.num_nodes).cuda()
    
        return data
        


if __name__ == '__main__':
    n_points = 1000
    dist_sd_ratios = np.geomspace(.15, 1.2*np.sqrt(np.log(n_points)), num=40)

    data_generator = synthetic_dataset(n_points=1000, n_features=100, dist_sd_ratio=0.5, p=0.001, q=0.0005, nonlinear=True)   # , sig_sq=None
    # n_points=1000, n_features=100, dist_sd_ratio=0.5, p=0.001, q=0.0005, nonlinear=True
    # the average degree: 1.75
    # p=0.01, q=0.005
    # the average degree: 8.56
    
    data = data_generator.generate_data(seed=1)
    data = data_generator.generate_split(data, seed=1, ratios=[0.48,0.32])

    row, col = data.edge_index
    deg = scatter_add(data.edge_value, row, dim=0, dim_size=data.num_nodes)
    
    import ipdb; ipdb.set_trace()
    print()