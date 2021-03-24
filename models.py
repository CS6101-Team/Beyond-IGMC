import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, RGCNConv, global_sort_pool, global_add_pool, SAGEConv
from torch_geometric.utils import dropout_adj
from util_functions import *
import pdb
import time
import torch.nn.init as init

class GNN(torch.nn.Module):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(self, dataset, gconv=SAGEConv, latent_dim=[32, 32, 32, 1],
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(GNN, self).__init__()
        self.regression = regression
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1]))
        self.lin1 = Linear(sum(latent_dim), 128)
        if self.regression:
            self.lin2 = Linear(128, 1)
        else:
            self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# ------ GraphSAGE start -----------
class NeighborAggregator(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


class SageGCN(torch.nn.Module):
    # layer definition
    def __init__(self, input_dim, hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """SageGCN层定义
        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.b = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.b)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.b)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSage(torch.nn.Module):
    # # a base GraphSage class
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )


class DGCNN(GNN):
    # DGCNN from [Zhang et al. AAAI 2018], GCN message passing + SortPooling
    def __init__(self, dataset, gconv=SAGEConv, latent_dim=[32, 32, 32, 1], k=30,
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(DGCNN, self).__init__(
            dataset, gconv, latent_dim, regression, adj_dropout, force_undirected
        )
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums))) - 1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print('k used in sortpooling is:', self.k)
        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_sort_pool(concat_states, batch, self.k)  # batch * (k*hidden)
        x = x.unsqueeze(1)  # batch * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # flatten
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)


class DGCNN_RS(DGCNN):
    # A DGCNN model using RGCN convolution to take consideration of edge types.
    def __init__(self, dataset, gconv=RGCNConv, latent_dim=[32, 32, 32, 1], k=30,
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2,
                 force_undirected=False):
        super(DGCNN_RS, self).__init__(
            dataset,
            SAGEConv,
            latent_dim,
            k,
            regression,
            adj_dropout=adj_dropout,
            force_undirected=force_undirected
        )
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], num_relations, num_bases))

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_sort_pool(concat_states, batch, self.k)  # batch * (k*hidden)
        x = x.unsqueeze(1)  # batch * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # flatten
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)


class IGMC(GraphSage):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    def __init__(self, dataset, gconv=RGCNConv, latent_dim=[32, 32, 32, 32],
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2,
                 force_undirected=False, side_features=False, n_side_features=0,
                 multiply_by=1):
        super(IGMC, self).__init__(
            dataset, GraphSage, latent_dim
        )
        # delete：, regression, adj_dropout, force_undirected (no such parameters in GraphSage)
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i + 1], num_relations, num_bases))
        self.lin1 = Linear(2 * sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)

    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)
