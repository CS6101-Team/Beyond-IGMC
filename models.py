import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, global_sort_pool, global_add_pool, global_max_pool
from torch_geometric.utils import dropout_adj
from util_functions import *
import pdb
import time


class GNN(torch.nn.Module):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 1], 
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(GNN, self).__init__()
        self.regression = regression
        self.adj_dropout = adj_dropout 
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1]))
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
 

class DGCNN(GNN):
    # DGCNN from [Zhang et al. AAAI 2018], GCN message passing + SortPooling
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 1], k=30, 
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(DGCNN, self).__init__(
            dataset, gconv, latent_dim, regression, adj_dropout, force_undirected
        )
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
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
            GCNConv, 
            latent_dim, 
            k, 
            regression, 
            adj_dropout=adj_dropout, 
            force_undirected=force_undirected
        )
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))

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


class IGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    def __init__(self, dataset, gconv=RGCNConv, latent_dim=[32, 32, 32, 32], 
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1):
        super(IGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2*sum(latent_dim)+n_side_features, 128)

    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch

        # print('\n\n')
        # print('1x:')
        # print(list(x.shape))

        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []
        # print('2x:')
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            # print(list(x.shape))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        # print('1concat:')
        # print(list(concat_states.shape))

        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        # print('users')
        # print(list(users.shape))
        # print(list(concat_states[users].shape))
        # print('items')
        # print(list(items.shape))
        # print(list(concat_states[items].shape))
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        # print('3x:')
        # print(list(x.shape))
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        # print('4x:')
        # print(list(x.shape))

        x = F.relu(self.lin1(x))

        # print('5x:')
        # print(list(x.shape))

        x = F.dropout(x, p=0.5, training=self.training)

        # print('6x:')
        # print(list(x.shape))

        x = self.lin2(x)

        # print('7x:')
        # print(list(x.shape))

        if self.regression:
            # print('8a:')
            # print(list((x[:, 0] * self.multiply_by).shape))
            return x[:, 0] * self.multiply_by
        else:
            # print('8b:')
            # print(list((F.log_softmax(x, dim=-1)).shape))
            return F.log_softmax(x, dim=-1)

# MaxPool instead of Concat

class MaxPoolIGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    def __init__(self, dataset, gconv=RGCNConv, latent_dim=[32, 32, 32, 32], 
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1):
        super(MaxPoolIGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))
        self.lin1 = Linear(int(2*sum(latent_dim) / 4), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear((2*sum(latent_dim)+n_side_features) / 4, 128)
        self.maxpool1d = nn.MaxPool1d(4)

    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch

        # print('\n\n')
        # print('1x:')
        # print(list(x.shape))

        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []
        # print('2x:')
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            # print(list(x.shape))
            concat_states.append(x)

        x = torch.stack(concat_states, 2)
        # print('3x:')
        # print(list(x.shape))

        x = self.maxpool1d(x)
        # print('4x:')
        # print(list(x.shape))

        maxpool_states = x.squeeze(-1)
        # print('5x:')
        # print(list(maxpool_states.shape))

        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        # print('users')
        # print(list(users.shape))
        # print(list(maxpool_states[users].shape))
        # print('items')
        # print(list(items.shape))
        # print(list(maxpool_states[items].shape))
        x = torch.cat([maxpool_states[users], maxpool_states[items]], 1)
        # print('6x:')
        # print(list(x.shape))

        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        # print('7x:')
        # print(list(x.shape))

        x = F.relu(self.lin1(x))

        # print('8x:')
        # print(list(x.shape))

        x = F.dropout(x, p=0.5, training=self.training)

        # print('9x:')
        # print(list(x.shape))

        x = self.lin2(x)

        # print('10x:')
        # print(list(x.shape))

        if self.regression:
            # print('11a:')
            # print(list((x[:, 0] * self.multiply_by).shape))
            return x[:, 0] * self.multiply_by
        else:
            # print('11b:')
            # print(list((F.log_softmax(x, dim=-1)).shape))
            return F.log_softmax(x, dim=-1)

# LSTM Attention Aggregation IGMC

class LSTMAttentionIGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    def __init__(self, dataset, gconv=RGCNConv, latent_dim=[32, 32, 32, 32], 
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1, input_size = 32, hidden_size = 32, dropout = 0.5):
        super(LSTMAttentionIGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2*sum(latent_dim)+n_side_features, 128)
        
        self.hidden_size = 16
        self.bi_lstm = torch.nn.LSTM(input_size = 32, hidden_size = self.hidden_size, bidirectional = True)
        self.lin3 = Linear(32, 1).cuda()
        self.softmax = torch.nn.Softmax(dim = 0)
        self.num_layers = len(latent_dim)


    def forward(self, data):

        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch

        # print('\n\n')
        # print('1x:')
        # print(list(x.shape))

        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []
        # print('2x:')
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            # print(list(x.shape))
            concat_states.append(x)
        x = torch.stack(concat_states, 0)
        # print('3x:')
        # print(list(x.shape))

        # Bi LSTM

        output, _ = self.bi_lstm(x)
        # print('1output:')
        # print(list(output.shape))
        batch_size = list(output.shape)[1]
        # # Reference: https://stackoverflow.com/questions/50856936/taking-the-last-state-from-bilstm-bigru-in-pytorch/50914946
        output = output.view(self.num_layers, batch_size, 2, self.hidden_size)   # (seq_len, batch_size, num_directions, hidden_size)
        output_forward = output[:, :, 0, :]   # (seq_len, batch_size, hidden_size)
        output_backward = output[:, :, 1, :]
        output_concatenated = torch.cat([output_forward, output_backward], dim = 2)
        # print('2(output, output_forward, output_backward, output_concatenated):')
        # print(list(output.shape))
        # print(list(output_forward.shape))
        # print(list(output_backward.shape))
        # print(list(output_concatenated.shape))

        # Dense + Softmax
        # Reshape
        dense_input = output_concatenated.permute((1, 0, 2))
        dense_output = self.lin3(dense_input)
        # Average across 0th dimension
        dense_output = torch.mean(dense_output, dim = 0).reshape((self.num_layers, ))

        # print('1Dense:')
        # print(list(dense_input.shape))
        # print(list(dense_output.shape))

        # print('1Softmax:')
        softmax_output = self.softmax(dense_output)
        # print(softmax_output)

        # Multiply weights to concat_states
        concat_states = [concat_states[i] * softmax_output[i] for i in range(self.num_layers)]

        # The rest is the same as IGMC

        concat_states = torch.cat(concat_states, 1)
        # print('1concat:')
        # print(list(concat_states.shape))

        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        # print('users')
        # print(list(users.shape))
        # print(list(concat_states[users].shape))
        # print('items')
        # print(list(items.shape))
        # print(list(concat_states[items].shape))
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        # print('4x:')
        # print(list(x.shape))
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        # print('5x:')
        # print(list(x.shape))

        x = F.relu(self.lin1(x))

        # print('6x:')
        # print(list(x.shape))

        x = F.dropout(x, p=0.5, training=self.training)

        # print('7x:')
        # print(list(x.shape))

        x = self.lin2(x)

        # print('8x:')
        # print(list(x.shape))

        if self.regression:
            # print('9a:')
            # print(list((x[:, 0] * self.multiply_by).shape))
            return x[:, 0] * self.multiply_by
        else:
            # print('9b:')
            # print(list((F.log_softmax(x, dim=-1)).shape))
            return F.log_softmax(x, dim=-1)

     
