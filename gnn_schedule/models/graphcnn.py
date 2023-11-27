import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, learn_eps, device, n_j, n_m):
        """
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        device: which device to use
        """

        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.mlps_precedent = torch.nn.ModuleList()
        self.mlps_succeedent = torch.nn.ModuleList()
        self.mlps_disjunctive = torch.nn.ModuleList()
        self.mlps_node = torch.nn.ModuleList()
        self.batch_norms_precedent = torch.nn.ModuleList()
        self.batch_norms_succeedent = torch.nn.ModuleList()
        self.batch_norms_disjunctive = torch.nn.ModuleList()
        self.batch_norms_node = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps_precedent.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.mlps_succeedent.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.mlps_disjunctive.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.mlps_node.append(MLP(num_mlp_layers, 152, hidden_dim, hidden_dim))
            else:
                self.mlps_precedent.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.mlps_succeedent.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.mlps_disjunctive.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.mlps_node.append(MLP(num_mlp_layers, 264, hidden_dim, hidden_dim))

            self.batch_norms_precedent.append(nn.BatchNorm1d(hidden_dim))
            self.batch_norms_succeedent.append(nn.BatchNorm1d(hidden_dim))
            self.batch_norms_disjunctive.append(nn.BatchNorm1d(hidden_dim))
            self.batch_norms_node.append(nn.BatchNorm1d(hidden_dim))

    def next_layer(self, x, h, layer, adj_tuples, use_eps=False):
        adj_precedent = adj_tuples[0]
        # adj_succeedent = adj_tuples[1]
        adj_disjunctive = adj_tuples[2]
        adj_all = adj_tuples[3]

        # pooling neighboring nodes and center nodes altogether
        pooled_precedent = torch.mm(adj_precedent, h)
        # pooled_succeedent = torch.mm(adj_succeedent, h)
        pooled_disjunctive = torch.mm(adj_disjunctive, h)
        pooled_all = torch.mm(adj_all, h)

        # representation of neighboring and center nodes
        pooled_precedent_rep = self.mlps_precedent[layer](pooled_precedent)
        h_precedent = F.relu(self.batch_norms_precedent[layer](pooled_precedent_rep))
        # pooled_succedent_rep = self.mlps_succeedent[layer](pooled_succeedent)
        # h_succedent = F.relu(self.batch_norms_succeedent[layer](pooled_succedent_rep))
        pooled_disjunctive_rep = self.mlps_disjunctive[layer](pooled_disjunctive)
        h_disjunctive = F.relu(self.batch_norms_disjunctive[layer](pooled_disjunctive_rep))
        h_all = F.relu(pooled_all)

        h_cat = torch.cat([h_precedent, h_disjunctive, h_all, h, x], dim=1)
        pooled_rep = self.mlps_node[layer](h_cat)
        h_k = F.relu(self.batch_norms_node[layer](pooled_rep))

        return h_k

    def forward(self, x, graph_pool, adj):

        graph_pool = graph_pool

        # list of hidden representation at each layer (including input)
        h = x
        for layer in range(self.num_layers - 1):
            h = self.next_layer(x, h, layer, adj)

        h_nodes = h.clone()
        pooled_h = torch.sparse.mm(graph_pool, h)
        return pooled_h, h_nodes
