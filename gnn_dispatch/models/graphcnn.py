import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn_dispatch.models.mlp import MLP

# import sys
# sys.path.append("models/")

"""
class Attention(nn.Module):
    def __init__(self): super(Attention, self).__init__()

    def forward(self, g_fea, candidates_feas):
        attention_score = torch.mm(candidates_feas, g_fea.t())
        attention_weight = F.softmax(attention_score, dim=0)
        representation_weighted = torch.mm(attention_weight.t(), candidates_feas)
        feas_final = torch.cat((g_fea, representation_weighted), dim=1)
        return feas_final
"""


class GraphCNN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        learn_eps,
        device,
    ):
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

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        # sum pooling
        pooled = torch.mm(Adj_block, h)

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, adj, use_eps=False):

        # pooling neighboring nodes and center nodes altogether
        pooled = torch.mm(adj, h)

        if use_eps and self.ep:
            pooled = pooled + (1 + self.eps[layer]) * h

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, x, graph_pool, adj):

        graph_pool = graph_pool

        # list of hidden representation at each layer (including input)
        h = x
        for layer in range(self.num_layers - 1):
            h = self.next_layer(h, layer, adj=adj)

        h_nodes = h.clone()
        pooled_h = torch.sparse.mm(graph_pool, h)
        return pooled_h, h_nodes
