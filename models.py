import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

from layers import BiGCNConv, indBiGCNConv, BiSAGEConv, BiGraphConv
from function import BinActive, BinLinear


class BiGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layers, dropout, print_log=True):
        super(BiGCN, self).__init__()

        if print_log:
            print("Create a {:d}-layered Bi-GCN.".format(layers))

        self.layers = layers
        self.dropout = dropout
        self.bn1 = torch.nn.BatchNorm1d(in_channels, affine=False)

        convs = []
        for i in range(self.layers):
            in_dim = hidden_channels if i > 0 else in_channels
            out_dim = hidden_channels if i < self.layers - 1 else out_channels
            if print_log:
                print("Layer {:d}, in_dim {:d}, out_dim {:d}".format(i, in_dim, out_dim))
            convs.append(BiGCNConv(in_dim, out_dim, cached=True, bi=True))
        self.convs = torch.nn.ModuleList(convs)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.bn1(x)

        for i, conv in enumerate(self.convs):
            x = BinActive()(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)

        return F.log_softmax(x, dim=1)


# indGCN and GraphSAGE
class NeighborSamplingGCN(torch.nn.Module):
    def __init__(self, model: str, in_channels, hidden_channels, out_channels, binarize, dropout=0.):
        super(NeighborSamplingGCN, self).__init__()

        assert model in ['indGCN', 'GraphSAGE'], 'Only indGCN and GraphSAGE are available.'
        GNNConv = indBiGCNConv if model == 'indGCN' else BiSAGEConv

        self.num_layers = 2
        self.model = model
        self.binarize = binarize
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GNNConv(in_channels, hidden_channels, binarize=binarize))
        self.convs.append(GNNConv(hidden_channels, out_channels, binarize=binarize))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            if(self.binarize):
                x = x - x.mean(dim=1, keepdim=True)
                x = x / (x.std(dim=1, keepdim=True) + 0.0001)
                x = BinActive()(x)

                x_target = x_target - x_target.mean(dim=1, keepdim=True)
                x_target = x_target / (x_target.std(dim=1, keepdim=True) + 0.0001)
                x_target = BinActive()(x_target)
            # if self.model == 'GraphSAGE':
            #     edge_index, _ = add_self_loops(edge_index, num_nodes=x[0].size(0))
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                if self.binarize:
                    # bn x
                    x = x - x.mean(dim=1, keepdim=True)
                    x = x / (x.std(dim=1, keepdim=True) + 0.0001)
                    x = BinActive()(x)

                    # bn x_target
                    x_target = x_target - x_target.mean(dim=1, keepdim=True)
                    x_target = x_target / (x_target.std(dim=1, keepdim=True) + 0.0001)
                    x_target = BinActive()(x_target)
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all


class SAINT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, binarize):
        super(SAINT, self).__init__()
        self.dropout = dropout
        self.binarize = binarize
        self.conv1 = BiGraphConv(in_channels, hidden_channels, binarize=self.binarize)
        self.conv2 = BiGraphConv(hidden_channels, hidden_channels, binarize=self.binarize)
        # if self.binarize:
        #     self.lin = BinLinear(2 * hidden_channels, out_channels)
        # else:
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x0, edge_index, edge_weight=None):
        if self.binarize:
            x0 = x0 - x0.mean(dim=1, keepdim=True)
            x0 = x0 / (x0.std(dim=1, keepdim=True) + 0.0001)
            x0 = BinActive()(x0)
        x1 = self.conv1(x0, edge_index, edge_weight)
        if not self.binarize:
            x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        if self.binarize:
            x2 = BinActive()(x1)
        else:
            x2 = x1
        x2 = self.conv2(x2, edge_index, edge_weight)
        if not self.binarize:
            x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = torch.cat([x1, x2], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)