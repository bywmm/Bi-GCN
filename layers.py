from torch_geometric.nn.conv import MessagePassing, SAGEConv, GraphConv
from torch_geometric.utils import add_self_loops, degree
import torch
import math
from torch.nn import Linear
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter_add

from function import BinLinear


class BiGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=True, bi=False):
        super(BiGCNConv, self).__init__(aggr="add")
        self.cached = cached
        self.bi = bi
        if bi:
            self.lin = BinLinear(in_channels, out_channels)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None

    def forward(self, x, edge_index):
        x = self.lin(x)

        if not self.cached or self.cached_result is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

            # Compute normalization
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            # normalization of each edge
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        x = self.propagate(edge_index,size=(x.size(0), x.size(0)),
                              x=x, norm=norm)
        return x

    def message(self, x_j, norm):

        # Normalize node features
        return norm.view(-1, 1) * x_j


class indBiGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, binarize=False):
        super(indBiGCNConv, self).__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.binarize = binarize
        if binarize:
            self.lin = BinLinear(in_channels, out_channels)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        # shape of x: [N, in_channels]
        # shape of edge_index: [2, E]
        if torch.is_tensor(x):
            x = (x, x)

        out = self.propagate(edge_index, x=x, norm=None)
        out = self.lin(out)

        return out


class BiSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False, binarize=False,
                 **kwargs):
        super(BiSAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        if binarize:
            self.lin_rel = BinLinear(in_channels, out_channels)
            self.lin_root = BinLinear(in_channels, out_channels)
        else:
            self.lin_rel = Linear(in_channels, out_channels, bias=True)
            self.lin_root = Linear(in_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index):
        """"""

        if torch.is_tensor(x):
            x = (x, x)

        out = self.propagate(edge_index, x=x)
        out = self.lin_rel(out) + self.lin_root(x[1])

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class BiGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, binarize=False, aggr='add', bias=True,
                 **kwargs):
        super(BiGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if binarize:
            self.lin = BinLinear(in_channels, out_channels)
            self.lin_root = BinLinear(in_channels, out_channels)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
            self.lin_root = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_root.reset_parameters()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        h = self.lin(x)
        edge_index, edge_weight = self.norm(edge_index, x.size(0))
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def propagate(self, edge_index, size, x, h, edge_weight):

        # message and aggregate
        if size is None:
            size = [x.size(0), x.size(0)]

        adj = torch_sparse.SparseTensor(row=edge_index[0], rowptr=None, col=edge_index[1], value=edge_weight,
                     sparse_sizes=torch.Size(size), is_sorted=True)  # is_sorted=True
        out = torch_sparse.matmul(adj, h, reduce='sum')
        # out = torch.cat([out, self.lin_root(x)], dim=1)
        out = out + self.lin_root(x)
        return out


# Initialization functions
def zeros_init(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones_init(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def glorot_init(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
