from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
import torch
import math
import torch_sparse
from torch_scatter import scatter,scatter_add

from module import BinaryLinear


class myBiGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=True, bi=False):
        super(myBiGCNConv, self).__init__(aggr="add")
        self.cached = cached
        self.bi = bi
        if bi:
            self.lin = BinaryLinear(in_channels, out_channels, scalar=True)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None

    def forward(self, x, edge_index):
        # shape of x: [N, in_channels]
        # shape of edge_index: [2, E]
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

        # Start propagating message
        x = self.propagate(edge_index,size=(x.size(0), x.size(0)),
                              x=x, norm=norm)
        return x

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Normalize node features
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # if self.bias is not None:
        #     aggr_out = aggr_out + self.bias
        return aggr_out

class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, normalize=False, bi=False,
                 **kwargs):
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        if bi:
            self.lin_rel = BinaryLinear(in_channels, out_channels, scalar=True)
            self.lin_root = BinaryLinear(in_channels, out_channels, scalar=True)
        else:
            self.lin_rel = Linear(in_channels, out_channels, bias=True)
            self.lin_root = Linear(in_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""

        if torch.is_tensor(x):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.lin_rel(out) + self.lin_root(x[1])

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class indGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bi=False):
        super(indGCNConv, self).__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bi = bi
        if bi:
            self.lin = BinaryLinear(in_channels, out_channels, scalar=True)
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

        # edge_index = add_self_loops_mini_batch(edge_index, num_nodes=x[0].size(0))
        #
        # Compute normalization
        # row, col = edge_index
        # deg = degree(row, x[0].size(0), dtype=x[0].dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # # normalization of each edge
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        #
        # print(x[0].shape, x[1].shape)
        # Start propagating message
        out = self.propagate(edge_index, x=x, norm=None)
        out = self.lin(out)

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Normalize node features
        # return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        # if self.bias is not None:
        #     aggr_out = aggr_out + self.bias
        return aggr_out

class GraphConv(MessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}_2 \mathbf{x}_j.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, bi=False, aggr='add', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if bi:
            self.lin = BinaryLinear(in_channels, out_channels, scalar=True)
            self.lin_root = BinaryLinear(in_channels, out_channels, scalar=True)
            # self.lin_root = torch.nn.Linear(in_channels, out_channels, bias=bias)
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

        # fill_value = 1 if not improved else 2
        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)

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

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




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
