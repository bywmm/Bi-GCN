from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import math

from tmp.module import BinaryLinear


class myBiGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=True, bi=False):
        super(myBiGCNConv, self).__init__(aggr="add")
        self.cached = cached
        self.bi = bi
        if bi:
            self.lin = BinaryLinear(in_channels, out_channels)
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

        # x = F.relu(x)
        # print("W:",self.weight,self.weight.shape)
        # print("b:",self.bias,self.bias.shape)
        # print("Z:",x,x.shape)
        # x_mean = x.mean(dim=1, keepdim=True).expand_as(x)
        # x = x - x_mean
        # x = BinActive(quantify=False)(x)

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
