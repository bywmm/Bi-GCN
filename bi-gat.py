import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='8', type=str, help='gpu id')
parser.add_argument('--exp_name', default=None, type=str)
parser.add_argument('--dataset', type=str, default="Cora")  # Cora/CiteSeer/PubMed
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=0)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--binarized', action='store_true')
args = parser.parse_args()

assert args.exp_name is not None, "Pls inter the exp_name!"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


import torch
import torch.nn.functional as F
from layers import BiGATConv, BiGCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric import transforms as T

from train_eval import run

from function import BinActive


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = BiGATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout,
            bi=args.binarized)
        self.conv2 = BiGATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout,
            bi=args.binarized)

        # self.conv1 = BiGCNConv(dataset.num_features, args.hidden*args.heads, cached=True, bi=args.binarized)
        # self.conv2 = BiGCNConv(args.hidden*args.heads, dataset.num_classes, cached=True, bi=args.binarized)

        # Note that for GAT, we set the affine=True for bn layers.
        self.bn1 = torch.nn.BatchNorm1d(dataset.num_features, affine=False)
        #
        # self.float_params1 = list(map(id, self.conv1.float_parameter))
        # self.float_params2 = list(map(id, self.conv2.float_parameter))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if not args.binarized:
            x = x / x.sum(1, keepdim=True).clamp(min=1)
        else:
            x = self.bn1(x)
            x = BinActive()(x)

        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv1(x, edge_index)

        if not args.binarized:
            x = F.elu(x)
        else:
            # x = self.bn2(x)
            x = BinActive()(x)

        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


root = '/home/wangjunfu/dataset/graph/Planetoid'
dataset = Planetoid(root, args.dataset)
run(args.exp_name, dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping)
