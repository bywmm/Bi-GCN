import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='8', type=str, help='gpu id')
parser.add_argument('--exp_name', default=None, type=str)
parser.add_argument('--dataset', type=str, default='Cora')  # Cora/CiteSeer/PubMed
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)  # 100
parser.add_argument('--epochs', type=int, default=1000)  # 200
parser.add_argument('--lr', type=float, default=0.1)  # 0.1
parser.add_argument('--weight_decay', type=float, default=0.005)  # 0.0005
parser.add_argument('--early_stopping', type=int, default=100)  # 10
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--binarized', action='store_true')

args = parser.parse_args()

assert args.exp_name is not None, "Pls inter the exp_name!"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

print("K=", args.K)


import torch
import torch.nn.functional as F
from layers import mySGConv as SGConv
from torch_geometric.datasets import Planetoid

from function import BinActive
from train_eval import run


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = SGConv(dataset.num_features, dataset.num_classes, K=args.K, cached=True, bi=args.binarized)
        self.bn1 = torch.nn.BatchNorm1d(dataset.num_features, affine=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if not args.binarized:
            # x = x / x.sum(1, keepdim=True).clamp(min=1)
            pass
        else:
            x = self.bn1(x)
            x = BinActive()(x)
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


root = '/home/wangjunfu/dataset/graph/Planetoid'
dataset = Planetoid(root, args.dataset)
run(args.exp_name, dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping)
