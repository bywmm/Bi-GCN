import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='gpu id')
parser.add_argument('--dataset', type=str, default='Cora')  # Cora/CiteSeer/PubMed/NELL.0.001
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005)  # 5e-4 / 1e-5 for nell
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.4)  # 0.5
parser.add_argument('--normalize_features', type=bool, default=False)
parser.add_argument('--plot_loss', type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if args.plot_loss:
    args.runs = 1


import torch
import torch.nn.functional as F
from tmp.layers import myBiGCNConv as GCNConv
from torch_geometric.datasets import Planetoid

from tmp.train_eval import run

from tmp.module import BinActive


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden, cached=True, bi=True)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes, cached=True, bi=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 0.0001)

        x = BinActive()(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv1(x, edge_index)

        x = BinActive()(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


root = '/home/wangjunfu/dataset/graph/Planetoid'
dataset = Planetoid(root, args.dataset)

data = dataset[0]
print(data)
print("Size of train set:", data.train_mask.sum().item())
print("Size of val set:", data.val_mask.sum().item())
print("Size of test set:", data.test_mask.sum().item())
print("Num classes:", dataset.num_classes)
print("Num features:", dataset.num_features)
permute_masks = None
print(args)
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping)

