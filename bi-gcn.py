import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='gpu id')
parser.add_argument('--dataset', type=str, default='Cora')  # Cora/CiteSeer/PubMed
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005)  # 5e-4
parser.add_argument('--early_stopping', type=int, default=0)  # 100
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.4)  # 0.5
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


import torch
import datetime
import torch.nn.functional as F
from layers import myBiGCNConv as BiGCNConv
from torch_geometric.datasets import Planetoid
from train_eval import run
from module import BinActive
from tensorboardX import SummaryWriter


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(dataset.num_features, affine=False)

        convs = []
        for i in range(args.layers):
            in_channels = args.hidden if i > 0 else dataset.num_features
            out_channels = args.hidden if i < args.layers - 1 else dataset.num_classes
            print("layers {:d}, in {:d}, out {:d}".format(i, in_channels, out_channels))
            convs.append(BiGCNConv(in_channels, out_channels, cached=True, bi=True))
        self.convs = torch.nn.ModuleList(convs)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.bn1(x)

        for i, conv in enumerate(self.convs):
            x = BinActive(scalar=True)(x)
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = conv(x, edge_index)

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
print(args)
# writer = SummaryWriter('logs/bigcn_loaded_test_log_{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()))
writer = None
run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, writer)

