import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='gpu id')
parser.add_argument('--dataset', type=str, default='Reddit')  # Reddit or Flickr
parser.add_argument('--batch', type=int, default=2000)  # Reddit:2000, Flickr:6000
parser.add_argument('--walk_length', type=int, default=4)  # Reddit:4, Flickr:2
parser.add_argument('--sample_coverage', type=int, default=50)  # Reddit:50, Flickr:100
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=100)  # 100, 50
parser.add_argument('--lr', type=float, default=0.01)  # 0.01, 0.001
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--hidden', type=int, default=128)  # 128, 256
parser.add_argument('--dropout', type=float, default=0.1)  # 0.1, 0.2
parser.add_argument('--use_normalization', action='store_true')
parser.add_argument('--binarized', action='store_true')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr, Reddit
from torch_geometric.data import GraphSAINTRandomWalkSampler
from layers import GraphConv
from torch_geometric.utils import degree
from module import BinActive

assert args.dataset in ['Flickr', 'Reddit']
path = '/home/wangjunfu/dataset/graph/'+str(args.dataset)
# path = '/home/dingsd/junbb/dataset/'+str(args.dataset)
if args.dataset == 'Flickr':
    dataset = Flickr(path)
else:
    dataset = Reddit(path)
data = dataset[0]
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch, walk_length=args.walk_length,
                                     num_steps=5, sample_coverage=args.sample_coverage,
                                     save_dir=dataset.processed_dir,
                                     num_workers=0)

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels, bi=args.binarized)
        self.conv2 = GraphConv(hidden_channels, hidden_channels, bi=args.binarized)
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x0, edge_index, edge_weight=None):
        if args.binarized:
            x0 = x0 - x0.mean(dim=1, keepdim=True)
            x0 = x0 / (x0.std(dim=1, keepdim=True) + 0.0001)
            x0 = BinActive(scalar=True)(x0)
        x1 = self.conv1(x0, edge_index, edge_weight)
        if not args.binarized:
            x1 = F.relu(x1)
        x1 = F.dropout(x1, p=args.dropout, training=self.training)

        if args.binarized:
            x2 = BinActive(scalar=True)(x1)
        else:
            x2 = x1
        x2 = self.conv2(x2, edge_index, edge_weight)
        if not args.binarized:
            x2 = F.relu(x2)
        x2 = F.dropout(x2, p=args.dropout, training=self.training)

        x = torch.cat([x1, x2], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=args.hidden).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()
    # model.set_aggr('add' if args.use_normalization else 'mean')
    model.set_aggr('add')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    # model.set_aggr('add' if args.use_normalization else 'mean')
    model.set_aggr('add')

    if args.use_normalization:
        out = model(data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device))
    else:
        out = model(data.x.to(device), data.edge_index.to(device))

    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs

val_f1s, test_f1s = [], []
for run in range(1, args.runs+1):
    best_val, best_test = 0, 0
    model.reset_parameters()
    for epoch in range(1, args.epochs+1):
        loss = train()
        accs = test()
        if accs[1] > best_val:
            best_val = accs[1]
            best_test = accs[2]
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
              f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
    test_f1s.append(best_test)
    print("Run: {:d}, best val: {:.4f}, best test: {:.4f}".format(run, best_val, best_test))

test_f1s = torch.tensor(test_f1s)
print("{:.4f} ± {:.4f}".format(test_f1s.mean(), test_f1s.std()))
