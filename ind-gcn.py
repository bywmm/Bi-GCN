import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='gpu id')
parser.add_argument('--dataset', type=str, default='Reddit')  # Reddit; Flickr
parser.add_argument('--batch', type=int, default=512)  # 512; 1024
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)  # 5e-4 / 1e-5 for nell
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)  # 0.5
parser.add_argument('--use_normalization', action='store_true')
parser.add_argument('--binarized', action='store_true')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Reddit, Flickr
from torch_geometric.data import NeighborSampler
from layers import indGCNConv as GCNConv
from module import BinActive
from sklearn.metrics import f1_score


assert args.dataset in ['Flickr', 'Reddit']
path = '/home/wangjunfu/dataset/graph/'+str(args.dataset)

if args.dataset == 'Flickr':
    dataset = Flickr(path)
else:
    dataset = Reddit(path)

data = dataset[0]

train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[25, 10], batch_size=args.batch, shuffle=True,
                               num_workers=0)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=args.batch, shuffle=False,
                                  num_workers=0)


class Ind_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Ind_GCN, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, bi=args.binarized))
        self.convs.append(GCNConv(hidden_channels, out_channels, bi=args.binarized))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            # print(x.shape, x_target.shape)
            if(args.binarized):
                x = x - x.mean(dim=1, keepdim=True)
                x = x / (x.std(dim=1, keepdim=True) + 0.0001)
                x = BinActive(scalar=True)(x)

                x_target = x_target - x_target.mean(dim=1, keepdim=True)
                x_target = x_target / (x_target.std(dim=1, keepdim=True) + 0.0001)
                x_target = BinActive(scalar=True)(x_target)
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=args.dropout, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                if args.binarized:
                    # bn x
                    x = x - x.mean(dim=1, keepdim=True)
                    x = x / (x.std(dim=1, keepdim=True) + 0.0001)
                    x = BinActive(scalar=True)(x)

                    # bn x_target
                    x_target = x_target - x_target.mean(dim=1, keepdim=True)
                    x_target = x_target / (x_target.std(dim=1, keepdim=True) + 0.0001)
                    x_target = BinActive(scalar=True)(x_target)
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Ind_GCN(dataset.num_features, args.hidden, dataset.num_classes).to(device)

def train():
    model.train()

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)


    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results.append(int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum()))

    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results.append(f1_score(y_true[mask], y_pred[mask], average='micro') if y_pred[mask].sum() > 0 else 0)
    return results

test_accs = []
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    best_test = 0.0
    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        loss, acc = train()
        train_acc, val_acc, test_acc, train_f1, val_f1, test_f1 = test()
        if val_f1 > best_val:
            best_val = val_f1
            best_test = test_f1
        print("Epoch: {:d}, Loss:{:.4f}, Train f1: {:.4f}, Val f1: {:.4f}, Test f1: {:.4f}".format(epoch, loss, train_f1,
                                                                                               val_f1, test_f1))
    test_accs.append(best_test)
    print("Run: {:d}, best_test: {:.4f}".format(run, best_test))

test_accs = torch.tensor(test_accs)
print("avg test f1 score:{:.4f} ± {:.4f}".format(test_accs.mean(), test_accs.std()))

