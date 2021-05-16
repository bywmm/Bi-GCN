# Reaches around 0.7870 ± 0.0036 test accuracy.

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help='gpu id')
parser.add_argument('--batch', type=int, default=1024)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=20)  # 10, 100
parser.add_argument('--lr', type=float, default=0.003)  # 0.01, 0.001
# parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--use_normalization', action='store_true')
parser.add_argument('--binarized', action='store_true')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler
from layers import BiSAGEConv as SAGEConv
import time
from function import BinActive

root = '/home/wangjunfu/dataset/graph/OGB/'
dataset = PygNodePropPredDataset(name='ogbn-products', root=root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]
# print(dataset.num_features, dataset.num_classes)

train_idx = split_idx['train']
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=args.batch,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, bi=args.binarized))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, bi=args.binarized))
        self.convs.append(SAGEConv(hidden_channels, out_channels, bi=args.binarized))
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
                x = BinActive()(x)

                x_target = x_target - x_target.mean(dim=1, keepdim=True)
                x_target = x_target / (x_target.std(dim=1, keepdim=True) + 0.0001)
                x_target = BinActive()(x_target)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x[0].size(0))
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


device = f'cuda' if torch.cuda.is_available() else 'cpu'
model = SAGE(dataset.num_features, args.hidden, dataset.num_classes, num_layers=args.num_layers)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
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
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


test_accs = []
for run in range(1, args.runs+1):
    print('')
    print(f'Run {run:02d}:')
    print('')
    run_st = time.time()

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs+1):
        epoch_st = time.time()
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}', "Time: {:d}s".format(int(time.time()-epoch_st)))

        if epoch > 5:
            eval_st = time.time()
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}',
                  "Time: {:d}s".format(int(time.time()-eval_st)))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)
    print("Runs: {:d}, Test_acc: {:.4f}, Time: {:d}".format(run, final_test_acc, int(time.time()-run_st)))

test_acc = torch.tensor(test_accs)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')