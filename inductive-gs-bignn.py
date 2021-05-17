import time
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr, Reddit
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.utils import degree

from models import SAINT


def train(model, loader, optimizer, device, use_normalization=False):
    model.train()
    # model.set_aggr('add' if args.use_normalization else 'mean')
    model.set_aggr('add')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if use_normalization:
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
def test(model, data, device, use_normalization=False):
    model.eval()
    # model.set_aggr('add' if args.use_normalization else 'mean')
    model.set_aggr('add')

    if use_normalization:
        out = model(data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device))
    else:
        out = model(data.x.to(device), data.edge_index.to(device))

    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument('--model', type=str, default='GraphSAINT')
    parser.add_argument('--dataset', type=str, default='Reddit')  # Reddit or Flickr
    parser.add_argument('--batch', type=int, default=2000)  # Reddit:2000, Flickr:6000
    parser.add_argument('--walk_length', type=int, default=4)  # Reddit:4, Flickr:2
    parser.add_argument('--sample_coverage', type=int, default=50)  # Reddit:50, Flickr:100
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)  # 100, 50
    parser.add_argument('--lr', type=float, default=0.01)  # 0.01, 0.001
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--hidden', type=int, default=256)  # 128, 256
    parser.add_argument('--dropout', type=float, default=0.1)  # 0.1, 0.2
    parser.add_argument('--use_normalization', action='store_true')
    parser.add_argument('--binarize', action='store_true')
    args = parser.parse_args()

    assert args.model in ['GraphSAINT']
    assert args.dataset in ['Flickr', 'Reddit']
    path = '/home/wangjunfu/dataset/graph/' + str(args.dataset)
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

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model = SAINT(data.num_node_features, args.hidden, dataset.num_classes, args.dropout, args.binarize).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_f1s, test_f1s = [], []
    for run in range(1, args.runs+1):
        best_val, best_test = 0, 0
        model.reset_parameters()
        start_time = time.time()
        for epoch in range(1, args.epochs+1):
            loss = train(model, loader, optimizer, device, args.use_normalization)
            accs = test(model, data, device, args.use_normalization)
            if accs[1] > best_val:
                best_val = accs[1]
                best_test = accs[2]
            if args.runs == 1:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
                      f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
        test_f1s.append(best_test)
        print("Run: {:d}, best val: {:.4f}, best test: {:.4f}, time cost: {:d}s".format(run, best_val, best_test, int(time.time()-start_time)))

    test_f1s = torch.tensor(test_f1s)
    print("{:.4f} ± {:.4f}".format(test_f1s.mean(), test_f1s.std()))


if __name__ == '__main__':
    main()
