from __future__ import division

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        plot_val=False, permute_masks=None, logger=None):

    val_losses, accs, durations = [], [], []
    avg_best_logits = None
    for run_num in range(runs):
        data = dataset[0]
        # get teacher logits
        # data.teacher_target = torch.from_numpy(np.load("logits_gat_citeseer.npy"))
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        # data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        # A = to_dense_adj(data.edge_index).view(data.x.size(0), data.x.size(0))
        # A = torch.matmul(A, A)
        # data.edge_index, _ = dense_to_sparse(A)s

        data = data.to(device)

        # new_train_mask = data.val_mask + data.test_mask
        # data.train_mask = (new_train_mask == False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # bn_lr = 0. * lr
        # optimizer = torch.optim.Adam([
        #     dict(params=model.bn1.parameters(), lr=bn_lr),
        #     dict(params=model.bn2.parameters(), lr=bn_lr),
        #     dict(params=model.conv1.parameters()),
        #     dict(params=model.conv2.parameters())
        # ], weight_decay=weight_decay, lr=lr)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_logits = None # use for distillation
        best_val_loss = float('inf')
        train_acc = 0
        val_acc = 0
        test_acc = 0
        val_loss_history = []
        tr_loss_history = []
        val_acc_history = []
        tr_acc_history = []
        te_acc_history = []
        epoch_count = -1
        patience = 0

        for epoch in range(1, epochs + 1):
            # print("epochs:",epoch)
            alpha = 0.4
            train(model, optimizer, data, alpha)
            eval_info = evaluate(model, data, alpha)
            eval_info['epoch'] = epoch
            update_flag = False

            if logger is not None:
                logger(eval_info)

            if eval_info['val_loss'] < best_val_loss:
                best_logits = eval_info['logits']
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']
                train_acc = eval_info['train_acc']
                val_acc = eval_info['val_acc']
                epoch_count = epoch
                update_flag = True

            if epoch % 20 == 0:
                tr_loss_history.append(eval_info['train_loss'])
                val_loss_history.append(eval_info['val_loss'])
                tr_acc_history.append(eval_info['train_acc'])
                val_acc_history.append(eval_info['val_acc'])
                te_acc_history.append(eval_info['test_acc'])
            if runs == 1:
                print(epoch, "train: {:.4f}, {:.4f}".format(eval_info['train_loss'], eval_info['train_acc']),
                      "val: {:.4f}, {:.4f}".format(eval_info['val_loss'],eval_info['val_acc']),
                      "test: {:.4f}".format(eval_info['test_acc']))
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break
                # if update_flag:
                #     patience = 0
                # else:
                #     patience += 1
                # if patience >= early_stopping:
                #     break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        # save model
        # torch.save(model, 'model.pkl')
        # torch.save(model.state_dict(), 'params.pkl')

        import numpy as np
        np.savetxt("val_loss_gcn_8.csv", np.array(val_loss_history), delimiter=",")
        np.savetxt("train_loss_gcn_8.csv", np.array(tr_loss_history), delimiter=",")

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        print("Run: {:d}, train_acc: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, epoch_cnt: {:d}".format(run_num+1, train_acc, val_acc, test_acc, epoch_count))
        if avg_best_logits is None:
            avg_best_logits = best_logits
        else:
            avg_best_logits = avg_best_logits + best_logits

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.4f}, std: {:.4f}, Duration: {:.4f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
    # print("avg_best_logits:", avg_best_logits.shape)
    # print(avg_best_logits)
    avg_best_logits = avg_best_logits.log()
    data = dataset[0].to(device)
    mask = data.test_mask
    pred = avg_best_logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    print("avg_test_acc: {:.4f}".format(acc))
    # np.save("logits_gat_citeseer.npy", avg_best_logits.cpu().numpy())
    # print(duration)


def train(model, optimizer, data, alpha=0.4):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    origin_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # teacher_loss = torch.nn.KLDivLoss()(out.exp()[data.train_mask], data.teacher_target[data.train_mask])
    # teacher_loss = torch.nn.MSELoss()(out.exp()[data.train_mask], data.teacher_target[data.train_mask])
    # loss = alpha * origin_loss + (1-alpha) * teacher_loss # hard loss + soft loss
    loss = origin_loss
    loss.backward()

    optimizer.step()


def evaluate(model, data, alpha=0.4):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    outs['logits'] = logits.exp()
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]

        origin_loss = F.nll_loss(logits[mask], data.y[mask]).item()
        # teacher_loss = torch.nn.KLDivLoss()(logits.exp()[mask], data.teacher_target[mask])
        # loss = alpha * origin_loss + (1 - alpha) * teacher_loss  # hard loss + soft loss
        loss = origin_loss
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs
