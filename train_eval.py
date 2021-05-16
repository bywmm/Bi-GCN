from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam, SGD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(exp_name, dataset, model, runs, epochs, lr, weight_decay, early_stopping):
    val_losses, accs, durations = [], [], []
    for run_num in range(runs):
        data = dataset[0]
        data = data.to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # model_float_parameters = filter(lambda p: id(p) in model.float_params1 + model.float_params2, model.parameters())
        # model_binary_parameters = filter(lambda p: id(p) not in model.float_params1 + model.float_params2, model.parameters())
        # optimizer = Adam([
        #     {'params': model_float_parameters, 'lr': 0.0*lr},
        #     {'params': model_binary_parameters}]
        #     , lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

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

        for epoch in range(1, epochs + 1):
            # print("epochs:",epoch)
            train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']
                train_acc = eval_info['train_acc']
                val_acc = eval_info['val_acc']
                epoch_count = epoch
                # torch.save(model.state_dict(), 'gcn_cora.pkl')
                # print("model saved!")

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        print("Run: {:d}, train_acc: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, epoch_cnt: {:d}".format(run_num+1, train_acc, val_acc, test_acc, epoch_count))

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Experiment:', exp_name)
    print('Val Loss: {:.4f}, Test Accuracy: {:.4f}, std: {:.4f}, Duration: {:.4f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs
