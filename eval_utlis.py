import os
import math
import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
from sklearn import metrics

import torch
import torch.nn.functional as F

from qlib.data import D


def get_windows(inputs, targets, mode, split_time, final_timestamps, args, shuffle=False):
    gamma = 1.8
    if mode == 'train':
        ## Attention
        train_timestamps = list(D.calendar(start_time=split_time[0], end_time=split_time[1]))
        start = final_timestamps.index(train_timestamps[0])
        end = final_timestamps.index(train_timestamps[-1])
    elif mode == 'valid':
        valid_timestamps = list(D.calendar(start_time=split_time[2], end_time=split_time[3]))
        start = final_timestamps.index(valid_timestamps[0])
        end = final_timestamps.index(valid_timestamps[-1])
    elif mode == 'test':
        test_timestamps = list(D.calendar(start_time=split_time[4], end_time=split_time[5]))
        start = final_timestamps.index(test_timestamps[0])
        end = final_timestamps.index(test_timestamps[-1])
    else:
        raise Exception('Unknown mode.')
    length = end - start + 1
    if shuffle == True:
        indexs = torch.randperm(length)
    elif shuffle == False:
        indexs = range(length)

    for index in indexs:
        i = index + start - (args.window_size + args.window_size_o) + 1
        window = inputs[i:(i+(args.window_size + args.window_size_o)), :, :]
        x_window = window.permute(1, 0, 2).to(args.device) # [296, Tr+To, 6]
        y_window = targets[i+(args.window_size + args.window_size_o)-1].to(args.device) # [296]

        ## ---------
        adj_o = []
        for j in range(0, args.window_size_o):
            adj_day = datetime.datetime.strftime(final_timestamps[i+args.window_size+j-1], '%Y-%m-%d')
            adj_list = []
            for dir in os.listdir(args.adj_path):
                adj_temp = np.load(os.path.join(args.adj_path, dir, adj_day + '.npy'))

                ## When test baseline models please annotate the code of following two lines, 
                ## because the correlation graph is not hoped to be normalized in this method.
                # adj_temp[np.where(adj_temp >= 0)] = 1-np.exp(-gamma*adj_temp[np.where(adj_temp >= 0)])
                # adj_temp[np.where(adj_temp < 0)] = np.exp(gamma*adj_temp[np.where(adj_temp < 0)]) - 1

                adj_temp = (adj_temp.T / (np.sum(np.abs(adj_temp), axis=1) + 1e-7)).T
                adj_list.append(adj_temp)

            adj = torch.Tensor(np.stack(adj_list, axis=0))
            adj_o.append(adj)
        adj_all = torch.stack(adj_o, dim=0)

        yield x_window, y_window, adj_all

def train(model, features, labels, split_time, final_timestamps, args, rmv_feature_num, criterion, optimizer):
    model.train()
    totals_loss = np.array([])
    for x, y, adj in get_windows(features, labels, 'train', split_time, final_timestamps, args, shuffle=True):
        adj = adj.to(args.device)
        logits = model(x[:,:,rmv_feature_num:], adj)
        total_loss = criterion(logits, y)
        total_loss.backward()
        ## weights update
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
        optimizer.zero_grad()
        totals_loss = np.append(totals_loss, total_loss.item())
    return totals_loss.mean()

def test(model, mode, maxK, minN, features, labels, split_time, final_timestamps, args, rmv_feature_num):
    logitses = torch.Tensor([])
    ys = torch.Tensor([])
    topK_logitses = torch.Tensor([])
    topK_ys = torch.Tensor([])
    binary_logitses = torch.Tensor([])
    binary_ys = torch.Tensor([])
    topK_binary_logitses = torch.Tensor([])
    topK_binary_ys = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for x, y, adj in get_windows(features, labels, mode, split_time, final_timestamps, args, shuffle=False):
            adj = adj.to(args.device)
            logits = model(x[:,:,rmv_feature_num:], adj)

            binary_logits = torch.where(logits>=0, torch.ones_like(logits), torch.zeros_like(logits))
            binary_y = torch.where(y>=0, torch.ones_like(y), torch.zeros_like(y))

            ## topK id
            _, maxK_id = logits.topk(maxK)
            _, minN_id = logits.topk(minN, largest=False)
            topK_id = torch.cat([maxK_id, minN_id], dim=0)
            topK_id = maxK_id
            
            ## total mse loss
            logitses = torch.cat([logitses, logits.cpu()], dim=0)
            ys = torch.cat([ys, y.cpu()], dim=0)

            ## topK mse loss
            topK_logitses = torch.cat([topK_logitses, logits[topK_id].cpu()], dim=0)
            topK_ys = torch.cat([topK_ys, y[topK_id].cpu()], dim=0)

            ## total acc
            binary_logitses = torch.cat([binary_logitses, binary_logits.cpu()], dim=0)
            binary_ys = torch.cat([binary_ys, binary_y.cpu()], dim=0)

            ## topK acc
            topK_binary_logitses = torch.cat([topK_binary_logitses, binary_logits[topK_id].cpu()], dim=0)
            topK_binary_ys = torch.cat([topK_binary_ys, binary_y[topK_id].cpu()], dim=0)

        total_rmse_loss = torch.sqrt(F.mse_loss(input=logitses, target=ys)).cpu().numpy()
        topK_rmse_loss = torch.sqrt(F.mse_loss(input=topK_logitses, target=topK_ys)).cpu().numpy()
        total_mape_loss = torch.abs((logitses - ys)).mean().cpu().numpy()
        topK_mape_loss = torch.abs((topK_logitses - topK_ys)).mean().cpu().numpy()
        total_accuracy = torch.eq(binary_logitses, binary_ys).float().mean().cpu().numpy()
        topK_accuracy = torch.eq(topK_binary_logitses, topK_binary_ys).float().mean().cpu().numpy()
        total_MCC = metrics.matthews_corrcoef(binary_ys, binary_logitses)
        topK_MCC = metrics.matthews_corrcoef(topK_binary_ys, topK_binary_logitses)
        return total_rmse_loss, topK_rmse_loss, total_mape_loss, topK_mape_loss, total_accuracy, topK_accuracy, total_MCC, topK_MCC
    
def get_split_time(test_start, test_months_size=1):
    ## 23 months for training, 1 months for validation, 1 months for testing
    test_start_dt = datetime.datetime.strptime(test_start, '%Y-%m-%d')
    test_end_dt = test_start_dt + relativedelta(months=test_months_size) - relativedelta(days=1)
    train_start_dt = test_start_dt - relativedelta(years=2)
    train_end_dt = train_start_dt + relativedelta(months=23) - relativedelta(days=1)
    valid_start_dt = train_start_dt + relativedelta(months=23)
    valid_end_dt = test_start_dt + - relativedelta(days=1)

    test_end = test_end_dt.strftime('%Y-%m-%d')
    train_start = train_start_dt.strftime('%Y-%m-%d')
    train_end = train_end_dt.strftime('%Y-%m-%d')
    valid_start = valid_start_dt.strftime('%Y-%m-%d')
    valid_end = valid_end_dt.strftime('%Y-%m-%d')

    return train_start, train_end, valid_start, valid_end, test_start, test_end

def evaluate(model, criterion, optimizer, scheduler, features, binary_labels, final_timestamps, args, rmv_feature_num, split_time, train_log_filename, model_filename, total_epoch, pprint):
    with open(train_log_filename, 'w', encoding='utf-8') as f:
        f.write('Train Log:' + '\n')

    maxK = 4
    minN = 4
    best_val = -math.inf

    for epoch in range(1, total_epoch+1):
        # ---------training------------
        train_totalloss = train(model, features, binary_labels, split_time, final_timestamps, args, rmv_feature_num, criterion, optimizer)
        lr_temp = optimizer.param_groups[-1]['lr']
        scheduler.step()
        # --------evaluation-----------
        train_loss = test(model, 'train', maxK, minN, features, binary_labels, split_time, final_timestamps, args, rmv_feature_num)
        val_loss = test(model, 'valid', maxK, minN, features, binary_labels, split_time, final_timestamps, args, rmv_feature_num)
        if pprint:
            print("| Epoch {:3d} |".format(epoch))
            print("| Train | RMSE {:6.8f} | TopkRMSE {:6.8f} | MAE {:6.8f} | TopkMAE {:6.8f} | ACC {:6.8f} | TopkACC {:6.8f} | MCC {:6.8f} | TopkMCC {:6.8f} |".format(train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], train_loss[6], train_loss[7]))
            print("| Valid | RMSE {:6.8f} | TopkRMSE {:6.8f} | MAE {:6.8f} | TopkMAE {:6.8f} | ACC {:6.8f} | TopkACC {:6.8f} | MCC {:6.8f} | TopkMCC {:6.8f} |".format(val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_loss[4], val_loss[5], val_loss[6], val_loss[7]))
        with open(train_log_filename, 'a', encoding='utf-8') as f:
            f.write("| Epoch {:3d} |".format(epoch) + '\n')
            f.write("| Train | RMSE {:6.8f} | TopkRMSE {:6.8f} | MAE {:6.8f} | TopkMAE {:6.8f} | ACC {:6.8f} | TopkACC {:6.8f} | MCC {:6.8f} | TopkMCC {:6.8f} |".format(train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], train_loss[6], train_loss[7]) + '\n')
            f.write("| Valid | RMSE {:6.8f} | TopkRMSE {:6.8f} | MAE {:6.8f} | TopkMAE {:6.8f} | ACC {:6.8f} | TopkACC {:6.8f} | MCC {:6.8f} | TopkMCC {:6.8f} |".format(val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_loss[4], val_loss[5], val_loss[6], val_loss[7]) + '\n')
        test_loss = test(model, 'test', maxK, minN, features, binary_labels, split_time, final_timestamps, args, rmv_feature_num)
        if pprint:
            print("| Test  | RMSE {:6.8f} | TopkRMSE {:6.8f} | MAE {:6.8f} | TopkMAE {:6.8f} | ACC {:6.8f} | TopkACC {:6.8f} | MCC {:6.8f} | TopkMCC {:6.8f} |".format(test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5], test_loss[6], test_loss[7]))
        with open(train_log_filename, 'a', encoding='utf-8') as f:
            f.write("| Test  | RMSE {:6.8f} | TopkRMSE {:6.8f} | MAE {:6.8f} | TopkMAE {:6.8f} | ACC {:6.8f} | TopkACC {:6.8f} | MCC {:6.8f} | TopkMCC {:6.8f} |".format(test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5], test_loss[6], test_loss[7]) + '\n')
        # -----------------------------

        mae = val_loss[4]
        if (epoch % 5 == 0) or (mae > best_val):
            if (mae > best_val):
                torch.save(model, model_filename)
                best_val = mae
                if pprint:
                    print('Model has been saved.')
                with open(train_log_filename, 'a', encoding='utf-8') as f:
                    f.write('Model has been saved.' + '\n')
