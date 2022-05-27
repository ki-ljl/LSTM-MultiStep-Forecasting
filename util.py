# -*- coding:utf-8 -*-
"""
@Time：2022/04/15 16:06
@Author：KI
@File：util.py
@Motto：Hungry And Humble
"""
from itertools import chain

import torch
from scipy.interpolate import make_interp_spline
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import LSTM, BiLSTM
from data_process import nn_seq_mmss, nn_seq_mo, nn_seq_sss, device, get_mape, setup_seed, MyDataset, nn_seq2seq
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

setup_seed(20)


def load_data(args, flag, batch_size):
    if flag == 'mms' or flag == 'mmss':
        Dtr, Dte, lis1, lis2 = nn_seq_mmss(B=batch_size, pred_step_size=args.pred_step_size)
    elif flag == 'mo':
        Dtr, Dte, lis1, lis2 = nn_seq_mo(B=batch_size, num=args.output_size)
    elif flag == 'seq2seq':
        Dtr, Dte, lis1, lis2 = nn_seq2seq(B=batch_size)
    else:
        Dtr, Dte, lis1, lis2 = nn_seq_sss(B=batch_size)

    return Dtr, Dte, lis1, lis2


def train(args, Dtr, path):
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    loss = 0
    for i in tqdm(range(args.epochs)):
        cnt = 0
        for (seq, label) in Dtr:
            cnt += 1
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if cnt % 100 == 0:
            #     print('epoch', i, ':', cnt - 100, '~', cnt, loss.item())
        print('epoch', i, ':', loss.item())

        scheduler.step()

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)


def test(args, Dte, lis, path):
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading model...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    m, n = lis[0], lis[1]
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()


def m_test(args, Dtes, lis, PATHS):
    pred = []
    y = []
    print('loading model...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    # model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    Dtes = [[x for x in iter(Dte)] for Dte in Dtes]
    models = []
    for path in PATHS:
        if args.bidirectional:
            model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        else:
            model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        model.load_state_dict(torch.load(path)['model'])
        model.eval()
        models.append(model)
    print('predicting...')
    for i in range(len(Dtes[0])):
        for j in range(len(Dtes)):
            model = models[j]
            seq, label = Dtes[j][i][0], Dtes[j][i][1]
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            seq = seq.to(device)
            with torch.no_grad():
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    m, n = lis[0], lis[1]
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print(y, pred)
    print('mape:', get_mape(y, pred))
    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()


def list_of_groups(data, sub_len):
    groups = zip(*(iter(data),) * sub_len)
    end_list = [list(i) for i in groups]
    count = len(data) % sub_len
    end_list.append(data[-count:]) if count != 0 else end_list
    return end_list


def ss_rolling_test(args, Dte, lis, path):
    """
    Single step scrolling.
    :param args:
    :param path:
    :param flag:
    :return:
    """
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, batch_size=1)
    pred = []
    y = []
    print('loading model...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    print('predicting...')
    Dte = [x for x in iter(Dte)]
    Dte = list_of_groups(Dte, args.pred_step_size)
    #
    for sub_item in tqdm(Dte):
        sub_pred = []
        for seq_idx, (seq, label) in enumerate(sub_item, 0):
            # 每个seq的batch都为1
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            if seq_idx != 0:
                seq = seq.cpu().numpy().tolist()[0]
                # 如果当前预测长度已经大于seq，直接用pred的后几个将seq中每个数组的第一个数字替换掉
                # 需要重新处理，只将几个负荷值进行替换
                if len(sub_pred) >= len(seq):
                    for t in range(len(seq)):
                        seq[t][0] = sub_pred[len(sub_pred) - len(seq) + t]
                else:
                    # 否则, seq的后几个用pred代替
                    for t in range(len(sub_pred)):
                        seq[len(seq) - len(sub_pred) + t][0] = sub_pred[t]
            else:
                # 第一个直接预测
                seq = seq.cpu().numpy().tolist()[0]
            # print(new_seq)
            seq = [seq]
            seq = torch.FloatTensor(seq)
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0)
            # print(new_seq)
            seq = [x for x in iter(seq)][0]
            # print(new_seq)
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                # print(y_pred)
                sub_pred.extend(y_pred)

        pred.extend(sub_pred)

    y, pred = np.array(y), np.array(pred)
    m, n = lis[0], lis[1]
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)


# multiple model scrolling
def mms_rolling_test(args, Dte, lis, PATHS):
    pred = []
    y = []
    print('loading model...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    models = []
    for path in PATHS:
        if args.bidirectional:
            model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        else:
            model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        model.load_state_dict(torch.load(path)['model'])
        model.eval()
        models.append(model)

    Dte = [x for x in iter(Dte)]
    Dte = list_of_groups(Dte, args.pred_step_size)
    #
    for sub_item in tqdm(Dte):
        sub_pred = []
        for seq_idx, (seq, label) in enumerate(sub_item, 0):
            # 每个seq的batch都为1
            model = models[seq_idx]
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            if seq_idx != 0:
                seq = seq.cpu().numpy().tolist()[0]
                # 如果当前预测长度已经大于seq，直接用pred的后几个将seq中每个数组的第一个数字替换掉
                # 需要重新处理，只将几个负荷值进行替换
                if len(sub_pred) >= len(seq):
                    for t in range(len(seq)):
                        seq[t][0] = sub_pred[len(sub_pred) - len(seq) + t]
                else:
                    # 否则, seq的后几个用pred代替
                    for t in range(len(sub_pred)):
                        seq[len(seq) - len(sub_pred) + t][0] = sub_pred[t]
            else:
                # 第一个直接预测
                seq = seq.cpu().numpy().tolist()[0]
            # print(new_seq)
            seq = [seq]
            seq = torch.FloatTensor(seq)
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0)
            # print(new_seq)
            seq = [x for x in iter(seq)][0]
            # print(new_seq)
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                # print(y_pred)
                sub_pred.extend(y_pred)

        pred.extend(sub_pred)

    y, pred = np.array(y), np.array(pred)
    m, n = lis[0], lis[1]
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    plot(y, pred)


def rolling_test(args, path, flag):
    """
    SingleStep Rolling Predicting.
    :param args:
    :param path:
    :param flag:
    :return:
    """
    Dtr, Dte, m, n = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading model...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    print('predicting...')
    for batch_idx, (seq, target) in enumerate(tqdm(Dte), 0):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        if batch_idx != 0:
            seq = seq.cpu().numpy().tolist()[0]
            if len(pred) >= len(seq):
                x = [[x] for x in pred]
                new_seq = x[-len(seq):]
            else:
                new_seq = seq[:(len(seq) - len(pred))]
                x = [[x] for x in pred]
                new_seq.extend(x)
        else:
            new_seq = seq.cpu().numpy().tolist()[0]
        # print(new_seq)
        new_seq = [new_seq]
        new_seq = torch.FloatTensor(new_seq)
        new_seq = MyDataset(new_seq)
        new_seq = DataLoader(dataset=new_seq, batch_size=1, shuffle=False, num_workers=0)
        # print(new_seq)
        new_seq = [x for x in iter(new_seq)][0]
        # print(new_seq)
        with torch.no_grad():
            new_seq = new_seq.to(device)
            y_pred = model(new_seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            # print(y_pred)
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)


def plot(y, pred):
    # plot
    x = [i for i in range(1, 150 + 1)]
    # print(len(y))
    x_smooth = np.linspace(np.min(x), np.max(x), 500)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()
