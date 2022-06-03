# -*- coding:utf-8 -*-
"""
@Time：2022/04/15 16:06
@Author：KI
@File：model_train.py
@Motto：Hungry And Humble
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data_process import nn_seq_mmss, nn_seq_mo, nn_seq_sss, device, setup_seed
from models import LSTM, BiLSTM, Seq2Seq

setup_seed(20)


def load_data(args, flag, batch_size):
    if flag == 'mms' or flag == 'mmss':
        Dtr, Dte, lis1, lis2 = nn_seq_mmss(B=batch_size, pred_step_size=args.pred_step_size)
    elif flag == 'mo':
        Dtr, Dte, lis1, lis2 = nn_seq_mo(B=batch_size, num=args.output_size)
    elif flag == 'seq2seq':
        Dtr, Dte, lis1, lis2 = nn_seq_mo(B=batch_size, num=args.output_size)
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

    state = {'models': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)


def seq2seq_train(args, Dtr, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    batch_size = args.batch_size
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
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
    # save
    state = {'models': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)
