# -*- coding:utf-8 -*-
"""
@Time: 2022/03/01 20:11
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data():
    """
    :return: dataframe
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/data.csv'
    df = pd.read_csv(path, encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# Multiple outputs data processing.
def nn_seq_mo(B, num):
    data = load_data()

    train = data[:int(len(data) * 0.7)]
    test = data[int(len(data) * 0.7):len(data)]

    def process(dataset, batch_size):
        load = dataset[dataset.columns[1]]
        load = load.tolist()
        m, n = np.max(load), np.min(load)
        load = (load - n) / (m - n)
        dataset = dataset.values.tolist()
        seq = []
        for i in range(0, len(dataset) - 24 - num, num):
            train_seq = []
            train_label = []

            for j in range(i, i + 24):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)

            for j in range(i + 24, i + 24 + num):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        return seq, [m, n]

    Dtr, lis1 = process(train, B)
    Dte, lis2 = process(test, B)

    return Dtr, Dte, lis1, lis2


# Single step scrolling data processing.
def nn_seq_sss(B):
    data = load_data()

    train = data[:int(len(data) * 0.7)]
    test = data[int(len(data) * 0.7):len(data)]

    def process(dataset, batch_size):
        load = dataset[dataset.columns[1]]
        load = load.tolist()
        m, n = np.max(load), np.min(load)
        load = (load - n) / (m - n)
        dataset = dataset.values.tolist()
        seq = []
        for i in range(len(dataset) - 24):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)
            train_label.append(load[i + 24])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        return seq, [m, n]

    Dtr, lis1 = process(train, B)
    Dte, lis2 = process(test, B)

    return Dtr, Dte, lis1, lis2


# Multiple models single step data processing.
def nn_seq_mmss(B, pred_step_size):
    data = load_data()

    train = data[:int(len(data) * 0.7)]
    test = data[int(len(data) * 0.7):len(data)]

    def process(dataset, batch_size):
        load = dataset[dataset.columns[1]]
        load = load.tolist()
        m, n = np.max(load), np.min(load)
        load = (load - n) / (m - n)
        dataset = dataset.values.tolist()
        #
        seqs = [[] for i in range(pred_step_size)]
        for i in range(0, len(dataset) - 24 - pred_step_size, pred_step_size):
            train_seq = []
            for j in range(i, i + 24):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)
            for j, ind in zip(range(i + 24, i + 24 + pred_step_size), range(pred_step_size)):
                #
                train_label = [load[j]]
                seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seqs[ind].append((seq, train_label))
        #
        res = []
        for seq in seqs:
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
            res.append(seq)

        return res, [m, n]

    Dtrs, lis1 = process(train, B)
    Dtes, lis2 = process(test, B)

    return Dtrs, Dtes, lis1, lis2


def nn_seq2seq():
    print('')


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))
