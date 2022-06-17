# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:20
@Author：KI
@File：seq2seq.py
@Motto：Hungry And Humble
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import seq2seq_args_parser
from model_train import seq2seq_train, load_data
from model_test import seq2seq_test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/seq2seq.pkl'

if __name__ == '__main__':
    args = seq2seq_args_parser()
    flag = 'seq2seq'
    Dtr, Val, Dte, m, n = load_data(args, flag, args.batch_size)
    seq2seq_train(args, Dtr, Val, LSTM_PATH)
    seq2seq_test(args, Dte, LSTM_PATH, m, n)
