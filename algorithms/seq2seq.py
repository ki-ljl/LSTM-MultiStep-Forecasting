# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:20
@Author：KI
@File：seq2seq.py
@Motto：Hungry And Humble
"""
import os
from args import seq2seq_args_parser
from util import seq2seq_train, seq2seq_test, load_data

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/model/seq2seq.pkl'

if __name__ == '__main__':
    args = seq2seq_args_parser()
    flag = 'seq2seq'
    Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    seq2seq_train(args, Dtr, LSTM_PATH)
    seq2seq_test(args, Dte, lis2, LSTM_PATH)
