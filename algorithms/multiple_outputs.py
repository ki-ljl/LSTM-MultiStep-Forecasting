# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:11
@Author：KI
@File：multiple_outputs.py
@Motto：Hungry And Humble
"""
import os
from args import mo_args_parser
from util import train, test, load_data

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/model/multiple_outputs.pkl'

if __name__ == '__main__':
    args = mo_args_parser()
    flag = 'mo'
    Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    train(args, Dtr, LSTM_PATH)
    test(args, Dte, lis2, LSTM_PATH)
