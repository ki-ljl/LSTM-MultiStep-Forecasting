# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:11
@Author：KI
@File：multiple_outputs.py
@Motto：Hungry And Humble
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import mo_args_parser
from model_train import train, load_data
from model_test import test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/multiple_outputs.pkl'

if __name__ == '__main__':
    args = mo_args_parser()
    flag = 'mo'
    Dtr, Val, Dte, m, n = load_data(args, flag, args.batch_size)
    train(args, Dtr, Val, LSTM_PATH)
    test(args, Dte, LSTM_PATH, m, n)
