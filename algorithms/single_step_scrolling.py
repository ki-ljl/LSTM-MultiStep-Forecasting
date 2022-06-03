# -*- coding: utf-8 -*-
"""
@Time：2022/1/18 14:27
@Author：KI
@File：single_step_scrolling.py
@Motto：Hungry And Humble

"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import sss_args_parser
from model_train import train, load_data
from model_test import ss_rolling_test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/single_step_scrolling.pkl'


if __name__ == '__main__':
    args = sss_args_parser()
    flag = 'sss'
    Dtr, _, _, _ = load_data(args, flag, batch_size=args.batch_size)
    train(args, Dtr, LSTM_PATH)
    _, Dte, _, lis2 = load_data(args, flag, batch_size=1)
    ss_rolling_test(args, Dte, lis2, LSTM_PATH)
