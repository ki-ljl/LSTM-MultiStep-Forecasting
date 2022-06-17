# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:20
@Author：KI
@File：multi_model_scrolling.py
@Motto：Hungry And Humble
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import mms_args_parser
from model_train import train, load_data
from model_test import mms_rolling_test

args = mms_args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATHS = [path + '/models/mms/' + str(i) + '.pkl' for i in range(args.pred_step_size)]

if __name__ == '__main__':
    flag = 'mms'
    Dtrs, Vals, Dtes, m, n = load_data(args, flag, batch_size=args.batch_size)
    for Dtr, Val, path in zip(Dtrs, Vals, LSTM_PATHS):
        train(args, Dtr, Val, path)
    Dtr, Val, Dte, m, n = load_data(args, flag='sss', batch_size=1)
    mms_rolling_test(args, Dte, LSTM_PATHS, m, n)
