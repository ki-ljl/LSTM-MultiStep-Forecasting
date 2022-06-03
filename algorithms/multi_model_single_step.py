# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:19
@Author：KI
@File：multi_model_single_step.py
@Motto：Hungry And Humble
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import mmss_args_parser
from util import train, m_test, load_data

args = mmss_args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATHS = [path + '/model/mmss/' + str(i) + '.pkl' for i in range(args.pred_step_size)]


if __name__ == '__main__':
    flag = 'mmss'
    Dtrs, Dtes, lis1, lis2 = load_data(args, flag, batch_size=args.batch_size)
    for Dtr, path in zip(Dtrs, LSTM_PATHS):
        train(args, Dtr, path)
    Dtrs, Dtes, lis1, lis2 = load_data(args, flag, batch_size=1)
    m_test(args, Dtes, lis2, LSTM_PATHS)
