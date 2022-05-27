# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:20
@Author：KI
@File：multi_model_scrolling.py
@Motto：Hungry And Humble
"""
import os
from args import mms_args_parser
from util import train, mms_rolling_test, load_data

args = mms_args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATHS = [path + '/model/mms/' + str(i) + '.pkl' for i in range(args.pred_step_size)]

if __name__ == '__main__':
    flag = 'mms'
    Dtrs, Dtes, lis1, lis2 = load_data(args, flag, batch_size=args.batch_size)
    for Dtr, path in zip(Dtrs, LSTM_PATHS):
        train(args, Dtr, path)
    Dtr, Dte, lis1, lis2 = load_data(args, flag='sss', batch_size=1)
    mms_rolling_test(args, Dte, lis2, LSTM_PATHS)
