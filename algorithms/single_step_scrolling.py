# -*- coding: utf-8 -*-
"""
@Time：2022/1/18 14:27
@Author：KI
@File：single_step_scrolling.py
@Motto：Hungry And Humble

"""
import os
from args import sss_args_parser
from util import train, ss_rolling_test, load_data

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/model/single_step_scrolling.pkl'


if __name__ == '__main__':
    args = sss_args_parser()
    flag = 'sss'
    Dtr, Dte, lis1, lis2 = load_data(args, flag, batch_size=1)
    ss_rolling_test(args, Dte, lis2, LSTM_PATH)
