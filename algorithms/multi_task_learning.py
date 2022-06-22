# -*- coding:utf-8 -*-
"""
@Time：2022/06/22 10:40
@Author：KI
@File：multi_task_learning.py
@Motto：Hungry And Humble
"""
import os
import sys

from data_process import nn_seq_mtl

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import multi_task_args_parser
from model_train import mtl_train
from model_test import mtl_test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/mtl.pkl'

if __name__ == '__main__':
    args = multi_task_args_parser()
    Dtr, Val, Dte, scaler = nn_seq_mtl(seq_len=args.seq_len, B=args.batch_size, pred_step_size=args.pred_step_size)
    mtl_train(args, Dtr, Val, LSTM_PATH)
    mtl_test(args, Dte, scaler, LSTM_PATH)
