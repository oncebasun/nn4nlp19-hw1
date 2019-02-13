# -*- coding: UTF-8 -*-
def get_default_config():
    args = lambda: None

    # Data path
    args.datadir = '../data'
    # Initial learning rate
    args.lr = 0.1
    # Store model of the best epoch
    args.best = False
    return args
