# -*- coding: utf-8 -*-
import os
import time
import logging


def make_out_paths(args):
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    data_ciph = args.datadir.replace('.', '_').replace('/', '-')
    if not args.test:
        if not args.test:
            args.modeldir = os.path.join('../saved_models/', timestamp + data_ciph)
        if not os.path.exists(args.modeldir):
            os.makedirs(args.modeldir)
        args.logdir = os.path.join(args.modeldir, 'logs/')
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
    else:
        args.testdir = os.path.join('../test_results/', timestamp + data_ciph)
        if not os.path.exists(args.testdir):
            os.makedirs(args.testdir)
        args.logdir = os.path.join(args.testdir, 'logs/')
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)

def get_logger(args):
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.logdir, 'main.log'))
    ch = logging.StreamHandler()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
    