# -*- coding: UTF-8 -*-
import os
import time
import random
import logging
import datetime
import argparse
import importlib
import default_config
import configs.config as configuration
import utils


if __name__ == '__main__':
    ###############################################
    #                 Preparation                 #
    ###############################################
    # Configuration priority 1 > 2 > 3 > 4 > 5:
    # 1. command line options
    # 2. default command line options
    # 3. command line config file 
    # 4. default config file 
    # 5. defult 
    args = default_config.get_default_config()
    configuration.update_config(args)

    parser = argparse.ArgumentParser(description='CNN text multi-class classificer')
    # options
    parser.add_argument('--config', type=str, metavar='CONFIG', help='Use this configuration file instead of the default config. E.g., "configs.config"')
    parser.add_argument('--test', action='store_true', default=False, help='train | test')
    parser.add_argument('--debug', action='store_true', default=False, help='Show DEBUG outputs')
    # data
    parser.add_argument('-d', '--datadir', type=str, help='path to the data directory [default: "%s"]' % args.datadir)
    # learning
    parser.add_argument('-lr', type=float, metavar='FLOAT', help='Initial learning rate [default: %f]' % args.lr)
    new_args = parser.parse_args()

    # Update config
    if new_args.config is not None:
        new_config = importlib.import_module(new_args.config)
        new_config.update_config(args)
    for key in new_args.__dict__:
        if key is not 'config' and new_args.__dict__[key] is not None:
            setattr(args, key, new_args.__dict__[key])

    # Make up output paths
    utils.make_out_paths(args)

    # Loggers
    logger = utils.get_logger(args)
    logger.info('%%% Task start %%%')
    logger.info('Parsing arguments ...')
    logger.debug('Logger is in DEBUG mode.')

    print(args.__dict__)

    