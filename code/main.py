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


def load_model_conf(model_dir, conf):
    with open(os.path.join(model_dir, 'conf.txt')) as f:
        exec(f.read())


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
    parser.add_argument('--config', type=str, metavar='CONFIG', help='use this configuration file instead of the default config, like "configs.config"')
    parser.add_argument('--test', action='store_true', default=False, help='train | test')
    parser.add_argument('--load', type=str, default=None, help='dir of model to load [default: None]')
    parser.add_argument('--debug', action='store_true', default=False, help='show DEBUG outputs')
    # data
    parser.add_argument('--datadir', '-d', type=str, help='path to the data directory [default: "%s"]' % args.datadir)
    # training
    parser.add_argument('-lr', type=float, metavar='FLOAT', help='initial learning rate [default: %f]' % args.lr)
    parser.add_argument('-best', action='store_true', help='store model of the best epoch [default: %s]' %str(args.best))
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
    logger.debug('Logger is in DEBUG mode.')

    # Run mode
    if not args.test:
        logger.info('Running in TRAINING mode.')
        if args.best:
            logger.info('Train in BEST mode')
        else:
            logger.info('Train in LAST mode')
        if hasattr(args, 'load') and args.load is not None:
            logger.info('Loading model: %s' % args.load)
            load_model_conf(args.load, args)
    elif args.test and args.load is not None:
        if args.best:
            logger.warning('In test mode, --best should not be set')
        logger.info('Running in TESTING mode.')
        logger.info('Loading model: %s' % args.load)
        load_model_conf(args.load, args)
    elif args.test and (not hasattr(args, 'load') or args.load is None):
        logger.error('Running in Test mode and --load is not set')
        quit()

    