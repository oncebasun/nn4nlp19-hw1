# -*- coding: UTF-8 -*-
import os
import time
import random
import logging
import datetime
import argparse
import importlib
import torch
import torchtext.data as data
import default_config
import configs.config as configuration
import utils


# TODO: design config setting system in a better way
MODEL_CONFIGS = ['seed']


def load_model_conf(model_dir, conf):
    with open(os.path.join(model_dir, 'conf.txt')) as f:
        exec(f.read())


def save_model_conf(model_dir, conf):
    with open(os.path.join(model_dir, 'conf.txt'), 'w') as f:
        for key in MODEL_CONFIGS:
            if hasattr(conf, key):
                f.write('args.%s = %s' %(key, getattr(conf, key)))


if __name__ == '__main__':
    ###############################################
    #                 Preparation                 #
    ###############################################
    # Configuration priority 0 > 1 > 2 > 3 > 4 > 5:
    # 0. (only for model configs) loaded model config
    # 1. command line options
    # 2. default command line options
    # 3. command line config file 
    # 4. main config file 
    # 5. defult 
    args = default_config.get_default_config()
    configuration.update_config(args)

    parser = argparse.ArgumentParser(description='CNN text multi-class classificer')
    # options
    parser.add_argument('--config', type=str, metavar='CONFIG', help='use this configuration file instead of the default config, like "configs.config"')
    parser.add_argument('--test', action='store_true', default=False, help='train | test')
    parser.add_argument('--load', type=str, default=None, help='dir of model to load [default: None]')
    parser.add_argument('--debug', action='store_true', default=False, help='show DEBUG outputs')
    # device
    parser.add_argument('--device', type=str, default='cpu', help='device to use for iterate data. cpu | cudaX (e.g. cuda0) [default: cpu]')
    # data
    parser.add_argument('--datadir', '-d', type=str, help='path to the data directory [default: "%s"]' % args.datadir)
    # model
    parser.add_argument('--seed', type=int, help='manual seed [default: random seed or from config file]')
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

    # Device
    if args.device == 'cpu':
        args.cuda = False
        logger.info('Device: CPU')
    elif args.device[:4] == 'cuda':
        if torch.cuda.is_available():
            args.cuda = True
            gpuid = int(args.device[4:])
            torch.cuda.set_device(gpuid)
            logger.info('Device: CUDA #%d' % gpuid)
        else:
            args.cuda = False
            logger.warning('CUDA is not available now. Automatically switched to using CPU.')
    else:
        logging.error('Invalid device: %s !' % args.device)
        quit()

    # Seeding
    if args.seed is None:
        args.seed = random.randint(1, 100000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Show configurations
    logger.info('%Configuration%')
    for key in args.__dict__:
        logger.info('  %s: %s' %(key, str(args.__dict__[key])))

    # Save model configs
    logger.info('Saving configs ... ')
    save_model_conf(args.modeldir, args)

    # Load dataset
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=True)
