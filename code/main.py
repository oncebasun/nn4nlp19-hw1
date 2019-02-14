# -*- coding: UTF-8 -*-
import os
import time
import random
import pickle
import logging
import datetime
import argparse
import importlib
import torch
import torchtext.data as data
import default_config
import configs.config as configuration
import utils
import datasets
from model.cnn import CNN
from solver import Solver
from evaluate import predict

# TODO: design config setting system in a better way
MODEL_CONFIGS = ['seed', 'alpha', 'dropout', 'embed_dim', 'kernel_num', 
                 'kernel_sizes', 'static']


def load_model_conf(model_dir, conf):
    with open(os.path.join(model_dir, 'conf.txt')) as f:
        exec(f.read())


def save_model_conf(model_dir, conf):
    with open(os.path.join(model_dir, 'conf.txt'), 'w') as f:
        for key in MODEL_CONFIGS:
            if hasattr(conf, key):
                f.write('args.%s = %s' %(key, getattr(conf, key)))

def load_model(model_dir, cnn):
    cnn.load_state_dict(torch.load(os.path.join(model_dir, 'cnn.pkl')))

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
    parser.add_argument('--verbose', action='store_true', default=False, help='show more detailed output')
    # device
    parser.add_argument('--device', type=str, default='cpu', help='device to use for iterate data. cpu | cudaX (e.g. cuda0) [default: cpu]')
    # data
    parser.add_argument('--datadir', '-d', type=str, help='path to the data directory [default: "%s"]' % args.datadir)
    parser.add_argument('--incorp_val', action='store_true', help='incorporate validation data into vocabulary' )
    # model
    parser.add_argument('--seed', type=int, help='manual seed [default: random seed or from config file]')
    parser.add_argument('--alpha', type=float, help='alpha of leaky relu [default: %f]' %args.alpha)
    parser.add_argument('--dropout', type=float, help='the probability for dropout [default: %f]' %args.dropout)
    parser.add_argument('--embed_dim', type=int, help='number of embedding dimension [default: %d]' %args.embed_dim)
    parser.add_argument('--kernel_num', type=int, help='number of each kind of kernel [default: %d]' %args.kernel_num)
    parser.add_argument('--kernel_sizes', type=str, help='comma-separated kernel size to use for convolution [default: "%s"]' %args.kernel_sizes)
    parser.add_argument('--static', action='store_true', help='fix the embeddings [default: %s]' %str(args.static))
    # training
    parser.add_argument('-lr', type=float, metavar='FLOAT', help='initial learning rate [default: %f]' % args.lr)
    parser.add_argument('--epochs', '-e', type=int, help='number of epochs for train [default: %d]' %args.epochs)
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    parser.add_argument('--batchsize', type=int, help='batch size for training [default: %d]' % args.batchsize)
    parser.add_argument('--best', action='store_true', help='store model of the best epoch [default: %s]' %str(args.best))
    parser.add_argument('--clip', type=float, help='clips gradient norm of an iterable of parameters [default: %f]' %args.clip)
    new_args = parser.parse_args()

    # Update config
    if new_args.config is not None:
        new_config = importlib.import_module(new_args.config)
        new_config.update_config(args)
    for key in new_args.__dict__:
        if key is not 'config' and new_args.__dict__[key] is not None and not(key in args.__dict__ and new_args.__dict__[key] == False):
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
    label_field = data.Field(sequential=False)
    if not args.test:
        train_data = datasets.DB(args.datadir, text_field, label_field, args.label_num, sect='train')
        if args.incorp_val:
            val_data = datasets.DB(args.datadir, text_field, label_field, args.label_num, sect='val')
            text_field.build_vocab(train_data, val_data)
            label_field.build_vocab(train_data, val_data)
        else:
            text_field.build_vocab(train_data)
            label_field.build_vocab(train_data)
            val_data = datasets.DB(args.datadir, text_field, label_field, args.label_num, sect='val')
        test_data = datasets.DB(args.datadir, text_field, label_field, args.label_num, sect='test')
        pickle.dump(text_field.vocab, open(os.path.join(args.modeldir, 'text_field.vocab'), 'wb'))
        pickle.dump(label_field.vocab, open(os.path.join(args.modeldir, 'label_field.vocab'), 'wb'))
        train_iter, val_iter = data.BucketIterator.splits(
                                (train_data, val_data),
                                batch_sizes=(args.batchsize, args.batchsize),
                                device = -1 if not args.cuda else None, 
                                repeat=False, shuffle=args.shuffle)
        test_iter = data.BucketIterator(test_data, batch_size=args.batchsize, device = -1 if not args.cuda else None, repeat=False, shuffle=False, sort=False)
    else:
        text_field.vocab = pickle.load(open(os.path.join(args.modeldir, 'text_field.vocab'), 'rb'))
        label_field.vocab = pickle.load(open(os.path.join(args.modeldir, 'label_field.vocab'), 'rb'))
        test_data = datasets.DB(args.datadir, text_field, label_field, args.label_num, sect='test')
        test_iter = data.BucketIterator(test_data, batch_size=args.batchsize, device = -1 if not args.cuda else None, repeat=False, shuffle=False, sort=False)
    logger.info('Text vocab size: %d' %len(text_field.vocab.itos))
    logger.info('Label vocab size: %d' %len(label_field.vocab.itos))

    # update args
    args.vocab_size = len(text_field.vocab)
    args.pad_idx = text_field.vocab.stoi['<pad>']

    ###############################################
    ##            Constrcuting Models            ##
    ###############################################
    cnn = CNN(args.vocab_size, args.embed_dim, args.label_num, 
              args.kernel_sizes, args.kernel_num, alpha=args.alpha, 
              static=args.static, dropout=args.dropout, pad_idx=args.pad_idx)
    if hasattr(args, 'load') and args.load is not None:
        load_model(args.load, cnn)

    ###############################################
    ##                 Training                  ##
    ###############################################
    if not args.test:
        # Keep track of time elapsed and running averages
        solver = Solver(optim_args={'lr':args.lr})
        solver.train(cnn,
                 train_iter, val_iter,
                 text_field, label_field,
                 args.epochs, args.clip,
                 cuda=args.cuda, best=args.best,
                 model_dir=args.modeldir, log_dir=args.logdir, 
                 verbose=args.verbose)

    ###############################################
    ##                 Predict                   ##
    ###############################################
    load_model(args.modeldir, cnn)
    predict(cnn, test_iter, text_field, label_field, args.predout, 
            cuda=args.cuda, verbose=args.verbose)
    