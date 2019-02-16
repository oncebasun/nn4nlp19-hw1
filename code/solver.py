import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import logging
from evaluate import test


class Solver(object):
    default_adam_args = {"lr": 0.001,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={}):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim

        self.logger = logging.getLogger()

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    #@profile
    def train(self, cnn, train_iter, val_iter, text_field, label_field, 
              num_epochs=10, clip=0.5, reg_lambda=0.0, cuda=False, best=True, 
              model_dir='../model/', log_dir='./logs', verbose=False):
        # Zero gradients of both optimizers
        optim = self.optim(cnn.parameters(), **self.optim_args)

        #self.tf_logger = Logger(log_dir)
        self._reset_histories()

        if cuda:
            cnn.cuda()

        self.logger.info('START TRAIN')
        self.logger.info('CUDA = ' + str(cuda))

        torch.save(cnn.state_dict(), os.path.join(model_dir, 'cnn.pkl'))

        best_val_acc = 0.0
        best_epoch = 0

        criterion = nn.CrossEntropyLoss()
        if cuda:
            criterion.cuda()

        ss = 0
        for epoch in range(num_epochs):
            self.logger.info('Epoch: %d start ...' % (epoch + 1))
            cnn.train()
            for batch in train_iter:
                ss += 1
                input, target = batch.text, batch.label # input: len x N; target: N

                # Reset
                optim.zero_grad()
                loss = 0

                # Setup data
                input.data.t_()  # N x len
                if cuda:
                    input = input.cuda()
                    target = target.cuda()

                # Run words through cnn
                scores = cnn(input)

                l2_reg = None
                for W in cnn.parameters():
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)

                loss = criterion(scores, target) + l2_reg * reg_lambda

                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn.parameters(), clip)
                optim.step()
                '''
                info = {
                    'Loss': loss.data[0],
                }
                for tag, value in info.items():
                    self.tf_logger.scalar_summary(tag, value, ss)
                '''
                if verbose:
                    self.logger.info('Epoch: %d, Iteration: %d, loss: %f' % (epoch + 1, ss, loss.item()))

            val_acc = test(cnn, val_iter, text_field, label_field, cuda=cuda, verbose=verbose)
            '''
            info = {
                'val_acc': val_acc
            }
            for tag, value in info.items():
                self.tf_logger.scalar_summary(tag, value, epoch)
            '''
            if best:
                if val_acc > best_val_acc:
                    torch.save(cnn.state_dict(), os.path.join(model_dir, 'cnn.pkl'))
                    best_val_acc = val_acc
            else:
                torch.save(cnn.state_dict(), os.path.join(model_dir, 'cnn.pkl'))
