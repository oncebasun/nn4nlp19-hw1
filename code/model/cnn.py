import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, embed_num, embed_dim, n_class, kernel_sizes, n_channel=100, alpha=0.2, dropout=0.5, static=False, pad_idx=1):
        super(CNN, self).__init__()

        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.n_channel = n_channel
        self.n_class = n_class
        self.kernel_sizes = [int(k) for k in kernel_sizes.split(',')]
        self.alpha = alpha
        self.dropout = dropout
        self.static = static

        self.embedding = nn.Embedding(self.embed_num, self.embed_dim, padding_idx=pad_idx)

        self.convs1 = nn.ModuleList([nn.Conv2d(1, self.n_channel, (k, self.embed_dim), padding=(k-1, 0)) for k in self.kernel_sizes])

        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.n_channel, self.n_class)


    def forward(self, word_inputs, test=False):
        x = self.embedding(word_inputs) # (N, len, embed_dim)

        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, 1, len, embed_dim)

        x = [F.leaky_relu(conv(x), self.alpha).squeeze(3) for conv in self.convs1]  # [(N, n_channel, len), ...]*len(kernel_sizes)
                
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, n_channel), ...]*len(kernel_sizes)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(kernel_sizes)*n_channel)

        x = self.fc1(x)  # (N, n_class)

        return x

    def get_embedding(self):
        return self.embedding



