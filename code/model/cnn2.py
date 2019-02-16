import codecs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


def load_external_embeddings(emb_layer, extern_emb, text_field):
    with codecs.open(extern_emb) as f:
        next(f)
        for line in f:
            l = line.split()
            word = l[0].lower()
            if word in text_field.vocab.stoi:
                embed = torch.Tensor([float(x) for x in l[1:]])
                idx = text_field.vocab.stoi[word]
                emb_layer.weight[idx].data.copy_(embed)


class CNN(nn.Module):
    def __init__(self, embed_num, embed_dim, n_class, kernel_sizes, n_channel=100, alpha=0.2, dropout=0.5, static=False, pad_idx=1, extern_emb=None, text_field=None):
        super(CNN, self).__init__()

        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.n_channel = n_channel
        self.n_class = n_class
        if isinstance(kernel_sizes, str):
            self.kernel_sizes = [int(k) for k in kernel_sizes.split(',')]
        elif isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes]
        else:
            self.kernel_sizes = kernel_sizes
        self.alpha = alpha
        self.dropout = dropout
        self.static = static

        self.embedding = nn.Embedding(self.embed_num, self.embed_dim, padding_idx=pad_idx)

        if extern_emb is not None:
            load_external_embeddings(self.embedding, extern_emb, text_field)

        if self.static:
            self.embedding.weight.requires_grad = False

        self.channel1 = 300
        self.convs1 = nn.Sequential(
                        nn.Conv1d(self.embed_dim, self.channel1, 3, padding=1),
                        nn.BatchNorm1d(self.channel1),
                        nn.ReLU(),
                        nn.Conv1d(self.channel1, self.channel1, 3, padding=1),
                        nn.BatchNorm1d(self.channel1),
                        nn.ReLU()
                      )

        self.pool1 = nn.MaxPool1d(2, stride=2)

        self.channel2 = 300
        self.convs2 = nn.Sequential(
                        nn.Conv1d(self.channel2, self.channel2, 3, padding=1),
                        nn.BatchNorm1d(self.channel2),
                        nn.ReLU(),
                        nn.Conv1d(self.channel2, self.channel2, 3, padding=1),
                        nn.BatchNorm1d(self.channel2),
                        nn.ReLU()
                      )

        self.pool2 = nn.MaxPool1d(2, stride=2)

        self.channel3 = 300
        self.convs3 = nn.Sequential(
                        nn.Conv1d(self.channel3, self.channel3, 3, padding=1),
                        nn.BatchNorm1d(self.channel3),
                        nn.ReLU(),
                        nn.Conv1d(self.channel3, self.channel3, 3, padding=1),
                        nn.BatchNorm1d(self.channel3),
                        nn.ReLU()
                      )

        self.pool3 = nn.MaxPool1d(2, stride=2)

        self.channel4 = 300
        self.convs4 = nn.Sequential(
                        nn.Conv1d(self.channel4, self.channel4, 3, padding=1),
                        nn.BatchNorm1d(self.channel4),
                        nn.ReLU(),
                        nn.Conv1d(self.channel4, self.channel4, 3, padding=1),
                        nn.BatchNorm1d(self.channel4),
                        nn.ReLU()
                      )

        self.dropout = nn.Dropout(self.dropout)

        self.hidden1 = 512
        self.fcs1 = nn.Sequential(
                        nn.Linear(self.channel2, self.hidden1),
                        nn.BatchNorm1d(self.hidden1),
                        nn.ReLU()
                    )

        self.fcs2 = nn.Sequential(
                        nn.Linear(self.hidden1, self.n_class),
                        nn.ReLU()
                    )


    def forward(self, word_inputs, test=False):
        x = self.embedding(word_inputs) # (N, len, embed_dim)

        x = x.transpose(1, 2)  # (N, embed_dim, len)

        #if self.static:
        #    x = Variable(x)

        x = x + self.convs1(x)   # (N, C, len)

        length = x.size(2)  
        x = F.pad(x, (0, length%2), "constant", 0)  # (N, C, len)
        x = self.pool1(x)  # (N, C, len/2)

        x = x + self.convs2(x)  # (N, C, len/2)

        #length = x.size(2)  
        #x = F.pad(x, (0, length%2), "constant", 0)  # (N, C, len/2)
        #x = self.pool2(x)  # (N, C, len/4)

        #x = x + self.convs3(x)  # (N, C, len/4)

        #length = x.size(2)  
        #x = F.pad(x, (0, length%2), "constant", 0)  # (N, C, len/4)
        #x = self.pool3(x)  # (N, C, len/8)

        #x = x + self.convs4(x)  # (N, C, len/8)

        length = x.size(2)  
        x = F.max_pool1d(x, x.size(2))  # (N, C, 1)
        x = x.squeeze(2)  # (N, C)

        x = self.dropout(x)  # (N, C)

        x = self.fcs1(x)  # (N, H1)

        x = self.fcs2(x)  # (N, n_class)

        return x

    def get_embedding(self):
        return self.embedding



