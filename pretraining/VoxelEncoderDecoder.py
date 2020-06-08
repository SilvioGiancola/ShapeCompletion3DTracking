'''
Created on February 4, 2017

@author: optas

'''
import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, n_filters=[4, 8, 16, 32, 64, 64, 128, 128],pool=[False,True,False,True,False,True,False,True],
                 pool_sizes=[[1,1,1], [2,2,2], [1,1,1], [2,3,2], [1,1,1], [2,3,2], [1,1,1], [3,3,3]], pool_layer=F.max_pool3d,
                 kernel_size=3, stride=1, padding=1, verbose=False, fc_features = 2*2*2*128, bneck_size = 10):
        super(Encoder, self).__init__()
        if verbose:
            print('Building Encoder')
        self.pool = pool
        self.pool_sizes = pool_sizes
        self.pool_layer = pool_layer
        n_layers = len(n_filters)
        self.bneck_size = bneck_size
        self.conv = []
        self.bn = []
        for i in range(n_layers):
            if i==0:
                in_channels = 1
            else:
                in_channels=n_filters[i-1]
            self.conv.append(nn.Conv3d(in_channels=in_channels, out_channels=n_filters[i], kernel_size=kernel_size, stride=stride, padding=padding))
            self.bn.append(nn.BatchNorm3d(num_features=n_filters[i], eps=1e-05, momentum=None, affine=True))
        self.fc1 = nn.Linear(in_features = fc_features, out_features = bneck_size)
        for i, layer in enumerate(self.conv, 1):
            setattr(self, 'conv_{}'.format(i), layer)
        for i, layer in enumerate(self.bn, 1):
            setattr(self, 'bn_{}'.format(i), layer)

    def forward(self, x, verbose=False):
        n_layers = len(self.conv)
        pool_idx = []
        for i in range(n_layers):
            x = self.conv[i](x)
            x = F.relu(x)
            x = self.bn[i](x)
            if self.pool[i]:
                x, cur_pool_idx = self.pool_layer(x, kernel_size=self.pool_sizes[i], return_indices=True)
                pool_idx.append(cur_pool_idx)
            else:
                pool_idx.append(None)
            if verbose:
                print(x.size())

        x = x.view(x.shape[0], -1)
        if verbose:
                print(x.size())
        x = self.fc1(x)
        if verbose:
                print(x.size())
        return x, pool_idx

class Decoder(nn.Module):
    def __init__(self, n_filters=[4, 8, 16, 32, 64, 64, 128, 128],pool=[False,True,False,True,False,True,False,True],
                 pool_sizes=[[1,1,1], [2,2,2], [1,1,1], [2,3,2], [1,1,1], [2,3,2], [1,1,1], [3,3,3]], pool_layer=F.max_unpool3d,
                 kernel_size=3, stride=1, padding=1, verbose=False, fc_features = 2*2*2*128, bneck_size = 10):
        super(Decoder, self).__init__()
        if verbose:
            print('Building Decoder')

        self.pool = pool
        self.pool_sizes = pool_sizes
        self.pool_layer = pool_layer
        n_layers = len(n_filters)
        self.bneck_size = bneck_size
        self.conv = []
        self.bn = []
        for i in range(n_layers):
            if i==0:
                in_channels = 1
            else:
                in_channels=n_filters[i-1]
            self.conv.append(nn.Conv3d(in_channels=n_filters[i], out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.bn.append(nn.BatchNorm3d(num_features=in_channels, eps=1e-05, momentum=None, affine=True))
        self.fc1 = nn.Linear(in_features = bneck_size, out_features = fc_features)
        for i, layer in enumerate(self.conv, 1):
            setattr(self, 'conv_{}'.format(i), layer)
        for i, layer in enumerate(self.bn, 1):
            setattr(self, 'bn_{}'.format(i), layer)

    def forward(self, x, pool_idx, verbose=False):
        n_layers = len(self.conv)
        x = self.fc1(x)
        if verbose:
                print(x.size())
        x = x.view(-1,128,2,2,2)
        if verbose:
                print(x.size())
        for i in range(n_layers):

            curIdx = n_layers - i - 1
            if self.pool[curIdx]:
                x = self.pool_layer(x, pool_idx[curIdx], kernel_size=self.pool_sizes[curIdx])
            x = self.conv[curIdx](x)
            if curIdx == 0:
                x = torch.sigmoid(x)
            else:
                x = F.relu(x)
                x = self.bn[curIdx](x)
            if verbose:
                print(x.size())
        return x
