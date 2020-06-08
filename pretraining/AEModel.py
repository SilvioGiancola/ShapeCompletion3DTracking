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

import PCEncoderDecoder
import VoxelEncoderDecoder

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, chkpt_file = None):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bneck_size = encoder.bneck_size
        self.load_chkpt(chkpt_file=chkpt_file)

    def load_chkpt(self, chkpt_file=None):
        if(chkpt_file is not None):
            print("=> loading checkpoint '{}'".format(chkpt_file))
            checkpoint = torch.load(chkpt_file)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(chkpt_file, checkpoint['epoch']))

    def forward(self, X):
        return self.decoder(self.encoder(X))

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)

class PCAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''
    def __init__(self, bneck_size=128, chkpt_file = None):
        super().__init__(PCEncoderDecoder.Encoder(bneck_size=bneck_size), PCEncoderDecoder.Decoder(bneck_size=bneck_size), chkpt_file=chkpt_file)




class VoxelAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for voxels.
    '''
    def __init__(self, bneck_size=128, chkpt_file = None):
        super().__init__(VoxelEncoderDecoder.Encoder(bneck_size=bneck_size), VoxelEncoderDecoder.Decoder(bneck_size=bneck_size), chkpt_file=chkpt_file)

    def forward(self, X):
        X, pool_idx = self.encoder(X)
        return self.decoder(X, pool_idx)

    def encode(self, X):
        X, pool_idx = self.encoder(X)
        return X