'''
Created on February 4, 2017

@author: optas

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from AEModel import PCAutoEncoder


class Model(nn.Module):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self,
                 bneck_size=128,
                 chkpt_file=None,
                 AE_chkpt_file=None):
        super(Model, self).__init__()
        self.AE = PCAutoEncoder(bneck_size, AE_chkpt_file)
        self.input_size = self.AE.input_size

        self.bneck_size = bneck_size
        self.score = nn.Linear(bneck_size * 2, 1)
        self.load_chkpt(chkpt_file=chkpt_file)

    def load_chkpt(self, chkpt_file=None):
        if chkpt_file is not None:
            print("=> loading checkpoint '{}'".format(chkpt_file))
            checkpoint = torch.load(chkpt_file)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(chkpt_file, checkpoint['epoch']))

    def forward(self, this_PC, prev_PC):
        X = self.AE.encode(this_PC)
        Y = self.AE.encode(prev_PC)
        Y_AE = self.AE.forward(prev_PC)
        Sim = F.cosine_similarity(X, Y, dim=1)
        # X = self.score(torch.cat((X,Y),dim=1)).squeeze()
        return Sim, Y_AE
        # return Sim

    def encode(self, X):
        return self.AE.encode(X)

    def decode(self, X):
        return self.AE.decode(X)
