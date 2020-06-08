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

#Calculate the pairwise distance between point clouds in a batch
def batch_pairwise_dist(x,y,use_cuda=True):
    x = x.transpose(2,1)
    y = y.transpose(2,1)
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P

#Calculate Chamfer Loss
class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()        

    def forward(self,preds,gts):
        P = batch_pairwise_dist(preds, gts, self.use_cuda)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2

#Calculate accuracy and coverage between two point clouds
def acc_cov(preds, gts, rho=0.02):
        num_preds = preds.size()[2]
        num_gts = gts.size()[2]
        P = batch_pairwise_dist(preds, gts)
        pred_mins, _ = torch.min(P, 1)
        gts_mins, _ = torch.min(P, 2)
        acc = (pred_mins<rho**2).sum(dim=1,dtype=torch.float)/num_preds
        cov = (gts_mins<rho**2).sum(dim=1,dtype=torch.float)/num_gts
        return acc, cov