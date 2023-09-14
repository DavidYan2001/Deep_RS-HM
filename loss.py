import torch
from torch import nn, optim
from torch.utils.data import Dataset
import os
import numpy as np



class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()


    def forward(self,img,pt_pred,homo_flow_gt,rho):
        loss = loss_caculation(img,pt_pred,homo_flow_gt,rho)
        return loss

def loss_caculation(img,pt_pred,homo_flow_gt,rho):


    batchsize,h,w = img.shape

    homo_flow_gt =homo_flow_gt.cuda()

    pt_pred = pt_pred*16
    pt_pred =pt_pred.reshape(batchsize,1,4,2).cuda()


    src = torch.tensor([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=torch.float32)
    src = src.repeat(batchsize,1,1)
    src = src.reshape((batchsize,1,4,2))

    src =src.cuda()

    homo_pred = DLT_solve(src,pt_pred)
    homo_pred = homo_pred.cuda()

    homo_pred_map = homo_pred.reshape((batchsize,1,1,3,3)).repeat(1,1,w,1,1)
    homo_pred_map = homo_pred_map.repeat(1,h,1,1,1).cuda()

    map_x=torch.ones((batchsize,h,w,1))
    for i in range(w):
        map_x[:,:,i] = i
    map_y=torch.ones((batchsize,h,w,1))
    for i in range(h):
        map_y[:,i,:,:] = i

    map_z_homo = torch.ones((batchsize,h,w,1))

    map_xyz = torch.cat((map_x,map_y,map_z_homo),dim=3)

    map_xyz =map_xyz.reshape((batchsize,h,w,3,1)).cuda()
    # print(map_xyz[0,0,0])
    # print(map_xyz[0,0,1])
    transform_map = homo_pred_map[:,:,:,...]@ map_xyz[:,:,:,...]
    transform_map = transform_map.cuda()
    # print(transform_map[0, 0, 0])
    # print(transform_map[0, 0, 1])

    a = transform_map[:,:,:,0:2].reshape(batchsize
                                         ,h,w,2)
    b = transform_map[:,:,:,2]

    transform = torch.div(a,b)

    homo_flow_pred = (transform[:,:,:,...]-map_xyz[:,:,:,0:2].reshape(batchsize,h,w,2))/rho

    dis = torch.abs(homo_flow_gt-homo_flow_pred)

    dis = torch.sqrt(torch.pow(dis[:,:,:,0],2)+torch.pow(dis[:,:,:,1],2))
    dis = dis.reshape(batchsize,h,w)
    #dis = torch.sqrt(dis)
    flow_loss = torch.mean(dis)

    return flow_loss


def DLT_solve(src_p, off_set):


    #src_ps = src_p.clone().detach()
    #off_sets = off_set.clone().detach()

    bs, n, h, w = src_p.shape

    N = bs * n

    src_ps = src_p.reshape(N, h, w)
    off_sets = off_set.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = torch.ones(N, 4, 1)
    if torch.cuda.is_available():
        ones = ones.cuda()
    xy1 = torch.cat((src_ps, ones), 2)
    zeros = torch.zeros_like(xy1)
    if torch.cuda.is_available():
        zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(N, 3, 3)
    H = H.reshape(bs,  3, 3)
    return H


class myLoss_basis_flow(nn.Module):
    def __init__(self):
        super(myLoss_basis_flow, self).__init__()


    def forward(self,img,homo_flow_pred,homo_flow_gt):
        loss = loss_caculation_basis(img,homo_flow_pred,homo_flow_gt)
        return loss

def loss_caculation_basis(img,homo_flow_pred,homo_flow_gt):


    dis = torch.abs(homo_flow_gt - homo_flow_pred)

    dis = torch.sqrt(torch.pow(dis[:, :, :, 0], 2) + torch.pow(dis[:, :, :, 1], 2))
    #dis = dis.reshape(batchsize, h, w)
    basis_flow_loss =torch.mean(dis)
    return basis_flow_loss