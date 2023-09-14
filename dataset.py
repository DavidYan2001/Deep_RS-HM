import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import math

def generate_Gaussian_parameters(k, h):
    mu = np.zeros(k)
    block_height = h / k
    for i in range(k):
        mu[i] = i * block_height + (block_height - 1) / 2

    return mu, block_height


def homo_gt2dst(src,homo_gt,mu,sigma,num_block,h):

    dst = np.zeros((len(src),2))

    for j in range(len(src)):

        Homo = np.zeros((3, 3))
        weight = np.zeros(num_block)

        for i in range(num_block):
            weight[i] = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((src[j,1] - mu[i]) / h) ** 2 / (2 * sigma ** 2))

        weight = weight / sum(weight)

        for i in range(num_block):
            Homo += weight[i] * homo_gt[i].reshape(3, 3)

        dst[j]= (Homo@ [src[j,0],src[j,1],1])[0:2]
        # transform_coordinate_gt[k, j] = Homo @ np.transpose([j, k, 1])
        # transform_coordinate_gt[k, j] = transform_coordinate_gt[k, j] / transform_coordinate_gt[k, j, 2]

    return dst

def seed_points_28pt(h,w,num_block,A,W,Fai):

    src =np.zeros((7*num_block,2))
    dst =np.zeros((7*num_block,2))
    amp=A*w
    block_height =h/num_block

    for i in range(num_block):
        src[7*i]=[0,block_height*i]
        dst[7*i]=[amp*math.sin(W * math.pi * block_height*i / h + Fai * math.pi),block_height*i]
        src[7*i+1]=[w-1,block_height*i]
        dst[7 * i+1] = [w-1+amp * math.sin(W * math.pi * block_height * i / h + Fai * math.pi), block_height * i]

        src[7*i+5]=[0,block_height*(i+1)-1]
        dst[7 * i+5] = [amp * math.sin(W * math.pi * (block_height*(i+1)-1)/ h + Fai * math.pi), block_height*(i+1)-1]

        src[7*i+6]=[w-1,block_height*(i+1)-1]
        dst[7* i+6] = [w-1+amp * math.sin(W * math.pi * (block_height*(i+1)-1) / h + Fai * math.pi), block_height*(i+1)-1]

        src[7*i+2]=[0,(block_height*i+block_height*(i+1)-1)/2]
        dst[7*i+2]=[amp * math.sin(W * math.pi * ((block_height*i+block_height*(i+1)-1)/2)/ h + Fai * math.pi),(block_height*i+block_height*(i+1)-1)/2]

        src[7 * i + 3] = [w-1, (block_height * i + block_height * (i + 1) - 1) / 2]
        dst[7 * i + 3] = [w-1+amp * math.sin(W * math.pi * ((block_height*i+block_height*(i+1)-1)/2)/ h + Fai * math.pi), (block_height * i + block_height * (i + 1) - 1) / 2]

        src[7 * i + 4] = [(w-1)/2, (block_height * i + block_height * (i + 1) - 1) / 2]
        dst[7 * i + 4] = [(w-1)/2+amp * math.sin(W * math.pi * ((block_height * i + block_height * (i + 1) - 1) / 2) / h + Fai * math.pi),(block_height * i + block_height * (i + 1) - 1) / 2]


    return src,dst


def seed_points_24pt(h,w,num_block,A,W,Fai):

    src =np.zeros((6*num_block,2))
    dst =np.zeros((6*num_block,2))
    amp=A*w
    block_height =h/num_block

    for i in range(num_block):
        src[6*i]=[0,block_height*i]
        dst[6*i]=[amp*math.sin(W * math.pi * block_height*i / h + Fai * math.pi),block_height*i]
        src[6*i+1]=[w-1,block_height*i]
        dst[6 * i+1] = [w-1+amp * math.sin(W * math.pi * block_height * i / h + Fai * math.pi), block_height * i]

        src[6*i+4]=[0,block_height*(i+1)-1]
        dst[6 * i+4] = [amp * math.sin(W * math.pi * (block_height*(i+1)-1)/ h + Fai * math.pi), block_height*(i+1)-1]

        src[6*i+5]=[w-1,block_height*(i+1)-1]
        dst[6* i+5] = [w-1+amp * math.sin(W * math.pi * (block_height*(i+1)-1) / h + Fai * math.pi), block_height*(i+1)-1]

        src[6*i+2]=[0,(block_height*i+block_height*(i+1)-1)/2]
        dst[6*i+2]=[amp * math.sin(W * math.pi * ((block_height*i+block_height*(i+1)-1)/2)/ h + Fai * math.pi),(block_height*i+block_height*(i+1)-1)/2]

        src[6 * i + 3] = [w-1, (block_height * i + block_height * (i + 1) - 1) / 2]
        dst[6 * i + 3] = [w-1+amp * math.sin(W * math.pi * ((block_height*i+block_height*(i+1)-1)/2)/ h + Fai * math.pi), (block_height * i + block_height * (i + 1) - 1) / 2]

    return src,dst


def seedpoints2pt(dst,H):

    Dst = np.zeros((len(dst),3))
    for i in range(len(dst)):
        Dst[i]=H@[dst[i,0],dst[i,1],1]
        Dst[i]=Dst[i]/Dst[i,2]

    return Dst

# Create a customized dataset class in pytorch
class RS_Homo_dataset(Dataset):
    def __init__(self, path, rho=16):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        #ori_images, pts1, delta = np.load(self.data[index], allow_pickle=True)


        idx = self.data[index]
        idx = idx.split('/')[-1]
        idx = idx.split('_')[0]

        test = np.load(self.data[index], allow_pickle=True)
        warp_image_crop, ori_image_crop,warp_flow_crop, inverse_flow_crop,RT_label= np.load(self.data[index], allow_pickle=True)
        #warp_image_crop = np.load(self.data[index], allow_pickle=True)


        #input_patch = ori_images[:, pts1[0, 1]: pts1[2, 1], pts1[0, 0]: pts1[2, 0]]

        #delta = delta.astype(float) / self.rho
        #normalization？
        # mu , block_height = generate_Gaussian_parameters(num_block,h)
        # weight = np.zeros(num_block)
        #homo_mix_flow = homo_mix_flow/self.rho

        # R_mean: 118.27573081719484, G_mean: 117.04811159232413, B_mean: 113.27866948803826
        # R_std: 65.25416418319284, G_std: 63.874639496256584, B_std: 66.86849563941627

        # R_mean: 118.01782032993862, G_mean: 116.55064222935268, B_mean: 112.7471807686942
        # R_std: 65.21290006911839, G_std: 64.08452383586653, B_std: 67.30414192051826

        warp_image = warp_image_crop

        mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        # mean_I = np.array([118.02, 116.55, 112.74]).reshape(1, 1, 3)
        # std_I = np.array([65.21, 64.08, 67.30]).reshape(1, 1, 3)
        # mean_I = np.array([118.27, 117.05, 113.28]).reshape(1, 1, 3)
        # std_I = np.array([65.25, 63.87, 66.87]).reshape(1, 1, 3)
        warp_image_crop = (warp_image_crop - mean_I) / std_I
        #warp_image_crop = np.mean(warp_image_crop, axis=2, keepdims=True)
        warp_image_crop =np.transpose(warp_image_crop,[2,0,1])

        return warp_image_crop,warp_flow_crop,warp_image,ori_image_crop,idx
        #分别是，归一化后的图像，输入的patch，四个采样点，坐标变换label

    def __len__(self):
        return len(self.data)
