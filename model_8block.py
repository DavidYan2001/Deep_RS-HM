import numpy as np
import  os
import torch.nn.functional as F
from gen_bases import gen_basis
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
idx = [0]
import torch.nn as nn
import torch

import math



class HomographyNet_basis(nn.Module):
    def __init__(self,h,w,batchsize):
        super(HomographyNet_basis, self).__init__()

        #self.basis = gen_basis(h,w).unsqueeze(0).reshape(1, 8, -1) # 8,2,h,w --> 1, 8, 2*h*w
        self.weight_map = gen_weightmap(h,w,mu=[15.5,47.5,79.5,111.5,143.5,175.5,207.5,239.5]).cuda()
        #self.weight_map_large = gen_weightmap(1080, 1920, mu=[67, 202, 337, 472, 607, 742, 877, 1012]).cuda()
        self.weight_map_large = gen_weightmap(720, 1280, mu=[44.5, 134.5, 224.5, 314.5, 404.5, 494.5, 584.5, 664.5]).cuda()
        self.basis = torch.load('../Datasets/CA/Data/learned_basis_8_256*256.pt').permute(0,3,1,2).unsqueeze(0).reshape(1, 8, -1).cuda() # 8,2,h,w --> 1, 8, 2*h*w
        self.basis_large = torch.load('../Datasets/CA/Data/learned_basis_8_1280_720.pt').permute(0,3,1,2).unsqueeze(0).reshape(1, 8, -1).cuda() # 8,2,h,w --> 1, 8, 2*h*w

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            #nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(131072, 1024),
            nn.ReLU(True),
            #nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(1024, 64)
        )

    def forward(self, x,h,w):

        batch_size,_,_,_ = x.shape
        #print(x.shape)

        out =nn.parallel.data_parallel(self.layer1,x,idx)
        out =nn.parallel.data_parallel(self.layer2,out,idx)
        out =nn.parallel.data_parallel(self.layer3,out,idx)
        out =nn.parallel.data_parallel(self.layer4,out,idx)
        out = out.contiguous().view(x.size(0), -1)
        #print(out.shape)
        out = self.fc(out).reshape(-1,64,1)
        # print(out.shape)
        # print(self.basis.shape)
        # print((self.basis * out.squeeze(3)).shape)
        # H_flow = (self.basis * out).sum(1).reshape(batch_size, 2, h, w).permute(0,2,3,1) #bs*h*w*2
        # H_flow = (self.basis * out) #bs*h*w*2
        # flow_x = H_flow[:,:12,:].sum(1).reshape(batch_size, 1, h, w).permute(0,2,3,1)
        # flow_y = H_flow[:,12:,:].sum(1).reshape(batch_size, 1, h, w).permute(0,2,3,1)
        # H_flow = torch.cat((flow_x,flow_y),dim=3)

        H_flow_1 = (self.basis * out[:,0:8,:]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        H_flow_2 = (self.basis * out[:,8:16,:]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        H_flow_3 = (self.basis * out[:,16:24,:]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        H_flow_4 = (self.basis * out[:,24:32,:]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        H_flow_5 = (self.basis * out[:,32:40,:]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        H_flow_6 = (self.basis * out[:,40:48,:]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        H_flow_7 = (self.basis * out[:,48:56,:]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        H_flow_8 = (self.basis * out[:,56:64,:]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)

        #
        # H_flow_1 = (self.basis * out[:, 0:12, :]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        # H_flow_2 = (self.basis * out[:, 12:24, :]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        # H_flow_3 = (self.basis * out[:, 24:36, :]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        # H_flow_4 = (self.basis * out[:, 36:48, :]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        # H_flow_5 = (self.basis * out[:, 48:60, :]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        # H_flow_6 = (self.basis * out[:, 60:72, :]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        # H_flow_7 = (self.basis * out[:, 72:84, :]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        # H_flow_8 = (self.basis * out[:, 84:96, :]).sum(1).reshape(batch_size, 2, h, w).permute(0, 2, 3, 1)
        weight_map = self.weight_map.reshape(1, h, w, 8, 1).repeat(batch_size, 1, 1, 1, 1)
        weight_map_large = self.weight_map_large.reshape(1, 720, 1280, 8, 1).repeat(batch_size, 1, 1, 1, 1)
        # weight_map = torch.zeros(batch_size, h, w, 8, 1).cuda()
        # for j in range(8):
        #      weight_map[:,32*j:32*j+32,:,j,:] = 1
        #weight_map = (torch.ones(batch_size,h,w,8,1)*(1/8)).cuda()
        H_flow = H_flow_1[:,:,:,...]*weight_map[:,:,:,0]+\
                 H_flow_2[:,:,:,...]*weight_map[:,:,:,1]+\
                 H_flow_3[:,:,:,...]*weight_map[:,:,:,2]+\
                 H_flow_4[:,:,:,...]*weight_map[:,:,:,3]+\
                 H_flow_5[:,:,:,...]*weight_map[:,:,:,4]+\
                 H_flow_6[:,:,:,...]*weight_map[:,:,:,5]+\
                 H_flow_7[:,:,:,...]*weight_map[:,:,:,6]+\
                 H_flow_8[:,:,:,...]*weight_map[:,:,:,7]

        H_flow_1 = (self.basis_large * out[:, 0:8, :]).sum(1).reshape(batch_size, 2, 720, 1280).permute(0, 2, 3, 1)
        H_flow_2 = (self.basis_large * out[:, 8:16, :]).sum(1).reshape(batch_size, 2, 720, 1280).permute(0, 2, 3, 1)
        H_flow_3 = (self.basis_large * out[:, 16:24, :]).sum(1).reshape(batch_size, 2, 720, 1280).permute(0, 2, 3, 1)
        H_flow_4 = (self.basis_large * out[:, 24:32, :]).sum(1).reshape(batch_size, 2, 720, 1280).permute(0, 2, 3, 1)
        H_flow_5 = (self.basis_large * out[:, 32:40, :]).sum(1).reshape(batch_size, 2, 720, 1280).permute(0, 2, 3, 1)
        H_flow_6 = (self.basis_large * out[:, 40:48, :]).sum(1).reshape(batch_size, 2, 720, 1280).permute(0, 2, 3, 1)
        H_flow_7 = (self.basis_large * out[:, 48:56, :]).sum(1).reshape(batch_size, 2, 720, 1280).permute(0, 2, 3, 1)
        H_flow_8 = (self.basis_large * out[:, 56:64, :]).sum(1).reshape(batch_size, 2, 720, 1280).permute(0, 2, 3, 1)

        H_flow_large = H_flow_1[:, :, :, ...] * weight_map_large[:, :, :, 0] + \
                 H_flow_2[:, :, :, ...] * weight_map_large[:, :, :, 1] + \
                 H_flow_3[:, :, :, ...] * weight_map_large[:, :, :, 2] + \
                 H_flow_4[:, :, :, ...] * weight_map_large[:, :, :, 3] + \
                 H_flow_5[:, :, :, ...] * weight_map_large[:, :, :, 4] + \
                 H_flow_6[:, :, :, ...] * weight_map_large[:, :, :, 5] + \
                 H_flow_7[:, :, :, ...] * weight_map_large[:, :, :, 6] + \
                 H_flow_8[:, :, :, ...] * weight_map_large[:, :, :, 7]




        output={}
        output['homoflow_pred'] = H_flow
        output['homoflow_pred_large']=H_flow_large
        output['weight'] = out.reshape(-1,64)

        return output


def gen_weightmap(h,w,mu):

    sigma = 0.1
    weight_map=torch.zeros(h,w,len(mu))
    for i in range(h):
        weight_1 = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((i - mu[0]) / h) ** 2 / (2 * sigma ** 2))
        weight_2 = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((i - mu[1]) / h) ** 2 / (2 * sigma ** 2))
        weight_3 = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((i - mu[2]) / h) ** 2 / (2 * sigma ** 2))
        weight_4 = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((i - mu[3]) / h) ** 2 / (2 * sigma ** 2))
        weight_5 = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((i - mu[4]) / h) ** 2 / (2 * sigma ** 2))
        weight_6 = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((i - mu[5]) / h) ** 2 / (2 * sigma ** 2))
        weight_7 = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((i - mu[6]) / h) ** 2 / (2 * sigma ** 2))
        weight_8 = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-((i - mu[7]) / h) ** 2 / (2 * sigma ** 2))

        sum = weight_1+weight_4+weight_3+weight_2+weight_8+weight_7+weight_6+weight_5
        weight_map[i,:,0] = weight_1/sum
        weight_map[i, :, 1] = weight_2/sum
        weight_map[i,:,2] = weight_3/sum
        weight_map[i,:,3] = weight_4/sum
        weight_map[i,:,4] = weight_5/sum
        weight_map[i,:,5] = weight_6/sum
        weight_map[i,:,6] = weight_7/sum
        weight_map[i,:,7] = weight_8/sum


    return weight_map

if __name__ == "__main__":
    from torchsummary import summary
    model = HomographyNet_basis(256,256,batchsize=1).cuda()
    print(HomographyNet_basis(256,256,batchsize=1))
    x=torch.ones(1,3, 256, 256).cuda()
    y =model(x,256,256)
    summary(model(h=256,w=256),input_size=(1,3,256,256))
