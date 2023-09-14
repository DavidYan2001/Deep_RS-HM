import cv2
import numpy as np

from scipy.interpolate import interp2d
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6,7"
idx = [0,1,2]

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class real_dataset(Dataset):
    def __init__(self, path, rho=16):
        lst = os.listdir(path)
        self.data = [path + i for i in lst]
        self.rho = rho

    def __getitem__(self, index):
        img_1,img_2,Homo,homoflow= np.load(self.data[index], allow_pickle=True)

        homoflow =torch.from_numpy(homoflow)

        return homoflow

    def __len__(self):
        return len(self.data)




def flow2basis(flowdata):

    N,h,w,_ = flowdata.shape

    flowdata = flowdata.view(N,-1).permute(1,0)

    if torch.cuda.is_available():
        flowdata=flowdata.cuda()
    U,sigma,V = torch.svd(flowdata)

    U = U.permute(1,0)

    basis = U[:8].view(8,h,w,2)

    max_value = basis.abs().reshape(8, -1).max(1)[0].reshape(8, 1, 1, 1)
    learned_basis = basis / max_value

    return learned_basis,sigma


if __name__ == "__main__":

    work_path = './homoflow_1088_1088/'

    flowdata = real_dataset(work_path)

    print('Found totally {} flow files '.format(len(flowdata)))

    test_loader = DataLoader(flowdata, batch_size=1, shuffle=True)

    FLOW =[]

    for i, batch_value in enumerate(test_loader):

        flow = np.array(batch_value[0])
        FLOW.append(flow)

    FLOW = np.array(FLOW)

    FLOW = torch.from_numpy(FLOW)


    learned_basis,sigma = flow2basis(FLOW)

    #torch.save(learned_basis, "./learned_basis_8_1088_1088.pt")

    learned_basis=learned_basis.cpu().numpy()

    # 定义输入的二维数组
    resize_basis=[]

    # 定义输出的二维数组尺寸
    out_shape = (720, 1280)
    for i in range(8):
        # 定义x轴和y轴的坐标
        a=learned_basis[i]
        x_axis = np.linspace(0, learned_basis[i].shape[1] - 1, learned_basis[i].shape[1])
        y_axis = np.linspace(0, learned_basis[i].shape[0] - 1, learned_basis[i].shape[0])

        # 使用interp2d函数进行插值操作
        f_x = interp2d(x_axis, y_axis, learned_basis[i,:,:,0], kind='cubic')
        f_y = interp2d(x_axis, y_axis, learned_basis[i,:,:,1], kind='cubic')

        # 生成输出的二维数组
        out_x = np.linspace(0, learned_basis[i].shape[1] - 1, out_shape[1])
        out_y = np.linspace(0, learned_basis[i].shape[0] - 1, out_shape[0])
        out_X = f_x(out_x, out_y)
        out_X = np.expand_dims(out_X,axis=2)

        out_Y = f_y(out_x, out_y)
        out_Y = np.expand_dims(out_Y, axis=2)

        # 输出结果
        #print(out)
        out= np.concatenate((out_X, out_Y), axis=2, out=None)

        resize_basis.append(out)

    resize_basis =np.array(resize_basis)
    resize_basis=torch.from_numpy(resize_basis)

    torch.save(resize_basis, "./learned_basis_8_256_256.pt")




