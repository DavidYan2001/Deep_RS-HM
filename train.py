import args as args
import numpy as np

from dataset import CocoDdataset
from model_8block import HomographyNet_basis
from loss import myLoss_basis_flow
from torch.utils.data import DataLoader
import argparse
import time
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
idx = [0,1,2]
from torch import nn, optim




import torch
def train(args):
    MODEL_SAVE_DIR = '../checkpoints/folder_oxbuilding/'#存储已训练好的模型

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    model = HomographyNet_basis(256,256,args.batch_size)#确定模型


    # MODEL_SAVE_DIR = '../checkpoints/folder_oxbuilding/'
    # checkpoint = 'rolling_oxbuilding_iter_156000.pth'
    # model_path = os.path.join(MODEL_SAVE_DIR, checkpoint)  # 训练好的模型文件路径
    # state = torch.load(model_path, map_location='cpu')  # 加载训练好的模型文件
    # model.load_state_dict(state['state_dict'])  # 加载文件内的模型参数到model

    TrainingData = CocoDdataset(args.train_path)#在main中定义好了数据路径

    ValidationData = CocoDdataset(args.val_path)

    print('Found totally {} training files and {} validation files'.format(len(TrainingData), len(ValidationData)))

    train_loader = DataLoader(TrainingData, batch_size=args.batch_size, shuffle=True, num_workers=16)

    val_loader = DataLoader(ValidationData, batch_size=args.batch_size, num_workers=4)
    #进行数据采样（构造batch）

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = myLoss_basis_flow()#考虑regression，用的是均方误差
    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # decrease the learning rate after every 1/3 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(2), gamma=0.8)
    #每隔1/3的训练进程降低学习率

    print("start training")
    glob_iter = 0 # 初始化迭代次数
    t0 = time.time() # 记录开始时间
    MSE = list()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0  # 初始化loss
        for i, batch_value in enumerate(train_loader):
            # save model
            if ((glob_iter + 1) % 4000 == 0 and glob_iter != 0):
                # glob_iter += 8000
                filename = 'rolling_oxbuilding' + '_iter_' + str(
                    glob_iter + 1) + '.pth'
                model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                state = {'epoch': args.epochs, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, model_save_path)
            # 保存model状态，optimizer状态


            inputs = batch_value[0].float()


            homoflow_gt = batch_value[1].float()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                homoflow_gt = homoflow_gt.cuda()

            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs,256, 256)
            homoflow_pred = outputs['homoflow_pred']

            loss = criterion(inputs[:, 0, ...], homoflow_pred, homoflow_gt)

            loss.requires_grad_(True)

            loss.backward()
            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
            #           ' -->grad_value:', parms.grad)
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 200 == 0:
                print("Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Mean Squared Error: {:.4f} lr={:.6f}".format(
                    epoch + 1, args.epochs, i + 1, len(train_loader), train_loss / 200, scheduler.get_last_lr()[0]))
                train_loss = 0.0

            if (i + 1) == len(train_loader):
                print("Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Mean Squared Error: {:.4f} lr={:.6f}".format(
                    epoch + 1, args.epochs, i + 1, len(train_loader), train_loss / ((i + 1) % 200),
                    scheduler.get_last_lr()[0]))
                train_loss = 0.0

            glob_iter += 1
        scheduler.step()

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0

            for i, batch_value in enumerate(val_loader):
                inputs = batch_value[0].float()

                homoflow_gt = batch_value[1].float()

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    homoflow_gt = homoflow_gt.cuda()
                outputs = model(inputs, 256, 256)
                homoflow_pred = outputs['homoflow_pred']
                # loss = criterion(outputs, target.view(-1, 8))
                loss = criterion(inputs[:, 0, ...], homoflow_pred, homoflow_gt)
                # img,homo_flow_pred,homo_flow_gt,rho
                val_loss += loss.item()

            MSE.append(val_loss / len(val_loader))
            print("Validation: Epoch[{:0>3}/{:0>3}] Mean Squared Error:{:.4f}, epoch time: {:.1f}s".format(
                epoch + 1, args.epochs, val_loss / len(val_loader), time.time() - epoch_start))

    elapsed_time = time.time() - t0
    print("Finished Training in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))





if __name__ == "__main__":
    train_path = '../data/training_oxbuilding/'
    val_path = '../data/validation_oxbuildin/'

    total_iteration = 160000
    batch_size = 16
    num_samples = 160000
    steps_per_epoch = num_samples // batch_size

    epochs = int(total_iteration / steps_per_epoch) #总迭代次数/batch个数==epoch个数

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=batch_size, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=epochs, help="number of epochs")
    parser.add_argument("--rho", type=int, default=16, help="rho")
    parser.add_argument("--train_path", type=str, default=train_path, help="path to training imgs")
    parser.add_argument("--val_path", type=str, default=val_path, help="path to validation imgs")
    args = parser.parse_args()
    train(args)
