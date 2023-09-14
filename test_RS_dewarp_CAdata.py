import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
from torch.utils.data import DataLoader
from dataset import RS_Homo_dataset
from model_8block import HomographyNet_basis
import argparse

import numpy as np
import cv2
from rollingshutter_basis.loss import  DLT_solve
from get_PSNR_SSIM import psnr,calculate_ssim


def warp_pts(H, src_pts):
    src_homo = np.hstack((src_pts, np.ones((4, 1)))).T
    #将元组按水平方向拼接，即在每个点坐标后面+1，构造齐次坐标
    dst_pts = np.matmul(H, src_homo)
    #test函数里，传入的H实际是逆变换，求解逆变换后的四个点坐标
    dst_pts = dst_pts / dst_pts[-1]
    #齐次坐标特性，可成比例缩放
    return dst_pts.T[:, :2]


def get_flow_gt(pt_gt,img,src):

    batchsize,_, h, w = img.shape

    src = torch.tensor([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=torch.float32)
    src = src.repeat(batchsize, 1, 1)
    src = src.reshape((batchsize, 1, 4, 2)).cuda()

    pt_gt = pt_gt * 16
    pt_gt = pt_gt.reshape(batchsize, 1, 4, 2).cuda()

    homo_gt = DLT_solve(src, pt_gt)

    homo_gt = homo_gt.cuda()

    homo_gt_map = homo_gt.reshape((batchsize, 1, 1, 3, 3)).repeat(1, 1, w, 1, 1)
    homo_gt_map = homo_gt_map.repeat(1, h, 1, 1, 1).cuda()

    map_x = torch.ones((batchsize, h, w, 1))
    for i in range(w):
        map_x[:, :, i] = i
    map_y = torch.ones((batchsize, h, w, 1))
    for i in range(h):
        map_y[:, i, :, :] = i

    map_z_homo = torch.ones((batchsize, h, w, 1))

    map_xyz = torch.cat((map_x, map_y, map_z_homo), dim=3)

    map_xyz = map_xyz.reshape((batchsize, h, w, 3, 1)).cuda()

    transform_map = homo_gt_map[:, :, :, ...] @ map_xyz[:, :, :, ...]
    transform_map = transform_map.cuda()

    a = transform_map[:, :, :, 0:2].reshape(batchsize
                                            , h, w, 2)
    b = transform_map[:, :, :, 2]

    homo_flow_gt = torch.div(a, b)

    homo_flow_gt = homo_flow_gt[:, :, :, ...] - map_xyz[:, :, :, 0:2].reshape(batchsize, h, w, 2)


    return homo_flow_gt

def test(args):
    MODEL_SAVE_DIR = '../checkpoints/'
    model_path = os.path.join(MODEL_SAVE_DIR, args.checkpoint) #训练好的模型文件路径
    result_dir = '../results/'  #存储测试结果路径
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']

    EPE_RE = []
    EPE_LT = []
    EPE_LL = []
    EPE_SF = []
    EPE_LF = []

    PSNR_RE =[]
    PSNR_LT =[]
    PSNR_LL =[]
    PSNR_SF =[]
    PSNR_LF =[]

    SSIM_RE=[]
    SSIM_LT=[]
    SSIM_LL=[]
    SSIM_SF=[]
    SSIM_LF=[]

    model = HomographyNet_basis(256,256,batchsize=1)
    state = torch.load(model_path, map_location='cpu') #加载训练好的模型文件
    model.load_state_dict(state['state_dict']) #加载文件内的模型参数到model
    if torch.cuda.is_available():
        model = model.cuda()

    TestingData = RS_Homo_dataset(args.test_path)
    test_loader = DataLoader(TestingData, batch_size=1,shuffle=False)#采样制作test数据集



    print("start testing")
    with torch.no_grad():
        model.eval() #测试模式
        flow_loss=0

        PSNR=0
        SSIM=0
        for i, batch_value in enumerate(test_loader):
            #注意是1个batch1个batch地送入迭代


            inputs = batch_value[0].float() #patches
            homo_flow_gt = batch_value[1].float() #坐标增量label
            warp_img = batch_value[2]
            ori_image_crop = batch_value[3].squeeze(0)
            #plt.imshow(ori_image_crop)
            # plt.show()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                homo_flow_gt =homo_flow_gt.cuda()

            #starter.record()
            outputs = model(inputs,256,256)
            

            homo_flow_pred = outputs['homoflow_pred']



            dis = torch.abs(homo_flow_gt - homo_flow_pred)

            dis = torch.sqrt(torch.pow(dis[:, :, :, 0], 2) + torch.pow(dis[:, :, :, 1], 2))

            flow_loss = np.float(torch.mean(dis).cpu())



            warp_img = np.array(warp_img.squeeze(0))

            h, w,c = warp_img.shape

            xx = torch.arange(0, w).cuda()
            yy = torch.arange(0, h).cuda()
            xx = xx.view(1, -1).repeat(h, 1).reshape(h, w, 1)
            yy = yy.view(-1, 1).repeat(1, w).reshape(h, w, 1)
            xy_map=torch.cat((xx,yy),dim=2)
            transform_map = xy_map+homo_flow_pred.squeeze(0)
            transform_map_gt = xy_map+homo_flow_gt.squeeze(0)
            map_x = np.float32(transform_map[:,:,0].cpu())
            map_y = np.float32(transform_map[:, :, 1].cpu())

            ori_image_pred = cv2.remap(warp_img, map_x, map_y, interpolation=cv2.INTER_CUBIC)


            map_x_gt = np.float32(transform_map_gt[:, :, 0].cpu())
            map_y_gt = np.float32(transform_map_gt[:, :, 1].cpu())

            # plt.imshow(ori_image_crop[:, :, [2, 1, 0]])
            # plt.show()

            ori_image_crop = cv2.remap(warp_img, map_x_gt, map_y_gt, interpolation=cv2.INTER_CUBIC)

            ori_image_test = np.array(ori_image_crop[:200,56:200,:])
            ori_image_pred_test = ori_image_pred[:200, 56:200,:]
            ori_image_pred_test=cv2.resize(ori_image_pred_test,(256,256))
            ori_image_test=cv2.resize(ori_image_test,(256,256))
            # ori_image_pred_test = ori_image_pred
            # ori_image_test = np.array(ori_image_crop)


            PSNR =psnr(ori_image_test,ori_image_pred_test)
            #(f'PSNR===={PSNR}')
            SSIM =calculate_ssim(ori_image_test,ori_image_pred_test)
            #print(f'SSIM===={SSIM}')

            video_name = batch_value[4][0]

            if video_name in RE:
                EPE_RE.append(flow_loss)
                PSNR_RE.append(PSNR)
                SSIM_RE.append(SSIM)
            elif video_name in LT:
                EPE_LT.append(flow_loss)
                PSNR_LT.append(PSNR)
                SSIM_LT.append(SSIM)
            elif video_name in LL:
                EPE_LL.append(flow_loss)
                PSNR_LL.append(PSNR)
                SSIM_LL.append(SSIM)
            elif video_name in SF:
                EPE_SF.append(flow_loss)
                PSNR_SF.append(PSNR)
                SSIM_SF.append(SSIM)
            elif video_name in LF:
                EPE_LF.append(flow_loss)
                PSNR_LF.append(PSNR)
                SSIM_LF.append(SSIM)
            print(i)



            # plt.imshow(ori_image_pred_test[:,:,[2,1,0]])
            # plt.show()
            #
            # plt.imshow(ori_image_test[:,:,[2,1,0]])
            # plt.show()
            # plt.imshow(warp_img[:, :, [2, 1, 0]])
            # plt.show()
            # plt.imshow(ori_image_crop[:,:,[2,1,0]])
            # plt.show()
            # plt.imshow(ori_image_pred[:,:,[2,1,0]])

            plt.show()
            cv2.imwrite(f"rb_depth/pred_{i}.jpg", ori_image_pred)
            cv2.imwrite(f"rb_depth/warp_{i}.jpg", warp_img)
            cv2.imwrite(f"rb_depth/ori_{i}.jpg", ori_image_crop)

            a=1

        # avg = timings.sum() / repetitions
        # print('\navg={}\n'.format(avg))

        EPE_RE_avg = np.mean(EPE_RE)
        print(f'EPE_RE_avg={EPE_RE_avg}')
        EPE_LT_avg = np.mean(EPE_LT)
        print(f'EPE_LT_avg={EPE_LT_avg}')
        EPE_LL_avg = np.mean(EPE_LL)
        print(f'EPE_LL_avg={EPE_LL_avg}')
        EPE_SF_avg = np.mean(EPE_SF)
        print(f'EPE_SF_avg={EPE_SF_avg}')
        EPE_LF_avg = np.mean(EPE_LF)
        print(f'EPE_LF_avg={EPE_LF_avg}')

        PSNR_RE_avg = np.mean(PSNR_RE)
        print(f'PSNR_RE_avg={PSNR_RE_avg}')
        PSNR_LT_avg = np.mean(PSNR_LT)
        print(f'PSNR_LT_avg={PSNR_LT_avg}')
        PSNR_LL_avg = np.mean(PSNR_LL)
        print(f'PSNR_LL_avg={PSNR_LL_avg}')
        PSNR_SF_avg = np.mean(PSNR_SF)
        print(f'PSNR_SF_avg={PSNR_SF_avg}')
        PSNR_LF_avg = np.mean(PSNR_LF)
        print(f'PSNR_LF_avg={PSNR_LF_avg}')

        SSIM_RE_avg = np.mean(SSIM_RE)
        print(f'SSIM_RE_avg={SSIM_RE_avg}')
        SSIM_LT_avg = np.mean(SSIM_LT)
        print(f'SSIM_LT_avg={SSIM_LT_avg}')
        SSIM_LL_avg = np.mean(SSIM_LL)
        print(f'SSIM_LL_avg={SSIM_LL_avg}')
        SSIM_SF_avg = np.mean(SSIM_SF)
        print(f'SSIM_SF_avg={SSIM_SF_avg}')
        SSIM_LF_avg = np.mean(SSIM_LF)
        print(f'SSIM_LF_avg={SSIM_LF_avg}')

        l =len(EPE_RE)+len(EPE_LT)+len(EPE_LL)+len(EPE_SF)+len(EPE_LF)
        EPE_avg = (np.sum(EPE_RE)+np.sum(EPE_LT)+np.sum(EPE_LL)+np.sum(EPE_SF)+np.sum(EPE_LF))/l
        EPE_avg = (EPE_RE_avg + EPE_LT_avg +EPE_LL_avg + EPE_SF_avg+ EPE_LF_avg)/5
        print(f'EPE_avg===={EPE_avg}')
        PSNR_avg = (np.sum(PSNR_RE) + np.sum(PSNR_LT) + np.sum(PSNR_LL) + np.sum(PSNR_SF) + np.sum(PSNR_LF)) / l
        PSNR_avg = (PSNR_RE_avg + PSNR_LT_avg + PSNR_LL_avg + PSNR_SF_avg + PSNR_LF_avg) / 5
        print(f'PSNR_avg===={PSNR_avg}')
        SSIM_avg = (np.sum(SSIM_RE) + np.sum(SSIM_LT) + np.sum(SSIM_LL) + np.sum(SSIM_SF) + np.sum(SSIM_LF)) / l
        SSIM_avg = (SSIM_RE_avg + SSIM_LT_avg + SSIM_LL_avg + SSIM_SF_avg + SSIM_LF_avg) / 5
        print(f'SSIM_avg===={SSIM_avg}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="folder_rolling_learn_8basis_block=8_RS/homographymodel_new_homoflow_loss(80000,RMSE+gen__8_basis,80000)_iter_272000.pth")
    parser.add_argument("--test_path", type=str, default="../data/testing_mix_RS(with_flow_and_RT_label)_depth/", help="path to test images")
    parser.add_argument("--rho", type=float, default=16, help="path to test images")
    args = parser.parse_args()
    test(args)