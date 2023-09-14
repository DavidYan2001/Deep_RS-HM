import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset


def homo2flow(img,homo,flowsize):
    w,h = flowsize

    homoflow = np.zeros((h, w, 2))

    for i in range(h):
        for j in range(w):
            coordinate = homo @ np.array([j, i, 1])
            coordinate = coordinate / coordinate[2]

            homoflow[i, j, 0] = coordinate[0] - j
            homoflow[i, j, 1] = coordinate[1] - i

    return homoflow

def savedata(source_path, img_path,new_path, imsize):

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    pair_list = list(open(source_path))
    npy_path ='./Coordinate-v2/'

    count = 0

    for i in range(len(pair_list)):

        img_pair = pair_list[i]
        video_name = img_pair.split('/')[0]
        pari_id = img_pair.split(' ')


        if video_name in RE:

            if pari_id[0][-1] == 'M':
                img_1 = cv2.imread(img_path + pari_id[0][:-2],0)
                npy_name_1 =pari_id[0][:-2]
            else:
                img_1 = cv2.imread(img_path + pari_id[0],0)
                npy_name_1 = pari_id[0]

            # load img2
            if pari_id[1][-2] == 'M':
                img_2 = cv2.imread(img_path + pari_id[1][:-3],0)
                npy_name_2 = pari_id[1][:-3]
            else:
                img_2 = cv2.imread(img_path + pari_id[1][:-1],0)
                npy_name_2 = pari_id[1][:-1]

            npy_name = npy_name_1.split('/')[1] + '_' + npy_name_2.split('/')[1] + '.npy'
            npy_id = npy_path + npy_name

            SURF = cv2.xfeatures2d.SURF_create()
            keypoints_1 = SURF.detectAndCompute(img_1, mask=None)
            keypoints_2 = SURF.detectAndCompute(img_2, mask=None)

            matcher = cv2.BFMatcher_create()
            match_points = matcher.knnMatch(keypoints_1[1], keypoints_2[1], k=2)

            well_match_points = list()
            for first, second in match_points:
                if (first.distance < 0.7 * second.distance):
                    well_match_points.append(first)

            src = list()
            dst = list()
            for m in well_match_points:
                src.append(keypoints_1[0][m.queryIdx].pt)
                dst.append(keypoints_2[0][m.trainIdx].pt)

            h, w = img_2.shape

            src = np.array(src)
            dst = np.array(dst)

            Homo, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5)

            homoflow = homo2flow(img_1,Homo,imsize)
            img_1_ =torch.tensor(img_1).unsqueeze(2)
            img_1_ =torch.tensor(img_1).unsqueeze(2)


            #imgs_ori = torch.tensor(np.concatenate([img_1, img_2], axis=2)).permute(2, 0, 1).float()

            img_1 =cv2.resize(img_1,(1088,1088))
            img_2 =cv2.resize(img_2,(1088,1088))
            training_image = np.dstack((img_1, img_2))
            #training_image = np.resize(training_image,imsize)

            point_dic = np.load(npy_id, allow_pickle=True)
            data = point_dic.item()

            match_points =data['matche_pts']
            match_points = np.array(match_points)

            datum = (img_1,img_2,Homo,homoflow)

            if not os.path.exists(new_path):
                os.makedirs(new_path)

            np.save(new_path + '/' + ('%s' % (i)).zfill(6), datum)

            if (count + 1) % 50 == 0:
                print('--image number ', count + 1)

            count+=1

    print("Generate {} {} files from {} raw data...".format(count, new_path, len(pair_list)))





if __name__ == "__main__":

    work_dir = './Test_List.txt'
    img_path = './Test/'


    imsize = (256, 256) #图片尺寸调整
    savedata(work_dir,img_path, './homoflow_1088_1088',  imsize)

