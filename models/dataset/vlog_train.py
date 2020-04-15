from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random

from utils.imutils2 import *
from utils.transforms import *
import torchvision.transforms as transforms

import scipy.io as sio
import scipy.misc
from geotnf.transformation import GeometricTnf

import torchvision.transforms as transforms

class VlogSet(data.Dataset):
    def __init__(self, params, is_train=True, frame_gap=1, augment=['crop', 'flip', 'frame_gap']):

        self.filelist = params['filelist']
        self.batchSize = params['batchSize']
        self.imgSize = params['imgSize']
        self.cropSize = params['cropSize']
        self.cropSize2 = params['cropSize2']
        self.videoLen = params['videoLen']
        # prediction distance, how many frames far away
        # yidadaa: predDistance 参数的作用是什么？
        # 通读代码后发现，predDistance 除了提供一个边界之外，似乎并没有用处
        self.predDistance = params['predDistance']
        # offset x,y parameters
        self.offset = params['offset']
        # gridSize = 3
        self.gridSize = params['gridSize']

        self.is_train = is_train
        self.frame_gap = frame_gap

        self.augment = augment

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.fnums = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            fnum = int(rows[1])

            self.jpgfiles.append(jpgfile)
            self.fnums.append(fnum)

        f.close()
        # 使用 GeometricTnf 执行可微采样，这里主要根据变换矩阵从图像中获取图像块
        self.geometricTnf = GeometricTnf('affine', out_h=params['cropSize2'], out_w=params['cropSize2'], use_cuda = False)

    def cropimg(self, img, offset_x, offset_y, cropsize):

        img = im_to_numpy(img)
        cropim = np.zeros([cropsize, cropsize, 3])
        cropim[:, :, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize, :]
        cropim = im_to_torch(cropim)

        return cropim

    def cropimg_np(self, img, offset_x, offset_y, cropsize):

        cropim = np.zeros([cropsize, cropsize])
        cropim[:, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize]

        return cropim

    def processflow(self, flow):
        '''把 flow scale 到 [-60, 60] 之间'''
        boundnum = 60
        flow = flow.astype(np.float)
        flow = flow / 255.0
        flow = flow * boundnum * 2
        flow = flow - boundnum

        flow = np.abs(flow)

        return flow


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        fnum = self.fnums[index]

        imgs = torch.Tensor(self.videoLen, 3, self.cropSize, self.cropSize)
        imgs_target  = torch.Tensor(2, 3, self.cropSize, self.cropSize)
        patch_target = torch.Tensor(2, 3, self.cropSize, self.cropSize)
        patch_target2 = torch.Tensor(2, 3, self.cropSize2, self.cropSize2)

        future_imgs  = torch.Tensor(2, 3, self.cropSize, self.cropSize)

        toflip = False
        if random.random() <= 0.5:
            toflip = True

        # 从数据集中选取同一个视频中的 videoLen + predDistance 帧连续帧
        frame_gap = self.frame_gap
        current_len = (self.videoLen  + self.predDistance) * frame_gap
        startframe = 0
        future_idx = current_len

        if fnum >= current_len:
            diffnum = fnum - current_len
            startframe = random.randint(0, diffnum)
            future_idx = startframe + current_len - 1
        else:
            newLen = int(fnum * 2.0 / 3.0)
            diffnum = fnum - newLen
            startframe = random.randint(0, diffnum)
            frame_gap = float(newLen - 1) / float(current_len)
            future_idx = int(startframe + current_len * frame_gap)

        # 默认裁剪偏移为 -1
        crop_offset_x = -1
        crop_offset_y = -1
        # 控制裁剪比例在 4:3 到 3:4之间
        ratio = random.random() * (4/3 - 3/4) + 3/4
        # reading video
        # 先读取长度为 videoLen 的视频帧
        for i in range(self.videoLen):

            nowid = int(startframe + i * frame_gap)
            # img_path = folder_path + "{:02d}.jpg".format(nowid)
            # specialized for fouhey format
            newid = nowid + 1
            img_path = folder_path + "{:06d}.jpg".format(newid)

            img = load_image(img_path)  # CxHxW

            ht, wd = img.size(1), img.size(2)
            if ht <= wd:
                ratio  = float(wd) / float(ht)
                # width, height
                img = resize(img, int(self.imgSize * ratio), self.imgSize)
            else:
                ratio  = float(ht) / float(wd)
                # width, height
                img = resize(img, self.imgSize, int(self.imgSize * ratio))


            if crop_offset_x == -1:
                crop_offset_x = random.randint(0, img.size(2) - self.cropSize - 1)
                crop_offset_y = random.randint(0, img.size(1) - self.cropSize - 1)

            img = self.cropimg(img, crop_offset_x, crop_offset_y, self.cropSize)
            assert(img.size(1) == self.cropSize)
            assert(img.size(2) == self.cropSize)

            if self.is_train:
                # Flip
                if toflip:
                    img = torch.from_numpy(fliplr(img.numpy())).float()

            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img = color_normalize(img, mean, std)

            imgs[i] = img.clone()

        # reading img
        # img_path = folder_path + "{:02d}.jpg".format(future_idx)
        # specialized for fouhey format

        # 选取之后的两帧作为预测帧，imgs_target
        for i in range(2):

            newid = int(future_idx + 1 + i * frame_gap)
            if newid > fnum:
                newid = fnum
            img_path = folder_path + "{:06d}.jpg".format(newid)

            img = load_image(img_path)  # CxHxW
            ht, wd = img.size(1), img.size(2)
            newh, neww = ht, wd
            if ht <= wd:
                ratio  = float(wd) / float(ht)
                # width, height
                img = resize(img, int(self.imgSize * ratio), self.imgSize)
                newh = self.imgSize
                neww = int(self.imgSize * ratio)
            else:
                ratio  = float(ht) / float(wd)
                # width, height
                img = resize(img, self.imgSize, int(self.imgSize * ratio))
                newh = int(self.imgSize * ratio)
                neww = self.imgSize

            img = self.cropimg(img, crop_offset_x, crop_offset_y, self.cropSize)
            assert(img.size(1) == self.cropSize)
            assert(img.size(2) == self.cropSize)

            if self.is_train:
                if toflip:
                    img = torch.from_numpy(fliplr(img.numpy())).float()

            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            img = color_normalize(img, mean, std)

            future_imgs[i] = img

        for i in range(2):
            imgs_target[i] = future_imgs[i].clone()

        # 根据帧间的像素绝对误差，来确定应该取哪个图像块来作为追踪块
        flow_cmb = future_imgs[0] - future_imgs[1]
        flow_cmb = im_to_numpy(flow_cmb)
        flow_cmb = flow_cmb.astype(np.float)
        flow_cmb = np.abs(flow_cmb)

        # cropSize 用来确定在原始图像中的裁剪大小，保证图像是方形的
        # 此数据集中是 240 * 240
        side_edge = self.cropSize
        box_edge  = int(side_edge / self.gridSize)

        # 遍历所有图像块
        lblxset = []
        lblyset = []
        scores  = []

        # 疑惑，如果这里 gridSize 为 3，那么此处的循环是没有意义的
        # 除非 gridSize 默认为 9，然后才能和 box_edge * 3 搭配使用，从而生成 80 * 80 的图像块
        # 所以根据推测，此处 gridSize 应该为 9，即 parser 中的默认参数
        for i in range(self.gridSize - 2):
            for j in range(self.gridSize - 2):

                offset_x1 = i * box_edge
                offset_y1 = j * box_edge
                lblxset.append(i)
                lblyset.append(j)
                # tpatch 是指某个图像块之间图像的 L1 灰度距离
                tpatch = flow_cmb[offset_y1: offset_y1 + box_edge * 3, offset_x1: offset_x1 + box_edge * 3].copy()
                tsum = tpatch.sum()
                scores.append(tsum)

        # 根据灰度距离从小到大排序，取 top10 差距最大的
        # 然后在 top10 中随机选取一个图像块作为最终 patch
        scores = np.array(scores)
        ids = np.argsort(scores)
        ids = ids[-10: ]
        lbl = random.randint(0, 9)
        lbl = ids[lbl]

        lbl_x = lblxset[lbl]
        lbl_y = lblyset[lbl]


        if self.is_train:
            if toflip:
                lbl_x = self.gridSize - 3 - lbl_x

        # lbl   = lbl_x * (self.gridSize - 2) + lbl_y

        # 缩放到 [0, 1]，用来确定具体的像素位置
        xloc = lbl_x / 6.0
        yloc = lbl_y / 6.0

        theta_aff = np.random.rand(6)
        scale = 1.0 - 1.0 / 3.0
        # 为图像块的起始位置加上随机偏移
        randnum = (np.random.rand(2) - 0.5) / 6.0
        xloc = xloc + randnum[0]
        yloc = yloc + randnum[1]

        if xloc < 0:
            xloc = 0.0
        if xloc > 1:
            xloc = 1.0

        if yloc < 0:
            yloc = 0.0
        if yloc > 1:
            yloc = 1.0

        # 根据选取的图像块，生成变换矩阵
        # [-45, 45]
        alpha = (np.random.rand(1)-0.5)*2*np.pi*(1.0/4.0)

        theta_aff[2] = (xloc * 2.0 - 1.0) * scale
        theta_aff[5] = (yloc * 2.0 - 1.0) * scale
        theta_aff[0] = 1.0 / 3.0 *np.cos(alpha)
        theta_aff[1] = 1.0 / 3.0 *(-np.sin(alpha))
        theta_aff[3] = 1.0 / 3.0 *np.sin(alpha)
        theta_aff[4] = 1.0 / 3.0 *np.cos(alpha)

        theta = torch.Tensor(theta_aff.astype(np.float32))
        theta = theta.view(1, 2, 3)
        theta_batch = theta.repeat(2, 1, 1) # repeat 两次，因为 imgs_target.size[0] == 2
        # 根据 theta 变换矩阵执行采样，生成图像块
        patch_target = self.geometricTnf(image_batch=imgs_target, theta_batch=theta_batch)
        theta = theta.view(2, 3)

        # TODO: 疑惑，既然这里只需要输出一个 target image，为啥上面的采样过程要对两张图像都做呢？
        imgs_target = imgs_target[0:1]
        patch_target = patch_target[0:1]


        meta = {'folder_path': folder_path, 'startframe': startframe, 'future_idx': future_idx, 'frame_gap': float(frame_gap), 'crop_offset_x': crop_offset_x, 'crop_offset_y': crop_offset_y, 'dataset': 'vlog'}

        return imgs, imgs_target, patch_target, theta, meta

    def __len__(self):
        return len(self.jpgfiles)
