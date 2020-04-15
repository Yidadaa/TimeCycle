# from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable

import math
import copy
import numpy as np
from . import resnet_res4s1
from . import inflated_resnet
import torchvision

import torch.nn.functional as F
from geotnf.transformation import GeometricTnfAffine
from geotnf.loss import TransformedGridLoss, WeakInlierCountPool
from utils.torch_util import expand_dim

import random
import utils.imutils2


import time
import sys

class CycleTime(nn.Module):

    def __init__(self, class_num=8, dim_in=2048, trans_param_num=3, detach_network=False, pretrained=True, temporal_out=4, T=None, hist=1):
        '''
        Args:
            class_num: 类别数
            dim_in: 输入特征维度
            trans_param_num: 描述变换的参数数量，此处只需要三个参数即可
            temporal_out: 视频时序长度
        '''
        
        super(CycleTime, self).__init__()

        dim = 512
        print(pretrained)

        resnet = resnet_res4s1.resnet50(pretrained=pretrained)
        self.encoderVideo = inflated_resnet.InflatedResNet(copy.deepcopy(resnet))
        self.detach_network = detach_network
        self.hist = hist

        self.div_num = 512
        self.T = self.div_num**-.5 if T is None else T
        print('self.T:', self.T)

        self.afterconv1 = nn.Conv3d(1024, 512, kernel_size=1, bias=False)

        self.spatial_out1 = 30
        self.spatial_out2 = 10
        self.temporal_out = temporal_out

        self.afterconv3_trans = nn.Conv2d(self.spatial_out1 * self.spatial_out1, 128, kernel_size=4, padding=0, bias=False)
        self.afterconv4_trans = nn.Conv2d(128, 64, kernel_size=4, padding=0, bias=False)

        corrdim = 64 * 4 * 4
        corrdim_trans = 64 * 4 * 4

        self.linear2 = nn.Linear(corrdim_trans, trans_param_num)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.avgpool3d = nn.AvgPool3d((4, 2, 2), stride=(1, 2, 2))
        self.maxpool2d = nn.MaxPool2d(2, stride=2)


        # initialization

        nn.init.kaiming_normal_(self.afterconv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.afterconv3_trans.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.afterconv4_trans.weight, mode='fan_out', nonlinearity='relu')

        # assuming no fc pre-training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # transformation
        self.geometricTnf = GeometricTnfAffine(geometric_model='affine',
                                         tps_grid_size=3,
                                         tps_reg_factor=0.2,
                                         out_h=self.spatial_out2, out_w=self.spatial_out2,
                                         offset_factor=227/210)

        xs = np.linspace(-1,1,80)
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        self.xs = xs

        self.criterion_inlier = WeakInlierCountPool(geometric_model='affine', tps_grid_size=3, tps_reg_factor=0.2, h_matches=30, w_matches=30, use_conv_filter=False, dilation_filter=0, normalize_inlier_count=True)
        self.criterion_synth  = TransformedGridLoss(use_cuda=True, geometric_model='affine')


    def compute_corr_softmax(self, patch_feat1, r50_feat2, detach_corrfeat=False):
        '''计算两个特征图的响应矩阵，使用 softmax 做归一化，从而反应各个位置的特征相关性

        Args:
            patch_feat1: 图像块特征图
            r50_feat2: 图像特征图

        Return:
            corrfeat: 相关性矩阵
        '''
        T = r50_feat2.shape[2]

        if detach_corrfeat is True:
            r50_feat2 = r50_feat2.detach()

        r50_feat2 = r50_feat2.transpose(3, 4) # for the inlier counter
        r50_feat2 = r50_feat2.contiguous()
        r50_feat2_vec = r50_feat2.view(r50_feat2.size(0), r50_feat2.size(1), -1)
        r50_feat2_vec = r50_feat2_vec.transpose(1, 2)

        patch_feat1_vec = patch_feat1.view(patch_feat1.size(0), patch_feat1.size(1), -1)
        corrfeat = torch.matmul(r50_feat2_vec, patch_feat1_vec)

        corrfeat = torch.div(corrfeat, self.T)

        corrfeat  = corrfeat.view(corrfeat.size(0), T, self.spatial_out1 * self.spatial_out1, self.spatial_out2, self.spatial_out2)
        corrfeat  = F.softmax(corrfeat, dim=2)
        corrfeat  = corrfeat.view(corrfeat.size(0), T * self.spatial_out1 * self.spatial_out1, self.spatial_out2, self.spatial_out2)

        return corrfeat
    #
    def compute_corr_softmax2(self, patch_feat1, r50_feat2):
        T = r50_feat2.shape[2]

        # bs, channels, time, h2, w2
        r50_feat2 = r50_feat2.contiguous()
        r50_feat2_vec = r50_feat2.view(r50_feat2.size(0), r50_feat2.size(1), -1)
        r50_feat2_vec = r50_feat2_vec.transpose(1, 2)

        # bs, channels, h1, w1
        patch_feat1 = patch_feat1.transpose(2, 3)
        patch_feat1 = patch_feat1.contiguous()
        patch_feat1_vec = patch_feat1.view(patch_feat1.size(0), patch_feat1.size(1), -1)

        corrfeat = torch.matmul(r50_feat2_vec, patch_feat1_vec)
        corrfeat  = torch.div(corrfeat, self.T)
        corrfeat  = corrfeat.contiguous()

        corrfeat  = corrfeat.view(corrfeat.size(0), T, self.spatial_out2 * self.spatial_out2, self.spatial_out1 * self.spatial_out1)
        corrfeat  = F.softmax(corrfeat, dim=3)
        corrfeat  = corrfeat.transpose(2, 3)
        corrfeat  = corrfeat.contiguous()
        corrfeat  = corrfeat.view(corrfeat.size(0) * T, self.spatial_out1 * self.spatial_out1, self.spatial_out2, self.spatial_out2)

        return corrfeat


    def transform_trans_out(self, trans_out1):
        # 将 [tx, ty, s] 转换为矩阵形式，其中 s 是旋转角度
        # 1/3 * cos(s), -1/3 * sin(s), tx
        # 1/3 * sin(s),  1/3 * cos(s), ty
        # 之所以要乘上 1/3，是因为生成的 patch 是原图的 1/3
        # image(240 * 240) -> patch(80 * 80)
        trans_out1 = trans_out1.view(-1, 3)

        trans_out1_theta = trans_out1[:, 2]
        trans_out1_2 = trans_out1[:, 0].unsqueeze(1)
        trans_out1_5 = trans_out1[:, 1].unsqueeze(1)
        trans_out1_0 = 1.0 / 3.0 * torch.cos(trans_out1_theta).unsqueeze(1)
        trans_out1_1 = - 1.0 / 3.0 * torch.sin(trans_out1_theta).unsqueeze(1)
        trans_out1_3 = 1.0 / 3.0 * torch.sin(trans_out1_theta).unsqueeze(1)
        trans_out1_4 = 1.0 / 3.0 * torch.cos(trans_out1_theta).unsqueeze(1)
        trans_out1 = torch.cat((trans_out1_0, trans_out1_1, trans_out1_2, trans_out1_3, trans_out1_4, trans_out1_5), dim=1)

        trans_out1 = trans_out1.view(-1, 2, 3)

        return trans_out1


    def forward_base(self, x, contiguous=False, can_detach=True):
        '''
        Return:
            x: (, 512) 维度的特征
            x_pre: resnet 提取到的特征
            x_norm: x 的 l2 norm 之后的值
        '''
        # import pdb; pdb.set_trace()

        # patch feature
        x = x.transpose(1, 2)
        x_pre = self.encoderVideo(x)

        if self.detach_network and can_detach:
            x_pre = x_pre.detach()

        x = self.afterconv1(x_pre)
        x = self.relu(x)

        if contiguous:
            x = x.contiguous()
            x = x.view(x.size(0), x.size(1), x.size(3), x.size(4))

        x_norm = F.normalize(x, p=2, dim=1)

        return x, x_pre, x_norm

    def compute_transform_img_to_patch(self, query, base, temporal_out=1, detach_corrfeat=False):
        # 计算 query 块相对于 base 的变换参数
        # 参考论文的 section 3.2.1 localizer 部分，通过两个简单的卷积层生成转换参数 [tx, ty, s]，表示旋转平移和缩放
        # query is a patch, base is a volume with time depth 1
        corrfeat = self.compute_corr_softmax(query, base, detach_corrfeat=detach_corrfeat)

        # 2. predict transform with affinity as input
        corrfeat_mat = corrfeat.view(corrfeat.size(0) * temporal_out, self.spatial_out1 * self.spatial_out1, self.spatial_out2, self.spatial_out2)
        corrfeat_trans  = self.afterconv3_trans(corrfeat_mat)
        corrfeat_trans  = self.leakyrelu(corrfeat_trans)
        corrfeat_trans  = self.afterconv4_trans(corrfeat_trans)
        corrfeat_trans  = self.leakyrelu(corrfeat_trans)
        corrfeat_trans  = corrfeat_trans.view(corrfeat_trans.shape[0], -1)

        trans_theta = self.linear2(corrfeat_trans).contiguous().view(corrfeat_trans.shape[0], 3)
        trans_theta = self.transform_trans_out(trans_theta) # 预测出来的转换参数通过此函数转换为转换矩阵，方便下一步使用

        return corrfeat, corrfeat_mat, corrfeat_trans, trans_theta

    def forward(self, ximg1, patch2, img2, theta):
        '''

        Args:
            ximg1: 待追踪图像
            patch2: 初始图像中的 patch 块
            img2: 初始图像
        '''
        B, T = ximg1.shape[:2]

        # base features
        r50_feat1, r50_feat1_pre, r50_feat1_norm = self.forward_base(ximg1)

        # target patch feature
        patch2_feat2, patch2_feat2_pre, patch_feat2_norm = self.forward_base(patch2, contiguous=True)

        # target image feature
        img_feat2, img_feat2_pre, img_feat2_norm = self.forward_base(img2, contiguous=True, can_detach=False)


        # base features to crop with transformations
        r50_feat1_transform = r50_feat1.transpose(1, 2)
        channels = r50_feat1_transform.size(2)
        r50_feat1_transform = r50_feat1_transform.contiguous()

        # add original code
        # 使用卷积层预测图像块相对于参考图像的变换
        _, corrfeat_trans_matrix2, corrfeat_trans1, trans_out2 = self.compute_transform_img_to_patch(patch_feat2_norm, r50_feat1_norm, temporal_out=self.temporal_out)
        bs2 = corrfeat_trans1.size(0)

        # 使用预测出来的变换矩阵，从参考图像的特征图中抽取对应的特征块
        r50_feat1_transform_ori = r50_feat1_transform.view(bs2, channels, self.spatial_out1, self.spatial_out1)
        r50_feat1_transform_ori = self.geometricTnf(r50_feat1_transform_ori, trans_out2)

        # r50_feat1_transform_ori = r50_feat1_transform_ori.transpose(1, 2)

        # 此处的代码结构和 compute_transform_img_to_patch 中类似
        def skip_prediction(img_feat2_norm, r50_feat1_transform_ori):
            r50_feat1_transform_ori = r50_feat1_transform_ori.contiguous()
            r50_feat1_transform_ori = r50_feat1_transform_ori.view(B, self.temporal_out, r50_feat1_transform_ori.size(1),  self.spatial_out2, self.spatial_out2)
            r50_feat1_transform_ori = r50_feat1_transform_ori.transpose(1, 2)

            r50_feat1_transform_norm = F.normalize(r50_feat1_transform_ori, p=2, dim=1)
            corrfeat_trans_matrix_reverse = self.compute_corr_softmax2(img_feat2_norm, r50_feat1_transform_norm)

            corrfeat_trans_reverse  = self.afterconv3_trans(corrfeat_trans_matrix_reverse)
            corrfeat_trans_reverse  = self.leakyrelu(corrfeat_trans_reverse)
            corrfeat_trans_reverse  = self.afterconv4_trans(corrfeat_trans_reverse)
            corrfeat_trans_reverse  = self.leakyrelu(corrfeat_trans_reverse)
            corrfeat_trans_reverse  = corrfeat_trans_reverse.contiguous()
            corrfeat_trans_reverse  = corrfeat_trans_reverse.view(bs2, -1)

            trans_out3  = self.linear2(corrfeat_trans_reverse)
            trans_out3  = trans_out3.contiguous()
            trans_out3  = self.transform_trans_out(trans_out3)

            return trans_out3, corrfeat_trans_matrix_reverse

        # 数据流走向
        # (base_img) -> img_feat1
        # (patch_img) -> patch_feat2
        # (target_img) -> img_feat2
        # (patch_feat2_norm, img_feat1_norm) -> trans_out2
        # (img_feat1, transout2) -> transformed_patch_ori
        # (transformed_patch_ori, img_feat2_norm) -> trans_out3

        # 这里的 skip_prediction 相当于在 target image 和每一张 source image 间做 two-step skip cycle
        # 见论文 section 3.1.2 skip cycle 部分
        trans_out3, _ = skip_prediction(img_feat2_norm, r50_feat1_transform_ori)

        # 循环执行追踪策略
        def recurrent_align(init_query, idx):
            # global ximg1
            # global patch2

            # corr_feat_mats = []
            trans_thetas = []
            trans_feats = []

            # should be normalized patch query
            if self.hist > 1:
                cur_query = torch.stack([init_query]*self.hist)
            else:
                cur_query = init_query

            crops = []
            for t in idx:

                # 1. get affinity of current patch on current frame
                cur_base_norm = r50_feat1_norm[:, :, t:t+1]
                cur_base_feat = r50_feat1_transform[:, t]

                # 2. predict transform with affinity as input
                corrfeat, corrfeat_mat, corrfeat_trans, trans_theta = self.compute_transform_img_to_patch(
                    cur_query if self.hist == 1 else cur_query.mean(0),
                    cur_base_norm)

                # 3. get cropped features with transform
                cur_base_crop = self.geometricTnf(cur_base_feat, trans_theta)

                # bs, channels, time, h2, w2
                cur_base_crop_norm = F.normalize(cur_base_crop, p=2, dim=1)

                # cur_query = cur_base_crop_norm

                if self.hist > 1:
                    cur_query[:-1] = cur_query[1:]
                    cur_query[-1] = cur_base_crop_norm
                else:
                    cur_query = cur_base_crop_norm
                # cur_query = torch.stack([cur_])

                trans_thetas.append(trans_theta)
                trans_feats.append(cur_base_crop)
                # corr_feat_mats.append(corrfeat_mat)

            return trans_thetas, trans_feats #, trans_feats, corr_feat_mats

        # 开始实施环形追踪策略，分别在 T 个时间步内进行正向和反向追踪
        # backward:
        #   1   2   3   4   5   6
        #    <-  <-  <-  <-  <-
        #   |-     IMG     -|  TARGET 
        # forward:
        #   1   2   3   4   5   6
        #    ->  ->  ->  ->  ->
        #   |-     IMG     -|  TARGET 
        def cycle(TT=None):
            if TT is None:
                TT = T

            # propagate backward
            # back_trans_thetas, back_trans_feats, back_corr_feat_mats = \
            back_trans_thetas, back_trans_feats = \
                recurrent_align(patch_feat2_norm, list(range(T))[::-1][:TT])
            # propagate forward
            # forw_trans_thetas, forw_trans_feats, forw_corr_feat_mats = \
            forw_trans_thetas, forw_trans_feats = \
                recurrent_align(F.normalize(back_trans_feats[-1], p=2, dim=1), list(range(T))[T-TT+1:])

            # cycle back from last base frame to target
            last_ = forw_trans_feats[-1] if len(forw_trans_feats) > 0 else back_trans_feats[0]
            last_corrfeat, last_corrfeat_mat, last_corrfeat_trans, last_trans_theta = self.compute_transform_img_to_patch(
                F.normalize(last_, p=2, dim=1), img_feat2_norm.unsqueeze(2))
            last_trans_feat = self.geometricTnf(img_feat2, last_trans_theta)
            last_trans_feat_norm = F.normalize(last_trans_feat, p=2, dim=1)

            forw_trans_thetas.append(last_trans_theta)

            return back_trans_thetas, forw_trans_thetas, back_trans_feats

        # back_trans_thetas, back_trans_feats, back_corr_feat_mats, forw_trans_thetas, forw_trans_feats, forw_corr_feat_mats = \
        #     [], [], [], [], [], []

        outputs = [[], [], []]

        for c in range(1, T+1):
            # _back_trans_thetas, _back_trans_feats, _back_corr_feat_mats, _forw_trans_thetas, _forw_trans_feats, _forw_corr_feat_mats = cycle(c)
            _outputs = cycle(c)
            for i, o in enumerate(_outputs):
                outputs[i] += o

            if c == T:
                back_trans_feats = _outputs[-1]

        # 抽取长时间步的 track 中的 backward 追踪时得到的所有特征块，然后用其与 target image feature 做 skip prediction
        back_trans_feats = torch.stack(back_trans_feats).transpose(0,1).contiguous()
        back_trans_feats = back_trans_feats.view(-1, *back_trans_feats.shape[2:])
        # 理解此处的 skip prediction 和上面的 skip prediction 的联系：
        # 1. 上一次 skip prediction 是 source image 和 target image 的两两之间，先做图像块匹配，然后计算 cycle loss
        # 2. 本次 skip prediction 是在对 [source image, target image] 做 backward 追踪时，对 source image 中追踪到的
        # 每一个图像块，计算其相对于 target image 的 cycle loss
        skip_trans, skip_corrfeat_mat = skip_prediction(img_feat2_norm, back_trans_feats)

        # 输出：
        # outputs[:2]: cycle alignment 中的 backward thetas 和 forward thetas
        # patch2_feat2: target image 中的 patch image feature
        # theta: patch image 相对于 target image 原始 theta 变换矩阵
        # trans_out2: patch image 相对于每一张 source image 的 theta 变换矩阵
        # trans_out3: source images 中单点匹配到的每一个 matched patch 相对于 target image 的 theta 变换矩阵
        # skip_trans: source images 中连续匹配到的每一个 matched patch 相对于 target image 的 theta 变换矩阵
        # skip_corrfeat_mat： 计算 skip_trans 时产生的相关性矩阵，后续计算 loss 时并没有用到
        # corrfeat_trans_matrix2: 计算 trans_out2 时产生的相关性矩阵，用于计算 inlier loss
        return outputs[:2], patch2_feat2, theta, trans_out2, trans_out3, skip_trans, skip_corrfeat_mat, corrfeat_trans_matrix2


    def loss(self, outputs, patch_feat, theta, trans_out2, trans_out3, skip_trans, skip_corrfeat_mat, corrfeat_trans_matrix2):
        # patch_feat is patch of target frame, theta is crop transform for patch

        back_trans_thetas, forw_trans_thetas = outputs

        loss_targ_theta = []
        loss_targ_theta_skip = []
        loss_back_inliers = []

        # TODO: 这几行是干啥的？
        nn = list(range(len(forw_trans_thetas)))
        nn = [ii for ii in [sum(nn[:i]) - 1 for i in nn][2:] if ii < len(forw_trans_thetas)]

        # criterion_synth 就是 alignment objective
        # 直接对两个 theta 矩阵在 grid 上计算他们的对齐程度
        for i in nn:
            loss_targ_theta.append(self.criterion_synth(forw_trans_thetas[i], theta))

        theta2 = theta.unsqueeze(1)
        theta2 = theta2.repeat(1, self.temporal_out, 1, 1)
        theta2 = theta2.view(-1, 2, 3)

        # 连续匹配的 skip trans 矩阵计算 loss 时并没有用到
        # 因为中间可能会出现目标丢失的情况，所以使用单点匹配的 skip trans
        loss_targ_theta_skip.append(self.criterion_synth(trans_out3, theta2))

        # 这里的 inliner loss 是 semantic level 的特征匹配 loss
        # 参考：End-to-end weakly-supervised semantic alignment
        loss_inlier = self.criterion_inlier(matches=corrfeat_trans_matrix2, theta=trans_out2)
        loss_inlier = torch.mean(-loss_inlier)
        loss_back_inliers.append(loss_inlier)

        return (loss_targ_theta, loss_targ_theta_skip, loss_back_inliers)
