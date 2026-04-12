# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from ..loss.dice_loss import BEC_Jaccard_Loss,category_BEC_Jaccard_Loss,MMD_Loss
from torch.nn import init
from Core.loss import lovasz_losses as L

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class NestedUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super(NestedUNet, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.deepsupervision = True
        self.register_buffer('device_id', torch.IntTensor(1))

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)

    def forward(self, input, mask=None):

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))#512,512*32
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))#256.256*64
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))#128,128
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))#64,64
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            #print(output4.shape)
            # loss1 = self.compute_multi_loss(output1, mask)
            # loss2 = self.compute_multi_loss(output2, mask)
            # loss3 = self.compute_multi_loss(output3, mask)
            # loss4 = self.compute_multi_loss(output4, mask)
            # self._loss = (loss1 + loss2 + loss3 + loss4) / 4

            return output4

        else:
            output = self.final(x0_4)
            #self._loss = self.compute_multi_loss(output, mask)

            return output

    @property
    def loss(self):
        return self._loss

    def compute_multi_loss_categroy(self, outputs, targets):
        criterion = category_BEC_Jaccard_Loss()
        criterion_bce=BEC_Jaccard_Loss()
        loss = 0
        avg_feature_final_array=np.zeros(9)
        for i in range(self.num_classes):
            avg_feature_final_c=self.category_anchor_construction(outputs[:, i, :, :], targets[:, i, :, :])
            avg_feature_final_array[i] = avg_feature_final_c

        for i in range(self.num_classes):
            if i == 0:
                loss_tpc = criterion_bce(outputs=outputs[:, i, :, :], targets=targets[:, i, :, :])
                loss_tpr = criterion_bce(outputs=outputs[:, 8, :, :], targets=targets[:, 8, :, :])
                loss+=3*(loss_tpc+loss_tpr)/2

            elif i == 1:
                loss_tnc= criterion(outputs=outputs[:, i, :, :], targets=targets[:, i, :, :],avg_feature_final=avg_feature_final_array[i])
                loss+=loss_tnc
            elif i != 0 and i != 1 and i <= 7:
                loss_c = criterion_bce(outputs=outputs[:, i, :, :], targets=targets[:, i, :, :])
                loss+=loss_c

        loss /= self.num_classes-2
        return loss

    def compute_multi_loss_region(self, outputs, targets):
        criterion_bce=BEC_Jaccard_Loss()
        criterion_ce=BEC_Jaccard_Loss()
        loss = 0
        #print(avg_feature_final_array)
        for i in range(self.num_classes):
            if i == 0:
                loss_tpc = criterion_bce(outputs=outputs[:, i, :, :], targets=targets[:, i, :, :])
                loss_tpr = criterion_bce(outputs=outputs[:, 8, :, :], targets=targets[:, 8, :, :])
                loss+=loss_tpc+loss_tpr
            elif i != 0  and i <= 7:
                loss_c = criterion_bce(outputs=outputs[:, i, :, :], targets=targets[:, i, :, :])
                loss+=loss_c
        loss /= self.num_classes
        loss_lovasz = 0
        for i in range(self.num_classes):
            loss_lovasz += L.lovasz_hinge(outputs[:, i, :, :], targets[:, i, :, :], per_image=True)
        loss_lovasz /= self.num_classes
        loss = loss + loss_lovasz
        return loss
    def compute_multi_loss_region_NEW(self, outputs, targets):
        criterion_bce=BEC_Jaccard_Loss()
        loss = 0
        #print(avg_feature_final_array)

        outputs=torch.softmax(outputs, dim=1)

        for i in range(self.num_classes):
                loss_now = criterion_bce(outputs=outputs[:, i, :, :], targets=targets[:, i, :, :])

                loss+=loss_now
        loss /= self.num_classes
        loss_lovasz=0
        for i in range(self.num_classes):
            loss_lovasz+=L.lovasz_hinge(outputs[:, i, :, :],targets[:, i, :, :],per_image=True)
        loss_lovasz/=self.num_classes
        loss=loss+loss_lovasz
        return loss

    def compute_multi_loss(self, outputs, targets):
        criterion = BEC_Jaccard_Loss()
        loss = 0
        for i in range(self.num_classes):
            loss+=criterion(outputs=outputs[:,i,:,:], targets=targets[:,i,:,:])
        loss/=self.num_classes
        return loss
    def category_anchor_construction(self,outputs, targets):
        jaccard_target = (targets).float()
        avg_feature_final = (outputs * jaccard_target).sum() / (jaccard_target.sum()+1e-15)
        return avg_feature_final