from __future__ import print_function, division
import torch.nn as nn
import torch

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        self.max_pool = torch.nn.MaxPool3d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()


        fout = 15

        self.f = [32, 64, 128]#, 256, 512, 1024]

        self.down_block1 = UNet_down_block(15, self.f[0], False)
        self.down_block2 = UNet_down_block(self.f[0], self.f[1], True)
        self.down_block3 = UNet_down_block(self.f[1], self.f[2], True)
        # self.down_block4 = UNet_down_block(self.f[2], self.f[3], True)
        # self.down_block5 = UNet_down_block(self.f[3], self.f[4], True)
        # self.down_block6 = UNet_down_block(self.f[4], self.f[5], True)

        mf = self.f[-1]
        self.mid_conv1 = torch.nn.Conv3d(mf, mf, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(mf)
        self.mid_conv2 = torch.nn.Conv3d(mf, mf, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(mf)
        self.mid_conv3 = torch.nn.Conv3d(mf, mf, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(mf)

        # self.up_block5 = UNet_up_block(self.f[4], self.f[5], self.f[4])
        # self.up_block4 = UNet_up_block(self.f[3], self.f[4], self.f[3])
        # self.up_block3 = UNet_up_block(self.f[2], self.f[3], self.f[2])
        self.up_block2 = UNet_up_block(self.f[1], self.f[2], self.f[1])
        self.up_block1 = UNet_up_block(self.f[0], self.f[1], fout)

        self.last_conv1 = torch.nn.Conv3d(fout, fout, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm3d(fout)
        self.last_conv2 = torch.nn.Conv3d(fout, fout, 1, padding=0)

        self.drop = torch.nn.Dropout3d(p=0.4)
        self.relu = torch.nn.ReLU()

    def set_size(self, size):
        self.size = size

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.drop(self.x1))
        # self.x3 = self.down_block3(self.drop(self.x2))
        # self.x4 = self.down_block4(self.drop(self.x3))
        # self.x5 = self.down_block5(self.drop(self.x4))
        x = self.down_block3(self.drop(self.x2))

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        # x = self.up_block5(self.x5, self.drop(x))
        # del self.x5

        # x = self.up_block4(self.x4, self.drop(x))
        # del self.x4
        #
        # x = self.up_block3(self.x3, self.drop(x))
        # del self.x3

        x = self.up_block2(self.x2, self.drop(x))
        del self.x2

        x = self.up_block1(self.x1, self.drop(x))
        del self.x1

        x = self.relu(self.last_bn(self.last_conv1(self.drop(x))))
        x = self.last_conv2(self.drop(x))

        return x