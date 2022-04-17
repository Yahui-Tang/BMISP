import torch
import torch.nn as nn
import numpy as np


class ColoringNet(nn.Module):
    def __init__(self):
        super(ColoringNet, self).__init__()
        self.condition_net = ConditionNet()
        self.render_net = RenderingNet()

    def forward(self, srgb, xyz, manu_xyz):
        condition = self.condition_net(srgb, xyz)
        out = self.render_net(manu_xyz, condition)
        return out


class Resample(nn.Module):
    def __init__(self, size=256):
        super(Resample, self).__init__()
        self.size = size

    def forward(self, x):
        inds_1 = torch.LongTensor(np.linspace(0, x.size(2), self.size, endpoint=False)).to(
            device=x.device)
        inds_2 = torch.LongTensor(np.linspace(0, x.size(3), self.size, endpoint=False)).to(
            device=x.device)
        resample_x = x.index_select(2, inds_1)
        resample_x = resample_x.index_select(3, inds_2)
        return resample_x


class Res_module(nn.Module):
    def __init__(self, inch):
        super(Res_module, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(inch, inch, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inch, inch, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, input_map):
        res = self.res(input_map)
        return input_map + res


class ConditionNet(nn.Module):
    def __init__(self):
        super(ConditionNet, self).__init__()
        self.net = nn.Sequential(
            Resample(),
            nn.Conv2d(6, 32, (7, 7), (1, 1), (3, 3)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 48, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(48, 64, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1, True),
            Res_module(64),
            Res_module(64),
            Res_module(64),
            Res_module(64),
            nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 320, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(320, 256, (1, 1), (1, 1), (0, 0)),
        )

    def forward(self, srgb, xyz):
        return self.net(torch.cat((xyz, srgb), dim=1))


class CFTLayer(nn.Module):
    def __init__(self):
        super(CFTLayer, self).__init__()
        self.CFT_scale_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (1, 1), (1, 1), (0, 0))
        )

        self.CFT_shift_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (1, 1), (1, 1), (0, 0))
        )

    def forward(self, feature, condition):
        scale = self.CFT_scale_conv(condition)
        shift = self.CFT_shift_conv(condition)
        return feature * (scale + 1) + shift


class ResBlock_CFT(nn.Module):
    def __init__(self):
        super(ResBlock_CFT, self).__init__()
        self.cft0 = CFTLayer()
        self.conv0 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True)
        )
        self.cft1 = CFTLayer()
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

    def forward(self, feature, condition):
        fea = self.cft0(feature, condition[:, :32, :, :])
        fea = self.conv0(fea)
        fea = self.cft1(fea, condition[:, 32:, :, :])
        fea = self.conv1(fea)
        return fea + feature


class RenderingNet(nn.Module):
    def __init__(self):
        super(RenderingNet, self).__init__()
        self.ResBlock_1 = ResBlock_CFT()
        self.ResBlock_2 = ResBlock_CFT()
        self.ResBlock_3 = ResBlock_CFT()
        self.ResBlock_4 = ResBlock_CFT()

        self.head_conv = nn.Sequential(
            nn.Conv2d(3, 64, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(inplace=True)
        )

        self.tail_conv = nn.Sequential(
            nn.Conv2d(64, 3, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid()
        )

    def forward(self, input, condition):
        fea_head = self.head_conv(input)
        fea_1 = self.ResBlock_1(fea_head, condition[:, :64, :, :])
        fea_2 = self.ResBlock_2(fea_1, condition[:, 64:128, :, :])
        fea_3 = self.ResBlock_3(fea_2, condition[:, 128:192, :, :])
        fea_4 = self.ResBlock_4(fea_3, condition[:, 192:, :, :])
        out = self.tail_conv(fea_4 + fea_head)
        return out
