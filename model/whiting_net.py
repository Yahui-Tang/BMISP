import torch
import numpy as np
import torch.nn as nn


class Resample(nn.Module):
    def __init__(self, size=128):
        super(Resample, self).__init__()
        self.size = size

    def forward(self, x):
        inds_1 = torch.LongTensor(np.linspace(0, x.size(2), self.size, endpoint=False)).to(device=x.device)
        inds_2 = torch.LongTensor(np.linspace(0, x.size(3), self.size, endpoint=False)).to(device=x.device)
        resample_x = x.index_select(2, inds_1)
        resample_x = resample_x.index_select(3, inds_2)
        return resample_x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SPALayer(nn.Module):
    def __init__(self, channel):
        super(SPALayer, self).__init__()
        self.spa = nn.Sequential(
            nn.Conv2d(channel, channel, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), padding=(6, 6), dilation=(6, 6)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), padding=(6, 6), dilation=(6, 6)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, 1, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid()
        )

    def forward(self, input):
        return input * self.spa(input)


class GAM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GAM, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim, out_dim, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.CA = CALayer(out_dim)
        self.SPA = SPALayer(out_dim)

    def forward(self, feature_map):
        out = self.net(feature_map)
        out = self.CA(out)
        out = self.SPA(out)
        return out


class Global_Net_Att(nn.Module):
    def __init__(self, in_dim):
        super(Global_Net_Att, self).__init__()
        self.net = nn.Sequential(
            Resample(128),
            GAM(in_dim, in_dim // 2),
            GAM(in_dim // 2, in_dim // 4),
            GAM(in_dim // 4, in_dim // 8),
            nn.Conv2d(in_dim // 8, 9, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.AdaptiveAvgPool2d(1),
            nn.Tanh()
        )

    def forward(self, feature_map):
        return self.net(feature_map)


class RCAB(nn.Module):
    def __init__(self, indim):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(indim, indim, (3, 3), (1, 1), (1, 1)),
            nn.InstanceNorm2d(indim, affine=True),
            nn.ReLU(),
            nn.Conv2d(indim, indim, (3, 3), (1, 1), (1, 1)),
            nn.InstanceNorm2d(indim, affine=True),
        )

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Multi_RCAB(nn.Module):
    def __init__(self, indim, depth=3):
        super(Multi_RCAB, self).__init__()
        rcab = []
        for i in range(depth):
            rcab.append(RCAB(indim))
        self.multi_rcab = nn.Sequential(*rcab)
        self.tail_conv = nn.Sequential(
            nn.Conv2d(indim, indim, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        res = self.multi_rcab(x)
        res += x
        out = self.tail_conv(res)
        return out


class DownSample_module(nn.Module):
    def __init__(self, inch, outch):
        super(DownSample_module, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inch, outch, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(outch, outch, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input_map):
        return self.net(input_map)


class RCAB_Upsample_module(nn.Module):
    def __init__(self, in_indim, out_indim, rcab_depth=3):
        super(RCAB_Upsample_module, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_indim, in_indim, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_indim, out_indim, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            Multi_RCAB(out_indim, rcab_depth),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(out_indim, out_indim, (5, 5), (1, 1), (2, 2)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        out = self.net(input)
        return out


class Local_Net(nn.Module):
    def __init__(self, in_dim, mid_dim=32):
        super(Local_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid_dim, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, feature_map):
        return self.net(feature_map)


class Encoder(nn.Module):
    def __init__(self, shape=[32, 64, 128]):
        super(Encoder, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, shape[0], (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(shape[0], shape[0], (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
        )
        self.downsample_block1 = DownSample_module(shape[0], shape[1])
        self.downsample_block2 = DownSample_module(shape[1], shape[2])

    def forward(self, img):
        x1 = self.head(img)
        x2 = self.downsample_block1(x1)
        x3 = self.downsample_block2(x2)

        return x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, shape=[32, 64, 128]):
        super(Decoder, self).__init__()
        self.upsample_block1 = RCAB_Upsample_module(shape[2], shape[1], rcab_depth=5)
        self.upsample_block2 = RCAB_Upsample_module(shape[1] * 2, shape[0], rcab_depth=5)
        self.tail_block = nn.Sequential(
            nn.Conv2d(shape[0] * 2, shape[0] * 2, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(shape[0] * 2, shape[0], (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x1, x2, x3):
        y3 = self.upsample_block1(x3)
        y2 = self.upsample_block2(torch.cat((y3, x2), dim=1))
        y1 = self.tail_block(torch.cat((y2, x1), dim=1))

        return y1


class Unet(nn.Module):
    def __init__(self, shape=[32, 64, 128]):
        super(Unet, self).__init__()
        self.encoder = Encoder(shape)
        self.decoder = Decoder(shape)

    def forward(self, img):
        x1, x2, x3 = self.encoder(img)
        y = self.decoder(x1, x2, x3)
        return x3, y


class whiting_Net(nn.Module):
    def __init__(self, shape=[32, 64, 128]):
        super(whiting_Net, self).__init__()
        self.unet = Unet(shape)
        self.global_net = Global_Net_Att(shape[2])

        self.local_affine_net = Local_Net(shape[0])

    def forward(self, srgb):
        x3, y = self.unet(srgb)
        M = self.local_affine_net(y)
        V = self.global_net(x3)

        x_vector = V[:, :3, :, :]
        y_vector = V[:, 3:6, :, :]
        z_vector = V[:, 6:, :, :]

        srgb_img = srgb * M[:, :3, :, :]

        x_m = torch.sum(x_vector * srgb_img, dim=1)
        y_m = torch.sum(y_vector * srgb_img, dim=1)
        z_m = torch.sum(z_vector * srgb_img, dim=1)

        xyz = torch.cat([x_m.unsqueeze(1), y_m.unsqueeze(1), z_m.unsqueeze(1)], dim=1)

        return xyz, M





