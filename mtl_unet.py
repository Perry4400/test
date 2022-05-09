# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:37:44 2022

@author: admin
"""

import torch
import torch.nn as nn


def make_model(args):
    return MultiTaskLossWrapper(1, 2, args)


class ChannelAttention(nn.Module):
    # CBAM
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.active = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = self.active(x)
        return out


class conv(nn.Module):
    # 卷积层
    def __init__(self, in_channels, out_channels, args, flow):
        super(conv, self).__init__()
        self.dp = 0.3
        self.flow = flow
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels)
        self.pconv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.flow == 'M':
            self.prelu = nn.PReLU(in_channels)
        else:
            self.prelu = nn.PReLU(out_channels)
        self.dropout = nn.Dropout2d(p=self.dp)

    def forward(self, x):
        if self.flow == 'M':
            out = self.bn(self.conv(self.dropout(self.prelu(x))))
        elif self.flow == 'EX':
            out = self.dropout(self.prelu(self.bn(self.pconv1(self.conv1(x)))))
        else:
            out = self.dropout(self.prelu(self.bn(self.conv(x))))
        return out


# 双卷积
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, args, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            conv(in_channels, mid_channels, args=args, flow='E'),
            conv(mid_channels, out_channels, args=args, flow='E'),
        )

    def forward(self, x):
        return self.double_conv(x)


# 下采样层
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, args=args)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 提取特征
class UNet_Encoder(nn.Module):
    def __init__(self, n_channels, inchannels, args, bilinear=True):
        super(UNet_Encoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, inchannels * 2, args=args)
        self.down1 = Down(inchannels * 2, inchannels * 4, args=args)
        self.down2 = Down(inchannels * 4, inchannels * 8, args=args)
        self.down3 = Down(inchannels * 8, inchannels * 16, args=args)
        factor = 2 if bilinear else 1
        self.down4 = Down(inchannels * 16, inchannels * 32 // factor, args=args)
        self.args = args
        if self.args.ABCD[1] == '1':
            self.ca1 = ChannelAttention(inchannels * 2, ratio=4)
            self.sa1 = SpatialAttention()
            self.ca2 = ChannelAttention(inchannels * 4, ratio=4)
            self.sa2 = SpatialAttention()
            self.ca3 = ChannelAttention(inchannels * 8, ratio=4)
            self.sa3 = SpatialAttention()
            self.ca4 = ChannelAttention(inchannels * 16, ratio=4)
            self.sa4 = SpatialAttention()
            self.ca5 = ChannelAttention(inchannels * 16, ratio=4)
            self.sa5 = SpatialAttention()

    def forward(self, x):
        x1 = self.inc(x)
        if self.args.ABCD[1] == '1':
            x1 = self.ca1(x1) * x1
            x1 = self.sa1(x1) * x1
            x2 = self.down1(x1)
            x2 = self.ca2(x2) * x2
            x2 = self.sa2(x2) * x2
            x3 = self.down2(x2)
            x3 = self.ca3(x3) * x3
            x3 = self.sa3(x3) * x3
            x4 = self.down3(x3)
            x4 = self.ca4(x4) * x4
            x4 = self.sa4(x4) * x4
            x5 = self.down4(x4)
            x5 = self.ca5(x5) * x5
            x5 = self.sa5(x5) * x5
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        return x5


# 输出层输出结果
class Local(nn.Module):
    def __init__(self, in_channel, inchannels, args):
        super(Local, self).__init__()
        self.args = args
        self.UE = UNet_Encoder(in_channel, inchannels, args=args)
        self.conv_EX1r = conv(inchannels * 16, inchannels * 24, args=args, flow='E')
        self.conv_EX2r = conv(inchannels * 24, inchannels * 32, args=args, flow='E')
        if self.args.ABCD[1] == '1':
            self.ca1 = ChannelAttention(inchannels * 32, ratio=4)
            self.sa1 = SpatialAttention()
        self.avgr = nn.AdaptiveAvgPool2d((1, 1))
        self.fcr = nn.Linear(inchannels * 32, 1)

        self.conv_EX1d = conv(inchannels * 16, inchannels * 24, args=args, flow='E')
        self.conv_EX2d = conv(inchannels * 24, inchannels * 32, args=args, flow='E')
        if self.args.ABCD[1] == '1':
            self.ca2 = ChannelAttention(inchannels * 32, ratio=4)
            self.sa2 = SpatialAttention()
        self.avgd = nn.AdaptiveAvgPool2d((1, 1))
        self.fcd = nn.Linear(inchannels * 32, 1)

    def forward(self, x):
        out = self.UE(x)
        out_r = self.avgr(self.conv_EX2r(self.conv_EX1r(out)))
        if self.args.ABCD[1] == '1':
            out_r = self.ca1(out_r) * out_r
            out_r = self.sa1(out_r) * out_r
        r = self.fcr(out_r.view(out_r.size(0), out_r.size(1))).view(-1)
        out_d = self.avgd(self.conv_EX2d(self.conv_EX1d(out)))
        if self.args.ABCD[1] == '1':
            out_d = self.ca2(out_d) * out_d
            out_d = self.sa2(out_d) * out_d
            # 由于空间维度为1x1，这里只有通道注意力机制起作用
        d = self.fcd(out_d.view(out_d.size(0), out_d.size(1))).view(-1)
        return [r, d]


# 多任务损失
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, in_channel, task_num, args):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = Local(in_channel, inchannels=16, args=args)
        self.task_num = task_num
        self.log_vars2 = nn.Parameter(torch.zeros((task_num)))

    def forward(self, inputs, targets):
        outputs = self.model(inputs)

        precision1 = torch.exp(-self.log_vars2[0])
        loss = torch.sum(0.5 * precision1 * (targets[0] - outputs[0]) ** 2., -1) + 0.5 * self.log_vars2[0]

        precision2 = torch.exp(-self.log_vars2[1])
        loss += torch.sum(0.5 * precision2 * (targets[1] - outputs[1]) ** 2., -1) + 0.5 * self.log_vars2[1]

        return loss, self.log_vars2.data.tolist(), outputs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Radar Target Detection Based on Deep Neural Networks')
    parser.add_argument('--ABCD', type=str, default='11', help='丢弃率')
    # ABCD[0]没有意义，若ABCD[1]='1'，则加入注意力机制，否则不加入
    args = parser.parse_args()
    lenf = 101
    mtl = MultiTaskLossWrapper(1, 2, args=args)

    inputs = torch.randn(5, 1, lenf * 2, 18 * 18)
    r = torch.rand(5)
    z = torch.rand(5)
    targets = [r, z]
    loss, log_vars2, output = mtl(inputs, [r, z])
    print(loss, log_vars2, output)
