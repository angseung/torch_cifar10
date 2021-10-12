'''
CrossLink Network
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU


def swish(x):
    return x * x.sigmoid()


def mish(x):
    return x * torch.tanh(F.softplus(x))


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class CrossLinkBlock(nn.Module):
    '''Cross-Link Block'''

    def __init__(self, in_channels, out_channels, kernel_size, group_size, pool_enable):
        super(CrossLinkBlock, self).__init__()

        self.pool_enable = pool_enable
        self.group_size = group_size

        # basic blocks
        self.dconv1_1 = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=kernel_size[0],
                                  stride=1,
                                  padding='same',
                                  groups=in_channels,
                                  bias=False)

        self.dconv1_2 = nn.Conv2d(out_channels,
                                  out_channels,
                                  kernel_size=kernel_size[1],
                                  stride=1,
                                  padding='same',
                                  groups=out_channels,
                                  bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.pconv1_1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               groups=group_size,
                               bias=False)

        self.pconv1_2 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               groups=group_size,
                               bias=False)

        self.activation = ReLU()

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        '''add forward here'''

        out1 = self.dconv1_1(x)
        out2 = channel_shuffle(x, self.group_size)

        out1 = self.activation(out1)
        out2 = self.pconv1_1(out2)

        out1 = channel_shuffle(out1, self.group_size)
        out2 = self.dconv1_2(out2)

        out1 = self.pconv1_2(out1)
        out2 = self.activation(out2)

        out = self.bn1(out1 + out2)

        if self.pool_enable:
            out = self.maxpool(out)

        return out


class CrossLinkBlock_bn(nn.Module):
    '''Cross-Link Block'''

    def __init__(self, in_channels, out_channels, kernel_size, group_size, pool_enable):
        super(CrossLinkBlock_bn, self).__init__()

        self.pool_enable = pool_enable
        self.group_size = group_size

        # basic blocks
        self.dconv1_1 = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=kernel_size[0],
                                  stride=1,
                                  padding='same',
                                  groups=in_channels,
                                  bias=False)

        self.dconv1_2 = nn.Conv2d(out_channels,
                                  out_channels,
                                  kernel_size=kernel_size[1],
                                  stride=1,
                                  padding='same',
                                  groups=out_channels,
                                  bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.bn3 = nn.BatchNorm2d(out_channels)

        self.pconv1_1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               groups=group_size,
                               bias=False)

        self.pconv1_2 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding='same',
                               groups=group_size,
                               bias=False)

        self.activation = ReLU()

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        '''add forward here'''

        out1 = self.dconv1_1(x)
        out2 = channel_shuffle(x, self.group_size)

        out1 = self.activation(self.bn1(out1))
        out2 = self.pconv1_1(out2)

        out1 = channel_shuffle(out1, self.group_size)
        out2 = self.dconv1_2(self.bn2(out2))

        out1 = self.pconv1_2(out1)
        out2 = self.activation(out2)

        out = self.bn3(out1 + out2)

        if self.pool_enable:
            out = self.maxpool(out)

        return out


class CLNET(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(CLNET, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32,
                               16,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.layers = self._make_layers(in_channels=16)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['out_channels', 'kernel_size', 'pool_enable', 'group_size']]


        for out_channels, kernel_size, pool_enable, group_size in zip(*cfg):
            layers.append(
                CrossLinkBlock(in_channels,
                               out_channels,
                               kernel_size,
                               group_size,
                               pool_enable))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = mish(self.bn1(self.pool1(self.conv1(x))))  # conv block
        out = self.conv3(swish(self.bn2(self.conv2(out))))  # sep block
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


class CLNET_bn(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(CLNET_bn, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32,
                               16,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.layers = self._make_layers(in_channels=16)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['out_channels', 'kernel_size', 'pool_enable', 'group_size']]


        for out_channels, kernel_size, pool_enable, group_size in zip(*cfg):
            layers.append(
                CrossLinkBlock_bn(in_channels,
                               out_channels,
                               kernel_size,
                               group_size,
                               pool_enable))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = mish(self.bn1(self.pool1(self.conv1(x))))  # conv block
        out = self.conv3(swish(self.bn2(self.conv2(out))))  # sep block
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def CLNet_V10(num_classes):
    cfg = {
        'out_channels': [24, 40, 40, 80, 80, 80, 112, 160, 160, 320],
        'kernel_size': [(3, 5), (5, 5), (5, 3), (3, 3), (3, 3), (3, 3), (3, 5), (5, 5), (5, 3), (3, 3)],
        'pool_enable': [True, False, True, False, False, False, True, True, False, False],
        'dropout_rate': 0.2,
        'group_size' : [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    }
    return CLNET(cfg, num_classes=num_classes)


def CLNet_V11(num_classes):
    cfg = {
        'out_channels': [24, 24, 40, 40, 80, 80, 80, 112, 160, 160, 320],
        'kernel_size': [(3, 3), (3, 5), (5, 5), (5, 3), (3, 3), (3, 3), (3, 3), (3, 5), (5, 5), (5, 3), (3, 3)],
        'pool_enable': [False, True, False, True, False, False, False, True, True, False, False],
        'dropout_rate': 0.2,
        'group_size' : [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    }
    return CLNET(cfg, num_classes=num_classes)


def CLNet_V11_bn(num_classes):
    cfg = {
        'out_channels': [24, 24, 40, 40, 80, 80, 80, 112, 160, 160, 320],
        'kernel_size': [(3, 3), (3, 5), (5, 5), (5, 3), (3, 3), (3, 3), (3, 3), (3, 5), (5, 5), (5, 3), (3, 3)],
        'pool_enable': [False, True, False, True, False, False, False, True, True, False, False],
        'dropout_rate': 0.2,
        'group_size' : [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    }
    return CLNET_bn(cfg, num_classes=num_classes)


def CLNet_V12_bn(num_classes):
    cfg = {
        'out_channels': [24, 24, 24, 40, 40, 80, 80, 80, 112, 160, 160, 320],
        'kernel_size': [(3, 3), (3, 3), (3, 5), (5, 5), (5, 3), (3, 3), (3, 3), (3, 3), (3, 5), (5, 5), (5, 3), (3, 3)],
        'pool_enable': [False, False, True, False, True, False, False, False, True, True, False, False],
        'dropout_rate': 0.2,
        'group_size' : [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    }
    return CLNET_bn(cfg, num_classes=num_classes)


import torchinfo


def test():
    net = CLNet_V12_bn(10)
    torchinfo.summary(net, (1, 3, 32, 32))
    x = torch.randn(3, 3, 224, 224, device='cuda')
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()
