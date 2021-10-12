'''
CrossLink Network
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class SparseCrossLinkBlock(nn.Module):
    '''Depthwise Shuffle block'''

    def __init__(self, in_channels, out_channels, kernel_size, pool_enable, pgroup, shortcut_enable):
        super(SparseCrossLinkBlock, self).__init__()

        self.pool_enable = pool_enable
        self.shortcut_enable = shortcut_enable

        # basic blocks
        self.dconv1 = nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=kernel_size[0],
                                stride=1,
                                padding='same',
                                groups=in_channels,
                                bias=False
                                )

        self.dconv2 = nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=kernel_size[1],
                                stride=1,
                                padding='same',
                                groups=in_channels,
                                bias=False
                                )

        self.bn = nn.BatchNorm2d(out_channels)

        self.pconv1 = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=pgroup,
                                bias=False)

        self.pconv_shortcut = nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        '''add forward here'''

        out1 = self.dconv1(x)
        out2 = self.dconv2(x)
        out = out2 * F.softsign(out1) + out1 * F.softsign(out2)
        out = self.bn(self.pconv1(out))

        if self.shortcut_enable:
            out = out + self.pconv_shortcut(x)

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
        cfg = [self.cfg[k] for k in ['out_channels',
                                     'kernel_size',
                                     'pool_enable',
                                     'pgroup',
                                     'shortcut_enable']]

        for out_channels, kernel_size, pool_enable, pgroup, shortcut_enable in zip(*cfg):
            layers.append(
                SparseCrossLinkBlock(in_channels,
                                     out_channels,
                                     kernel_size,
                                     pool_enable,
                                     pgroup,
                                     shortcut_enable))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.pool1(self.conv1(x))))  # conv block
        out = self.conv3(F.relu(self.bn2(self.conv2(out))))  # sep block
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def CLNetV1_C1B0(num_classes):
    cfg = {
        'out_channels': [24, 40, 40, 80, 80, 80, 112, 160, 160, 320],
        'kernel_size': [(3, 5), (5, 5), (5, 3), (3, 3), (3, 3), (3, 3), (3, 5), (5, 5), (5, 3), (3, 3)],
        'pool_enable': [True, False, True, False, False, False, True, True, False, False],
        'pgroup': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'shortcut_enable': [True, True, True, True, True, True, True, True, True, True],
        'dropout_rate': 0.2
    }
    return CLNET(cfg, num_classes=num_classes)


def CLNetV1_C1B1(num_classes):
    cfg = {
        'out_channels': [24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 320],
        'kernel_size': [(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5),(3, 5)],
        'pool_enable': [False, True, False, False, True, False, False, False, False, True, False, True, False, False,
                        False],
        'pgroup': [2]*15,
        'shortcut_enable': [True]*15,
        'dropout_rate': 0.2
    }
    return CLNET(cfg, num_classes=num_classes)


import torchinfo


def test():
    net = CLNetV1_C1B1(10)
    torchinfo.summary(net, (1, 3, 32, 32))
    x = torch.randn(3, 3, 32, 32, device='cuda')
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()
