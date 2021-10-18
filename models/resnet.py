'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SwishBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(SwishBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.silu(out)
        return out


class MishBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MishBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.mish(out)
        return out


class MobileBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MobileBasicBlock, self).__init__()

        self.dconv1 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.pconv1 = nn.Conv2d(in_planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.dconv2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding='same',
                                groups=in_planes,
                                bias=False
                                )

        self.pconv2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.dconv1(x)
        out = self.pconv1(out)
        out = torch.relu(self.bn1(out))

        out = self.dconv2(out)
        out = self.pconv2(out)
        out = torch.relu(self.bn2(out))

        out += self.shortcut(x)
        out = torch.relu(out)

        return out

class MobileBasicBlockSep(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MobileBasicBlockSep, self).__init__()

        self.dconv1_1 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.dconv1_2 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.pconv1 = nn.Conv2d(in_planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn1_1 = nn.BatchNorm2d(in_planes)

        self.bn1_2 = nn.BatchNorm2d(in_planes)

        self.dconv2_1 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.dconv2_2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.pconv2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn2_1 = nn.BatchNorm2d(planes)

        self.bn2_2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = self.bn1_1(self.dconv1_1(0.5 * x))
        out2 = self.bn1_2(self.dconv1_2(0.5 * x))

        out = self.pconv1(out1 + out2)

        out = torch.relu(out)

        out1 = self.bn2_1(self.dconv2_1(0.5 * out))
        out2 = self.bn2_2(self.dconv2_2(0.5 * out))

        out = self.pconv2(out1 + out2)

        out = torch.relu(out + self.shortcut(x))

        return out


class MobileBasicBlockComb(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MobileBasicBlockComb, self).__init__()

        self.dconv1_1 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.dconv1_2 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.pconv1 = nn.Conv2d(in_planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.dconv2_1 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.dconv2_2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.pconv2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = self.dconv1_1(0.5 * x)
        out2 = self.dconv1_2(0.5 * x)

        out = self.bn1(self.pconv1(out1 + out2))

        out = torch.relu(out)

        out1 = self.dconv2_1(0.5 * out)
        out2 = self.dconv2_2(0.5 * out)

        out = self.bn2(self.pconv2(out1 + out2))

        out = torch.relu(out + self.shortcut(x))

        return out

class MobileBasicBlockSepMish(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MobileBasicBlockSepMish, self).__init__()

        self.dconv1_1 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.dconv1_2 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.pconv1 = nn.Conv2d(in_planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn1_1 = nn.BatchNorm2d(in_planes)

        self.bn1_2 = nn.BatchNorm2d(in_planes)

        self.dconv2_1 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.dconv2_2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.pconv2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn2_1 = nn.BatchNorm2d(planes)

        self.bn2_2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = self.bn1_1(self.dconv1_1(0.5 * x))
        out2 = self.bn1_2(self.dconv1_2(0.5 * x))

        out = self.pconv1(out1 + out2)

        out = torch.relu(out)

        out1 = self.bn2_1(self.dconv2_1(0.5 * out))
        out2 = self.bn2_2(self.dconv2_2(0.5 * out))

        out = self.pconv2(out1 + out2)

        out = torch.relu(out + self.shortcut(x))

        return out


class MobileBasicBlockCombMish(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MobileBasicBlockCombMish, self).__init__()

        self.dconv1_1 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.dconv1_2 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.pconv1 = nn.Conv2d(in_planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.dconv2_1 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.dconv2_2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.pconv2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = self.dconv1_1(0.5 * x)
        out2 = self.dconv1_2(0.5 * x)

        out = self.bn1(self.pconv1(out1 + out2))

        out = torch.relu(out)

        out1 = self.dconv2_1(0.5 * out)
        out2 = self.dconv2_2(0.5 * out)

        out = self.bn2(self.pconv2(out1 + out2))

        out = torch.relu(out + self.shortcut(x))

        return out

class MobileBasicBlockSepSwish(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MobileBasicBlockSepSwish, self).__init__()

        self.dconv1_1 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.dconv1_2 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.pconv1 = nn.Conv2d(in_planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn1_1 = nn.BatchNorm2d(in_planes)

        self.bn1_2 = nn.BatchNorm2d(in_planes)

        self.dconv2_1 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.dconv2_2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.pconv2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn2_1 = nn.BatchNorm2d(planes)

        self.bn2_2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = self.bn1_1(self.dconv1_1(0.5 * x))
        out2 = self.bn1_2(self.dconv1_2(0.5 * x))

        out = self.pconv1(out1 + out2)

        out = torch.relu(out)

        out1 = self.bn2_1(self.dconv2_1(0.5 * out))
        out2 = self.bn2_2(self.dconv2_2(0.5 * out))

        out = self.pconv2(out1 + out2)

        out = torch.relu(out + self.shortcut(x))

        return out


class MobileBasicBlockCombSwish(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MobileBasicBlockCombSwish, self).__init__()

        self.dconv1_1 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.dconv1_2 = nn.Conv2d(in_planes,
                                in_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=in_planes,
                                bias=False
                                )

        self.pconv1 = nn.Conv2d(in_planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.dconv2_1 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.dconv2_2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=planes,
                                bias=False
                                )

        self.pconv2 = nn.Conv2d(planes,
                                planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=1,
                                bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = self.dconv1_1(0.5 * x)
        out2 = self.dconv1_2(0.5 * x)

        out = self.bn1(self.pconv1(out1 + out2))

        out = torch.relu(out)

        out1 = self.dconv2_1(0.5 * out)
        out2 = self.dconv2_2(0.5 * out)

        out = self.bn2(self.pconv2(out1 + out2))

        out = torch.relu(out + self.shortcut(x))

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(MobileBasicBlockCombSwish, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    torchinfo.summary(net, (1, 3, 32, 32))
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
if __name__ == '__main__':
    test()
