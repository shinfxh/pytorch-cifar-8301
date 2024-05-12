'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class BasicBlockNeg(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, a=False, last=False):
        super(BasicBlockNeg, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if a:
            self.a = nn.Parameter(torch.tensor(1.0))
        else:
            self.a = torch.tensor(1.0)
        self.last = last
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
        if self.last:
            out += self.shortcut(x) * (-self.a)
        else:
            out += self.shortcut(x) * self.a 
        out = F.relu(out)
        return out


class BottleneckNeg(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, a=False, last=False):
        super(BottleneckNeg, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()

        if a:
            self.a = nn.Parameter(torch.tensor(1.0))
        else:
            self.a = torch.tensor(1.0)
        self.last = last
        
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
        if self.last:
            out += self.shortcut(x) * (-self.a)
        else:
            out += self.shortcut(x) * self.a 
        out = F.relu(out)
        return out


class NegNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, a=False):
        super(NegNet, self).__init__()
        self.a = a
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, last=True)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride, last=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            if last and i == len(strides) - 1:
                layers.append(block(self.in_planes, planes, stride, a=self.a, last=last))
            else:
                layers.append(block(self.in_planes, planes, stride, a=self.a, last=False))
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


def NegNet18(num_classes=10, a=False):
    return NegNet(BasicBlockNeg, [2, 2, 2, 2], num_classes=num_classes, a=a)


def NegNet34(num_classes=10, a=False):
    return NegNet(BasicBlockNeg, [3, 4, 6, 3], num_classes=num_classes, a=a)


def NegNet50(num_classes=10, a=False):
    return NegNet(BottleneckNeg, [3, 4, 6, 3], num_classes=num_classes, a=a)


def NegNet101(num_classes=10, a=False):
    return NegNet(BottleneckNeg, [3, 4, 23, 3], num_classes=num_classes, a=a)


def ResNet152():
    return NegNet(BottleneckNeg, [3, 8, 36, 3])


def test():
    net = NegNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
