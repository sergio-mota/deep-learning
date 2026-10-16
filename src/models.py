import torch
import torch.nn as nn
import torch.nn.functional as F

from src.basic_models import CNN, FC


class CNNCIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = CNN(3, 32, 3, padding=1, use_batchnorm=True, use_relu=True, pool_size=2, pool_stride=2)
        self.cnn2 = CNN(32, 64, 3, padding=1, use_batchnorm=True, use_relu=True, pool_size=2, pool_stride=2)
        self.cnn3 = CNN(64, 128, 3, padding=1, use_batchnorm=True, use_relu=True, pool_size=2, pool_stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = FC(128 * 4 * 4, 256, True, use_batchnorm=True, use_relu=True)
        self.fc2 = FC(256, 10, True)

    def forward(self, x):
        x = self.cnn3(self.cnn2(self.cnn1(x)))
        x = self.dropout(torch.flatten(x, 1))
        x = self.fc2(self.fc1(x))
        return x

class BasicBlock(nn.Module):
    """
    The standard residual block used in ResNet-18.
    Consists of two 3x3 convolutional layers with a shortcut connection.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolution layer of the block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer of the block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection if dimensions change (due to stride > 1 or channel mismatch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # The skip connection addition
        out = F.relu(out)
        return out


class CifarResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # 1. CIFAR-Specific Stem Layer: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 2. The 4 ResNet stages (2 blocks per stage for ResNet-18)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 3. Final pooling and classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # The first block handles downsampling via its stride parameter
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # Resolves to [Batch, 64, 32, 32]
        out = self.layer1(out)                # Resolves to [Batch, 64, 32, 32]
        out = self.layer2(out)                # Resolves to [Batch, 128, 16, 16]
        out = self.layer3(out)                # Resolves to [Batch, 256, 8, 8]
        out = self.layer4(out)                # Resolves to [Batch, 512, 4, 4]
        out = self.avgpool(out)               # Resolves to [Batch, 512, 1, 1]
        out = torch.flatten(out, 1)           # Resolves to [Batch, 512]
        out = self.linear(out)                # Resolves to [Batch, 10]
        return out


def ResNet18_Cifar(num_classes=10):
    return CifarResNet18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
