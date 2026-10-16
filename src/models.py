import torch

from src.basic_models import CNN, FC


class CIFAR10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = CNN(3, 32, 3, padding=1, use_batchnorm=True, use_relu=True, pool_size=2, pool_stride=2)
        self.cnn2 = CNN(32, 64, 3, padding=1, use_batchnorm=True, use_relu=True, pool_size=2, pool_stride=2)
        self.cnn3 = CNN(64, 128, 3, padding=1, use_batchnorm=True, use_relu=True, pool_size=2, pool_stride=2)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = FC(128 * 4 * 4, 256, True, use_batchnorm=True, use_relu=True)
        self.fc2 = FC(256, 10, True)

    def forward(self, x):
        x = x + self.cnn1(x)
        x = x + self.cnn2(x)
        x = x + self.cnn3(x)
        x = self.dropout(torch.flatten(x, 1))
        x = self.fc2(self.fc1(x))
        return x
