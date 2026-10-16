import torch


class CNN(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            *,
            padding: int = 0,
            use_batchnorm: bool = False,
            use_relu: bool = False,
            pool_size: int | None = None,
            pool_stride: int | None = None,
        ):
        super().__init__()
        use_pool = pool_size is not None and pool_stride is not None

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_batchnorm)
        self.bn = torch.nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.relu = torch.nn.ReLU() if use_relu else None
        self.pool = torch.nn.MaxPool2d(pool_size, pool_stride) if use_pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.relu(x) if self.relu is not None else x
        x = self.pool(x) if self.pool is not None else x
        return x

class FC(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_bias: bool,
            *,
            use_batchnorm: bool = False,
            use_relu: bool = False
        ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=use_bias and not use_batchnorm)
        self.bn = torch.nn.BatchNorm1d(out_features) if use_batchnorm else None
        self.relu = torch.nn.ReLU() if use_relu else None

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.relu(x) if self.relu is not None else x
        return x
