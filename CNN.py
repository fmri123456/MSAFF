import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n):
        super(CNN, self).__init__()
        self.n = n
        self.conv_layers = nn.ModuleList()
        for i in range(n):  # 第一次到第n次卷积
            self.conv_layers.append(nn.Conv2d(1, 2**i, kernel_size=3))

    def forward(self, x):
        # 第一次到第n次卷积
        for i in range(self.n):
            x = F.relu(self.conv_layers[i](x))
            # 最大池化
            x, _ = torch.max(x,dim=1,keepdim=True)
            # 填充
            x = F.pad(x, (1, 1, 1, 1),mode='constant', value=1)
        return x
