from .linear import CustomLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import ChebyKANLayer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # MNIST图像大小是28x28
        self.linear1 = CustomLinear(28*28, 128)
        self.linear2 = CustomLinear(128, 64)
        self.linear3 = CustomLinear(64, 10)  # 输出层，10类数字

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ChebyNet(nn.Module):
    def __init__(self):
        super(ChebyNet, self).__init__()
        self.chebykan1 = ChebyKANLayer(28*28, 32, 4)
        self.ln1 = nn.LayerNorm(32) # To avoid gradient vanishing caused by tanh
        self.chebykan2 = ChebyKANLayer(32, 16, 4)
        self.ln2 = nn.LayerNorm(16)
        self.chebykan3 = ChebyKANLayer(16, 10, 4)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x

