import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import ChebyKANLayer

class ChebyNet(nn.Module):
    def __init__(self):
        super(ChebyNet, self).__init__()
        self.chebykan1 = ChebyKANLayer(28*28, 256, 4)
        self.ln1 = nn.LayerNorm(256) # To avoid gradient vanishing caused by tanh
        self.chebykan2 = ChebyKANLayer(256, 256, 4)
        self.ln2 = nn.LayerNorm(256)
        self.chebykan3 = ChebyKANLayer(256, 10, 4)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x

