import torch
import torch.nn as nn
import numpy as np

from .cheby import ChebyFunction
import einops as ein


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(degree + 1, input_dim, output_dim))
        self.tanh_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.tanh_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Chebyshev polynomial is defined in [-1, 1]

        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x * self.tanh_scale + self.tanh_bias)
        
        # Initialize Chebyshev polynomial tensors
        cheby = ChebyFunction.apply(x, self.degree); 

        # print(type(cheby))

        '''
        Due to some strange bug here, einsum does not backprop properly.
        It took me ages to find this strange bug.
        y = torch.einsum('dbi,dio->bo', cheby, self.cheby_coeffs)  # (d b i), (d i o) -> (d b o)
        y = ein.einsum(cheby, self.cheby_coeffs, 'd b i, d i o -> b o')
        '''

        # y = ein.einsum(cheby, self.cheby_coeffs, 'd b i, d i o -> b o')

        # Compute the Chebyshev interpolation
        y = torch.bmm(cheby, self.cheby_coeffs)  # (d b i), (d i o) -> (d b o)
        y = torch.sum(y, dim=0)  # (d b o) -> (b o), sum

        y = y.view(-1, self.outdim)
        return y