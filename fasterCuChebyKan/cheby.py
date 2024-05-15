import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import faster_cheby_ops


class ChebyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, degree):
        """
        returns: cheby
        Note: degree does not require grad
        """
        # ctx.save_for_backward(x)

        batch_size, in_feats = x.size()
        cheby = x.new_ones((degree + 1, batch_size, in_feats))
        faster_cheby_ops.forward(x, cheby, degree)

        ctx.save_for_backward(x, cheby)

        return cheby


    @staticmethod
    def backward(ctx, grad_output): 
        # print(f'{grad_output.size()=}')
        x, cheby = ctx.saved_tensors

        grad_x = x.new_zeros(x.size())

        # print(f'{grad_output.size()}')

        faster_cheby_ops.backward(grad_output, x, cheby, grad_x)

        return grad_x, None # None for degree

















