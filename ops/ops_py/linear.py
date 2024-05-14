import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import linear_op


class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)

        batch_size = input.size()[0]
        out_feats = weight.size()[0]

        # output = input.mm(weight.t())
        ##################################
        result = input.new_zeros((batch_size, out_feats))

        linear_op.forward(input, weight, bias, result)
        ##################################

        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.t().mm(input)

            ##################################
            batch_size, out_features = grad_output.size()
            _, in_features = input.size()
            grad_weight = grad_output.new_zeros((out_features, in_features))
            linear_op.backward_weight(grad_output, input, grad_weight)
            
            ##################################


        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

# 方便使用的帮助函数
def custom_linear(input, weight, bias=None):
    return CustomLinearFunction.apply(input, weight, bias)


class CustomLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(CustomLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # 初始化权重和偏置
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 权重初始化
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # 调用自定义函数
        return CustomLinearFunction.apply(input, self.weight, self.bias)





