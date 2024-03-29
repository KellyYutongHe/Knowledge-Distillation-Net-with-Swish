import math
from torch import nn
from torch.autograd import Function
import torch

import swish_cpp

torch.manual_seed(42)


class SWISHFunction(Function):
    @staticmethod
    def forward(ctx, x, beta):
        output = swish_cpp.forward(x, beta)[0]
        variables = [x, beta]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_y):
        grad_x, grad_beta = swish_cpp.backward(grad_y, *ctx.saved_variables)
        return grad_x, grad_beta
        # grad_x = swish_cpp.backward(grad_y, *ctx.saved_variables)[0]
        # return grad_x, None


class SWISH(nn.Module):
    def __init__(self):
        super(SWISH, self).__init__()
        # self.input_feature = input_feature
        # self.output_feature = output_feature
        self.beta = nn.Parameter(torch.Tensor(1))
        self.beta.data.uniform_(0, 1.5)

    def forward(self, x):
        return SWISHFunction.apply(x, self.beta)
