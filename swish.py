import math
from torch import nn
from torch.autograd import Function
import torch

import swish_cpp

torch.manual_seed(42)


# class SWISHFunction(Function):
#     @staticmethod
#     def forward(ctx, x, beta):
#         output = swish_cpp.forward(x, beta)[0]
#         variables = [x, beta]
#         ctx.save_for_backward(*variables)
#
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_y):
#         # grad_x, grad_beta = swish_cpp.backward(grad_y, *ctx.saved_variables)
#         # return grad_x, grad_beta
#         grad_x = swish_cpp.backward(grad_y, *ctx.saved_variables)[0]
#         return grad_x, None


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


# class SWISH(nn.Module):
#     def __init__(self, input_features, state_size):
#         super(LLTM, self).__init__()
#         self.input_features = input_features
#         self.state_size = state_size
#         self.weights = nn.Parameter(
#             torch.Tensor(3 * state_size, input_features + state_size))
#         self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.state_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, +stdv)
#
#     def forward(self, input, state):
#         return LLTMFunction.apply(input, self.weights, self.bias, *state)
