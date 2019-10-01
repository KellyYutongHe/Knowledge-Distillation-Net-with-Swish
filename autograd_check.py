from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.autograd import gradcheck
from swish import SWISHFunction


device = torch.device("cpu")

x = torch.randn(3, 3, dtype = torch.float64, device = device, requires_grad = True)
beta = torch.randn(1, dtype = torch.float64, device = device, requires_grad = True)


if gradcheck(SWISHFunction.apply, (x, beta), eps = 1e-3):
    print("OK")
