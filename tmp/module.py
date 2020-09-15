# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import math

import torch
import torch.nn as nn

from tmp.function import binary_linear, binary_linear_scalar, bst, sbst

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scalar=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scalar = scalar
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (math.sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        # 感觉bias is not None不用单独分出来，直接传就行。后面验证一下再改。
        if self.bias is not None:
            if self.scalar:
                return binary_linear_scalar(input, self.weight, self.bias)
            else:
                return binary_linear(input, self.weight, self.bias)
        return binary_linear(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class BinActive(nn.Module):
    def __init__(self, scalar=True):
        super(BinActive, self).__init__()
        self.scalar = scalar

    def forward(self, input):
        if self.scalar:
            return sbst(input)
        else:
            return bst(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

