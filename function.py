# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch.autograd import Function
import math


def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2


def halftone_error_diffusion(input, alpha=0.4375, beta=0.1875, gamma=0.3125, delta=0.0625):
    s = input.shape
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = torch.zeros(size=s).to('cuda')

    for i in range(0, s[0]):
        for j in range(0, s[1]):
            output[i, j] = input[i, j].sign()

            quant_err = input[i, j] - output[i, j]

            if j+1 < s[1]:
                input[i, j + 1] += quant_err * alpha
            if i+1 < s[0]:
                if j-1 > 0:
                    input[i + 1, j - 1] += quant_err * beta
                input[i + 1, j] += quant_err * gamma
                if j+1 < s[1]:
                    input[i + 1, j + 1] += quant_err * delta
    return output


def halftone_ordered_dithering(input):
    bayer_matrix = torch.tensor([[0., 32, 8, 40, 2, 34, 10, 42],
                      [48, 16, 56, 24, 50, 18, 58, 26],
                      [12, 44, 4, 36, 14, 46, 6, 38],
                      [60, 28, 52, 20, 62, 30, 54, 22],
                      [3, 35, 11, 43, 1, 33, 9, 41],
                      [51, 19, 59, 27, 49, 17, 57, 25],
                      [15, 47, 7, 39, 13, 45, 5, 37],
                      [63,31, 55, 23, 61, 29, 53, 21]]).to(input.device)
    n = 8
    # bayer_matrix = torch.tensor([[0., 2], [3, 1]]).to(input.device)
    # n = 2
    bayer_matrix = (bayer_matrix + 1.0) / (n**2 / 2) - 1.0
    # print(bayer_matrix)
    s = input.shape
    bayer_matrix_s = bayer_matrix.repeat((s[0]//n+1, s[1]//n+1))[:s[0], :s[1]]
    out = input - bayer_matrix_s

    return out.sign()


class BinaryLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # # mean center
        # mean_weight = weight.mean()
        # weight = weight - mean_weight
        # # clamp
        # weight = weight.clamp(-1.0, 1.0)
        ctx.save_for_backward(input, weight, bias)

        weight_b = weight.sign()
        # weight_b = halftone_ordered_dithering(weight)

        output = input.mm(weight_b.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        weight_b = weight.sign()
        # weight_b = halftone_ordered_dithering(weight)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_b)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias


class BinaryLinearScalarFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        # size of input: [n, in_channels]
        # size of weight: [out_channels, in_channels]
        # size of bias: [out_channels]

        s = weight.size()
        n = s[1]
        m = weight.norm(1, dim=1, keepdim=True).div(n)
        weight_hat = weight.sign().mul(m.expand(s))
        output = input.mm(weight_hat.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        grad_input = grad_bias = None

        grad_weight = grad_output.t().mm(input)

        s = weight.size()
        n = s[1]
        m = weight.norm(1, dim=1, keepdim=True).div(n).expand(s)
        # print(m.shape, m)
        m[weight.lt(-1.0)] = 0
        m[weight.gt(1.0)] = 0
        m = m.mul(grad_weight)

        m_add = weight.sign().mul(grad_weight)
        m_add = m_add.sum(dim=1, keepdim=True).expand(s)
        m_add = m_add.mul(weight.sign()).div(n)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.sign())
        if ctx.needs_input_grad[1]:
            grad_weight = m.add(m_add)
            # grad_weight[weight.lt(-1.0)] = 0
            # grad_weight[weight.gt(1.0)] = 0
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias


class BinaryStraightThroughScalarD0Function(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # size of input: [n, in_channels]

        input_b = input.sign()
        s = input.size()
        alpha = input.norm(1, 0, keepdim=True).div(s[0]).expand(s)
        output = alpha.mul(input_b)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        grad_input = grad_output.clone()

        # grad_input = m.add(m_add)
        grad_input = grad_input * where(torch.abs(input[0]) <= 1, 1, 0)

        return grad_input


class BinaryStraightThroughScalarFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # size of input: [n, in_channels]

        input_b = input.sign()
        s = input.size()
        alpha = input.norm(1, 1, keepdim=True).div(s[1]).expand(s)
        output = alpha.mul(input_b)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        grad_input = grad_output.clone()

        # grad_input = m.add(m_add)
        grad_input = grad_input * where(torch.abs(input[0]) <= 1, 1, 0)

        return grad_input


class BinaryStraightThroughFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        grad_input = grad_output.clone()
        grad_input = grad_input * where(torch.abs(input[0]) <= 1, 1, 0)
        return grad_input


binary_linear = BinaryLinearFunction.apply
binary_linear_scalar = BinaryLinearScalarFunction.apply

bst = BinaryStraightThroughFunction.apply
sbst = BinaryStraightThroughScalarFunction.apply
sbst0 = BinaryStraightThroughScalarD0Function.apply
