import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import cffi
# from .._ext import depthconv
import torch.nn as nn

import torch.autograd as ag

try:
    from os.path import join as pjoin, dirname
    from torch.utils.cpp_extension import load as load_extension
    root_dir = pjoin(dirname(__file__), '../src_pytorch13')
    depthconv = load_extension(
        '_depthconv',
        [pjoin(root_dir, 'depthconv_cuda_redo.c'), pjoin(root_dir, 'depthconv_cuda_kernel.cu')],
        verbose=True
    )
except ImportError:
    raise ImportError('Can not compile depth-aware cnn library.')

__all__ = ['depth_conv']

def depth_conv(input,
                  depth,
                  weight,
                  bias,
                  stride=1,
                  padding=0,
                  dilation=1):

    if input is not None and input.dim() != 4:
        raise ValueError(
            "Expected 4D tensor as input, got {}D tensor instead.".format(
                input.dim()))

    f = DepthconvFunction()
    return f(input, weight, depth)
    # return DepthconvFunction.apply(input, weight, depth=depth)


class DepthconvFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, depth=None):
        ''' input: RGB features, [27, 64, 16, 16]
            weight: the DCF filter, []
            depth: Depth crops, [27, 1, 16, 16]

        '''
        print('in DepthconvFunction:', input.shape, weight.shape, depth.shape)
        def _output_size(input, weight, padding, dilation, stride):
            channels = weight.size(0)

            output_size = (input.size(0), channels)
            for d in range(input.dim() - 2):
                in_size = input.size(d + 2)
                pad = padding[d]
                kernel = dilation * (weight.size(d + 2) - 1) + 1
                stride = stride
                output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
                print('output_size: ', output_size)
            if not all(map(lambda s: s > 0, output_size)):
                raise ValueError(
                    "convolution input is too small (output would be {})".format(
                        'x'.join(map(str, output_size))))
            return output_size


        stride, padding, dilation = 1, (2, 2), 1
        bias = input.new(weight.size(0)).zero_()

        output_size = [int((input.size()[i + 2] + 2 * padding[i] - weight.size()[i + 2]) / stride + 1)
                       for i in range(2)]

        output = input.new(*_output_size(input, weight,  padding, dilation, stride)).zero_()
        columns = input.new(weight.size(1) * weight.size(2) * weight.size(3), output_size[0] * output_size[1]).zero_()
        ones = input.new(output_size[0] * output_size[1]).zero_()

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(input, torch.cuda.FloatTensor):
                raise NotImplementedError
            depthconv.depthconv_forward_cuda(
                    input, depth, weight, bias, output, columns, ones,
                    weight.size(3), weight.size(2), stride, stride,
                    padding[1], padding[0], dilation, dilation)

        ctx.save_for_backward(input, depth, weight, bias, columns, ones)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, depth, weight, bias, columns, ones = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        stride, padding, dilation = 1, (2, 2), 1
        output_size = [int((input.size()[i + 2] + 2 * padding[i] - weight.size()[i + 2]) / stride + 1)
                       for i in range(2)]

        if ctx.needs_input_grad[0]:
            grad_input = input.new(*input.size()).zero_()
            depthconv.depthconv_backward_input_cuda(
                input, depth, grad_output, grad_input,
                weight, columns,
                weight.size(3), weight.size(2), stride, stride,
                padding[1], padding[0], dilation, dilation)

        if ctx.needs_input_grad[2]:
            grad_weight = weight.new(*weight.size()).zero_()
            grad_bias = weight.new(*bias.size()).zero_()

            depthconv.depthconv_backward_parameters_cuda(
                input, depth, grad_output, grad_weight, grad_bias, columns,
                ones,
                weight.size(3), weight.size(2), stride, stride,
                padding[1], padding[0], dilation, dilation, 1)

        return grad_input, grad_weight
