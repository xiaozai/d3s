import torch
import torch.nn.functional as F
from pytracking.libs.tensorlist import tensor_operation, TensorList
from ltr.external.depthconv.functions import depth_conv


@tensor_operation
def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
            stride=1, padding=0, dilation=1, groups=1, mode=None, depth=None):
    """Standard conv2d. Returns the input if weight=None."""

    if weight is None:
        return input

    ind = None
    if mode is not None:
        if padding != 0:
            raise ValueError('Cannot input both padding and mode.')
        if mode == 'same':
            padding = (weight.shape[2]//2, weight.shape[3]//2)
            if weight.shape[2] % 2 == 0 or weight.shape[3] % 2 == 0:
                ind = (slice(-1) if weight.shape[2] % 2 == 0 else slice(None),
                       slice(-1) if weight.shape[3] % 2 == 0 else slice(None))
        elif mode == 'valid':
            padding = (0, 0)
        elif mode == 'full':
            padding = (weight.shape[2]-1, weight.shape[3]-1)
        else:
            raise ValueError('Unknown mode for padding.')

    if depth is None:
        out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    else:
        ''' DepthConv, same as DAL
        input : RGB features, during tracking is TensorList
        depth : depth crops, is TensorList
         '''
        if isinstance(depth, TensorList):
            depth = depth[0]
        if isinstance(input, TensorList):
            input = input[0]
        depth = F.interpolate(depth, size=(input.shape[-2], input.shape[-1]))
        # depth = depth.repeat(1, input.shape[1], 1, 1)
        depth = depth.to(input.device)
        # out = depth_conv(input, depth, weight, bias, stride=stride, padding=padding, dilation=dilation)
        out = depth_conv(input,depth,weight,None,stride=1,padding=padding,dilation=1)

    if ind is None:
        return out
    return out[:,:,ind[0],ind[1]]


@tensor_operation
def conv1x1(input: torch.Tensor, weight: torch.Tensor):
    """Do a convolution with a 1x1 kernel weights. Implemented with matmul, which can be faster than using conv."""

    if weight is None:
        return input

    return torch.matmul(weight.view(weight.shape[0], weight.shape[1]),
                        input.view(input.shape[0], input.shape[1], -1)).view(input.shape[0], weight.shape[0], input.shape[2], input.shape[3])
