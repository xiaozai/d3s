import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm

from torch.nn.modules.utils import _pair
from scipy import ndimage
import ml_collections
import copy
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# from torch.nn import Flatten
''' Torch 1.1.0, can not from torch.nn import Flatten '''
class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(1)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def conv_no_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes))

def conv3x3_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

def conv1x1_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)

def conv131_layer(input_dim, output_dim, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(conv1x1_layer(input_dim, output_dim),
                         conv(output_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                         conv1x1_layer(output_dim, output_dim))


'''In spired by ECCV2020
CBAM: Convolutional Block Attention Module
https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py#L80

channel attention + spatial attention

'''
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class spatial_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = conv_no_relu(2, 1, kernel_size=7, stride=1, padding=3)
    def forward(self, x):
        x_compress = self.compress(x)      # Bx2xHxW
        x_out = self.spatial(x_compress)   # Bx1xHxW
        scale = torch.sigmoid(x_out)       # Bx1xHxW
        return x * scale                   # BxCxHxW

class channel_attention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.input_channels = input_channels
        self.mlp = nn.Sequential(Flatten(),
                                 nn.Linear(input_channels, input_channels//reduction_ratio),
                                 nn.ReLU(),
                                 nn.Linear(input_channels//reduction_ratio, input_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                x_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'max':
                x_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'lp':
                x_pool = F.lp_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'lse':
                x_pool = logsumexp_2d(x)

            channel_att_raw = self.mlp(x_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class DepthNet(nn.Module):
    def __init__(self, d_dims, output_dims):
        super().__init__()
        self.conv = conv_no_relu(d_dims, output_dims, kernel_size=1, stride=1, padding=0)
        self.channel_attn = channel_attention(output_dims, reduction_ratio=output_dims//4) # 4, 16, 32, 64
        self.spatial_attn = spatial_attention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, f_d):
        f_d = self.conv(f_d)
        f_d = self.channel_attn(f_d) # BxCxHxW * BxCx1x1
        f_d = self.spatial_attn(f_d) # BxCxHxW * Bx1xHxW
        return f_d



class SegmNet(nn.Module):
    """ Network module for IoU prediction. Refer to the paper for an illustration of the architecture."""
    def __init__(self, mixer_channels=3, topk_pos=3, topk_neg=3):
        super().__init__()

        segm_input_dim = (64, 256, 512, 1024)
        segm_inter_dim = (4, 16, 32, 64)
        segm_dim = (64, 64)

        self.segment0 = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
        self.segment1 = conv_no_relu(segm_dim[0], segm_dim[1])

        self.mixer = conv(mixer_channels, segm_inter_dim[3])
        self.s3 = conv(segm_inter_dim[3], segm_inter_dim[2])

        self.s2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.s1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.s0 = conv(segm_inter_dim[0], segm_inter_dim[0])

        self.f2 = conv(segm_input_dim[2], segm_inter_dim[2])
        self.f1 = conv(segm_input_dim[1], segm_inter_dim[1])
        self.f0 = conv(segm_input_dim[0], segm_inter_dim[0])

        self.d3 = DepthNet(segm_input_dim[3], segm_inter_dim[3]) #
        self.d2 = DepthNet(segm_input_dim[2], segm_inter_dim[2])
        self.d1 = DepthNet(segm_input_dim[1], segm_inter_dim[1])
        self.d0 = DepthNet(segm_input_dim[0], segm_inter_dim[0])

        self.post2 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.post1 = conv(segm_inter_dim[1], segm_inter_dim[0])
        self.post0 = conv_no_relu(segm_inter_dim[0], 2)

        self.m2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.m1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.m0 = conv(segm_inter_dim[0], segm_inter_dim[0])

        self.pyramid_pred3 = conv_no_relu(segm_inter_dim[2], 2)
        self.pyramid_pred2 = conv_no_relu(segm_inter_dim[1], 2)
        self.pyramid_pred1 = conv_no_relu(segm_inter_dim[0], 2)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

    def forward(self, feat_test, feat_test_d, feat_train, feat_train_d, mask_train, test_dist=None, debug=False):

        f_test = self.segment1(self.segment0(feat_test[3]))    # [B, 1024, 24,24] -> [B, 64, 24, 24]
        f_train = self.segment1(self.segment0(feat_train[3]))  # [B, 1024, 24,24] -> [B, 64, 24, 24]

        f_test_d = self.d3(feat_test_d[3])    # [B, 1024, 24, 24] -> [1, 64, 24, 24]
        f_train_d = self.d3(feat_train_d[3])  # [B, 1024, 24, 24] -> [1, 64, 24, 24]

        # Fusion RGBD Features
        f_test = torch.max(f_test, f_test_d)
        f_train = torch.max(f_train, f_train_d)

        # reshape mask to the feature size
        mask_pos = F.interpolate(mask_train[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,384, 384] -> [1,1,24,24]
        mask_neg = 1 - mask_pos

        pred_pos, pred_neg = self.similarity_segmentation(f_test, f_train, mask_pos, mask_neg)

        pred_ = torch.cat((torch.unsqueeze(pred_pos, -1), torch.unsqueeze(pred_neg, -1)), dim=-1)
        pred_sm = F.softmax(pred_, dim=-1) # [1, 24, 24, 2]

        if test_dist is not None:
            ''' F + P + L , segm_layers: [B,3,24,24]'''
            dist = F.interpolate(test_dist[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,24,24]
            segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1),
                                     torch.unsqueeze(pred_pos, dim=1),
                                     dist), dim=1)
        else:
            ''' F + P , segm_layers: [B,2,24,24]'''
            segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1), torch.unsqueeze(pred_pos, dim=1)), dim=1)

        out = self.mixer(segm_layers)

        f_test_rgbd2 = torch.max(self.f2(feat_test[2]), self.d2(feat_test[2]))
        f_test_rgbd1 = torch.max(self.f1(feat_test[1]), self.d1(feat_test[1]))
        f_test_rgbd0 = torch.max(self.f0(feat_test[0]), self.d0(feat_test[0]))

        out3 = self.s3(F.interpolate(out, size=(f_test_rgbd2.shape[-2], f_test_rgbd2.shape[-1])))
        out2 = self.post2(F.interpolate(self.m2(f_test_rgbd2) + self.s2(out3), size=(f_test_rgbd1.shape[-2], f_test_rgbd1.shape[-1])))
        out1 = self.post1(F.interpolate(self.m1(f_test_rgbd1) + self.s1(out2), size=(f_test_rgbd0.shape[-2], f_test_rgbd0.shape[-1])))
        out0 = self.post0(F.interpolate(self.m0(f_test_rgbd0) + self.s0(out1), size=(f_test_rgbd0.shape[-2]*2, f_test_rgbd0.shape[-1]*2)))

        pred3 = self.pyramid_pred3(F.interpolate(out3, size=(out0.shape[-2], out0.shape[-1])))
        pred2 = self.pyramid_pred2(F.interpolate(out2, size=(out0.shape[-2], out0.shape[-1])))
        pred1 = self.pyramid_pred1(F.interpolate(out1, size=(out0.shape[-2], out0.shape[-1])))

        if not debug:
            return (out0, pred1, pred2, pred3)
        else:
            return (out0, pred1, pred2, pred3), None


    def similarity_segmentation(self, f_test, f_train, mask_pos, mask_neg):
        '''Song's comments:
            f_test / f_train: [1,64,24,24]
            mask_pos / mask_neg: [1,1,24,24] for each pixel, it belongs to foreground or background
            sim_resh : [1,24,24,576]
            sim_pos / sim_neg: [1,24,24,576] for each pixel, the similarity of f_test and f_train
            pos_map / pos_neg: [1,24,24]     for each pixel, the averaged TopK similarity score
        '''
        # first normalize train and test features to have L2 norm 1
        # cosine similarity and reshape last two dimensions into one
        sim = torch.einsum('ijkl,ijmn->iklmn',
                           F.normalize(f_test, p=2, dim=1),
                           F.normalize(f_train, p=2, dim=1))
        sim_resh = sim.view(sim.shape[0], sim.shape[1], sim.shape[2], sim.shape[3] * sim.shape[4])
        # reshape masks into vectors for broadcasting [B x 1 x 1 x w * h]
        # re-weight samples (take out positive ang negative samples)
        sim_pos = sim_resh * mask_pos.view(mask_pos.shape[0], 1, 1, -1) # [1,24,24,576] * [1,1,1,576] -> [1,24,24,576]
        sim_neg = sim_resh * mask_neg.view(mask_neg.shape[0], 1, 1, -1)
        # take top k positive and negative examples
        # mean over the top positive and negative examples
        pos_map = torch.mean(torch.topk(sim_pos, self.topk_pos, dim=-1).values, dim=-1)
        neg_map = torch.mean(torch.topk(sim_neg, self.topk_neg, dim=-1).values, dim=-1)
        return pos_map, neg_map
