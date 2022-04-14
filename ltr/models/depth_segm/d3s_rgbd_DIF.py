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

def conv1x1_no_relu(in_planes, out_planes):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(out_planes))

def conv131_layer(input_dim, output_dim, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(conv1x1_no_relu(input_dim, output_dim),
                         conv(output_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                         conv1x1_no_relu(output_dim, output_dim))


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
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return scale

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

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class DWNet(nn.Module):
    def __init__(self, rgb_dims, d_dims, output_dims):
        super().__init__()
        self.conv_rgb = conv_no_relu(rgb_dims, output_dims, kernel_size=1, stride=1, padding=0)
        self.conv_d = conv_no_relu(d_dims, output_dims, kernel_size=1, stride=1, padding=0)
        self.channel_attn_rgb = channel_attention(output_dims, reduction_ratio=output_dims//4) # 4, 16, 32, 64
        self.spatial_attn_d = spatial_attention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, f_rgb, f_d):
        ''' According to the visualization in d3s_rgbd_CBAM02,
        the spatial attention is mostly like Depth

        So, maybe we can directly use the depth feature map as the spatial weight on RGB feature
            f_d = conv(f_d) => Bx1xHxW
            f_rgb' = channel_attn(f_rgb, f_d) = f_rgb * f_d
        '''
        # Mapping the channels
        f_rgb = self.conv_rgb(f_rgb)
        f_rgb = self.channel_attn_rgb(f_rgb)

        # get spatial attn from D feature
        f_d = self.conv_d(f_d)
        f_d = self.channel_attn_rgb(f_d)
        attn_d = self.spatial_attn_d(f_d)

        # Intergration
        f_rgbd = f_rgb * attn_d + f_rgb

        return f_rgbd, attn_d

class DepthNet(nn.Module):
    def __init__(self, input_dim=1, inter_dim=(4, 16, 32, 64)):
        super().__init__()

        self.conv0 = conv131_layer(input_dim, inter_dim[0])
        self.conv1 = conv131_layer(inter_dim[0], inter_dim[1])
        self.conv2 = conv131_layer(inter_dim[1], inter_dim[2])
        self.conv3 = conv131_layer(inter_dim[2], inter_dim[3])

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dp):
        ''' It seems the DepthNet is too sallow and did not learn anything '''
        feat0 = self.conv0(dp)    # 192, 192
        feat1 = self.conv1(feat0) # 96, 96
        feat2 = self.conv2(feat1) # 48, 48
        feat3 = self.conv3(feat2) # 24, 24

        return [feat0, feat1, feat2, feat3] # [4, 16, 32, 64]


class SegmNet(nn.Module):
    """ Network module for IoU prediction. Refer to the paper for an illustration of the architecture."""
    def __init__(self, mixer_channels=3, topk_pos=3, topk_neg=3):
        super().__init__()

        segm_input_dim = (64, 256, 512, 1024)
        segm_inter_dim = (4, 16, 32, 64)
        segm_dim = (64, 64)

        self.depth_feat_extractor = DepthNet(input_dim=1, inter_dim=segm_inter_dim)

        self.segment0 = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
        self.segment1 = conv_no_relu(segm_dim[0], segm_dim[1])

        self.mixer = conv(mixer_channels, segm_inter_dim[2])
        self.s3 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.s0 = conv(segm_inter_dim[1], segm_inter_dim[0])

        self.f2 = conv(segm_input_dim[2], segm_inter_dim[2])
        self.f1 = conv(segm_input_dim[1], segm_inter_dim[1])
        self.f0 = conv(segm_input_dim[0], segm_inter_dim[0])

        self.post_f = conv_no_relu(segm_inter_dim[0], 2)
        self.post_i = conv_no_relu(segm_inter_dim[1], 2)
        # self.post_d = conv_no_relu(segm_inter_dim[0], 2)

        self.m2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.m1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.m0 = conv(segm_inter_dim[1], segm_inter_dim[0])

        # Convert Stacked RGB features to a single feature map
        self.conv_rgb = conv1x1_no_relu(32+16+4, segm_inter_dim[1])
        self.conv_d = conv1x1_no_relu(segm_inter_dim[1], segm_inter_dim[1])
        # self.conv_d2 = conv1x1_no_relu(65, segm_inter_dim[0])

        self.rgbd_fusion_i = DWNet(segm_inter_dim[3], segm_inter_dim[3], segm_inter_dim[3])
        self.rgbd_fusion_f = DWNet(segm_inter_dim[1], segm_inter_dim[1], segm_inter_dim[1])

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

    def forward(self, feat_test, feat_test_d, feat_train, feat_train_d, mask_train, test_dist=None, debug=False):
        # Fusion RGBD Features
        f_test = self.segment1(self.segment0(feat_test[3]))    # 1x1x64 conv + 3x3x64 conv -> [1, 1024, 24,24] -> [1, 64, 24, 24]
        f_train = self.segment1(self.segment0(feat_train[3]))  # 1x1x64 conv + 3x3x64 conv -> [1, 1024, 24,24] -> [1, 64, 24, 24]
        #
        f_test, attn_i = self.rgbd_fusion_i(f_test, feat_test_d[3])
        f_train, _ = self.rgbd_fusion_i(f_train, feat_train_d[3])
        #
        mask_pos = F.interpolate(mask_train[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,384, 384] -> [1,1,24,24]
        mask_neg = 1 - mask_pos

        ''' F + P + L , segm_layers: [B,3,24,24]'''
        pred_pos, pred_neg = self.similarity_segmentation(f_test, f_train, mask_pos, mask_neg)
        pred_ = torch.cat((torch.unsqueeze(pred_pos, -1), torch.unsqueeze(pred_neg, -1)), dim=-1)
        pred_sm = F.softmax(pred_, dim=-1)                                              # [1, 24, 24, 2]
        # distance map is give - resize for mixer
        dist = F.interpolate(test_dist[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,24,24]
        segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1),
                                 torch.unsqueeze(pred_pos, dim=1),
                                 dist), dim=1)

        ''' Initial Segment'''
        out_i = self.mixer(segm_layers)                         # B, 3, 24, 24    -> B, 32, 24, 24
        out_i = self.s3(F.upsample(out_i, scale_factor=8))      # B, 32, 96, 96 -> B, 16, 192, 192
        pred_i = self.post_i(F.upsample(out_i, scale_factor=2)) # B, 16, 192, 192 -> B, 2, 384, 384

        ''' Finetune Segment '''
        f_rgb = self.conv_rgb(torch.cat((F.upsample(self.f2(feat_test[2]), scale_factor=4),   # B, 32, 48, 48
                                         F.upsample(self.f1(feat_test[1]), scale_factor=2),   # B, 16, 96, 96
                                         self.f0(feat_test[0])), dim=1))                      # B, 4, 192, 192
                                                                                # B, 64+32+16, 192, 192 -> B, 16, 192, 192
        f_d = self.conv_d(F.upsample(feat_test_d[1], scale_factor=2))           # B, 16,  96,  96 -> B, 16, 192, 192
        f_rgbd, attn_f = self.rgbd_fusion_f(f_rgb, f_d)                         # B, 16, 192, 192 -> B, 16, 192, 192

        out_f = self.post_f(F.upsample(self.m0(f_rgbd) + self.s0(out_i), scale_factor=2)) # -> B, 4, 192, 192 -> B, 2, 384, 384

        # ''' DepthNet, weak-supervision '''
        # out_d = F.upsample(torch.cat((feat_test_d[3], dist), dim=1), scale_factor=8) # B, 65, 192, 192
        # out_d = self.post_d(F.upsample(self.conv_d2(out_d), scale_factor=2))

        if not debug:
            return (out_f, pred_i)
        else:
            return (out_f, pred_i), (attn_f, attn_i)

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
