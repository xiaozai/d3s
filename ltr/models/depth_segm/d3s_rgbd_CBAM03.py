import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm, Flatten
from torch.nn.modules.utils import _pair
from scipy import ndimage
import ml_collections
import copy
import math

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

def conv131_layer(input_dim, output_dim):
    return nn.Sequential(conv1x1_layer(input_dim, output_dim),
                         conv(output_dim, output_dim, kernel_size=3, stride=2),
                         conv1x1_layer(output_dim, output_dim))

# def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
#     valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
#             all(roi[:, 4] <= image_size[1]-1)
#     return valid


# def normalize_vis_img(x):
#     x = x - np.min(x)
#     x = x / np.max(x)
#     return (x * 255).astype(np.uint8)



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

        scale = F.sigmoid(channel_attm_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


# ''' Inspired by
#     ACNet : Attention based network to exploit complementary features for RGBD semantic segmentation
#     (ICIP2019)
#     https://github.com/anheidelonghu/ACNet
# '''
# def channel_attention(num_channel):
#     return nn.Sequential(nn.AdaptiveAvgPool2d(1),
#                          conv1x1_layer(num_channel, num_channel),
#                          nn.Sigmoid())

class CBAM(nn.Module):
    def __init__(self, rgb_dims, d_dims, output_dims):
        super().__init__()
        self.conv_rgb = conv(rgb_dims, output_dims, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_d = conv(d_dims, output_dims, kernel_size=1, stride=1, padding=0, dilation=1)

        self.channel_attn_rgb = channel_attention(output_dims, reduction_ratio=output_dims//4) # 4, 16, 32, 64
        self.channel_attn_d = channel_attention(output_dims, reduction_ratio=output_dims//4)

        self.spatial_attn_rgb = spatial_attention()
        self.spatial_attn_d = spatial_attention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, f_rgb, f_d):
        ''' channel attention + spatial attention '''
        f_rgb = self.conv_rgb(f_rgb)
        f_d = self.conv_d(f_d)

        f_rgb = self.channel_attn_rgb(f_rgb)
        f_d = self.channel_attn_rgb(f_d)

        ''' We calculate the weights for rgb and d features
            by using spatial attention for each pixel
            weights for each pixel across all channels (RGB, D)

            1) softmax(w_rgb, w_d)
            2) cat(f_rgb, f_d) along channels
            2) cat(channel_attn_rgb, channel_attn_d) -> spatial attention
        '''
        # spatial attention map, Bx1xHxW
        spa_rgb = self.spatial_attn_rgb(f_rgb)
        spa_d = self.spatial_attn_d(f_d)

        spatial_attn = F.softmax(torch.cat((spa_rgb, spa_d), dim=1), dim=1) # Bx2xHxW
        f_rgbd = f_rgb * spatial_attn[:, 0, :, :] + f_d * spatial_attn[:, 1, :, :]

        return f_rgbd, spatial_attn

class DepthNet(nn.Module):
    def __init__(self, input_dim=1, inter_dim=(4, 16, 32, 64)):
        super().__init__()

        self.conv0 = conv131_layer(input_dim, inter_dim[0])
        self.conv1 = conv131_layer(inter_dim[0], inter_dim[1])
        self.conv2 = conv131_layer(inter_dim[1], inter_dim[2])
        self.conv3 = conv131_layer(inter_dim[2], inter_dim[3])

        # self.layer0 = channel_attention(inter_dim[0]) # 192x192
        # self.layer1 = channel_attention(inter_dim[1]) # 96x96
        # self.layer2 = channel_attention(inter_dim[2]) # 48x48
        # self.layer3 = channel_attention(inter_dim[3]) # 24x24

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dp):
        feat0 = self.conv0(dp)
        # feat0 = self.layer0(feat0)

        feat1 = self.conv1(feat0)
        # feat1 = self.layer1(feat1)

        feat2 = self.conv2(feat1)
        # feat2 = self.layer2(feat2)

        feat3 = self.conv3(feat2)
        # feat3 = self.layer3(feat3)

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

        self.mixer = conv(mixer_channels, segm_inter_dim[3])
        self.s3 = conv(segm_inter_dim[3], segm_inter_dim[2])

        self.s2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.s1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.s0 = conv(segm_inter_dim[0], segm_inter_dim[0])

        self.f2 = conv(segm_input_dim[2], segm_inter_dim[2])
        self.f1 = conv(segm_input_dim[1], segm_inter_dim[1])
        self.f0 = conv(segm_input_dim[0], segm_inter_dim[0])

        self.post2 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.post1 = conv(segm_inter_dim[1], segm_inter_dim[0])
        self.post0 = conv_no_relu(segm_inter_dim[0], 2)


        self.m2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.m1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.m0 = conv(segm_inter_dim[0], segm_inter_dim[0])

        self.rgbd_fusion3 = CBAM(segm_inter_dim[3], segm_inter_dim[3], segm_inter_dim[3])
        self.rgbd_fusion2 = CBAM(segm_inter_dim[2], segm_inter_dim[2], segm_inter_dim[2])
        self.rgbd_fusion1 = CBAM(segm_inter_dim[1], segm_inter_dim[1], segm_inter_dim[1])
        self.rgbd_fusion0 = CBAM(segm_inter_dim[0], segm_inter_dim[0], segm_inter_dim[0])

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
        ''' Song's comments:
            just add rgbd-fusion,
        '''
        f_test = self.segment1(self.segment0(feat_test[3]))    # 1x1x64 conv + 3x3x64 conv -> [1, 1024, 24,24] -> [1, 64, 24, 24]
        f_train = self.segment1(self.segment0(feat_train[3]))  # 1x1x64 conv + 3x3x64 conv -> [1, 1024, 24,24] -> [1, 64, 24, 24]
        # Fusion RGBD Features
        f_test, attn03 = self.rgbd_fusion3(f_test, feat_test_d[3])
        f_train, _ = self.rgbd_fusion3(f_train, feat_train_d[3])

        # reshape mask to the feature size
        mask_pos = F.interpolate(mask_train[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,384, 384] -> [1,1,24,24]
        mask_neg = 1 - mask_pos

        pred_pos, pred_neg = self.similarity_segmentation(f_test, f_train, mask_pos, mask_neg)

        pred_ = torch.cat((torch.unsqueeze(pred_pos, -1), torch.unsqueeze(pred_neg, -1)), dim=-1)
        pred_sm = F.softmax(pred_, dim=-1) # [1, 24, 24, 2]
        if test_dist is not None:
            # distance map is give - resize for mixer
            dist = F.interpolate(test_dist[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,24,24]
            # concatenate inputs for mixer
            # softmaxed segmentation, positive segmentation and distance map
            ''' F + P + L , segm_layers: [B,3,24,24]'''
            segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1),
                                     torch.unsqueeze(pred_pos, dim=1),
                                     dist), dim=1)
        else:
            ''' F + P , segm_layers: [B,2,24,24]'''
            segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1), torch.unsqueeze(pred_pos, dim=1)), dim=1)

        out = self.mixer(segm_layers)
        out3 = self.s3(F.upsample(out, scale_factor=2))

        f_test_rgbd2, attn02 = self.rgbd_fusion2(self.f2(feat_test[2]), feat_test_d[2])
        out2 = self.post2(F.upsample(self.m2(f_test_rgbd2) + self.s2(out3), scale_factor=2))

        f_test_rgbd1, attn01 = self.rgbd_fusion1(self.f1(feat_test[1]), feat_test_d[1])
        out1 = self.post1(F.upsample(self.m1(f_test_rgbd1) + self.s1(out2), scale_factor=2))

        f_test_rgbd0, attn00 = self.rgbd_fusion0(self.f0(feat_test[0]), feat_test_d[0])
        out0 = self.post0(F.upsample(self.m0(f_test_rgbd0) + self.s0(out1), scale_factor=2))

        pred3 = self.pyramid_pred3(F.upsample(out3, scale_factor=8))
        pred2 = self.pyramid_pred2(F.upsample(out2, scale_factor=4))
        pred1 = self.pyramid_pred1(F.upsample(out1, scale_factor=2))

        if not debug:
            return (out0, pred1, pred2, pred3)
        else:
            return (out0, pred1, pred2, pred3), (attn00, attn01, attn02, attn03)


    def similarity_segmentation(self, f_test, f_train, mask_pos, mask_neg):
        '''Song's comments:
            f_test / f_train: [1,64,24,24]
            mask_pos / mask_neg: [1,1,24,24] for each pixel, it belongs to foreground or background
            sim_resh : [1,24,24,576]
            sim_pos / sim_neg: [1,24,24,576] for each pixel, the similarity of f_test and f_train
            pos_map / pos_neg: [1,24,24]     for each pixel, the averaged TopK similarity score

            cosine similarity between normalized train and test features,
            which is exactly same as transformer:
                similarity = softmax(norm(Q) * norm(K))
            for example : TREG's code, softmax(layernorm(1x1conv(softmax(QK)*V)+V))
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
