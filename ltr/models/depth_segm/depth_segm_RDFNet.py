
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


''' Inspired by
    RDFNet: RGB-D Multi-level Residual Feature Fusion for Indoor Semantic Segmentation
    their "Multi-modal feature fusion (MMF) network"
'''

class MMFNet(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.conv1x1_rgb = nn.Conv2d(input_dims, output_dims, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.res01_rgb = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))

        self.res02_rgb = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))

        self.conv3x3_rgb = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        self.conv1x1_d = nn.Conv2d(input_dims, output_dims, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.res01_d = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))

        self.res02_d = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))

        self.conv3x3_d = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.pool5x5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.conv3x3 = nn.Conv2d(output_dims, output_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, f_rgb, f_d):
        f_rgb = self.conv1x1_rgb(f_rgb)
        f_rgb = f_rgb + self.res01_rgb(f_rgb)
        f_rgb = f_rgb + self.res02_rgb(f_rgb)

        f_d = self.conv1x1_d(f_d)
        f_d = f_d + self.res01_d(f_d)
        f_d = f_d + self.res02_d(f_d)

        f_rgbd = self.relu(self.conv3x3_rgb(f_rgb) + self.conv3x3_d(f_d))
        f_rgbd = f_rgbd + self.conv3x3(self.pool5x5(f_rgbd))

        return f_rgbd

'''RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
    we use the simple version by ourselves
'''
class RCU_block(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.conv3x3_1 = nn.Conv2d(input_dims, input_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv3x3_2 = nn.Conv2d(input_dims, input_dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
    def forward(self, x):
        y = self.conv3x3_1(F.relu(x))
        y = self.conv3x3_2(F.relu(y))
        return x + y

class MRG_block(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv3x3_1 = nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv3x3_2 = nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
    def forward(self, x, y):
        x = F.upsample(self.conv3x3_1(x), scale_factor=2)
        y = F.upsample(self.conv3x3_2(y), scale_factor=2)
        return x + y

class CPR_block(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv3x3_1 = nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv3x3_2 = nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.pool5x5_1 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool5x5_2 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        x = F.relu(x)
        x1 = self.conv3x3_1(self.pool5x5_1(x))
        x2 = self.conv3x3_2(self.pool5x5_2(x1))

        return x + x1 + x2

class RefineNet(nn.Module):

    def __init__(self, input_dims):
        super().__init__()
        self.RCU01 = RCU_block(input_dims)
        self.RCU02 = RCU_block(input_dims)

        self.RCU03 = RCU_block(input_dims)
        self.RCU04 = RCU_block(input_dims)

        self.RCU05 = RCU_block(input_dims)

        self.MRF = MRG_block(input_dims)

        self.CPR = CPR_block(input_dims)


    def forward(self, pre_out, cur_out):
        x = self.RCU02(self.RCU01(pre_out))
        y = self.RCU03(self.RCU04(cur_out))
        out = self.RCU05(self.CPR(self.MRF(x, y)))
        return out

class DepthNet(nn.Module):
    def __init__(self, input_dim=1, inter_dim=(4, 16, 32, 64)):
        super().__init__()

        self.conv0 = conv(input_dim, inter_dim[0])    # 1  -> 4
        self.conv0_1 = conv(inter_dim[0]*2, inter_dim[0], kernel_size=1, stride=1, padding=0)

        self.conv1 = conv(inter_dim[0], inter_dim[1]) # 4 -> 16
        self.conv1_1 = conv(inter_dim[1]*2, inter_dim[1], kernel_size=1, stride=1, padding=0)

        self.conv2 = conv(inter_dim[1], inter_dim[2]) # 16 -> 32
        self.conv2_1 = conv(inter_dim[2]*2, inter_dim[2], kernel_size=1, stride=1, padding=0)

        self.conv3 = conv(inter_dim[2], inter_dim[3]) # 32 -> 64
        self.conv3_1 = conv(inter_dim[3]*2, inter_dim[3], kernel_size=1, stride=1, padding=0)


        # AvgPool2d , more smooth, MaxPool2d, more sharp
        self.maxpool0 = nn.MaxPool2d(2, stride=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.avgpool0 = nn.AvgPool2d(2, stride=2)
        self.avgpool1 = nn.AvgPool2d(2, stride=2)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.avgpool3 = nn.AvgPool2d(2, stride=2)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dp):
        feat0 = self.conv0(dp)
        feat0 = torch.cat((self.maxpool0(feat0), self.avgpool0(feat0)), dim=1)
        feat0 = self.conv0_1(feat0)

        feat1 = self.conv1(feat0)
        feat1 = torch.cat((self.maxpool1(feat1), self.avgpool1(feat1)), dim=1)
        feat1 = self.conv1_1(feat1)

        feat2 = self.conv2(feat1)
        feat2 = torch.cat((self.maxpool2(feat2), self.avgpool2(feat2)), dim=1)
        feat2 = self.conv2_1(feat2)

        feat3 = self.conv3(feat2)
        feat3 = torch.cat((self.maxpool3(feat3), self.avgpool3(feat3)), dim=1)
        feat3 = self.conv3_1(feat3)

        return [feat0, feat1, feat2, feat3] # [4, 16, 32, 64]


class DepthSegmNet(nn.Module):
    def __init__(self):
        super().__init__()

        segm_input_dim = (64, 256, 512, 1024)
        segm_inter_dim = (4, 16, 32, 64)
        segm_dim = (64, 64)  # convolutions before cosine similarity

        ''' Use colormap encoding for depth input '''
        self.depth_feat_extractor = DepthNet(input_dim=3, inter_dim=segm_inter_dim)

        # For previous out features
        self.s_layers = nn.ModuleList([conv(segm_inter_dim[0], segm_inter_dim[0]),
                                       conv(segm_inter_dim[1], segm_inter_dim[1]),
                                       conv(segm_inter_dim[2], segm_inter_dim[2]),
                                       conv(1, segm_inter_dim[3])])
        # For RGB features
        self.f_layers = nn.ModuleList([conv(segm_input_dim[0], segm_inter_dim[0]),
                                       conv(segm_input_dim[1], segm_inter_dim[1]),
                                       conv(segm_input_dim[2], segm_inter_dim[2]),
                                       conv(segm_input_dim[3], segm_inter_dim[3])])

        self.post_layers = nn.ModuleList([conv_no_relu(segm_inter_dim[0], 2),
                                          conv(segm_inter_dim[1], segm_inter_dim[0]),
                                          conv(segm_inter_dim[2], segm_inter_dim[1]),
                                          conv(segm_inter_dim[3], segm_inter_dim[2])])
        # Fuse RGB+D features
        self.fusion_layers = nn.ModuleList([MMFNet(segm_inter_dim[0], segm_inter_dim[0]),
                                            MMFNet(segm_inter_dim[1], segm_inter_dim[1]),
                                            MMFNet(segm_inter_dim[2], segm_inter_dim[2]),
                                            MMFNet(segm_inter_dim[3], segm_inter_dim[3])])
        # For RGBD features
        self.refine_layers = nn.ModuleList([RefineNet(segm_inter_dim[0]),
                                            RefineNet(segm_inter_dim[1]),
                                            RefineNet(segm_inter_dim[2]),
                                            RefineNet(segm_inter_dim[3])])


        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, test_dist=None, debug=False):
        ''' rgb features   : [conv1, layer1, layer2, layer3], Bx64x192x192  -> Bx256x96x96 -> Bx512x48x48 -> Bx1024x24x24
            depth features : [feat0, feat1, feat2, feat3],    Bx4x192x192 -> Bx16x96x96 -> Bx32x48x48 -> Bx64x24x24
        '''
        out = F.interpolate(test_dist[0], size=(feat_test_rgb[3].shape[-2], feat_test_rgb[3].shape[-1])) # [B, 1, 24, 24]

        out = self.layer_processing(out, feat_test_rgb, feat_test_d, layer=3)   # [B, 32, 48, 48]
        out = self.layer_processing(out, feat_test_rgb, feat_test_d, layer=2)   # [B, 16, 96, 96]
        out = self.layer_processing(out, feat_test_rgb, feat_test_d, layer=1)   # [B, 4, 192, 192]
        out = self.layer_processing(out, feat_test_rgb, feat_test_d, layer=0)   # [B, 2, 384, 384]

        if debug :
            return out, None
        else:
            return out

    def layer_processing(self, pre_out, feat_test_rgb, feat_test_d, layer=0):

        f_test_rgb, f_test_d = feat_test_rgb[layer], feat_test_d[layer]

        feat_rgbd = self.fusion_layers[layer](self.f_layers[layer](f_test_rgb), f_test_d)
        out = self.refine_layers[layer](self.s_layers[layer](pre_out), feat_rgbd)
        out = self.post_layers[layer](out)

        return out
