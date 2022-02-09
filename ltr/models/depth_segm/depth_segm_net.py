import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

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

def conv_sigmoid(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.Sigmoid())


class DepthNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=64):
        super().__init__()

        self.conv0 = conv(input_dim, output_dim)
        self.conv1 = conv(output_dim, output_dim)
        self.conv2 = conv(output_dim, output_dim)
        self.conv3 = conv(output_dim, output_dim)

        # AvgPool2d , more smooth, MaxPool2d, more sharp
        self.maxpool0 = nn.MaxPool2d(2, stride=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        # self.avgpool0 = nn.AvgPool2d(2, stride=2)
        # self.avgpool1 = nn.AvgPool2d(2, stride=2)
        # self.avgpool2 = nn.AvgPool2d(2, stride=2)
        # self.avgpool3 = nn.AvgPool2d(2, stride=2)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dp):
        feat0 = self.conv0(dp)
        feat0 = self.maxpool0(feat0) #  + self.avgpool0(feat0) # 384 -> 192

        feat1 = self.conv1(feat0)
        feat1 = self.maxpool1(feat1) #  + self.avgpool1(feat1) # 192 -> 96

        feat2 = self.conv2(feat1)
        feat2 = self.maxpool2(feat2) #  + self.avgpool2(feat2) # 96 -> 48

        feat3 = self.conv3(feat2)
        feat3 = self.maxpool3(feat3) #  + self.avgpool3(feat3) # 48 -> 24

        return [feat0, feat1, feat2, feat3]

class DepthSegmNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.depth_feat_extractor = DepthNet(input_dim=1, output_dim=64)

        self.f0 = conv(1024, 64, kernel_size=3, padding=1)
        self.f1 = conv(512, 64, kernel_size=3, padding=1)
        self.f2 = conv(256, 32, kernel_size=3, padding=1)
        self.f3 = conv(64, 16, kernel_size=3, padding=1)

        self.d0 = conv(64, 64, kernel_size=3, padding=1)
        self.d1 = conv(64, 64, kernel_size=3, padding=1)
        self.d2 = conv(64, 32, kernel_size=3, padding=1)
        self.d3 = conv(64, 16, kernel_size=3, padding=1)

        self.c0 = conv( 3, 64, kernel_size=3, padding=1)   # self.mixer : mix (p_rgb, p_d, dist)
        self.c1 = conv(64, 64, kernel_size=3, padding=1)
        self.c2 = conv(32, 32, kernel_size=3, padding=1)
        self.c3 = conv(16, 16, kernel_size=3, padding=1)

        self.s0 = conv(64, 64, kernel_size=3, padding=1)
        self.s1 = conv(64, 64, kernel_size=3, padding=1)
        self.s2 = conv(32, 32, kernel_size=3, padding=1)
        self.s3 = conv(16, 16, kernel_size=3, padding=1)

        self.post0 = conv(64, 64, kernel_size=3, padding=1)
        self.post1 = conv(64, 32, kernel_size=3, padding=1)
        self.post2 = conv(32, 16, kernel_size=3, padding=1)
        self.post3 = conv_no_relu(16, 2)   # GT is the pair of Foreground and Background segmentations

        # to get weights for RGB and D features
        self.w1_1 = conv_no_relu(1, 1, kernel_size=3, padding=1)
        self.w1_2 = conv_no_relu(1, 1, kernel_size=3, padding=1)
        self.w2 = conv_sigmoid(1, 1, kernel_size=3, padding=1)
        self.w3 = conv_no_relu(1, 1, kernel_size=3, padding=1)
        self.w4 = conv_no_relu(1, 1, kernel_size=3, padding=1)
        self.w5 = conv_sigmoid(1, 1, kernel_size=3, padding=1)

        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, test_dist=None):
        ''' rgb features   : [conv1, layer1, layer2, layer3], Bx64x192x192  -> Bx256x96x96 -> Bx512x48x48 -> Bx1024x24x24
            depth features : [feat0, feat1, feat2, feat3],    Bx64x192x192 -> Bx64x96x96 -> Bx64x48x48 -> Bx64x24x24
            based on D3S, feat = w_rgb * feat_rgb + w_d * feat_d
        '''

        pos_rgb, neg_rgb = self.cosine_similarity(self.f0(feat_test_rgb[3]), self.f0(feat_train_rgb[3]), mask_train[0])
        pos_d, neg_d = self.cosine_similarity(self.d0(feat_test_d[3]), self.d0(feat_train_d[3]), mask_train[0])
        dist = F.interpolate(test_dist[0], size=(feat_test_d[3].shape[-2], feat_test_d[3].shape[-1]))   # [B, 1, 24, 24]

        # weights for RGB features and Depth features
        w_rgb, w_d = self.mutual_guided_block(pos_rgb, pos_d) # [B, 1, 24, 24]
        feat_rgbd3 = self.f0(feat_test_rgb[3]) * w_rgb + self.d0(feat_test_d[3]) * w_d # [B, 64, 24, 24]
        feat_rgbd2 = self.f1(feat_test_rgb[2]) * F.upsample(w_rgb, scale_factor=2) + self.d1(feat_test_d[2]) * F.upsample(w_d, scale_factor=2)
        feat_rgbd1 = self.f2(feat_test_rgb[1]) * F.upsample(w_rgb, scale_factor=4) + self.d2(feat_test_d[1]) * F.upsample(w_d, scale_factor=4)
        feat_rgbd0 = self.f3(feat_test_rgb[0]) * F.upsample(w_rgb, scale_factor=8) + self.d3(feat_test_d[0]) * F.upsample(w_d, scale_factor=8)

        out0 = torch.cat((pos_rgb, pos_d, dist), dim=1)                                    # [B, 3, 24, 24]
        out0 = self.post0(F.upsample(self.s0(feat_rgbd3) + self.c0(out0), scale_factor=2)) # [B, 64, 24, 24]   -> [B, 64, 48, 48]
        out1 = self.post1(F.upsample(self.s1(feat_rgbd2) + self.c1(out0), scale_factor=2)) # [B, 64, 48, 48]   -> [B, 32, 96, 96]
        out2 = self.post2(F.upsample(self.s2(feat_rgbd1) + self.c2(out1), scale_factor=2)) # [B, 32, 96, 96]   -> [B, 16, 192, 192]
        out3 = self.post3(F.upsample(self.s3(feat_rgbd0) + self.c3(out2), scale_factor=2)) # [B, 16, 192, 192] -> [B, 2, 384, 384]

        return out3


    def mutual_guided_block(self, pos_rgb, pos_d):
        ''' rgb->tir, tir->rgb blocks in paper Temporal Aggregation for Adaptive RGBT Tracking
            we use it as d -> rgb, use depth to guide rgb
            rgb_pn : rgb positive response map [B, 1, H, W]
            depth_pn : depth positive maps, [B, 1, H, W]
        '''

        w_d1 = self.w1_2(self.w1_1(pos_d))
        w_d2 = self.w2(pos_d)

        w_rgb = self.w3(pos_rgb)

        w = self.w4(torch.mul(w_rgb, w_d2) + w_rgb)
        w = self.w5(w + w_d1)

        return w*pos_rgb, (1-w)*pos_d # [B, 1, 24, 24]

    def cosine_similarity(self, f_test, f_train, mask_train, topk=3):
        ''' D3S cosine similarity between test and train features'''
        mask_pos = F.interpolate(mask_train, size=(f_train.shape[-2], f_train.shape[-1]))
        mask_neg = 1 - mask_pos
        # first normalize train and test features to have L2 norm 1
        # cosine similarity and reshape last two dimensions into one
        sim = torch.einsum('ijkl,ijmn->iklmn',
                           F.normalize(f_test, p=2, dim=1),
                           F.normalize(f_train, p=2, dim=1))
        sim_resh = sim.view(sim.shape[0], sim.shape[1], sim.shape[2], sim.shape[3] * sim.shape[4]) # [B, H, W, M, N] -> [B,H,W,MN]
        # reshape masks into vectors for broadcasting [B x 1 x 1 x w * h],
        # re-weight samples (take out positive ang negative samples)
        sim_pos = sim_resh * mask_pos.view(mask_pos.shape[0], 1, 1, -1)         # [B,24,24,576]
        sim_neg = sim_resh * mask_neg.view(mask_pos.shape[0], 1, 1, -1)

        # take top k positive and negative examples mean over the top positive and negative examples
        pos_map = torch.mean(torch.topk(sim_pos, topk, dim=-1).values, dim=-1)  # [B, H, W]
        neg_map = torch.mean(torch.topk(sim_neg, topk, dim=-1).values, dim=-1)

        p = torch.cat((torch.unsqueeze(pos_map, -1), torch.unsqueeze(neg_map, -1)), dim=-1) # [B, H, W, 2]
        p = F.softmax(p, dim=-1)                                                # [B, 24, 24, 2]

        return torch.unsqueeze(p[:, :, :, 0], dim=1), torch.unsqueeze(p[:, :, :, 1], dim=1)     # [B, 1, H, W]
