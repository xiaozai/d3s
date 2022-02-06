import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
#

def conv_block(in_planes, out_planes=256):
    ''' 1x1x256 -> 3x3x256 -> 1x1x256 '''
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

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


class DepthNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=256):
        super().__init__()

        self.conv0 = conv_block(input_dim, output_dim)
        self.conv1 = conv_block(output_dim, output_dim)
        self.conv2 = conv_block(output_dim, output_dim)
        self.conv3 = conv_block(output_dim, output_dim)

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
        feat0 = self.maxpool0(feat0) + self.avgpool0(feat0) # 384 -> 192

        feat1 = self.conv1(feat0)
        feat1 = self.maxpool1(feat1) + self.avgpool1(feat1) # 192 -> 96

        feat2 = self.conv2(feat1)
        feat2 = self.maxpool2(feat2) + self.avgpool2(feat2) # 96 -> 48

        feat3 = self.conv3(feat2)
        feat3 = self.maxpool3(feat3) + self.avgpool3(feat3) # 48 -> 24

        return [feat0, feat1, feat2, feat3]

class DepthSegmNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.depth_feat_extractor = DepthNet(input_dim=1, output_dim=64)

        # 256 depth feat + 1 dist map + rgb similarity + depth similarity
        self.mixer = conv(3, 64, kernel_size=3, padding=1)

        self.f0 = conv(1024, 64, kernel_size=3, padding=1)

        self.f1 = conv(512, 64, kernel_size=3, padding=1)
        self.f2 = conv(256, 32, kernel_size=3, padding=1)
        self.f3 = conv(64, 16, kernel_size=3, padding=1)

        self.d1 = conv(64, 64, kernel_size=3, padding=1)
        self.d2 = conv(64, 32, kernel_size=3, padding=1)
        self.d3 = conv(64, 16, kernel_size=3, padding=1)

        self.post1 = conv(64, 32, kernel_size=3, padding=1)
        self.post2 = conv(32, 16, kernel_size=3, padding=1)
        self.post3 = conv_no_relu(16, 2)   # GT is the pair of Foreground and Background segmentations

        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat_test_rgb, depth_test_imgs, feat_train_rgb, feat_train_d, mask_train, test_dist=None):

        ''' rgb features   : [conv1, layer1, layer2, layer3], Bx64x192x192  -> Bx256x96x96 -> Bx512x48x48 -> Bx1024x24x24
            depth features : [feat0, feat1, feat2, feat3],    Bx64x192x192 -> Bx64x96x96 -> Bx64x48x48 -> Bx64x24x24

            segmantation is the combination of sementatic features
            out_{i+1} = pos_i(cat(f_i(p_rgb * f_rgb), d_i(p_d * f_d)) + out_i)
        '''

        feat_test_d = self.depth_feat_extractor(depth_test_imgs)

        # weights for RGB features and Depth features
        similarity_rgb = self.cosine_similarity(self.f0(feat_test_rgb[3]), self.f0(feat_train_rgb[3]), mask_train[0]) # [B, 1, 24, 24]
        similarity_d = self.cosine_similarity(feat_test_d[3], feat_train_d[3], mask_train[0])                         # [B, 1, 24, 24]
        similarity = F.softmax(torch.cat((similarity_rgb, similarity_d), dim=1), dim=1)                               # [B, 2, 24, 24]
        prob_rbg = torch.unsqueeze(similarity[:, 0, :, :], dim=1)                                                     # [B, 1, 24, 24]
        prob_d = torch.unsqueeze(similarity[:, 1, :, :], dim=1)                                                       # [B, 1, 24, 24]

        # distance map is give - resize for mixer
        dist = F.interpolate(test_dist[0], size=(feat_test_d[3].shape[-2], feat_test_d[3].shape[-1]))                 # [B, 1, 24, 24]

        out0 = torch.cat((prob_rgb, prob_d, dist), dim=1)                       # [B, 3, 24, 24]
        out0 = F.upsample(self.mixer(out0), scale_factor=2)                     # [B, 64, 48, 48]

        # merge with layer2 features
        p_rgb1 = F.interpolate(prob_rbg, size=(feat_test_rgb[2].shape[-2], feat_test_rgb[2].shape[-1]))  # [B, 1, 48, 48]
        p_d1 = F.interpolate(prob_d, size=(feat_test_d[2].shape[-2], feat_test_d[2].shape[-1]))          # [B, 1, 48, 48]

        out1 = torch.cat(self.f1(torch.mul(feat_test_rgb[2], p_rgb1)), self.d1(torch.mul(feat_test_d[2], p_d1)), dim=1) # [B, 64, 48, 48]
        out1 = self.post1(F.upsample(out1+out0, scale_factor=2))                # [B, 32, 96, 96]

        # merge with layer1 features
        p_rgb2 = F.interpolate(prob_rbg, size=(feat_test_rgb[1].shape[-2], feat_test_rgb[1].shape[-1]))  # [B, 1, 96, 96]
        p_d2 = F.interpolate(prob_d, size=(feat_test_d[1].shape[-2], feat_test_d[1].shape[-1]))          # [B, 1, 96, 96]

        out2 = torch.cat(self.f2(torch.mul(feat_test_rgb[1], p_rgb2)), self.d2(torch.mul(feat_test_d[1], p_d2)), dim=1) # [B, 32, 96, 96]
        out2 = self.post2(F.upsample(out2+out1, scale_factor=2))                # [B, 16, 192, 192]

        # merge with layer0 features
        p_rgb3 = F.interpolate(prob_rbg, size=(feat_test_rgb[0].shape[-2], feat_test_rgb[0].shape[-1]))  # [B, 1, 96, 96]
        p_d3 = F.interpolate(prob_d, size=(feat_test_d[0].shape[-2], feat_test_d[0].shape[-1]))          # [B, 1, 96, 96]

        out3 = torch.cat(self.f3(torch.mul(feat_test_rgb[0], p_rgb3)), self.d3(torch.mul(feat_test_d[0], p_d3)), dim=1) # [B, 16, 96, 96]
        out3 = self.post3(F.upsample(out3+out2, scale_factor=2))                # [B, 2, 384, 384]

        return out3


    def cosine_similarity(self, f_test, f_train, mask_train, topk=3):

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

        return torch.unsqueeze(p[:, :, :, 0], dim=1)                            # [B, 1, 24, 24]


    # def feature_correlation(self, feat_test, feat_train, mask_train):
    #     ''' feat_test = feat_test * similarity(feat_test, feat_train, mask_train)'''
    #     mask_pos = F.interpolate(mask_train, size=(feat_train.shape[-2], feat_train.shape[-1]))
    #     mask_neg = 1 - mask_pos
    #     feat_test = self.similarity_segmentation(feat_test, feat_train, mask_pos, mask_neg)
    #     return feat_test
    #
    # def similarity_segmentation(self, f_test, f_train, mask_pos, mask_neg, topk=3):
    #     # first normalize train and test features to have L2 norm 1
    #     # cosine similarity and reshape last two dimensions into one
    #     sim = torch.einsum('ijkl,ijmn->iklmn',
    #                        F.normalize(f_test, p=2, dim=1),
    #                        F.normalize(f_train, p=2, dim=1))
    #     sim_resh = sim.view(sim.shape[0], sim.shape[1], sim.shape[2], sim.shape[3] * sim.shape[4]) # [B, 24, 24, 24, 24] -> [B,24,24,576]
    #     # reshape masks into vectors for broadcasting [B x 1 x 1 x w * h]
    #     # re-weight samples (take out positive ang negative samples)
    #     sim_pos = sim_resh * mask_pos.view(mask_pos.shape[0], 1, 1, -1)         # [B,H,W,MN]
    #     sim_neg = sim_resh * mask_neg.view(mask_pos.shape[0], 1, 1, -1)
    #
    #     # take top k positive and negative examples
    #     # mean over the top positive and negative examples
    #
    #     pos_map = torch.mean(torch.topk(sim_pos, topk, dim=-1).values, dim=-1)  # [B, H, W]
    #     neg_map = torch.mean(torch.topk(sim_neg, topk, dim=-1).values, dim=-1)
    #
    #     p = torch.cat((torch.unsqueeze(pos_map, -1), torch.unsqueeze(neg_map, -1)), dim=-1) # [B, H, W, 2]
    #     p = F.softmax(p, dim=-1)                                                # [1, 24, 24, 2]
    #
    #     ''' feat * similarity of foreground '''
    #     # out_pos = f_test * torch.unsqueeze(p[:, :, :, 0], dim=1)
    #     # out_neg = f_test * torch.unsqueeze(p[:, :, :, 1], dim=1)
    #     # return out_pos
    #
    #     return torch.unsqueeze(p[:, :, :, 0], dim=1)
