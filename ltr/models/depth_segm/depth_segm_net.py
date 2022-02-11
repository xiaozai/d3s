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

class DepthNet(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()

        self.conv0 = conv(input_dim, 4)
        self.conv1 = conv(4, 16)
        self.conv2 = conv(16, 32)
        self.conv3 = conv(32, 64)

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

        return [feat0, feat1, feat2, feat3] # [4, 16, 32, 64]


class DepthSegmNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.depth_feat_extractor = DepthNet(input_dim=1)

        # 256 depth feat + 1 dist map + rgb similarity + depth similarity
        self.mixer = conv(5, 64, kernel_size=3, padding=1)

        # 1024， 512， 256， 64 -> 64, 32, 16, 4
        self.segment0_rgb = conv(1024, 64, kernel_size=1, padding=0)
        self.segment1_rgb = conv_no_relu(64, 64)

        self.segment0_d = conv(64, 64, kernel_size=1, padding=0)
        self.segment1_d = conv_no_relu(64, 64)

        self.mixer = conv(5, 64)
        self.s3 = conv(64, 64)

        self.s2 = conv(32, 32)
        self.s1 = conv(16, 16)
        self.s0 = conv(4, 4)

        self.f2 = conv(512, 32)
        self.f1 = conv(256, 16)
        self.f0 = conv(64, 4)

        self.d2 = conv(32, 32)
        self.d1 = conv(16, 16)
        self.d0 = conv(4, 4)

        self.post2 = conv(32, 16)
        self.post1 = conv(16, 4)
        self.post0 = conv_no_relu(4, 2)

        self.initialize_weights()

        self.topk_pos = 3
        self.topk_neg = 3

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, test_dist=None, debug=False):
        ''' rgb features   : [conv1, layer1, layer2, layer3], Bx64x192x192  -> Bx256x96x96 -> Bx512x48x48 -> Bx1024x24x24
            depth features : [feat0, feat1, feat2, feat3],    Bx4x192x192 -> Bx16x96x96 -> Bx32x48x48 -> Bx64x24x24

            depth_segm = d3s + depth feat
        '''

        f_test_rgb = self.segment1_rgb(self.segment0_rgb(feat_test_rgb[3]))
        f_train_rgb = self.segment1_rgb(self.segment0_rgb(feat_train_rgb[3]))

        f_test_d = self.segment1_d(self.segment0_d(feat_test_d[3]))
        f_train_d = self.segment1_d(self.segment0_d(feat_train_d[3]))

        mask_pos = F.interpolate(mask_train[0], size=(f_train_rgb.shape[-2], f_train_rgb.shape[-1])) # [1,1,384, 384] -> [1,1,24,24]
        mask_neg = 1 - mask_pos

        # rgb
        pred_pos_rgb, pred_neg_rgb = self.similarity_segmentation(f_test_rgb, f_train_rgb, mask_pos, mask_neg)

        pred_rgb = torch.cat((torch.unsqueeze(pred_pos_rgb, -1), torch.unsqueeze(pred_neg_rgb, -1)), dim=-1)
        pred_sm_rgb = F.softmax(pred_rgb, dim=-1) # [1, 24, 24, 2]

        # depth
        pred_pos_d, pred_neg_d = self.similarity_segmentation(f_test_d, f_train_d, mask_pos, mask_neg)

        pred_d = torch.cat((torch.unsqueeze(pred_pos_d, -1), torch.unsqueeze(pred_neg_d, -1)), dim=-1)
        pred_sm_d = F.softmax(pred_d, dim=-1) # [1, 24, 24, 2]

        # dist map (DCF response), in our case , it is response map, in d3s, it is distance map
        dist = F.interpolate(test_dist[0], size=(f_test_rgb.shape[-2], f_test_rgb.shape[-1]))                 # [B, 1, 24, 24]

        # mix
        segm_layers = torch.cat((torch.unsqueeze(pred_sm_rgb[:, :, :, 0], dim=1),
                                 torch.unsqueeze(pred_pos_rgb, dim=1),
                                 torch.unsqueeze(pred_sm_d[:, :, :, 0], dim=1), # depth may mislead
                                 torch.unsqueeze(pred_pos_d, dim=1),
                                 dist), dim=1)


        out = self.mixer(segm_layers)
        out = self.s3(F.upsample(out, scale_factor=2))

        out = self.post2(F.upsample(self.f2(feat_test_rgb[2]) + self.d2(feat_test_d[2]) + self.s2(out), scale_factor=2))
        out = self.post1(F.upsample(self.f1(feat_test_rgb[1]) + self.d1(feat_test_d[1]) + self.s1(out), scale_factor=2))
        out = self.post0(F.upsample(self.f0(feat_test_rgb[0]) + self.d0(feat_test_d[0]) + self.s0(out), scale_factor=2))

        if debug:
            return out3, (pred_pos_rgb,pred_neg_rgb,pred_pos_d,pred_neg_d)
        else:
            return out3

    def similarity_segmentation(self, f_test, f_train, mask_pos, mask_neg):
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
