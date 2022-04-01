import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def conv3x3_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

def conv1x1_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)

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


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
            all(roi[:, 4] <= image_size[1]-1)
    return valid


def normalize_vis_img(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)

''' Inspired by
    ACNet : Attention based network to exploit complementary features for RGBD semantic segmentation
    (ICIP2019)
    https://github.com/anheidelonghu/ACNet
'''

def channel_attention(num_channel):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1),
                         conv1x1_layer(num_channel, num_channel),
                         nn.Sigmoid())


class ACNet(nn.Module):
    def __init__(self, rgb_dims, d_dims, output_dims):
        super().__init__()
        self.conv1x1_rgb = conv1x1_layer(rgb_dims, output_dims)
        self.conv1x1_d = conv1x1_layer(d_dims, output_dims)

        self.conv1x1_rgb_w = conv1x1_layer(output_dims, output_dims)
        self.conv1x1_d_w = conv1x1_layer(output_dims, output_dims)

        self.bn_rgb = nn.BatchNorm2d(output_dims)
        self.bn_d = nn.BatchNorm2d(output_dims)

        self.relu_rgb = nn.ReLU(inplace=True)
        self.relu_d = nn.ReLU(inplace=True)

        self.attn_rgb = channel_attention(output_dims)
        self.attn_d = channel_attention(output_dims)

    def forward(self, f_rgb, f_d):
        f_rgb = self.conv1x1_rgb(f_rgb)
        f_rgb = self.bn_rgb(f_rgb)
        f_rgb = self.relu_rgb(f_rgb)

        f_d = self.conv1x1_d(f_d)
        f_d = self.bn_rgb(f_d)
        f_d = self.relu_d(f_d)

        weight_rgb = self.attn_rgb(f_rgb)
        weight_d = self.attn_d(f_d)

        f_rgbd = f_rgb.mul(weight_rgb) + f_d.mul(weight_d)

        return f_rgbd

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

class SegmNet(nn.Module):
    """ Network module for IoU prediction. Refer to the paper for an illustration of the architecture."""
    def __init__(self, segm_input_dim=(128,256), segm_inter_dim=(256,256), segm_dim=(64, 64), mixer_channels=2, topk_pos=3, topk_neg=3):
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

        self.post2 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.post1 = conv(segm_inter_dim[1], segm_inter_dim[0])
        self.post0 = conv_no_relu(segm_inter_dim[0], 2)

        self.rgbd_fusion3 = ACNet(segm_inter_dim[3], segm_inter_dim[3], segm_inter_dim[3])
        self.rgbd_fusion2 = ACNet(segm_inter_dim[2], segm_inter_dim[2], segm_inter_dim[2])
        self.rgbd_fusion1 = ACNet(segm_inter_dim[1], segm_inter_dim[1], segm_inter_dim[1])
        self.rgbd_fusion0 = ACNet(segm_inter_dim[0], segm_inter_dim[0], segm_inter_dim[0])

        self.depth_feat_extractor = DepthNet(input_dim=1)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

    def forward(self, feat_test, feat_test_d, feat_train, feat_train_d, mask_train, test_dist=None):
        ''' Song's comments:
            just add rgbd-fusion, 
        '''
        f_test = self.segment1(self.segment0(feat_test[3]))    # 1x1x64 conv + 3x3x64 conv -> [1, 1024, 24,24] -> [1, 64, 24, 24]
        f_train = self.segment1(self.segment0(feat_train[3]))  # 1x1x64 conv + 3x3x64 conv -> [1, 1024, 24,24] -> [1, 64, 24, 24]
        # reshape mask to the feature size
        mask_pos = F.interpolate(mask_train[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,384, 384] -> [1,1,24,24]
        mask_neg = 1 - mask_pos

        f_test = self.rgbd_fusion3(f_test, feat_test_d[3])
        f_train = self.rgbd_fusion3(f_train, feat_train_d[3])

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
        out = self.s3(F.upsample(out, scale_factor=2))

        out = self.post2(F.upsample(self.rgbd_fusion2(self.f2(feat_test[2]), feat_test_d[2]) + self.s2(out), scale_factor=2))
        out = self.post1(F.upsample(self.rgbd_fusion1(self.f1(feat_test[1]), feat_test_d[1]) + self.s1(out), scale_factor=2))
        out = self.post0(F.upsample(self.rgbd_fusion0(self.f0(feat_test[0]), feat_test_d[0]) + self.s0(out), scale_factor=2))

        return out


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
