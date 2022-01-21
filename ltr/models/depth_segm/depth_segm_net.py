import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
#
# import matplotlib.pyplot as plt
# def draw_axis(ax, img, title, show_minmax=False):
#     ax.imshow(img)
#     if show_minmax:
#         minval_, maxval_, _, _ = cv2.minMaxLoc(img)
#         title = '%s \n min=%.2f max=%.2f' % (title, minval_, maxval_)
#     ax.set_title(title, fontsize=9)


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


# def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
#     valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
#             all(roi[:, 4] <= image_size[1]-1)
#     return valid


# def normalize_vis_img(x):
#     x = x - np.min(x)
#     x = x / np.max(x)
#     return (x * 255).astype(np.uint8)

class DepthNet(nn.Module):
    def __init__(self, input_dim=1, dims=(64, 128, 256), kernels=(1,3,3), pads=(0, 1, 0)):
        super().__init__()
        ''' Assume that DepthFeat == P + F
            1) simple attempt : self.depth_feat_extractor = DepthNet(input_dim=1, dims=(64, 128, 256), kernels=(1,3,3), pads=(0, 1, 1))
            2) 1x1x32 -> 3x3x32 -> 1x1x2 ?
            3) 1x1x32 -> maxpool 384->192 -> 3x3x32 -> maxpool 192->96 -> 1x1x2 -> maxpool 96->48 -> softmax
        '''
        self.conv1 = conv(input_dim, dims[0], kernel_size=kernels[0], padding=pads[0])
        self.conv2 = conv(dims[0], dims[1], kernel_size=kernels[1], padding=pads[1])
        self.conv3 = conv_no_relu(dims[1], dims[2], kernel_size=kernels[2], padding=pads[2])

        # AvgPool2d , more smooth, MaxPool2d, more sharp
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dp):
        feat1 = self.conv1(dp)     # [B, C, 384, 384]
        # feat1 = F.interpolate(feat1, size=(192, 192))
        feat1 = self.pool1(feat1)  # B, C, 192, 192

        feat2 = self.conv2(feat1)
        # feat2 = F.interpolate(feat2, size=(96, 96))
        feat2 = self.pool2(feat2)  # B, C, 96, 96

        feat3 = self.conv3(feat2)
        # feat3 = F.interpolate(feat3, size=(48, 48))
        feat3 = self.pool3(feat3)  # B, C, 48, 48

        # feat3 = F.softmax(feat3, dim=1)  # ?

        return feat3

class DepthSegmNet(nn.Module):
    """
        Given the depth image and the dist map, predict the mask from depth
        depth images -> depth feat + dist map -> mask

        Assume that depth image is the initial mask ??

        segm_input_dim = (64, 64, 128, 256)
        segm_inter_dim = (4, 16, 32, 64)
        segm_dim = (64, 64)  # convolutions before cosine similarity

    """
    def __init__(self):
        super().__init__()
        '''
        1) simple : depth feat + L 256+1 -> mixer 3x3x128 -> 3x3x64 -> 3x3x32 -> 3x3x2, train loss = 0.048
        2)        : depth feat + L 2+1   -> mixer 1x1x32  -> 3x3x32 -> 3x3x32 -> 3x3x2
        3)        : depth feat (Bx2x48x48) * similarity(feat_test_d, feat_train_d)
                    -> depth feat + L
                    -> mix, 1x1x32
                    -> [feat_test_rgb[2] (BxCx48x48) * similarity(feat_test_rgb[2], feat_train_rgb[2])] -> 1x1x32 + mix_feat
                    -> feat_test_rgb[1] -> 1x1x32 + out1
                    -> feat_test_rgb[0] -> 1x1x32 + out2
        '''

        self.depth_feat_extractor = DepthNet(input_dim=1, dims=(32, 32, 2), kernels=(1,3,1), pads=(0, 1, 0))

        self.mixer = conv(3, 32, kernel_size=1, padding=0) # ???? 256 depth feat + 1 dist map ?

        self.f2 = conv(512, 32, kernel_size=1, padding=0)
        self.f1 = conv(256, 32, kernel_size=1, padding=0)
        self.f0 = conv(64, 32, kernel_size=1, padding=0)

        self.post2 = conv(32, 32)
        self.post1 = conv(32, 32)
        self.post0 = conv_no_relu(32, 2)      # GT is the pair of Foreground and Background segmentations

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, feat_test_rgb, depth_test_imgs, feat_train_rgb, feat_train_d, mask_train, test_dist=None):
        ''' Song's comments:
            depth test   : batch*1*384*384
            f_test_depth : batch*256*384*384
            feat_test_rgb layer0 : [8, 64, 192, 192]
                          layer1 : [8, 256, 96, 96]
                          layer2 : [8, 512, 48, 48]
            We assume that DepthFeat = F+P, concatenate with dist (location) map
            how about channel wise multiplication???
        '''

        ''' multi level depth feat ???? '''
        feat_test_d = self.depth_feat_extractor(depth_test_imgs)  # [B, 2, 384, 384] = F+B, [B, C, 48, 48]

        feat_test_d = self.feature_correlation(feat_test_d, feat_train_d, mask_train[0]) # [B, 2*2, H, W]

        feat_test_rgb[2] = self.feature_correlation(feat_test_rgb[2], feat_train_rgb[2], mask_train[0])
        # feat_test_rgb[1] = self.feature_correlation(feat_test_rgb[1], feat_train_rgb[1], mask_train[0])
        # feat_test_rgb[0] = self.feature_correlation(feat_test_rgb[0], feat_train_rgb[0], mask_train[0])

        if test_dist is not None:
            '''Song: we change the test_dist map into Guassian map instead of distance map '''
            # distance map is give - resize for mixer # concatenate inputs for mixer
            dist = F.interpolate(test_dist[0], size=(feat_test_d.shape[-2], feat_test_d.shape[-1])) # [B, 1,   384,384]
            segm_layers = torch.cat((feat_test_d, dist), dim=1)                                     # [B, C+1, 384, 384]
        else:
            segm_layers = torch.cat((feat_test_d, feat_test_d[:, 0, :, :]), dim=1)                  # [B, 3, 384, 384]

        # Mix DepthFeat and Location Map
        out0 = self.mixer(segm_layers) # [B, 32, 384, 384]

        # out1 = self.post2(F.upsample(self.f2(feat_test_rgb[2]), scale_factor=8) + F.upsample(out_mix, scale_factor=8))
        # out1 = self.post2(F.upsample(self.f2(feat_test_rgb[2]) + out, scale_factor=8))
        # out2 = self.post1(F.upsample(self.f1(feat_test_rgb[1]), scale_factor=4) + out1)
        # out3 = self.post0(F.upsample(self.f0(feat_test_rgb[0]), scale_factor=2) + out2)
        out1 = self.post2(F.upsample(self.f2(feat_test_rgb[2]) + out0, scale_factor=2))  # 48 ->  96
        out2 = self.post1(F.upsample(self.f1(feat_test_rgb[1]) + out1, scale_factor=2)) #  96 -> 192
        out3 = self.post0(F.upsample(self.f0(feat_test_rgb[0]) + out2, scale_factor=2)) # 192 -> 384

        return out3

    def feature_correlation(self, feat_test, feat_train, mask_train):
        ''' feat_test = feat_test * similarity(feat_test, feat_train, mask_train)'''
        mask_pos = F.interpolate(mask_train, size=(feat_train.shape[-2], feat_train.shape[-1]))
        mask_neg = 1 - mask_pos
        feat_test = self.similarity_segmentation(feat_test, feat_train, mask_pos, mask_neg)
        return feat_test

    def similarity_segmentation(self, f_test, f_train, mask_pos, mask_neg, topk=3):
        '''Song's comments:
            same as D3S similarity_segmentation,
            but out = f_test * pos_similarity_map
            softmax(Q*K)*V
        '''
        # first normalize train and test features to have L2 norm 1
        # cosine similarity and reshape last two dimensions into one
        sim = torch.einsum('ijkl,ijmn->iklmn',
                           F.normalize(f_test, p=2, dim=1),
                           F.normalize(f_train, p=2, dim=1))
        sim_resh = sim.view(sim.shape[0], sim.shape[1], sim.shape[2], sim.shape[3] * sim.shape[4]) # [B, 24, 24, 24, 24] -> [B,24,24,576]
        # reshape masks into vectors for broadcasting [B x 1 x 1 x w * h]
        # re-weight samples (take out positive ang negative samples)
        sim_pos = sim_resh * mask_pos.view(mask_pos.shape[0], 1, 1, -1)         # [B,H,W,MN]
        sim_neg = sim_resh * mask_neg.view(mask_pos.shape[0], 1, 1, -1)

        # take top k positive and negative examples
        # mean over the top positive and negative examples
        # sim_pos = F.softmax(sim_pos, dim=-1)                                  # [B, H, W, MN]
        pos_map = torch.mean(torch.topk(sim_pos, topk, dim=-1).values, dim=-1)  # [B, H, W]
        neg_map = torch.mean(torch.topk(sim_neg, topk, dim=-1).values, dim=-1)

        p = torch.cat((torch.unsqueeze(pos_map, -1), torch.unsqueeze(neg_map, -1)), dim=-1) # [B, H, W, 2]
        p = F.softmax(p, dim=-1)                                                # [1, 24, 24, 2]

        ''' feat * similarity of foreground '''
        out_pos = f_test * torch.unsqueeze(p[:, :, :, 0], dim=1)
        # out_neg = f_test * torch.unsqueeze(p[:, :, :, 1], dim=1)
        return out_pos
