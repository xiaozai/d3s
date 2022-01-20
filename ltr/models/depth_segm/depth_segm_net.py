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
        '''
        self.conv1 = conv(input_dim, dims[0], kernel_size=kernels[0], padding=pads[0])
        self.conv2 = conv(dims[0], dims[1], kernel_size=kernels[1], padding=pads[1])
        self.conv3 = conv_no_relu(dims[1], dims[2], kernel_size=kernels[2], padding=pads[2])

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dp):
        feat1 = self.conv1(dp)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
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

        self.depth_feat_extractor = DepthNet(input_dim=1, dims=(32, 32, 2), kernels=(1,3,1), pads=(0, 1, 0))

        '''
        1) simple : depth feat + L 256+1 -> mixer 3x3x128 -> 3x3x64 -> 3x3x32 -> 3x3x2, train loss = 0.048
        2)        : depth feat + L 2+1   -> mixer 1x1x32  -> 3x3x32 -> 3x3x32 -> 3x3x2
        '''
        self.mixer = conv(3, 32, kernel_size=1, padding=0) # ???? 256 depth feat + 1 dist map ?

        self.f2 = conv(512, 32, kernel_size=1, padding=0)
        self.f1 = conv(256, 32, kernel_size=1, padding=0)
        self.f0 = conv(64, 32, kernel_size=1, padding=0)

        self.post2 = conv(32, 32)
        self.post1 = conv(32, 32)
        self.post0 = conv_no_relu(32, 2) # GT is the pair of Foreground and Background segmentations

        self.initialize()

    def initialize(self):
        # Init weights
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
        '''
        f_test_depth = self.depth_feat_extractor(depth_test_imgs)  # [B, 2, 384, 384] = F+B
        # pred_sm = F.softmax(f_test_depth, dim=1)  # ????
        ''' We assume that DepthFeat = F+P, concatenate with dist (location) map
            how about channel wise multiplication???
        '''
        if test_dist is not None:
            '''Song: we change the test_dist map into Guassian map instead of distance map '''
            # distance map is give - resize for mixer # concatenate inputs for mixer
            dist = F.interpolate(test_dist[0], size=(f_test_depth.shape[-2], f_test_depth.shape[-1])) # [B,1,384,384]
            segm_layers = torch.cat((f_test_depth, dist), dim=1)                                      # [B, C+1, 384, 384]
        else:
            segm_layers = torch.cat((f_test_depth, f_test_depth[:, 0, :, :]), dim=1)                  # [B, 3, 384, 384]

        # Mix DepthFeat and Location Map
        out_mix = self.mixer(segm_layers) # [B, 32, 384, 384]

        ''' Song's comment:
            Do we use the RGB features ??? Yes
            1) only depth feature -> mask ! when depth is missing or can not distinguish background and target
            2) depth + rgb features [layer0, layer1, layer2] more semantic features
            3) channel correlation between f_train and f_test
               or similarity between test and train, same as D3S ???? == D3S + Depth branch ???  D3S used the layer3 features, 24*24
                   1)
                   similarity_rgb = cos(feat_train_rgb * feat_test_rgb) * train_mask
                   similarity_d   = cos(feat_train_d * feat_test_d) * train_mask
                   similarity = max(similarity_rgb, similarity_d)
                   2)
                   feat_test_rgb = softmax(similarity_rgb) * feat_test_rgb
                   feat_test_d = softmax(similarity_d) * feat_test_d
        '''
        out1 = self.post2(F.upsample(self.f2(feat_test_rgb[2]), scale_factor=8) + out_mix)
        out2 = self.post1(F.upsample(self.f1(feat_test_rgb[1]), scale_factor=4) + out1)
        out3 = self.post0(F.upsample(self.f0(feat_test_rgb[0]), scale_factor=2) + out2)

        return out3
