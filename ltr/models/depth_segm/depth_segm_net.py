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


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
            all(roi[:, 4] <= image_size[1]-1)
    return valid


def normalize_vis_img(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)

class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv(1, 64, kernel_size=1, padding=0)
        self.conv2 = conv(64, 128)
        self.conv3 = conv_no_relu(128, 256)

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
    def __init__(self, segm_input_dim=(128,256), segm_inter_dim=(256,256), segm_dim=(64, 64), mixer_channels=2, topk_pos=3, topk_neg=3):
        super().__init__()

        self.depth_feat_extractor = DepthNet()

        # self.segment0 = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
        # self.segment1 = conv_no_relu(segm_dim[0], segm_dim[1])
        #
        self.mixer = conv(256, 256)
        # self.s3 = conv(segm_inter_dim[3], segm_inter_dim[2])
        #
        # self.s2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        # self.s1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        # self.s0 = conv(segm_inter_dim[0], segm_inter_dim[0])
        #
        # self.f2 = conv(segm_input_dim[2], segm_inter_dim[2])
        # self.f1 = conv(segm_input_dim[1], segm_inter_dim[1])
        # self.f0 = conv(segm_input_dim[0], segm_inter_dim[0])
        #
        self.post2 = conv(256, 128)
        self.post1 = conv(128, 64)
        self.post0 = conv_no_relu(64, 2)


        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        # self.topk_pos = topk_pos
        # self.topk_neg = topk_neg

    def forward(self, feat_test, depth_test, feat_train, depth_train, mask_train, test_dist=None):
        ''' Song's comments:
            Given feat_test and feat_train and mask_train, and test_dist,
                Step 1: similarity maps between feat_test and feat_train
                Step 2: F + P + L
                Step 3: Conv -> Up -> segmentation outputs


            Simple Network 01 : Depth images -> depth feat + dist -> mask ???
        '''
        # f_test = self.segment1(self.segment0(feat_test[3]))    # 1x1x64 conv + 3x3x64 conv -> [1, 1024, 24,24] -> [1, 64, 24, 24]
        # f_train = self.segment1(self.segment0(feat_train[3]))  # 1x1x64 conv + 3x3x64 conv -> [1, 1024, 24,24] -> [1, 64, 24, 24]
        # # reshape mask to the feature size
        # mask_pos = F.interpolate(mask_train[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,384, 384] -> [1,1,24,24]
        # mask_neg = 1 - mask_pos
        #
        # pred_pos, pred_neg = self.similarity_segmentation(f_test, f_train, mask_pos, mask_neg)
        #
        # pred_ = torch.cat((torch.unsqueeze(pred_pos, -1), torch.unsqueeze(pred_neg, -1)), dim=-1)
        # pred_sm = F.softmax(pred_, dim=-1) # [1, 24, 24, 2]
        print(depth_test.shape)
        f_test_depth = self.depth_feat_extractor(depth_test)  # [N, 1, H, W] -> [B, 256, H, W]
        # f_train_depth = self.depth_feat_extractor(depth_train)
        print(f_test_depth.shape)
        if test_dist is not None:
            # distance map is give - resize for mixer
            # dist = F.interpolate(test_dist[0], size=(f_train.shape[-2], f_train.shape[-1])) # [1,1,24,24]
            dist = F.interpolate(test_dist[0], size=(f_test_depth.shape[-2], f_test_depth.shape[-1])) # [1,1,24,24]
            print('dist : ', dist.shape)
            # concatenate inputs for mixer
            # softmaxed segmentation, positive segmentation and distance map
            # ''' F + P + L '''
            # segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1),
            #                          torch.unsqueeze(pred_pos, dim=1),
            #                          dist), dim=1)
            ''' How to DepthF + Dist ???
                1) cat
                2) channel wise multiplication???
            '''
            segm_layers = torch.cat((torch.unsqueeze(f_test_depth, dim=1),
                                     dist), dim=1)
        else:
            # segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1), torch.unsqueeze(pred_pos, dim=1)), dim=1)
            segm_layers = f_test_depth

        print(segm_layers.shape)

        ''' Song's comment :
            segm_layers -> mask
            do we need the upsample ???
        '''
        out = self.mixer(segm_layers)
        print('out : ', out.shape)
        # out = self.s3(F.upsample(out, scale_factor=2))



        ''' Do we use the RGB features ???
            e.g. self.f2(feat_test[2]) + self.s2(out)

            1) only depth feature
            2) depth + rgb features [layer0, layer1, layer2] more semantic features

            3) channel correlation between f_train and f_test

            4) similarity between test and train, same as D3S ???? == D3S + Depth branch ???
        '''
        # out = self.post2(F.upsample(self.f2(feat_test[2]) + self.s2(out), scale_factor=2))
        # out = self.post1(F.upsample(self.f1(feat_test[1]) + self.s1(out), scale_factor=2))
        # out = self.post0(F.upsample(self.f0(feat_test[0]) + self.s0(out), scale_factor=2))
        out = self.post2(out)
        print('post02: ', out.shape)
        out = self.post1(out)
        print('post01: ', out.shape)
        out = self.post0(out)
        print('post0: ', out.shape)

        return out # [1,2,384,384]


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
