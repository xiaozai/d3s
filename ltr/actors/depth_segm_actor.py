from . import BaseActor
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import skimage.measure
import math

from scipy.special import softmax

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import ltr.models.depth_segm.depth_segm as segm_models

def draw_axis(ax, img, title, show_minmax=False):
    ax.imshow(img)
    if show_minmax:
        minval_, maxval_, _, _ = cv2.minMaxLoc(img)
        title = '%s \n min=%.2f max=%.2f' % (title, minval_, maxval_)
    ax.set_title(title, fontsize=9)

def process_attn_maps(att_mat, batch_element, layer=0): #, train_mask):
    # use batch 0
    if isinstance(att_mat, list) or isinstance(att_mat, tuple):
        att_mat = torch.stack(att_mat) # [layers=3, B, heads=3, Pq, P_kv]
    # att_mat = att_mat.detach().cpu()
    att_mat = att_mat[:, batch_element, ...].squeeze(1) # [layers=3,heads=3, P_q, P_kv]
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1) # [layers, P_q, P_kv]
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att # [layers, 144, 144]
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1] # last layer of multihead attention, [P_q, P_kv]

    if int(np.sqrt(aug_att_mat.size(-2))) ** 2 == aug_att_mat.size(-2):
        grid_size = int(np.sqrt(aug_att_mat.size(-2)))
        rows = 1
        cols = 1
    # elif int(np.sqrt(aug_att_mat.size(-2)//4)) ** 2 * 4 == aug_att_mat.size(-2):
    #     grid_size = int(np.sqrt(aug_att_mat.size(-2)//4))
    #     rows = 2
    #     cols = 2
    elif int(np.sqrt(aug_att_mat.size(-2)//2)) ** 2 * 2 == aug_att_mat.size(-2):
        grid_size = int(np.sqrt(aug_att_mat.size(-2)//2))
        rows = 2
        cols = 1

    out_img = np.zeros((v.shape[0],))
    for idx in range(v.shape[0]):
        pixel = v[idx, :].detach().numpy()
        out_img[idx] = pixel.max() # probability for FG

    out_img = out_img.reshape((grid_size*rows, grid_size*cols))

    return out_img

def save_debug(data, pred_mask, vis_data, batch_element = 0):

    data = data.detach().clone().cpu()
    pred_mask = pred_mask.detach().clone().cpu()
    if vis_data is not None:
        if isinstance(vis_data, list) or isinstance(vis_data, tuple):
            vis_data = [torch.stack(vid).detach().clone().cpu() if isinstance(vid, list) or isinstance(vid, tuple) \
                                                                else vid.detach().clone().cpu() \
                                                                for vid in vis_data]
        else:
            vis_data = vis_data.detach().clone().cpu()


        if len(vis_data) == 2:
            p_rgb, p_d = vis_data

            p_rgb = p_rgb[batch_element, ...] # [H, W, 2] F + B
            p_d = p_d[batch_element, ...]
            p_rgb = (p_rgb.numpy().squeeze()).astype(np.float32) # [H, W, 2]
            p_d = (p_d.numpy().squeeze()).astype(np.float32)

        elif len(vis_data) == 3:
            attn_weights2, attn_weights1, attn_weights0 = vis_data
            attn_weights3 = attn_weights2

            attn_weights3 = process_attn_maps(attn_weights3, batch_element)# train_mask)
            attn_weights2 = process_attn_maps(attn_weights2, batch_element)# train_mask)
            attn_weights1 = process_attn_maps(attn_weights1, batch_element)# train_mask)
            attn_weights0 = process_attn_maps(attn_weights0, batch_element)# train_mask)

        elif len(vis_data) == 4:
            attn_weights3, attn_weights2, attn_weights1, attn_weights0 = vis_data

            attn_weights3 = process_attn_maps(attn_weights3, batch_element, layer=3)#, train_mask)
            attn_weights2 = process_attn_maps(attn_weights2, batch_element, layer=2)#, train_mask)
            attn_weights1 = process_attn_maps(attn_weights1, batch_element, layer=1)#, train_mask)
            attn_weights0 = process_attn_maps(attn_weights0, batch_element, layer=0)#, train_mask)


    dir_path = data['settings'].env.images_dir

    train_img = data['train_images'][:, batch_element, :, :].permute(1, 2, 0)
    test_img = data['test_images'][:, batch_element, :, :].permute(1, 2, 0)

    mu = torch.Tensor(data['settings'].normalize_mean).view(1, 1, 3)
    std = torch.Tensor(data['settings'].normalize_std).view(1, 1, 3)

    train_img = 255 * (train_img * std + mu)
    test_img = 255 * (test_img * std + mu)

    train_img = train_img.numpy().astype(np.uint8)
    test_img = test_img.numpy().astype(np.uint8)

    ''' Song, when using depth colormap and normalization '''
    train_depth = data['train_depths'][:, batch_element, :, :].permute(1, 2, 0) # .numpy().squeeze().astype(np.float32)
    test_depth = data['test_depths'][:, batch_element, :, :].permute(1, 2, 0) #.numpy().squeeze().astype(np.float32)

    train_depth = train_depth.numpy().squeeze()
    test_depth = test_depth.numpy().squeeze()

    if train_depth.shape[-1] == 3:
        train_depth = 255 * (train_depth * std + mu)
        test_depth = 255 * (test_depth * std + mu)

        train_depth = train_depth.astype(np.uint8)
        test_depth = test_depth.astype(np.uint8)


    train_mask = data['train_masks'][0, batch_element, :, :].numpy().astype(np.float32)
    test_mask = data['test_masks'][0, batch_element, :, :].numpy().astype(np.float32)
    test_dist = data['test_dist'][0, batch_element, :, :].numpy().squeeze().astype(np.float32)


    # softmax on the mask prediction (since this is done internaly when calculating loss)
    mask = F.softmax(pred_mask, dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
    predicted_mask = (mask > 0.5).astype(np.float32) * mask


    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(9, 9))

    draw_axis(ax1, train_img, 'Train image')
    draw_axis(ax3, test_img, 'Test image')
    draw_axis(ax2, train_depth, 'Train depth')
    draw_axis(ax4, test_depth, 'Test depth')
    draw_axis(ax5, train_mask, 'train mask')
    draw_axis(ax6, test_mask, 'Ground-truth')
    draw_axis(ax7, predicted_mask, 'Prediction', show_minmax=True)
    draw_axis(ax8, test_dist, 'test_dist')


    if vis_data is not None:
        if len(vis_data) == 2:
            draw_axis(ax9, test_dist, 'test_dist')
            draw_axis(ax10, p_rgb, 'similarity rgb')
            draw_axis(ax11, p_d, 'similarity d')

        elif len(vis_data) == 4 or len(vis_data) == 3:
            draw_axis(ax9, attn_weights3, 'attn_weights3', show_minmax=True)
            draw_axis(ax10, attn_weights2, 'attn_weights2', show_minmax=True)
            draw_axis(ax11, attn_weights1, 'attn_weights1', show_minmax=True)
            draw_axis(ax12, attn_weights0, 'attn_weights0', show_minmax=True)


    save_path = os.path.join(data['settings'].env.images_dir, '%03d-%04d.png' % (data['epoch'], data['iter']))
    plt.savefig(save_path)
    plt.close(f)

def process_feature_weights(weight_rgb, weight_d, batch_element=0):
    # 64, 32, 16, 4, 2
    weight_rgb = weight_rgb.numpy().squeeze()
    weight_rgb = weight_rgb[batch_element, ...]
    weight_rgb = weight_rgb.squeeze()
    weight_d = weight_d.numpy().squeeze()
    weight_d = weight_d[batch_element, ...]
    weight_d = weight_d.squeeze()
    # channels = weight_rgb.shape[0]
    # row = int(math.sqrt(channels))
    weight_rgb = weight_rgb.reshape(4, -1)
    weight_d = weight_d.reshape(4, -1)

    weight = np.concatenate((weight_rgb, weight_d), axis=1)
    return weight

def save_debug_MP(data, pred_mask, vis_data, batch_element = 0):

    data = data.detach().clone().cpu()
    pred_mask = [pm.detach().clone().cpu() for pm in pred_mask]

    if vis_data is not None:
        if isinstance(vis_data, list) or isinstance(vis_data, tuple):
            vis_data = [torch.stack(vid).detach().clone().cpu() if isinstance(vid, list) or isinstance(vid, tuple) \
                                                                else vid.detach().clone().cpu() \
                                                                for vid in vis_data]
        else:
            vis_data = vis_data.detach().clone().cpu()

        if len(vis_data) == 32:
            attn_d = vis_data[batch_element, 0, ...].numpy().squeeze()

        elif len(vis_data) == 2:
            attn_weights1, attn_weights3 = vis_data
            attn_weights1 = attn_weights1[batch_element, 0, ...].numpy().squeeze()
            attn_weights3 = attn_weights3[batch_element, 0, ...].numpy().squeeze()

        elif len(vis_data) == 3:
            attn_weights2, attn_weights1, attn_weights0 = vis_data
            attn_weights3 = attn_weights2

            attn_weights3 = process_attn_maps(attn_weights3, batch_element)# train_mask)
            attn_weights2 = process_attn_maps(attn_weights2, batch_element)# train_mask)
            attn_weights1 = process_attn_maps(attn_weights1, batch_element)# train_mask)
            attn_weights0 = process_attn_maps(attn_weights0, batch_element)# train_mask)

        elif len(vis_data) == 4:
            attn_weights0, attn_weights1, attn_weights2, attn_weights3 = vis_data

            if isinstance(attn_weights3, list) or isinstance(attn_weights3, tuple) or len(attn_weights3.shape) == 5:
                ''' transformer attn maps '''
                attn_weights3 = process_attn_maps(attn_weights3, batch_element, layer=3)
            elif len(attn_weights3.shape) == 4:
                ''' spatial attn maps '''
                attn_weights3 = torch.mean(attn_weights3[batch_element, :, ...], dim=0).numpy().squeeze() # H * W for RGB weights


            if isinstance(attn_weights2, list) or isinstance(attn_weights2, tuple) or len(attn_weights2.shape) == 5:
                attn_weights2 = process_attn_maps(attn_weights2, batch_element, layer=2)
                attn_weights1 = process_attn_maps(attn_weights1, batch_element, layer=1)
                attn_weights0 = process_attn_maps(attn_weights0, batch_element, layer=0)

            elif len(attn_weights2.shape) == 4:
                attn_weights2 = attn_weights2[batch_element, 0, ...].numpy().squeeze()
                attn_weights1 = attn_weights1[batch_element, 0, ...].numpy().squeeze()
                attn_weights0 = attn_weights0[batch_element, 0, ...].numpy().squeeze()

        elif len(vis_data) == 5:
            attn_d, attn_weights3, attn_weights2, attn_weights1, attn_weights0 = vis_data
            attn_d = attn_d[batch_element, 0, ...].numpy().squeeze()
            attn_weights3 = attn_weights3[batch_element, 0, ...].numpy().squeeze()
            attn_weights2 = attn_weights2[batch_element, 0, ...].numpy().squeeze()
            attn_weights1 = attn_weights1[batch_element, 0, ...].numpy().squeeze()
            attn_weights0 = attn_weights0[batch_element, 0, ...].numpy().squeeze()

        elif len(vis_data) == 8:
            weight_rgb3, weight_rgb2, weight_rgb1, weight_rgb0, weight_d3, weight_d2, weight_d1, weight_d0 = vis_data
            # feature channel attention weights , B, C, 1, 1
            weight3 = process_feature_weights(weight_rgb3, weight_d3, batch_element=batch_element)
            weight2 = process_feature_weights(weight_rgb2, weight_d2, batch_element=batch_element)
            weight1 = process_feature_weights(weight_rgb1, weight_d1, batch_element=batch_element)
            weight0 = process_feature_weights(weight_rgb0, weight_d0, batch_element=batch_element)


    dir_path = data['settings'].env.images_dir

    train_img = data['train_images'][:, batch_element, :, :].permute(1, 2, 0)
    test_img = data['test_images'][:, batch_element, :, :].permute(1, 2, 0)

    mu = torch.Tensor(data['settings'].normalize_mean).view(1, 1, 3)
    std = torch.Tensor(data['settings'].normalize_std).view(1, 1, 3)

    train_img = 255 * (train_img * std + mu)
    test_img = 255 * (test_img * std + mu)

    train_img = train_img.numpy().astype(np.uint8)
    test_img = test_img.numpy().astype(np.uint8)

    ''' Song, when using depth colormap and normalization '''
    train_depth = data['train_depths'][:, batch_element, :, :].permute(1, 2, 0) # .numpy().squeeze().astype(np.float32)
    test_depth = data['test_depths'][:, batch_element, :, :].permute(1, 2, 0) #.numpy().squeeze().astype(np.float32)

    # if train_depth.shape[-1] == 3:
    #     train_depth = 255 * (train_depth * std + mu)
    #     test_depth = 255 * (test_depth * std + mu)

    if train_depth.shape[-1] == 3:
        train_depth = 255 * (train_depth * std + mu)
        test_depth = 255 * (test_depth * std + mu)

        train_depth = train_depth.numpy().squeeze().astype(np.uint8)
        test_depth = test_depth.numpy().squeeze().astype(np.uint8)
    else:
        train_depth = train_depth.numpy().squeeze() #.astype(np.uint8)
        test_depth = test_depth.numpy().squeeze() # .astype(np.uint8)

    train_mask = data['train_masks'][0, batch_element, :, :].numpy().astype(np.float32)
    test_mask = data['test_masks'][0, batch_element, :, :].numpy().astype(np.float32)
    test_dist = data['test_dist'][0, batch_element, :, :].numpy().squeeze().astype(np.float32)


    if len(pred_mask) == 4:
        # softmax on the mask prediction (since this is done internaly when calculating loss)
        mask0 = F.softmax(pred_mask[0], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask0 = (mask0 > 0.5).astype(np.float32) * mask0

        mask1 = F.softmax(pred_mask[1], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask1 = (mask1 > 0.5).astype(np.float32) * mask1

        mask2 = F.softmax(pred_mask[2], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask2 = (mask2 > 0.5).astype(np.float32) * mask2

        mask3 = F.softmax(pred_mask[3], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask3 = (mask3 > 0.5).astype(np.float32) * mask3

    elif len(pred_mask) == 2:
        mask1 = F.softmax(pred_mask[0], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask1 = (mask1 > 0.5).astype(np.float32) * mask1

        mask3 = F.softmax(pred_mask[1], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask3 = (mask3 > 0.5).astype(np.float32) * mask3

    elif len(pred_mask) == 3:
        mask1 = F.softmax(pred_mask[0], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask1 = (mask1 > 0.5).astype(np.float32) * mask1

        mask3 = F.softmax(pred_mask[1], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask3 = (mask3 > 0.5).astype(np.float32) * mask3

        mask_d = F.softmax(pred_mask[2], dim=1)[batch_element, 0, :, :].numpy().astype(np.float32)
        predicted_mask_d = (mask_d > 0.5).astype(np.float32) * mask_d


    f, ((ax1, ax2, ax3, ax4), \
        (ax5, ax6, ax7, ax8), \
        (ax9, ax10, ax11, ax12), \
        (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize=(12, 12))

    plt.axis('off')

    draw_axis(ax1, train_img, 'Train image')
    draw_axis(ax3, test_img, 'Test image')
    draw_axis(ax2, train_depth, 'Train depth')
    draw_axis(ax4, test_depth, 'Test depth')
    draw_axis(ax5, train_mask, 'train mask')
    draw_axis(ax6, test_mask, 'Ground-truth')
    # draw_axis(ax7, predicted_mask0, 'Prediction') # , show_minmax=True)
    draw_axis(ax8, test_dist, 'test_dist')


    draw_axis(ax14, predicted_mask1, 'Layer1') #, show_minmax=True)
    draw_axis(ax16, predicted_mask3, 'Layer3') #, show_minmax=True)

    if len(pred_mask) == 4:
        draw_axis(ax13, predicted_mask0, 'Layer0') #, show_minmax=True)
        draw_axis(ax15, predicted_mask2, 'Layer2') #, show_minmax=True)

    elif len(pred_mask) == 3:
        draw_axis(ax13, predicted_mask_d, 'DepthMask') #, show_minmax=True)



    if vis_data is not None:
        print(len(vis_data))
        if len(vis_data) == 32:
            draw_axis(ax7, attn_d, 'attn_d')

        elif len(vis_data) == 2:
            # draw_axis(ax9, test_dist, 'test_dist')
            draw_axis(ax11, attn_weights1, 'attn_weights 1')
            draw_axis(ax12, attn_weights3, 'attn_weights 3')

        elif len(vis_data) == 4 or len(vis_data) == 3:
            draw_axis(ax9, attn_weights0, 'attn_weights0')
            draw_axis(ax10, attn_weights1, 'attn_weights1')
            draw_axis(ax11, attn_weights2, 'attn_weights2')
            draw_axis(ax12, attn_weights3, 'attn_weights3')

        elif len(vis_data) == 5:
            draw_axis(ax7, attn_d, 'attn_d')
            draw_axis(ax9, attn_weights0, 'attn_weights0')
            draw_axis(ax10, attn_weights1, 'attn_weights1')
            draw_axis(ax11, attn_weights2, 'attn_weights2')
            draw_axis(ax12, attn_weights3, 'attn_weights3')

        elif len(vis_data) == 8:
            draw_axis(ax9, weight0, 'channel_weights0')
            draw_axis(ax10, weight1, 'channel_weights1')
            draw_axis(ax11, weight2, 'channel_weights2')
            draw_axis(ax12, weight3, 'channel_weights3')

    save_path = os.path.join(data['settings'].env.images_dir, '%03d-%04d.png' % (data['epoch'], data['iter']))
    plt.savefig(save_path)
    plt.close(f)

class DepthSegmActor(BaseActor):
    """ Actor for training the Segmentation in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """

        test_dist = None
        if 'test_dist' in data:
            test_dist = data['test_dist'].permute(1, 0, 2, 3)

        masks_pred, vis_data = self.net(data['train_images'].permute(1, 0, 2, 3), # batch*3*384*384
                                        data['train_depths'].permute(1, 0, 2, 3), # batch*1*384*384
                                        data['test_images'].permute(1, 0, 2, 3),
                                        data['test_depths'].permute(1, 0, 2, 3),
                                        data['train_masks'].permute(1, 0, 2, 3),
                                        test_dist=test_dist,
                                        debug=True) # Song :  vis pos and neg maps


        masks_gt = data['test_masks'].permute(1, 0, 2, 3) # C, B, H, W -> # B * 1 * H * W
        masks_gt_pair = torch.cat((masks_gt, 1 - masks_gt), dim=1)   # B * 2 * H * W

        loss = self.objective(masks_pred, masks_gt_pair)

        if torch.isnan(loss):
            print('loss segm is Nan .....')

        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss.item()}

        if 'iter' in data and (data['iter'] - 1) % 50 == 0:

            try:
                save_debug(data, masks_pred, vis_data)
            except:
                print('save_debug error ....')

        return loss, stats

class DepthSegmActor_MultiPred(BaseActor):
    """ Actor for training the Segmentation in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """

        test_dist = None
        if 'test_dist' in data:
            test_dist = data['test_dist'].permute(1, 0, 2, 3)

        masks_pred, vis_data = self.net(data['train_images'].permute(1, 0, 2, 3), # batch*3*384*384
                                        data['train_depths'].permute(1, 0, 2, 3), # batch*1*384*384
                                        data['test_images'].permute(1, 0, 2, 3),
                                        data['test_depths'].permute(1, 0, 2, 3),
                                        data['train_masks'].permute(1, 0, 2, 3),
                                        test_dist=test_dist,
                                        debug=True) # Song :  vis pos and neg maps


        masks_gt = data['test_masks'].permute(1, 0, 2, 3) # C, B, H, W -> # B * 1 * H * W
        masks_gt_pair = torch.cat((masks_gt, 1 - masks_gt), dim=1)   # B * 2 * H * W

        if len(masks_pred) == 4:
            loss0 = self.objective(masks_pred[0], masks_gt_pair)
            loss1 = self.objective(masks_pred[1], masks_gt_pair)
            loss2 = self.objective(masks_pred[2], masks_gt_pair)
            loss3 = self.objective(masks_pred[3], masks_gt_pair)

            if self.loss_weights is None:
                self.loss_weights = [1, 0.1, 0.1, 0.1]
            loss = loss0 * self.loss_weights[0] + loss1 * self.loss_weights[1] + loss2 * self.loss_weights[2] + loss3 * self.loss_weights[3]

            stats = {'Loss/total': loss.item(),
                     # 'Loss/segm': loss.item(),
                     'Layer0': loss0.item(),
                     'Layer1': loss1.item(),
                     'Layer2': loss2.item(),
                     'Layer3': loss3.item(),
                     }

        elif len(masks_pred) == 2:
            loss1 = self.objective(masks_pred[0], masks_gt_pair)
            loss2 = self.objective(masks_pred[1], masks_gt_pair)
            loss = loss1 + loss2

            stats = {'Loss/total': loss.item(),
                     'Layer1': loss1.item(),
                     'Layer2': loss2.item(),
                     }

        elif len(masks_pred) == 3:
            loss1 = self.objective(masks_pred[0], masks_gt_pair)
            loss3 = self.objective(masks_pred[1], masks_gt_pair)
            loss_d = self.objective(masks_pred[2], masks_gt_pair)
            loss = loss1 + loss3 * 0.5 + loss_d * 0.5

            stats = {'Loss/total': loss.item(),
                     'Layer1': loss1.item(),
                     'Layer3': loss3.item(),
                     'Loss D': loss_d.item()
                     }

        if 'iter' in data and (data['iter'] - 1) % 50 == 0:

            save_debug_MP(data, masks_pred, vis_data)

        return loss, stats
