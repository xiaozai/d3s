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

def cat_attn_feat(attn_weights):
    ''' attn_weights channels : 32, 16, 4, 2 => 4*8, 4*4, 2*2, 1*2'''
    C, H, W = attn_weights.shape
    if C == 32:
        edge = 4
        attn_maps = np.zeros((edge*H*2, edge*W), dtype=np.float32)
    elif C == 16:
        edge = 4
        attn_maps = np.zeros((edge*H, edge*W), dtype=np.float32)
    elif C == 4:
        edge = 2
        attn_maps = np.zeros((edge*H, edge*W), dtype=np.float32)
    elif C == 2:
        edge = 2
        attn_maps = np.zeros((H, edge*W), dtype=np.float32)

    # edge = int(math.sqrt(C))+1
    # attn_maps = np.zeros((edge*H, edge*W), dtype=np.float32)
    for idx in range(C):
        attn = attn_weights[idx] # H, W
        hid = idx // edge
        wid = idx % edge
        attn_maps[hid*H:(hid+1)*H, wid*W:(wid+1)*W] = attn
    return attn_maps

def process_attn_maps(att_mat, batch_element, train_mask):
    # use batch 0
    att_mat = torch.stack(att_mat) # [layers=3, B, heads=3, Pq, P_kv]
    att_mat = att_mat.detach().cpu()
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

    if int(np.sqrt(aug_att_mat.size(-2)//4)) ** 2 * 4 == aug_att_mat.size(-2):
        grid_size = int(np.sqrt(aug_att_mat.size(-2)//4))
        rows = 2
        cols = 2
    elif int(np.sqrt(aug_att_mat.size(-2)//2)) ** 2 * 2 == aug_att_mat.size(-2):
        grid_size = int(np.sqrt(aug_att_mat.size(-2)//2))
        rows = 2
        cols = 1
    # block_size = train_mask.shape[0]//grid_size
    # mask = skimage.measure.block_reduce(train_mask, (block_size, block_size), np.max)
    # mask = np.concatenate((mask, mask), axis=0)
    # mask = np.reshape(mask, (grid_size*grid_size*2,))

    out_img = np.zeros((v.shape[0],))
    for idx in range(v.shape[0]):
        # pixel = v[idx, :].detach().numpy() * mask
        pixel = v[idx, :].detach().numpy()
        # pixel = softmax(pixel)
        p_kv = len(pixel)
        prob_idx = np.argmax(pixel)
        if prob_idx < p_kv // 2:
            out_img[idx] = pixel.max() # probability for FG
        else:
            out_img[idx] = 0

    return out_img.reshape((grid_size*rows, grid_size*cols))

    # return out_img.reshape((grid_size*2, grid_size))
    # return out_img.reshape((grid_size*2, -1))

def save_debug(data, pred_mask, vis_data):

    batch_element = 0
    # vis_cosine_similarity = True

    if len(vis_data) == 2:
        p_rgb, p_d = vis_data

        p_rgb = p_rgb[batch_element, ...] # [H, W, 2] F + B
        p_d = p_d[batch_element, ...]
        p_rgb = (p_rgb.detach().cpu().numpy().squeeze()).astype(np.float32) # [H, W, 2]
        p_d = (p_d.detach().cpu().numpy().squeeze()).astype(np.float32)

    elif len(vis_data) == 3:
        attn_weights2, attn_weights1, attn_weights0 = vis_data
        attn_weights3 = attn_weights2
        train_mask = train_mask = data['train_masks'][0, batch_element, :, :]

        # attn_weights = [layers, B, Heads, P_q, P_kv]
        attn_weights3 = process_attn_maps(attn_weights3, batch_element, train_mask)
        attn_weights2 = process_attn_maps(attn_weights2, batch_element, train_mask)
        attn_weights1 = process_attn_maps(attn_weights1, batch_element, train_mask)
        attn_weights0 = process_attn_maps(attn_weights0, batch_element, train_mask)

    elif len(vis_data) == 4:
        attn_weights3, attn_weights2, attn_weights1, attn_weights0 = vis_data
        train_mask = train_mask = data['train_masks'][0, batch_element, :, :]
        attn_weights3 = process_attn_maps(attn_weights3, batch_element, train_mask)
        attn_weights2 = process_attn_maps(attn_weights2, batch_element, train_mask)
        attn_weights1 = process_attn_maps(attn_weights1, batch_element, train_mask)
        attn_weights0 = process_attn_maps(attn_weights0, batch_element, train_mask)


    dir_path = data['settings'].env.images_dir

    train_img = data['train_images'][:, batch_element, :, :].permute(1, 2, 0)
    train_depth = data['train_depths'][:, batch_element, :, :].permute(1, 2, 0)
    train_mask = data['train_masks'][0, batch_element, :, :]
    test_img = data['test_images'][:, batch_element, :, :].permute(1, 2, 0)
    test_depth = data['test_depths'][:, batch_element, :, :].permute(1, 2, 0)
    test_mask = data['test_masks'][0, batch_element, :, :]

    test_dist = data['test_dist'][0, batch_element, :, :] # song


    # softmax on the mask prediction (since this is done internaly when calculating loss)
    mask = F.softmax(pred_mask, dim=1)[batch_element, 0, :, :].cpu().detach().numpy().astype(np.float32)
    predicted_mask = (mask > 0.5).astype(np.float32) * mask

    mu = torch.Tensor(data['settings'].normalize_mean).to(torch.device('cuda')).view(1, 1, 3)
    std = torch.Tensor(data['settings'].normalize_std).to(torch.device('cuda')).view(1, 1, 3)

    train_img = 255 * (train_img * std + mu)
    test_img = 255 * (test_img * std + mu)

    train_img = (train_img.detach().cpu().numpy()).astype(np.uint8)
    train_depth = (train_depth.detach().cpu().numpy().squeeze()).astype(np.float32)
    test_img = (test_img.detach().cpu().numpy()).astype(np.uint8)
    test_depth = (test_depth.detach().cpu().numpy().squeeze()).astype(np.float32)
    train_mask = (train_mask.detach().cpu().numpy()).astype(np.float32)
    test_mask = (test_mask.detach().cpu().numpy()).astype(np.float32)
    test_dist = (test_dist.detach().cpu().numpy().squeeze()).astype(np.float32)
    test_conf = 1 - test_dist / np.max(test_dist)

    # Song
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(9, 9))

    draw_axis(ax1, train_img, 'Train image')
    draw_axis(ax3, test_img, 'Test image')
    draw_axis(ax2, train_depth, 'Train depth')
    draw_axis(ax4, test_depth, 'Test depth')
    draw_axis(ax5, train_mask, 'train mask')
    draw_axis(ax6, test_mask, 'Ground-truth')
    draw_axis(ax7, predicted_mask, 'Prediction', show_minmax=True)
    draw_axis(ax8, test_dist, 'test_dist')



    if len(vis_data) == 2:
        draw_axis(ax9, test_conf, 'test_dist')
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



def save_debug_attnweights(data, pred_mask, vis_data, batch_element=0):

    # detach from computation path
    data = data.detach().clone().cpu()
    pred_mask = pred_mask.detach().clone().cpu()
    vis_data = [x.detach().clone().cpu().numpy().astype(np.float32) for x in vis_data]
    dir_path = data['settings'].env.images_dir

    vis_data = [x[batch_element, :, :, :].squeeze() for x in vis_data]
    attn_maps = [cat_attn_feat(x) for x in vis_data]
    if len(attn_maps) == 4:
        attn_maps3, attn_maps2, attn_maps1, attn_maps0 = attn_maps # B,C,H,W
    elif len(attn_maps) == 3:
        attn_maps2, attn_maps1, attn_maps0 = attn_maps # B,C,H,W
        attn_maps3 = attn_maps2


    train_img = data['train_images'][:, batch_element, :, :].permute(1, 2, 0)
    test_img = data['test_images'][:, batch_element, :, :].permute(1, 2, 0)
    train_depth = data['train_depths'][:, batch_element, :, :].permute(1, 2, 0).numpy().squeeze().astype(np.float32)
    test_depth = data['test_depths'][:, batch_element, :, :].permute(1, 2, 0).numpy().squeeze().astype(np.float32)
    train_mask = data['train_masks'][0, batch_element, :, :].numpy().squeeze().astype(np.float32)
    test_mask = data['test_masks'][0, batch_element, :, :].numpy().squeeze().astype(np.float32)
    test_dist = data['test_dist'][0, batch_element, :, :].numpy().squeeze().astype(np.float32)


    mu = torch.Tensor(data['settings'].normalize_mean).view(1, 1, 3)
    std = torch.Tensor(data['settings'].normalize_std).view(1, 1, 3)
    train_img = 255 * (train_img * std + mu)
    test_img = 255 * (test_img * std + mu)
    train_img = train_img.numpy().astype(np.uint8)
    test_img = test_img.numpy().astype(np.uint8)


    # # # softmax on the mask prediction (since this is done internaly when calculating loss)
    mask = F.softmax(pred_mask, dim=1)[batch_element, 0, :, :].numpy().squeeze().astype(np.float32)
    predicted_mask = (mask > 0.5).astype(np.float32) * mask
    #
    # #
    plt.switch_backend('agg')
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(20, 20))
    # #
    draw_axis(ax1, train_img, 'Train image')
    draw_axis(ax3, test_img, 'Test image')
    draw_axis(ax2, train_depth, 'Train depth')
    draw_axis(ax4, test_depth, 'Test depth')
    draw_axis(ax5, train_mask, 'train mask')
    draw_axis(ax6, test_mask, 'Ground-truth')
    draw_axis(ax7, predicted_mask, 'Prediction', show_minmax=True)
    draw_axis(ax8, test_dist, 'test_dist')
    draw_axis(ax9, attn_maps3, 'attn_weights3', show_minmax=True)
    draw_axis(ax10, attn_maps2, 'attn_weights2', show_minmax=True)
    draw_axis(ax11, attn_maps1, 'attn_weights1', show_minmax=True)
    draw_axis(ax12, attn_maps0, 'attn_weights0', show_minmax=True)
    #
    save_path = os.path.join(data['settings'].env.images_dir, '%03d-%04d.png' % (data['epoch'], data['iter']))
    #
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
            save_debug(data,masks_pred, vis_data) # vis_data = (p_rgb, p_d) or  (pred_sm_d, attn_weights2, attn_weights1, attn_weights0)

        return loss, stats

class DepthSegmActor_no_targetsz(BaseActor):
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
        # masks_gt_pair = masks_gt_pair.to(masks_pred.device)

        loss = self.objective(masks_pred, masks_gt_pair)

        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss.item()}

        if 'iter' in data and (data['iter'] - 1) % 50 == 0:
            save_debug_attnweights(data, masks_pred, vis_data) # vis_data = (p_rgb, p_d) or  (pred_sm_d, attn_weights2, attn_weights1, attn_weights0)

        return loss, stats
