from . import BaseActor
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def draw_axis(ax, img, title, show_minmax=False):
    ax.imshow(img)
    if show_minmax:
        minval_, maxval_, _, _ = cv2.minMaxLoc(img)
        title = '%s \n min=%.2f max=%.2f' % (title, minval_, maxval_)
    ax.set_title(title, fontsize=9)

def process_attn_maps(att_mat, batch_element, train_mask):
    # use batch 0
    att_mat = torch.stack(att_mat)
    att_mat = att_mat[:, batch_element, ...].squeeze(1) # [layers=3, B, heads=3, 144, 144]
    att_mat = att_mat.cpu().detach()
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1) # [layers, 144, 144]
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

    # Attention from the output token to the input space.
    v = joint_attentions[-1] # last layer of multihead attention
    # print(v.shape) # 144x144, tokensxtokens in original papers, token0 is the class
    # select few tokens as output

    grid_size = int(np.sqrt(aug_att_mat.size(-1)//2)) # for each img,
    print(mask.max())
    mask = np.resize(train_mask, (grid_size, grid_size))
    mask = np.concatenate((mask, mask), axis=0)
    mask = np.reshape(mask, (grid_size*grid_size*2,))
    print(mask.max())
    out_img = np.zeros((v.shape[0],))
    for idx in range(v.shape[0]):
        # out_img[idx] = v[idx, :].detach().numpy().max() # 24*6
        pixel = v[idx, :].detach().numpy() * mask # (144, keep probs for foreground pixels
        out_img[idx] = pixel.max()
    # out_img = (out_img*255).astype(np.uint8)
    # print(out_img)
    return out_img.reshape((grid_size*2, grid_size))

def save_debug(data, pred_mask, vis_data):

    batch_element = 0
    # vis_cosine_similarity = True

    if len(vis_data) == 2:
        p_rgb, p_d = vis_data

        p_rgb = p_rgb[batch_element, ...] # [H, W, 2] F + B
        p_d = p_d[batch_element, ...]
        p_rgb = (p_rgb.detach().cpu().numpy().squeeze()).astype(np.float32) # [H, W, 2]
        p_d = (p_d.detach().cpu().numpy().squeeze()).astype(np.float32)

    elif len(vis_data) == 4:
        attn_weights3, attn_weights2, attn_weights1, attn_weights0 = vis_data
        train_mask = train_mask = data['train_masks'][0, batch_element, :, :].cpu().numpy().astype(np.float32)
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
    # print(test_dist.shape)

    # softmax on the mask prediction (since this is done internaly when calculating loss)
    mask = F.softmax(pred_mask, dim=1)[batch_element, 0, :, :].cpu().detach().numpy().astype(np.float32)
    predicted_mask = (mask > 0.5).astype(np.float32) * mask

    mu = torch.Tensor(data['settings'].normalize_mean).to(torch.device('cuda')).view(1, 1, 3)
    std = torch.Tensor(data['settings'].normalize_std).to(torch.device('cuda')).view(1, 1, 3)

    train_img = 255 * (train_img * std + mu)
    test_img = 255 * (test_img * std + mu)

    train_img = (train_img.cpu().numpy()).astype(np.uint8)
    train_depth = (train_depth.cpu().numpy().squeeze()).astype(np.float32)
    test_img = (test_img.cpu().numpy()).astype(np.uint8)
    test_depth = (test_depth.cpu().numpy().squeeze()).astype(np.float32)
    train_mask = (train_mask.cpu().numpy()).astype(np.float32)
    test_mask = (test_mask.cpu().numpy()).astype(np.float32)
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

    elif len(vis_data) == 4:
        draw_axis(ax9, attn_weights3, 'attn_weights3', show_minmax=True)
        draw_axis(ax10, attn_weights2, 'attn_weights2', show_minmax=True)
        draw_axis(ax11, attn_weights1, 'attn_weights1', show_minmax=True)
        draw_axis(ax12, attn_weights0, 'attn_weights0', show_minmax=True)


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

        if self.target_size:
            # Run network to obtain IoU prediction for each proposal in 'test_proposals'
            masks_pred, size_pred, vis_data = self.net(data['train_images'].permute(1, 0, 2, 3), # batch*3*384*384
                                                       data['train_depths'].permute(1, 0, 2, 3), # batch*1*384*384
                                                       data['test_images'].permute(1, 0, 2, 3),
                                                       data['test_depths'].permute(1, 0, 2, 3),
                                                       data['train_masks'].permute(1, 0, 2, 3),
                                                       test_dist=test_dist,
                                                       debug=True) # Song :  vis pos and neg
        else:
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
            # loss = torch.tensor(0.5)

        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss.item(),
                 'Loss/size': 0}

        if self.target_size:
            # train_masks = data['train_masks'].permute(1, 0, 2, 3)

            img_sz = masks_gt.shape[-1] * masks_gt.shape[-2] * 1.0 # H * W
            # train_sz = torch.sum(train_masks.view(train_masks.shape[0], -1), 1).unsqueeze(-1) # [B, 1]
            test_sz = torch.sum(masks_gt.view(masks_gt.shape[0], -1), 1).unsqueeze(-1) # [B, 1]

            loss_sz = self.target_sz_objective(test_sz/img_sz, size_pred/img_sz)

            loss = loss + 10 * loss_sz
            stats['Loss/total'] = loss.item()
            stats['Loss/size'] = loss_sz.item()


        if 'iter' in data and (data['iter'] - 1) % 50 == 0:
            save_debug(data,masks_pred, vis_data) # vis_data = (p_rgb, p_d) or  (pred_sm_d, attn_weights2, attn_weights1, attn_weights0)

        return loss, stats
