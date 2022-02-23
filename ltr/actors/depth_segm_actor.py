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

def process_attn_maps(att_mat):
    att_mat = torch.stack(att_mat).squeeze(1)
    print(att_mat.shape)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    print(att_mat.shape)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    print(residual_att.shape)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size*2, grid_size//2).detach().numpy()
    # mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    # result = (mask * im).astype("uint8")

    return mask

def save_debug(data, pred_mask, vis_data):

    batch_element = 0
    vis_cosine_similarity = True

    if len(vis_data) == 2:
        p_rgb, p_d = vis_data

        p_rgb = p_rgb[batch_element, ...] # [H, W, 2] F + B
        p_d = p_d[batch_element, ...]
        p_rgb = (p_rgb.detach().cpu().numpy().squeeze()).astype(np.float32) # [H, W, 2]
        p_d = (p_d.detach().cpu().numpy().squeeze()).astype(np.float32)

    elif len(vis_data) == 4:
        attn_weights3, attn_weights2, attn_weights1, attn_weights0 = vis_data
        # print(len(attn_weights3), attn_weights3[0].shape) # [B, 3, 144, 144]
        p_rgb = process_attn_maps(attn_weights3)
        p_d = process_attn_maps(attn_weights2)

    elif len(vis_data) == 5:
        p_rgb, p_d, attn_weights2, attn_weights1, attn_weights0 = vis_data


    dir_path = data['settings'].env.images_dir

    train_img = data['train_images'][:, batch_element, :, :].permute(1, 2, 0)
    train_depth = data['train_depths'][:, batch_element, :, :].permute(1, 2, 0)
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
    test_mask = (test_mask.cpu().numpy()).astype(np.float32)
    # predicted_mask = mask.astype(np.float32)

    # Song
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(8, 8))
    draw_axis(ax1, train_img, 'Train image')
    draw_axis(ax4, test_img, 'Test image')
    draw_axis(ax2, train_depth, 'Train depth')
    draw_axis(ax5, test_depth, 'Test depth')
    draw_axis(ax3, test_mask, 'Ground-truth')
    draw_axis(ax6, predicted_mask, 'Prediction', show_minmax=True)

    if vis_cosine_similarity:


        # empty_channel = np.zeros((p_rgb.shape[0], p_rgb.shape[1], 1), dtype=np.uint8)

        # p_rgb = (p_rgb * 255).astype(np.uint8)
        # p_rgb = np.concatenate((p_rgb, empty_channel), axis=-1) # [H, W, 3]

        # p_d = (p_d * 255).astype(np.uint8)
        # p_d = np.concatenate((p_d, empty_channel), axis=-1) # [H, W, 3]

        draw_axis(ax7, p_rgb, 'similarity rgb')
        draw_axis(ax8, p_d, 'similarity d')

    test_dist = (test_dist.detach().cpu().numpy().squeeze()).astype(np.float32)
    draw_axis(ax9, test_dist, 'test_dist')

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

        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        masks_pred, vis_data = self.net(data['train_images'].permute(1, 0, 2, 3), # batch*3*384*384
                                        data['train_depths'].permute(1, 0, 2, 3), # batch*1*384*384
                                        data['test_images'].permute(1, 0, 2, 3),
                                        data['test_depths'].permute(1, 0, 2, 3),
                                        data['train_masks'].permute(1, 0, 2, 3),
                                        test_dist=test_dist,
                                        debug=True) # Song :  vis pos and neg maps

        masks_gt = data['test_masks'].permute(1, 0, 2, 3)            # B * 1 * H * W
        masks_gt_pair = torch.cat((masks_gt, 1 - masks_gt), dim=1)   # B * 2 * H * W
        # Compute loss
        loss = self.objective(masks_pred, masks_gt_pair)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss.item()}

        if 'iter' in data and (data['iter'] - 1) % 50 == 0:
            save_debug(data,masks_pred, vis_data) # vis_data = (p_rgb, p_d) or  (pred_sm_d, attn_weights2, attn_weights1, attn_weights0)

        return loss, stats
