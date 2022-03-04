from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import torch.nn
import math
import time
import numpy as np
import cv2
import copy
from pytracking import dcf, fourier, TensorList, operation
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from pytracking.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2
from .optim import ConvProblem, FactorizedConvProblem
from pytracking.features import augmentation
import ltr.data.processing_utils as prutils
from ltr import load_network

from ltr.external.depthconv.functions import depth_conv

from pytracking.bbox_fit import fit_bbox_to_mask
from pytracking.mask_to_disk import save_mask


class DepthSegmST(BaseTracker):
    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.features_filter.initialize()
        self.features_initialized = True

    def initialize(self, image, state, init_mask=None, *args, **kwargs):
        # Initialize some stuff
        self.frame_num = 1
        self.frame_name = '%08d' % self.frame_num

        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize features for ResNet50
        self.initialize_features()

        # Check if image is color
        color, depth = image['color'], image['depth']

        self.params.features_filter.set_is_color(color.shape[2] == 3)

        # Get feature specific params
        self.fparams = self.params.features_filter.get_fparams('feature_params')

        self.time = 0
        tic = time.time()

        self.rotated_bbox = True

        if len(state) == 8:
            self.gt_poly = np.array(state)
            x_ = np.array(state[::2])
            y_ = np.array(state[1::2])

            self.pos = torch.Tensor([np.mean(y_), np.mean(x_)])
            # self.target_sz = torch.Tensor([np.max(y_) - np.min(y_), np.max(x_) - np.min(x_)])

            # overwrite state - needed for segmentation
            if self.params.vot_anno_conversion_type == 'preserve_area':
                state = self.poly_to_aabbox(x_, y_)
            else:
                state = np.array([np.min(x_), np.min(y_), np.max(x_) - np.min(x_), np.max(y_) - np.min(y_)])

            self.target_sz = torch.Tensor([state[3], state[2]])

            if init_mask is not None:
                self.rotated_bbox = False

        elif len(state) == 4:
            state[0] -= 1
            state[1] -= 1
            # Get position and size
            self.pos = torch.Tensor([state[1] + state[3] / 2, state[0] + state[2] / 2])
            self.pos_prev = [state[1] + state[3] / 2, state[0] + state[2] / 2]
            self.target_sz = torch.Tensor([state[3], state[2]])
            self.gt_poly = np.array([state[0], state[1],
                                     state[0] + state[2] - 1, state[1],
                                     state[0] + state[2] - 1, state[1] + state[3] - 1,
                                     state[0], state[1] + state[3] - 1])

            self.rotated_bbox = False

        # Set search area
        self.target_scale = 1.0
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        if search_area > self.params.max_image_sample_size:
            self.target_scale = math.sqrt(search_area / self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale = math.sqrt(search_area / self.params.min_image_sample_size)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features_filter.stride())
        if getattr(self.params, 'search_area_shape', 'square') == 'square':
            self.img_sample_sz = torch.round(
                torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
        elif self.params.search_area_shape == 'initrect':
            self.img_sample_sz = torch.round(self.base_target_sz * self.params.search_area_scale)
        else:
            raise ValueError('Unknown search area shape')
        if self.params.feature_size_odd:
            self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)
        else:
            self.img_sample_sz += feat_max_stride - (self.img_sample_sz + feat_max_stride) % (2 * feat_max_stride)

        # Set sizes
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features_filter.size(self.img_sample_sz)
        self.output_sz = self.params.score_upsample_factor * self.img_support_sz  # Interpolated size of the output
        self.kernel_size = self.fparams.attribute('kernel_size')

        # Optimization options
        self.params.precond_learning_rate = self.fparams.attribute('learning_rate')
        if self.params.CG_forgetting_rate is None or max(self.params.precond_learning_rate) >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - max(
                self.params.precond_learning_rate)) ** self.params.CG_forgetting_rate

        self.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(),
                                                        self.output_sz.long() * self.params.effective_search_area / self.params.search_area_scale,
                                                        centered=False).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=False).to(self.params.device)

        # Initialize some learning things
        self.init_learning()

        # Convert image
        im, dp = numpy_to_torch(color), numpy_to_torch(depth)
        self.im, self.dp, self.mask, self.score_map = im, dp, None, None  # For debugging only

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        x_rgb = self.generate_init_samples(im) # crops and augments and Generated ResNet50 feature maps for DCF

        # Initialize projection matrix
        self.init_projection_matrix(x_rgb)

        # Transform to get the training sample
        train_x_rgb = self.preprocess_sample(x_rgb) # x_rgb * self.feature_window

        # Generate label function
        init_y = self.init_label_function(train_x_rgb) # Gaussian map

        # Init memory
        self.init_memory(train_x_rgb) # No need for depth

        # Init optimizer and do initial optimization for DCF
        self.init_optimization(train_x_rgb, init_y) # Song, but Depth is not used for filter optimize

        if self.params.use_segmentation:
            self.init_segmentation(color, depth, state, init_mask=init_mask)

        # array of scores
        self.scores = np.array([1])

        toc_ = time.time() - tic
        self.time += toc_

        self.rgb_patches = None
        self.d_patches = None
        # vis the crop image patches or search region
        self.vis_search_center = None
        self.vis_search_scales = None
        self.vis_serach_size = None

    def init_optimization(self, train_x, init_y):
        # Initialize filter for DCF RGB based
        filter_init_method = getattr(self.params, 'filter_init_method', 'zeros')
        self.filter = TensorList(
            [x.new_zeros(1, cdim, sz[0], sz[1]) for x, cdim, sz in zip(train_x, self.compressed_dim, self.kernel_size)])
        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'randn':
            for f in self.filter:
                f.normal_(0, 1 / f.numel())
        else:
            raise ValueError('Unknown "filter_init_method"')

        # Get parameters
        self.params.update_projection_matrix = getattr(self.params, 'update_projection_matrix',
                                                       True) and self.params.use_projection_matrix
        optimizer = getattr(self.params, 'optimizer', 'GaussNewtonCG')

        # Setup factorized joint optimization
        if self.params.update_projection_matrix:
            self.joint_problem = FactorizedConvProblem(self.init_training_samples, init_y, self.filter_reg,
                                                       self.fparams.attribute('projection_reg'), self.params,
                                                       self.init_sample_weights,
                                                       self.projection_activation, self.response_activation)

            # Variable containing both filter and projection matrix
            joint_var = self.filter.concat(self.projection_matrix)

            # Initialize optimizer
            analyze_convergence = getattr(self.params, 'analyze_convergence', False)
            if optimizer == 'GaussNewtonCG':
                self.joint_optimizer = GaussNewtonCG(self.joint_problem, joint_var, plotting=(self.params.debug == 3), # SY >= 3 -> == 3
                                                     analyze=analyze_convergence, fig_num=(12, 13, 14))
            elif optimizer == 'GradientDescentL2':
                self.joint_optimizer = GradientDescentL2(self.joint_problem, joint_var,
                                                         self.params.optimizer_step_length,
                                                         self.params.optimizer_momentum,
                                                         plotting=(self.params.debug == 3), debug=analyze_convergence, # SY >= 3 -> == 3
                                                         fig_num=(12, 13))

            # Do joint optimization
            if isinstance(self.params.init_CG_iter, (list, tuple)):
                self.joint_optimizer.run(self.params.init_CG_iter)
            else:
                self.joint_optimizer.run(self.params.init_CG_iter // self.params.init_GN_iter, self.params.init_GN_iter)

            if analyze_convergence:
                opt_name = 'CG' if getattr(self.params, 'CG_optimizer', True) else 'GD'
                for val_name, values in zip(['loss', 'gradient'],
                                            [self.joint_optimizer.losses, self.joint_optimizer.gradient_mags]):
                    val_str = ' '.join(['{:.8e}'.format(v.item()) for v in values])
                    file_name = '{}_{}.txt'.format(opt_name, val_name)
                    with open(file_name, 'a') as f:
                        f.write(val_str + '\n')
                raise RuntimeError('Exiting')

        # Re-project samples with the new projection matrix
        compressed_samples = self.project_sample(self.init_training_samples, self.projection_matrix)
        for train_samp, init_samp in zip(self.training_samples, compressed_samples):
            train_samp[:init_samp.shape[0], ...] = init_samp

        self.hinge_mask = None

        # Initialize optimizer
        self.conv_problem = ConvProblem(self.training_samples, self.y, self.filter_reg, self.sample_weights,
                                        self.response_activation)

        if optimizer == 'GaussNewtonCG':
            self.filter_optimizer = ConjugateGradient(self.conv_problem, self.filter,
                                                      fletcher_reeves=self.params.fletcher_reeves,
                                                      direction_forget_factor=self.params.direction_forget_factor,
                                                      debug=(self.params.debug == 3), fig_num=(12, 13))
        elif optimizer == 'GradientDescentL2':
            self.filter_optimizer = GradientDescentL2(self.conv_problem, self.filter, self.params.optimizer_step_length,
                                                      self.params.optimizer_momentum, debug=(self.params.debug == 3),
                                                      fig_num=12)

        # Transfer losses from previous optimization
        if self.params.update_projection_matrix:
            self.filter_optimizer.residuals = self.joint_optimizer.residuals
            self.filter_optimizer.losses = self.joint_optimizer.losses

        if not self.params.update_projection_matrix:
            self.filter_optimizer.run(self.params.init_CG_iter)

        # Post optimization
        self.filter_optimizer.run(self.params.post_init_CG_iter)

        # Free memory
        del self.init_training_samples
        if self.params.use_projection_matrix:
            del self.joint_problem, self.joint_optimizer

    def track(self, image):
        self.frame_num += 1
        self.frame_name = '%08d' % self.frame_num

        self.pos_prev = [copy.copy(self.pos[0].item()), copy.copy(self.pos[1].item())]

        # Convert image
        color, depth = image['color'], image['depth']
        im, dp = numpy_to_torch(color), numpy_to_torch(depth)
        self.im, self.dp = im, dp  # For debugging only

        # ------- LOCALIZATION ------- #
        # Get sample
        sample_pos = copy.deepcopy(self.pos)
        sample_scales = self.target_scale * self.params.scale_factors
        test_x_rgb, test_x_d = self.extract_processed_sample(im, dp, sample_pos, sample_scales, self.img_sample_sz)

        ''' Song, use DAL instead of normal DCF'''
        # Compute scores
        if not self.params.use_dal:
            scores_raw = self.apply_filter(test_x_rgb)
        else:
            scores_raw = self.classify_target(test_x_rgb, test_x_d)


        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw)
        new_pos = sample_pos + translation_vec

        # Localization uncertainty
        max_score = torch.max(s).item()
        uncert_score = 0
        if self.frame_num > 5:
            uncert_score = np.mean(self.scores) / max_score

        self.uncert_score = uncert_score

        if uncert_score < self.params.tracking_uncertainty_thr:
            self.scores = np.append(self.scores, max_score)
            if self.scores.size > self.params.response_budget_sz:
                self.scores = np.delete(self.scores, 0)

        if flag == 'not_found':
            uncert_score = 100

        # Update position and scale
        # [AL] Modification
        # if flag != 'not_found':
        if uncert_score < self.params.tracking_uncertainty_thr:
            if getattr(self.params, 'use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])

        # if self.params.debug >= 2:
        if self.params.debug == 2:
            show_tensor(s[scale_ind, ...], 5, title='Max score = {:.2f}'.format(torch.max(s[scale_ind, ...]).item()))

        # Song
        self.score_map = s[scale_ind, ...].squeeze().cpu().detach().numpy() # if self.params.debug == 5 else None

        # just a sanity check so that it does not get out of image
        if new_pos[0] < 0:
            new_pos[0] = 0
        if new_pos[1] < 0:
            new_pos[1] = 0
        if new_pos[0] >= color.shape[0]:
            new_pos[0] = color.shape[0] - 1
        if new_pos[1] >= color.shape[1]:
            new_pos[1] = color.shape[1] - 1

        pred_segm_region = None
        if self.segmentation_task or (
            self.params.use_segmentation and uncert_score < self.params.uncertainty_segment_thr):
            pred_segm_region = self.segment_target(color, depth, new_pos, self.target_sz)
            # Song , do we need to consider about the mask sz? pixels in the pred_segm_region ?
            if pred_segm_region is None:
                print('segment_target is None, use DCF results...')
                self.pos = new_pos.clone()
        else:
            self.pos = new_pos.clone()

        # ------- UPDATE ------- #

        # Check flags and set learning rate if hard negative
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.hard_negative_learning_rate if hard_negative else None

        # [AL] Modification
        # if update_flag:
        if update_flag and uncert_score < self.params.tracking_uncertainty_thr:
            # Get train sample
            train_x_rgb = TensorList([x[scale_ind:scale_ind + 1, ...] for x in test_x_rgb])

            # Create label for sample
            train_y = self.get_label_function(sample_pos, sample_scales[scale_ind])

            # Update memory
            self.update_memory(train_x_rgb, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.filter_optimizer.run(self.params.hard_negative_CG_iter)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            self.filter_optimizer.run(self.params.CG_iter)

        if self.params.use_segmentation:
            if pred_segm_region is not None:
                # [x, y, w, h]
                # just a sanity check so that it does not get out of image
                if pred_segm_region[0] < 0:
                    pred_segm_region[0] = 0
                if pred_segm_region[1] < 0:
                    pred_segm_region[1] = 0
                if pred_segm_region[0] + pred_segm_region[2] >= color.shape[1]:
                    pred_segm_region[2] = color.shape[1] - pred_segm_region[0]
                if pred_segm_region[1] + pred_segm_region[3] >= color.shape[0]:
                    pred_segm_region[3] = color.shape[0] - pred_segm_region[1]

                return pred_segm_region, self.score_map.max()

        # Return new state
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))
        # [x, y, w, h]
        # just a sanity check so that it does not get out of image
        if new_state[0] < 0:
            new_state[0] = 0
        if new_state[1] < 0:
            new_state[1] = 0
        if new_state[0] + new_state[2] >= color.shape[1]:
            new_state[2] = color.shape[1] - new_state[0]
        if new_state[1] + new_state[3] >= color.shape[0]:
            new_state[3] = color.shape[0] - new_state[1]

        return new_state.tolist(), self.score_map.max()


    def classify_target(self, sample_x, sample_d):
        if sample_d is not None:
            alpha = 0.1
            scores = self.apply_filter_depthaware(sample_x[0], self.filter[0], sample_d, alpha)
        else:
            scores = self.apply_filter(sample_x)
        return scores

    def apply_filter(self, sample_x: TensorList):
        return operation.conv2d(sample_x, self.filter, mode='same')

    def apply_filter_depthaware(self, feat, filter, depth, alpha):
        '''DAL depth-aware DCF'''
        # for i in range(len(feat)):
        if feat.shape[2]!=depth.shape[2] or feat.shape[2]!=depth.shape[2]:
            depth=F.upsample(depth, size=(feat.shape[2],feat.shape[3]),mode='bilinear')

        multiple_filters = (feat.dim() == 5)
        padding = (filter.shape[-2] // 2, filter.shape[-1] // 2)
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        size_scoreoutput = int(feat.shape[-2] + 2 * padding[0] - filter.shape[-2] / 1 + 1)

        if multiple_filters:
            scores=torch.zeros(num_images, num_sequences,1,size_scoreoutput,size_scoreoutput).to(feat.device)
            for seq in range(num_sequences):
                feat_seq  =feat[:,seq,:,:,:]
                filter_seq=filter[:,seq,:,:,:]
                depth_seq =depth[:,seq,:,:,:]
                depth_seq =alpha*depth_seq.to(feat.device)

                scores_seq=depth_conv(feat_seq,depth_seq,filter_seq,None,stride=1,padding=padding,dilation=1)

                scores[:,seq,:, :,:]=scores_seq
            return scores.view(num_images, num_sequences, scores.shape[-2], scores.shape[-1])

        depth =alpha*depth
        depth =depth.to(feat.device)
        scores=depth_conv(feat,depth,filter,None,stride=1,padding=padding,dilation=1)
        return scores.view(num_images, num_sequences, scores.shape[-2], scores.shape[-1])

    def localize_target(self, scores_raw):
        # Weighted sum (if multiple features) with interpolation in fourier domain
        weight = self.fparams.attribute('translation_weight', 1.0)
        scores_raw = weight * scores_raw
        sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) * scores_raw.size(3))
        for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
            sf_weighted[i] = fourier.shift_fs(sf_weighted[i],
                                              math.pi * (1 - torch.Tensor([ksz[0] % 2, ksz[1] % 2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, self.output_sz)

        if self.output_window is not None and not getattr(self.params, 'perform_hn_without_windowing', False):
            scores *= self.output_window

        if getattr(self.params, 'advanced_localization', False):
            return self.localize_advanced(scores)

        # Get maximum
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp.float().cpu()

        # Convert to displacements in the base scale
        disp = (max_disp + self.output_sz / 2) % self.output_sz - self.output_sz / 2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind, ...].view(-1) * (self.img_support_sz / self.output_sz) * self.target_scale
        translation_vec *= self.params.scale_factors[scale_ind]

        # Shift the score output for visualization purposes
        if self.params.debug >= 2:
            sz = scores.shape[-2:]
            scores = torch.cat([scores[..., sz[0] // 2:, :], scores[..., :sz[0] // 2, :]], -2)
            scores = torch.cat([scores[..., :, sz[1] // 2:], scores[..., :, :sz[1] // 2]], -1)

        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores):
        """Does the advanced localization with hard negative detection and target not found."""

        sz = scores.shape[-2:]

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores_orig = scores.clone()

            scores_orig = torch.cat([scores_orig[..., (sz[0] + 1) // 2:, :], scores_orig[..., :(sz[0] + 1) // 2, :]],
                                    -2)
            scores_orig = torch.cat([scores_orig[..., :, (sz[1] + 1) // 2:], scores_orig[..., :, :(sz[1] + 1) // 2]],
                                    -1)

            scores *= self.output_window

        # Shift scores back
        scores = torch.cat([scores[..., (sz[0] + 1) // 2:, :], scores[..., :(sz[0] + 1) // 2, :]], -2)
        scores = torch.cat([scores[..., :, (sz[1] + 1) // 2:], scores[..., :, :(sz[1] + 1) // 2]], -1)

        # Find maximum
        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - self.output_sz // 2
        translation_vec1 = target_disp1 * (self.img_support_sz / self.output_sz) * self.target_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'not_found'

        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores = scores_orig

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * self.target_sz / self.target_scale
        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - self.output_sz // 2
        translation_vec2 = target_disp2 * (self.img_support_sz / self.output_sz) * self.target_scale

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1 ** 2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2 ** 2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores, 'hard_negative'

        return translation_vec1, scale_ind, scores, None

    def extract_sample(self, color: torch.Tensor, depth: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        return self.params.features_filter.extract(color, pos, scales, sz, dp=depth)

    def extract_processed_sample(self, color: torch.Tensor, depth: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> (
    TensorList, TensorList):
        x_rgb, d_crops, rgb_patches = self.extract_sample(color, depth, pos, scales, sz)
        self.rgb_patches = rgb_patches.clone().detach().cpu().numpy().squeeze()
        self.rgb_patches = np.swapaxes(np.swapaxes(self.rgb_patches, 0, 1), 1, 2).astype(int)
        self.d_patches = d_crops.clone().detach().cpu().numpy().squeeze()
        return self.preprocess_sample(self.project_sample(x_rgb)), d_crops

    def preprocess_sample(self, x: TensorList) -> (TensorList, TensorList):
        if getattr(self.params, '_feature_window', False):
            x = x * self.feature_window
        return x

    def project_sample(self, x: TensorList, proj_matrix=None):
        # Apply projection matrix
        if proj_matrix is None:
            proj_matrix = self.projection_matrix
        return operation.conv2d(x, proj_matrix).apply(self.projection_activation)

    def init_learning(self):
        # Get window function
        self.feature_window = TensorList([dcf.hann2d(sz).to(self.params.device) for sz in self.feature_sz])

        # Filter regularization
        self.filter_reg = self.fparams.attribute('filter_reg')

        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = getattr(self.params, 'projection_activation', 'none')
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.projection_activation = torch.nn.ReLU(inplace=True)
        elif projection_activation == 'elu':
            self.projection_activation = torch.nn.ELU(inplace=True)
        elif projection_activation == 'mlu':
            self.projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = getattr(self.params, 'response_activation', 'none')
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.response_activation = lambda x: x
        elif response_activation == 'relu':
            self.response_activation = torch.nn.ReLU(inplace=True)
        elif response_activation == 'elu':
            self.response_activation = torch.nn.ELU(inplace=True)
        elif response_activation == 'mlu':
            self.response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Generate augmented initial samples."""

        # Compute augmentation size
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift operator
        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor).long().tolist()

        # Create transofmations
        self.transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.params.augmentation:
            self.transforms.extend(
                [augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation['shift']])
        if 'relativeshift' in self.params.augmentation:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz / 2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in
                                    self.params.augmentation['relativeshift']])
        if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.params.augmentation:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in
                                    self.params.augmentation['blur']])
        if 'scale' in self.params.augmentation:
            self.transforms.extend(
                [augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in
                 self.params.augmentation['scale']])
        if 'rotate' in self.params.augmentation:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in
                                    self.params.augmentation['rotate']])

        init_samples = self.params.features_filter.extract_transformed(im, self.pos.round(), self.target_scale,
                                                                       aug_expansion_sz, self.transforms)


        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        # Add dropout samples
        if 'dropout' in self.params.augmentation:
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1] * num)
            for i, use_aug in enumerate(self.fparams.attribute('use_augmentation')):
                if use_aug:
                    init_samples[i] = torch.cat([init_samples[i],
                                                 F.dropout2d(init_samples[i][0:1, ...].expand(num, -1, -1, -1), p=prob,
                                                             training=True)])

        return init_samples

    def init_projection_matrix(self, x):
        # Set if using projection matrix
        self.params.use_projection_matrix = getattr(self.params, 'use_projection_matrix', True)

        if self.params.use_projection_matrix:
            self.compressed_dim = self.fparams.attribute('compressed_dim', None)

            proj_init_method = getattr(self.params, 'proj_init_method', 'pca')
            if proj_init_method == 'pca':
                x_mat = TensorList([e.permute(1, 0, 2, 3).reshape(e.shape[1], -1).clone() for e in x])
                x_mat -= x_mat.mean(dim=1, keepdim=True)
                cov_x = x_mat @ x_mat.t()
                self.projection_matrix = TensorList(
                    [None if cdim is None else torch.svd(C)[0][:, :cdim].t().unsqueeze(-1).unsqueeze(-1).clone() for
                     C, cdim in
                     zip(cov_x, self.compressed_dim)])
            elif proj_init_method == 'randn':
                self.projection_matrix = TensorList(
                    [None if cdim is None else ex.new_zeros(cdim, ex.shape[1], 1, 1).normal_(0,
                                                                                             1 / math.sqrt(ex.shape[1]))
                     for ex, cdim in
                     zip(x, self.compressed_dim)])
        else:
            self.compressed_dim = x.size(1)
            self.projection_matrix = TensorList([None] * len(x))

    def init_label_function(self, train_x):
        # Allocate label function
        self.y = TensorList([x.new_zeros(self.params.sample_memory_size, 1, x.shape[2], x.shape[3]) for x in train_x])

        # Output sigma factor
        output_sigma_factor = self.fparams.attribute('output_sigma_factor')
        self.sigma = (
                     self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(
            2)

        # Center pos in normalized coords
        target_center_norm = (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)

        # Generate label functions
        for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz, self.kernel_size, train_x):
            center_pos = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * sz
                y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center)

        # Return only the ones to use for initial training
        return TensorList([y[:x.shape[0], ...] for y, x in zip(self.y, train_x)])

    def init_memory(self, train_x):
        # Initialize first-frame training samples
        self.num_init_samples = train_x.size(0)
        self.init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        self.init_training_samples = train_x

        # Sample counters and weights
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, self.init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, cdim, x.shape[2], x.shape[3]) for x, cdim in
             zip(train_x, self.compressed_dim)])

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, learning_rate=None):
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind,
                                                 self.num_stored_samples, self.num_init_samples, self.fparams,
                                                 learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y
        if self.hinge_mask is not None:
            for m, y, ind in zip(self.hinge_mask, sample_y, replace_ind):
                m[ind:ind + 1, ...] = (y >= self.params.hinge_threshold).float()
        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, fparams,
                              learning_rate=None):
        # Update weights and get index to replace in memory
        replace_ind = []
        for sw, prev_ind, num_samp, num_init, fpar in zip(sample_weights, previous_replace_ind, num_stored_samples,
                                                          num_init_samples, fparams):
            lr = learning_rate
            if lr is None:
                lr = fpar.learning_rate

            init_samp_weight = getattr(fpar, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, sample_pos, sample_scale):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (self.pos - sample_pos) / (sample_scale * self.img_support_sz)
        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            center = sz * target_center_norm + 0.5 * torch.Tensor([(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            train_y.append(dcf.label_function_spatial(sz, sig, center))
        return train_y

    def update_state(self, new_pos, new_scale=None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = 0.2
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def create_dist(self, width, height, cx=None, cy=None):

        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2

        x_ = np.linspace(1, width, width) - cx
        y_ = np.linspace(1, width, width) - cy
        X, Y = np.meshgrid(x_, y_)

        return np.sqrt(np.square(X) + np.square(Y)).astype(np.float32)

    def create_dist_gauss(self, map_sz, w, h, cx=None, cy=None, p=4, sz_weight=0.7):
        # create a square-shaped distance map with a Gaussian function which can be interpreted as a distance
        # to the given bounding box (center [cx, cy], width w, height h)
        # p is power of a Gaussian function
        # sz_weight is a weight of a bounding box size in Gaussian denominator
        if cx is None:
            cx = map_sz / 2
        if cy is None:
            cy = map_sz / 2

        x_ = np.linspace(1, map_sz, map_sz) - 1 - cx
        y_ = np.linspace(1, map_sz, map_sz) - 1 - cy
        X, Y = np.meshgrid(x_, y_)
        # 1 - is needed since we need distance-like map (not Gaussian function)
        return 1 - np.exp(-((np.power(X, p) / (sz_weight * w ** p)) + (np.power(Y, p) / (sz_weight * h ** p))))

    def init_segmentation(self, color, depth, bb, init_mask=None):
        '''
        Song init mask sometimes is not correct
        '''
        init_patch_crop_rgb, f_ = prutils.sample_target(color, np.array(bb), self.params.segm_search_area_factor,
                                                    output_sz=self.params.segm_output_sz)

        init_patch_crop_d, _ = prutils.sample_target(depth, np.array(bb), self.params.segm_search_area_factor,
                                                    output_sz=self.params.segm_output_sz)
        init_patch_crop_d = np.expand_dims(init_patch_crop_d, axis=-1)

        self.segmentation_task = False
        if init_mask is not None:
            mask = copy.deepcopy(init_mask).astype(np.float32)
            self.segmentation_task = True
            # self.params.segm_optimize_polygon = True
            # segmentation videos are shorter - therefore larger scale change factor can be used
            self.params.min_scale_change_factor = 0.9
            self.params.max_scale_change_factor = 1.1
            self.params.segm_mask_thr = 0.2
        else:
            mask = np.zeros((color.shape[0], color.shape[1]), dtype=np.int32)
            if hasattr(self, 'gt_poly'):
                p1 = self.gt_poly[:2]
                p2 = self.gt_poly[2:4]
                p3 = self.gt_poly[4:6]
                p4 = self.gt_poly[6:]
                cv2.fillConvexPoly(mask, np.array([p1, p2, p3, p4], dtype=np.int32), 1)
                mask = mask.astype(np.float32)
            else:
                p1 = bb[:2]
                p2 = [bb[0] + bb[2], bb[1]]
                p3 = [bb[0] + bb[2], bb[1] + bb[3]]
                p4 = [bb[0], bb[1] + bb[3]]
                cv2.fillConvexPoly(mask, np.array([p1, p2, p3, p4], dtype=np.int32), 1)
                mask = mask.astype(np.float32)

        init_mask_patch_np, patch_factor_init = prutils.sample_target(mask, np.array(bb),
                                                                      self.params.segm_search_area_factor,
                                                                      output_sz=self.params.segm_output_sz, pad_val=0)

        # network was renamed therefore we need to specify constructor_module and constructor_fun_name
        segm_net, _ = load_network(self.params.segm_net_path, backbone_pretrained=False,
                                   constructor_module=self.params.constructor_module,
                                   constructor_fun_name=self.params.constructor_fun_name) #
                                   # constructor_module='ltr.models.depth_segm.depth_segm',
                                   # constructor_fun_name='depth_segm_attention1_resnet50') #

        if self.params.use_gpu:
            segm_net.cuda()
        segm_net.eval()

        for p in segm_net.segm_predictor.parameters():
            p.requires_grad = False

        self.params.segm_normalize_mean = np.array(self.params.segm_normalize_mean).reshape((1, 1, 3))
        self.params.segm_normalize_std = np.array(self.params.segm_normalize_std).reshape((1, 1, 3))

        # normalize input image RGB
        init_patch_norm_rgb = init_patch_crop_rgb.astype(np.float32) / float(255)
        init_patch_norm_rgb -= self.params.segm_normalize_mean
        init_patch_norm_rgb /= self.params.segm_normalize_std

        # Depth is already normalized when basetracker._read_image()
        init_patch_norm_d = init_patch_crop_d.astype(np.float32)

        # create distance map for discriminative segmentation
        if self.params.segm_use_dist:
            if self.params.segm_dist_map_type == 'center':
                # center-based dist map
                dist_map = self.create_dist(init_patch_crop_rgb.shape[0], init_patch_crop_rgb.shape[1])
            elif self.params.segm_dist_map_type == 'bbox':
                # bbox-based dist map
                dist_map = self.create_dist_gauss(self.params.segm_output_sz, bb[2] * patch_factor_init,
                                                  bb[3] * patch_factor_init)
            else:
                print('Error: Unknown distance map type.')
                exit(-1)

            dist_map = torch.Tensor(dist_map)

        # put image patch and mask to GPU
        init_patch_rgb = torch.Tensor(init_patch_norm_rgb)
        init_patch_d = torch.Tensor(init_patch_norm_d)
        init_mask_patch = torch.Tensor(init_mask_patch_np)
        if self.params.use_gpu:
            init_patch_rgb = init_patch_rgb.to(self.params.device)
            init_patch_d = init_patch_d.to(self.params.device)
            init_mask_patch = init_mask_patch.to(self.params.device)
            if self.params.segm_use_dist:
                dist_map = dist_map.to(self.params.device)
                dist_map = torch.unsqueeze(torch.unsqueeze(dist_map, dim=0), dim=0)
                test_dist_map = [dist_map]
            else:
                test_dist_map = None

        # reshape image for the feature extractor
        init_patch_rgb = torch.unsqueeze(init_patch_rgb, dim=0).permute(0, 3, 1, 2)
        init_patch_d = torch.unsqueeze(init_patch_d, dim=0).permute(0, 3, 1, 2)
        init_mask_patch = torch.unsqueeze(torch.unsqueeze(init_mask_patch, dim=0), dim=0) # [1, 1, 384,384]

        # extract features (extracting twice on the same patch - not necessary)
        train_feat_rgb = segm_net.extract_backbone_features(init_patch_rgb)

        # prepare features in the list (format for the network)
        train_feat_segm_rgb = [feat for feat in train_feat_rgb.values()]
        test_feat_segm_rgb = [feat for feat in train_feat_rgb.values()]
        train_masks = [init_mask_patch]

        # Song : extract depth features
        train_feat_segm_d = segm_net.segm_predictor.depth_feat_extractor(init_patch_d)

        ''' init_mask is 384*384 after segm_predictor
        Sometimes, init mask is not good
        Can we use K-Cluster to replace init_segmentation???

            rgb features   : [conv1, layer1, layer2, layer3], Bx64x192x192 -> Bx256x96x96 -> Bx512x48x48 -> Bx1024x24x24
            depth features : [feat0, feat1, feat2, feat3],    Bx4x192x192  -> Bx16x96x96  -> Bx32x48x48  -> Bx64x24x24
        '''
        # if init_mask is None:
        #     test_feat_segm_rgb
        if init_mask is None:
            iters = 0
            while iters < 1:
                # Obtain segmentation prediction
                # segm_pred = segm_net.segm_predictor(test_feat_segm, train_feat_segm, train_masks, test_dist_map)
                segm_pred = segm_net.segm_predictor(test_feat_segm_rgb, train_feat_segm_d,
                                                    train_feat_segm_rgb, train_feat_segm_d, # if we use the feature correlation
                                                    train_masks, test_dist_map)

                # softmax on the prediction (during training this is done internaly when calculating loss)
                # take only the positive channel as predicted segmentation mask
                mask = F.softmax(segm_pred, dim=1)[0, 0, :, :].cpu().numpy()
                mask = (mask > self.params.init_segm_mask_thr).astype(np.float32)

                if hasattr(self, 'gt_poly'):
                    # dilate polygon-based mask
                    # dilate only if given mask is made from polygon, not from axis-aligned bb (since rotated bb is much tighter)
                    dil_kernel_sz = max(5, int(round(0.05 * min(self.target_sz).item() * f_)))
                    kernel = np.ones((dil_kernel_sz, dil_kernel_sz), np.uint8)
                    mask_dil = cv2.dilate(init_mask_patch_np, kernel, iterations=1)
                    mask = mask * mask_dil
                else:
                    mask = mask * init_mask_patch_np

                target_pixels = np.sum((mask > 0.5).astype(np.float32))

                # self.mask = mask # Song
                self.segm_init_target_pixels = target_pixels

                if self.params.save_mask:
                    segm_crop_sz = math.ceil(math.sqrt(bb[2] * bb[3]) * self.params.segm_search_area_factor)
                    save_mask(None, mask, segm_crop_sz, bb, color.shape[1], color.shape[0],
                              self.params.masks_save_path, self.sequence_name, self.frame_name)

                mask_gpu = torch.unsqueeze(torch.unsqueeze(torch.tensor(mask), dim=0), dim=0).to(self.params.device)
                train_masks = [mask_gpu]

                iters += 1

            # Song what happend if init mask is not correct?
            # print('init mask : ', np.sum(mask), np.sum(init_mask_patch_np), np.sum(mask) / np.sum(init_mask_patch_np))
            # if np.sum(mask) > np.sum(init_mask_patch_np) or np.sum(mask) / (np.sum(init_mask_patch_np)+0.001) < 0.75:
            #     mask = init_mask_patch_np
            #     target_pixels = np.sum((init_mask_patch_np).astype(np.float32))
        else:
            # Song , we use init box as init mask to prevent bad init mask
            init_mask_patch_np = (init_mask_patch_np > 0.1).astype(np.float32)
            mask = init_mask_patch_np
            # self.mask = init_mask_patch_np
            target_pixels = np.sum((init_mask_patch_np).astype(np.float32))
            self.segm_init_target_pixels = target_pixels

            mask_gpu = torch.unsqueeze(torch.unsqueeze(torch.tensor(init_mask_patch_np), dim=0), dim=0).to(
                self.params.device)

        # store everything that is needed for later
        self.segm_net = segm_net
        self.train_feat_segm_rgb = train_feat_segm_rgb # ResNet50 features, in GIM, only train_feat_segm_rgb[3] is used
        self.train_feat_segm_d = train_feat_segm_d     # feat3 in DepthNet, for future if we use feature correlation
        self.init_mask_patch = mask_gpu
        if self.params.segm_use_dist:
            self.dist_map = dist_map

        mask = np.array(mask, dtype=np.uint8)
        self.init_mask = mask
        self.mask_pixels = np.array([np.sum(mask)])
        # self.segm_init_target_pixels = target_pixels
        self.mask = mask
        self.masked_img = init_patch_crop_rgb * np.expand_dims(mask, axis=-1)
        self.init_masked_img = init_patch_crop_rgb * np.expand_dims(mask, axis=-1)

        self.polygon = None
        self.prbox = None

    def segment_target(self, color, depth, pos, sz):
        # pos and sz are in the image coordinates
        # construct new bounding box first
        tlx_ = pos[1] - sz[1] / 2
        tly_ = pos[0] - sz[0] / 2
        w_ = sz[1]
        h_ = sz[0]
        bb = [tlx_.item(), tly_.item(), w_.item(), h_.item()]

        # extract patch
        patch_rgb, f_ = prutils.sample_target(color, np.array(bb), self.params.segm_search_area_factor,
                                          output_sz=self.params.segm_output_sz)
        patch_d, _ = prutils.sample_target(depth, np.array(bb), self.params.segm_search_area_factor,
                                          output_sz=self.params.segm_output_sz)
        patch_d = np.expand_dims(patch_d, axis=-1)

        segm_crop_sz = math.ceil(math.sqrt(bb[2] * bb[3]) * self.params.segm_search_area_factor)

        # normalize input image
        init_patch_norm_rgb = patch_rgb.astype(np.float32) / float(255)
        init_patch_norm_rgb -= self.params.segm_normalize_mean
        init_patch_norm_rgb /= self.params.segm_normalize_std

        init_patch_norm_d = patch_d.astype(np.float32)

        # put image patch and mask to GPU
        patch_gpu_rgb = torch.Tensor(init_patch_norm_rgb)
        patch_gpu_d = torch.Tensor(init_patch_norm_d)
        if self.params.use_gpu:
            patch_gpu_rgb = patch_gpu_rgb.to(self.params.device)
            patch_gpu_d = patch_gpu_d.to(self.params.device)

            # reshape image for the feature extractor
            patch_gpu_rgb = torch.unsqueeze(patch_gpu_rgb, dim=0).permute(0, 3, 1, 2)
            patch_gpu_d = torch.unsqueeze(patch_gpu_d, dim=0).permute(0, 3, 1, 2)

        # extract features (extracting twice on the same patch - not necessary)
        test_feat_rgb = self.segm_net.extract_backbone_features(patch_gpu_rgb)
        test_feat_d = self.segm_net.segm_predictor.depth_feat_extractor(patch_gpu_d)

        # prepare features in the list (format for the network)
        test_feat_segm_rgb = [feat for feat in test_feat_rgb.values()]
        train_masks = [self.init_mask_patch]
        if self.params.segm_use_dist:
            if self.params.segm_dist_map_type == 'center':
                # center-based distance map
                test_dist_map = [self.dist_map]
            elif self.params.segm_dist_map_type == 'bbox':
                # bbox-based distance map
                D = self.create_dist_gauss(self.params.segm_output_sz, w_.item() * f_, h_.item() * f_)
                test_dist_map = [torch.unsqueeze(torch.unsqueeze(torch.Tensor(D).to(self.params.device), dim=0), dim=0)]
        else:
            test_dist_map = None

        # Obtain segmentation prediction
        segm_pred = self.segm_net.segm_predictor(test_feat_segm_rgb, test_feat_d,
                                                 self.train_feat_segm_rgb, self.train_feat_segm_d,
                                                 train_masks, test_dist_map)
        # softmax on the prediction (during training this is done internaly when calculating loss)
        # take only the positive channel as predicted segmentation mask
        mask = F.softmax(segm_pred, dim=1)[0, 0, :, :].cpu().numpy() # [1,2,384, 384] -> [384,384]
        if self.params.save_mask:
            mask_real = copy.copy(mask)
        mask = (mask > self.params.segm_mask_thr).astype(np.uint8)

        # print('mean previous mask pixels : ', np.mean(self.mask_pixels), 'pred mask pixels:', np.sum(mask) )
        ''' Song , from the mask visualization, it seems that pred_mask is correct, but probx is wrong '''

        # self.mask = mask # predicted segmentation
        self.masked_img = patch_rgb * np.expand_dims(mask, axis=-1)


        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]

        if self.segmentation_task:
            mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, contours, -1, 1, thickness=-1)
            # save mask to disk
            # Note: move this below if evaluating on VOT
            if self.params.save_mask:
                save_mask(None, mask_real, segm_crop_sz, bb, color.shape[1], color.shape[0],
                          self.params.masks_save_path, self.sequence_name, self.frame_name)



        print('max contour area : ', np.max(cnt_area))

        if len(cnt_area) > 0 and len(contours) != 0 and np.max(cnt_area) > 1000:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2) # Song, checked already, here is correct

            prbox = np.reshape(cv2.boxPoints(cv2.minAreaRect(polygon)), (4, 2))  # Rotated Rectangle
            prbox_init = copy.deepcopy(prbox) # Song, checked already, here is correct

            self.polygon = polygon

            prbox_opt = np.array([])
            if self.params.segm_optimize_polygon:
                if not self.segmentation_task:
                    mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
                    self.mask = mask # song

                    # save mask to disk
                    # Note: move this below if evaluating on VOT
                    # if self.params.save_mask:
                    #     save_mask(mask, mask_real, segm_crop_sz, bb, color.shape[1], color.shape[0],
                    #               self.params.masks_save_path, self.sequence_name, self.frame_name)

                t_opt_start_ = time.time()
                prbox_opt_ = fit_bbox_to_mask(mask.astype(np.int32), rotated=self.rotated_bbox)
                bbox_opt_time = time.time() - t_opt_start_
                if prbox_opt_ is not None:
                    A1 = np.linalg.norm(np.array([prbox[0, 0], prbox[0, 1]]) - np.array([prbox[1, 0], prbox[1, 1]])) * \
                         np.linalg.norm(np.array([prbox[1, 0], prbox[1, 1]]) - np.array([prbox[2, 0], prbox[2, 1]]))
                    A_new = np.linalg.norm(np.array([prbox_opt_[0, 0], prbox_opt_[0, 1]]) - np.array(
                        [prbox_opt_[1, 0], prbox_opt_[1, 1]])) * \
                            np.linalg.norm(np.array([prbox_opt_[1, 0], prbox_opt_[1, 1]]) - np.array(
                                [prbox_opt_[2, 0], prbox_opt_[2, 1]]))
                    area_ratio = A_new / A1

                    if area_ratio > 0.1 and area_ratio < 2.5:  # 1.7
                        prbox_opt = prbox_opt_
                    else:
                        print('Bbox optimization has made too large difference.')

            displacement = np.mean(prbox, axis=0) - np.array([mask.shape[0] / 2, mask.shape[1] / 2])
            prbox = (prbox - np.mean(prbox, axis=0) + displacement) / f_ + np.array([pos[1].item(), pos[0].item()])


            if self.params.segm_scale_estimation:

                # use pixels_ratio to determine if new scale should be estimated or not
                mask_pixels_ = np.max(cnt_area)
                # Song target mask size changes should be in a range
                pixels_ratio = abs(np.mean(self.mask_pixels) - mask_pixels_) / np.mean(self.mask_pixels)
                # mask_pixels_ratio =  mask_pixels_ / (np.mean(self.mask_pixels)+0.001) # Song, prevent segmentation goes smaller
                # init_mask_pixels_ratio = mask_pixels_ / (np.sum(self.init_mask)+0.001)

                if self.uncert_score < self.params.uncertainty_segm_scale_thr:

                    if pixels_ratio < self.params.segm_pixels_ratio and pixels_ratio:

                        self.mask_pixels = np.append(self.mask_pixels, mask_pixels_)
                        if self.mask_pixels.size > self.params.mask_pixels_budget_sz:
                            self.mask_pixels = np.delete(self.mask_pixels, 0)

                        new_aabb = self.poly_to_aabbox(prbox[:, 0], prbox[:, 1])
                        new_target_scale = (math.sqrt(new_aabb[2] * new_aabb[3]) * self.params.search_area_scale) / \
                                           self.img_sample_sz[0]
                        rel_scale_ch = (abs(new_target_scale - self.target_scale) / self.target_scale).item()
                        if new_target_scale > self.params.segm_min_scale and rel_scale_ch < self.params.max_rel_scale_ch_thr:
                            self.target_scale = max(self.target_scale * self.params.min_scale_change_factor,
                                                    min(self.target_scale * self.params.max_scale_change_factor,
                                                        new_target_scale))

                        print('target scale : ', self.target_scale)
                    else:
                        print('pred_mask too large or too small, ', mask_pixels_, np.mean(self.mask_pixels), np.sum(self.init_mask))

            if not self.params.segm_scale_estimation or pixels_ratio < self.params.consider_segm_pixels_ratio:
                self.pos[0] = np.mean(prbox[:, 1])
                self.pos[1] = np.mean(prbox[:, 0])

            if not self.params.segm_scale_estimation or pixels_ratio < self.params.segm_pixels_ratio:
                if prbox_opt.size > 0:
                    displacement_opt = np.mean(prbox_opt, axis=0) - np.array([mask.shape[0] / 2, mask.shape[1] / 2])
                    prbox = (prbox_opt - np.mean(prbox_opt, axis=0) + displacement_opt) / f_ + np.array(
                        [pos[1].item(), pos[0].item()])

                if self.rotated_bbox:
                    pred_region = [prbox[0, 0], prbox[0, 1], prbox[1, 0], prbox[1, 1], prbox[2, 0], prbox[2, 1],
                                   prbox[3, 0], prbox[3, 1]]
                else:
                    pred_region = [np.min(prbox[:, 0]) + 1, np.min(prbox[:, 1]) + 1,
                                   np.max(prbox[:, 0]) - np.min(prbox[:, 0]) + 1,
                                   np.max(prbox[:, 1]) - np.min(prbox[:, 1]) + 1]

                print('is rotated_bbox : ', self.rotated_bbox)
                self.prbox = pred_region
                return pred_region

        return None

    def poly_to_aabbox(self, x_, y_):
        # keep the center and area of the polygon
        # change aspect ratio of the original bbox
        cx = np.mean(x_)
        cy = np.mean(y_)
        x1 = np.min(x_)
        x2 = np.max(x_)
        y1 = np.min(y_)
        y2 = np.max(y_)
        A1 = np.linalg.norm(np.array([x_[0], y_[0]]) - np.array([x_[1], y_[1]])) * \
             np.linalg.norm(np.array([x_[1], y_[1]]) - np.array([x_[2], y_[2]]))
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        return np.array([cx - w / 2, cy - h / 2, w, h])
