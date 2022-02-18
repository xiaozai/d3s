
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import ml_collections
import copy

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

''' Code from
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
'''
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        print('mixed q/k/v : ', mixed_query_layer.shape, mixed_key_layer.shape, mixed_value_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        print('query layer : ', query_layer.shape)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Song add mask here for background pixels, force background probs is 0
        print('attention_probs : ', attention_probs.shape)
        # attetion_probs[:, 8:, :] = 0

        attention_probs = self.softmax(attention_scores)

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)



        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()

        # self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        print('embeddings : patches size and num : ', patch_size, n_patches) # (12, 12) 64 patches

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1) # B, 1, C

        x = self.patch_embeddings(x) # [B, C, H/stride, W/stride]
        x = x.flatten(2)             # [B, C, H*W=patches]
        x = x.transpose(-1, -2)      # [B, tokens, C]
        # x = torch.cat((cls_tokens, x), dim=1) # [B, 1+N, C]

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x                        # Bx16xC
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h                    # Bx16xC

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)            # [B, N_Patches, C=768], class embeddings + patches
        print('embedding_output : ', embedding_output.shape)
        encoded, attn_weights = self.encoder(embedding_output)
        print('encoded : ', encoded.shape)
        print('attn_weights : ', len(attn_weights))              # [B, 1+N_Patches, C=768]
        return encoded, attn_weights



def get_b16_config(size=(16,16)):
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': size})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


class DepthNet(nn.Module):
    def __init__(self, input_dim=1, inter_dim=(4, 16, 32, 64)):
        super().__init__()

        self.conv0 = conv(input_dim, inter_dim[0])    # 1  -> 4
        self.conv1 = conv(inter_dim[0], inter_dim[1]) # 4 -> 16
        self.conv2 = conv(inter_dim[1], inter_dim[2]) # 16 -> 32
        self.conv3 = conv(inter_dim[2], inter_dim[3]) # 32 -> 64

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
        feat0 = self.maxpool0(feat0) + self.avgpool0(feat0) # 384 -> 192, 4

        feat1 = self.conv1(feat0)
        feat1 = self.maxpool1(feat1) + self.avgpool1(feat1) # 192 -> 96, 16

        feat2 = self.conv2(feat1)
        feat2 = self.maxpool2(feat2) + self.avgpool2(feat2) # 96 -> 48, 32

        feat3 = self.conv3(feat2)
        feat3 = self.maxpool3(feat3) + self.avgpool3(feat3) # 48 -> 24, 64

        return [feat0, feat1, feat2, feat3] # [4, 16, 32, 64]


class DepthSegmNetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        segm_input_dim = (64, 256, 512, 1024)
        segm_inter_dim = (4, 16, 32, 64)
        segm_dim = (64, 64)  # convolutions before cosine similarity

        self.depth_feat_extractor = DepthNet(input_dim=1, inter_dim=segm_inter_dim)


        self.rgbd_attention2 = Transformer(get_b16_config(size=(12, 12)), (96, 96), 32, True) # vis = True img_size = 384, patches=(16,16), in_channels=32
        self.rgbd_attention1 = Transformer(get_b16_config(size=(24, 24)), (192, 192), 16, True)
        self.rgbd_attention0 = Transformer(get_b16_config(size=(48, 48)), (384, 384), 4, True)


        # 1024， 512， 256， 64 -> 64, 32, 16, 4
        self.segment0_rgb = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
        self.segment1_rgb = conv_no_relu(segm_dim[0], segm_dim[1]) # 64

        self.segment0_d = conv(64, 64, kernel_size=1, padding=0)
        self.segment1_d = conv_no_relu(64, 64)

        # 256 depth feat + 1 dist map + rgb similarity + depth similarity
        self.mixer = conv(9, segm_inter_dim[3])
        self.s3 = conv(segm_inter_dim[3], segm_inter_dim[2]) # 64 -> 32

        self.s2 = conv(segm_inter_dim[2], segm_inter_dim[2]) # 32, 32
        self.s1 = conv(segm_inter_dim[1], segm_inter_dim[1]) # 16, 16
        self.s0 = conv(segm_inter_dim[0], segm_inter_dim[0]) # 4, 4

        self.f2 = conv(segm_input_dim[2], segm_inter_dim[2])
        self.f1 = conv(segm_input_dim[1], segm_inter_dim[1])
        self.f0 = conv(segm_input_dim[0], segm_inter_dim[0])

        self.d2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.d1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.d0 = conv(segm_inter_dim[0], segm_inter_dim[0])

        self.post2 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.post1 = conv(segm_inter_dim[1], segm_inter_dim[0])
        self.post0 = conv_no_relu(segm_inter_dim[0], 2)

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

            we also model background feat
        '''

        f_test_rgb = self.segment1_rgb(self.segment0_rgb(feat_test_rgb[3]))
        f_train_rgb = self.segment1_rgb(self.segment0_rgb(feat_train_rgb[3]))

        f_test_d = self.segment1_d(self.segment0_d(feat_test_d[3]))
        f_train_d = self.segment1_d(self.segment0_d(feat_train_d[3]))

        mask_pos = F.interpolate(mask_train[0], size=(f_train_rgb.shape[-2], f_train_rgb.shape[-1])) # [1,1,384, 384] -> [B,1,24,24]
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
        segm_layers = torch.cat((dist,
                                 torch.unsqueeze(pred_sm_rgb[:, :, :, 0], dim=1),
                                 torch.unsqueeze(pred_pos_rgb, dim=1),
                                 torch.unsqueeze(pred_sm_d[:, :, :, 0], dim=1), # depth may mislead
                                 torch.unsqueeze(pred_pos_d, dim=1),
                                 #
                                 torch.unsqueeze(pred_sm_rgb[:, :, :, 1], dim=1), # RGB BG
                                 torch.unsqueeze(pred_neg_rgb, dim=1),
                                 torch.unsqueeze(pred_sm_d[:, :, :, 1], dim=1),   # D BG
                                 torch.unsqueeze(pred_neg_rgb, dim=1)),
                                 dim=1)


        out = self.mixer(segm_layers)
        out = self.s3(F.upsample(out, scale_factor=2))

        # feat maps -> self.attention, -> B, tokens, C
        # Merge RGB and D feature maps into one image, add some background patch

        # feat_test_rgb[2], Bx512x48x48, mask , Bx1x24x24
        # [[F_rgb, F_d],
        #  [F_rgb_bg, F_d_bg]] ,  BxCx2Hx2W

        train_bg_mask2 = 1 - F.interpolate(mask_train[0], size=(feat_train_rgb[2].shape[-2], feat_train_rgb[2].shape[-1])) # Bx512x48x48
        feat_rgbd_stack2 = torch.cat((torch.cat((self.f2(feat_test_rgb[2]), self.d2(feat_test_d[2])), dim=3),
                                        torch.cat((self.f2(feat_train_rgb[2]*train_bg_mask2), self.d2(feat_train_d[2]*train_bg_mask2)), dim=3)),
                                        dim=2)
        print('stack rgbd feat : ', feat_test_rgb[2].shape, feat_test_d[2].shape, feat_rgbd_stack2.shape)
        feat_rgbd2, attn_weights2 = self.rgbd_attention2(feat_rgbd_stack2) # Bx1+TxC, the first is a classication learnable embedding
        print('feat_rgbd2 : ', feat_rgbd2.shape) # [B, 1+T, C]

        out = self.post2(F.upsample(self.f2(feat_rgbd2) + self.s2(out), scale_factor=2))

        feat_rgbd_stack1 = torch.cat(((feat_test_rgb[1], feat_test_d[1], feat_train_rgb[1]*mask_neg, feat_train_d[1]*mask_neg)))
        feat_rgbd1, attn_weights1 = self.rgbd_attention1(feat_rgbd_stack1)
        out = self.post1(F.upsample(self.f1(feat_rgbd1) + self.s1(out), scale_factor=2))

        feat_rgbd_stack0 = torch.cat(((feat_test_rgb[0], feat_test_d[0], feat_train_rgb[0]*mask_neg, feat_train_d[0]*mask_neg)))
        feat_rgbd0, attn_weights0 = self.rgbd_attention0(feat_rgbd_stack0)
        out = self.post0(F.upsample(self.f0(feat_rgbd0) + self.s0(out), scale_factor=2))


        if debug:
            return out, (pred_sm_rgb, pred_sm_d, attn_weights2, attn_weights1, attn_weights0)
        else:
            return out

    def similarity_segmentation(self, f_test, f_train, mask_pos, mask_neg):
        # first normalize train and test features to have L2 norm 1
        # cosine similarityand reshape last two dimensions into one
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
