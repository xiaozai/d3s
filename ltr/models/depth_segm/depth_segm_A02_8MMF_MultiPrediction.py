
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import ml_collections
import copy
import math

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


''' Code from
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
'''
class Attention(nn.Module):
    def __init__(self, config, vis, patches=64):
        super(Attention, self).__init__()
        self.vis = vis
        self.patches = patches
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 12 heads * 64

        self.query = Linear(config.hidden_size, self.all_head_size) # 768 -> 768
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

    def forward(self, hidden_states, kv=None, mask=None):
        mixed_query_layer = self.query(hidden_states)  # [B, Patches, C]

        if kv is not None:
            mixed_key_layer = self.key(kv)      # [B, Patches, C]
            mixed_value_layer = self.value(kv)  # [B, Patches, C]
        else:
            mixed_key_layer = self.key(hidden_states)      # [B, Patches, C]
            mixed_value_layer = self.value(hidden_states)  # [B, Patches, C]

        query_layer = self.transpose_for_scores(mixed_query_layer) # [B, patches, C] -> [B, num_attention_heads, patches, attention_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)     # [B, 12, 64, 64], 12 heads, 64 head size, 768 = 12 * 64
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # Bx12x64x64 * Bx12x64x64 = Bx12x64x64
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if mask is None:
            # Song add mask here for background pixels, force background probs is 0
            attention_scores[:, :, self.patches//2:, :] = 0
        else:
            # Song add mask for similarity between template and search region
            attention_scores = attention_scores * mask.view(mask.shape[0], 1, 1, -1) # head, layers, Pq, P_kv * [1, B, 1, 1, P_kv]

        attention_probs = self.softmax(attention_scores) # dim=-1 Bx12xpatchesx64

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)              # Bx12xPatchesx64 * Bx12x64x64
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()          # [B, Patches, heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [B, Patches, C=768]
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        # self.act_fn = torch.nn.functional.gelu
        self.act_fn = torch.nn.ReLU(inplace=True)
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

        self.n_patches = n_patches

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
    def __init__(self, config, vis, patches=64):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis, patches=patches)

    def forward(self, x, kv=None, mask=None):
        h = x                        # Bx16xC
        x = self.attention_norm(x)
        if kv is not None:
            kv = self.attention_norm(kv)
        x, weights = self.attn(x, kv=kv, mask=mask)
        x = x + h                    # Bx16xC

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis, patches=64):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, patches=patches)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, kv=None, mask=None):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states, kv=kv, mask=mask)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels, vis, mask_pool=False):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis, patches=self.embeddings.n_patches)

        if mask_pool:
            self.mask_pool = nn.MaxPool2d(config.patches['size'], stride=config.patches['size'])

    def forward(self, q_ids, kv=None, mask=None):

        embeddings_kv = self.embeddings(kv) if kv is not None else None
        mask = self.mask_pool(mask).view(mask.shape[0], -1) if mask is not None else None

        embedding_q = self.embeddings(q_ids)  # [B, N_Patches, C=768], class embeddings + patches

        encoded, attn_weights = self.encoder(embedding_q, kv=embeddings_kv, mask=mask)  # encoded [B, patches, C=768]
        return encoded, attn_weights



def get_b16_config(size=(16,16)):
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': size})
    config.hidden_size = 96 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024 # 3072
    config.transformer.num_heads = 3 # 12
    config.transformer.num_layers = 3 # 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

''' Inspired by
    RDFNet: RGB-D Multi-level Residual Feature Fusion for Indoor Semantic Segmentation
    their "Multi-modal feature fusion (MMF) network"
'''

class MMFNet(nn.Module):
    def __init__(self, rgb_dims, d_dims, output_dims):
        super().__init__()

        self.conv1x1_rgb = conv1x1_layer(rgb_dims, output_dims)
        self.res01_rgb = nn.Sequential(nn.ReLU(inplace=True),
                                       conv3x3_layer(output_dims, output_dims),
                                       nn.ReLU(inplace=True),
                                       conv3x3_layer(output_dims, output_dims))

        self.res02_rgb = nn.Sequential(nn.ReLU(inplace=True),
                                       conv3x3_layer(output_dims, output_dims),
                                       nn.ReLU(inplace=True),
                                       conv3x3_layer(output_dims, output_dims))

        self.conv3x3_rgb = conv3x3_layer(output_dims, output_dims)

        self.conv1x1_d = conv1x1_layer(d_dims, output_dims)
        self.res01_d = nn.Sequential(nn.ReLU(inplace=True),
                                     conv3x3_layer(output_dims, output_dims),
                                     nn.ReLU(inplace=True),
                                     conv3x3_layer(output_dims, output_dims))

        self.res02_d = nn.Sequential(nn.ReLU(inplace=True),
                                     conv3x3_layer(output_dims, output_dims),
                                     nn.ReLU(inplace=True),
                                     conv3x3_layer(output_dims, output_dims))

        self.conv3x3_d = conv3x3_layer(output_dims, output_dims)

        self.relu = nn.ReLU(inplace=True)

        self.pool5x5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.conv3x3 = conv3x3_layer(output_dims, output_dims)

        self.dropout = Dropout(0.3)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, f_rgb, f_d):
        f_rgb = self.conv1x1_rgb(f_rgb)
        f_rgb = f_rgb + self.res01_rgb(f_rgb)
        f_rgb = f_rgb + self.res02_rgb(f_rgb)

        f_d = self.conv1x1_d(f_d)
        f_d = f_d + self.res01_d(f_d)
        f_d = f_d + self.res02_d(f_d)

        f_rgbd = self.relu(self.conv3x3_rgb(f_rgb) + self.conv3x3_d(f_d))
        f_rgbd = f_rgbd + self.conv3x3(self.pool5x5(f_rgbd))
        f_rgbd = self.dropout(f_rgbd)

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


class DepthSegmNetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        segm_input_dim = (64, 256, 512, 1024)
        segm_inter_dim = (4, 16, 32, 64)
        segm_dim = (64, 64)

        self.depth_feat_extractor = DepthNet(input_dim=1, inter_dim=segm_inter_dim)

        config0 = get_b16_config(size=(12, 12))
        config1 = get_b16_config(size=(6, 6))
        config2 = get_b16_config(size=(3, 3))
        config3 = get_b16_config(size=(2, 2))
        self.transformers = nn.ModuleList([Transformer(config0, (384, 192), 4, True),
                                           Transformer(config1, (96, 96), 16, True, mask_pool=True),
                                           Transformer(config2, (48, 48), 32, True, mask_pool=True),
                                           Transformer(config3, (24, 24), 64, True, mask_pool=True)])

        self.MMF_layers = nn.ModuleList([MMFNet(segm_input_dim[0], segm_inter_dim[0], segm_inter_dim[0]),
                                         MMFNet(segm_input_dim[1], segm_inter_dim[1], segm_inter_dim[1]),
                                         MMFNet(segm_input_dim[2], segm_inter_dim[2], segm_inter_dim[2]),
                                         MMFNet(segm_input_dim[3], segm_inter_dim[3], segm_inter_dim[3])])

        self.s_layers = nn.ModuleList([conv(segm_inter_dim[0], segm_inter_dim[0]),
                                       conv(segm_inter_dim[1], segm_inter_dim[1]),
                                       conv(segm_inter_dim[2], segm_inter_dim[2]),
                                       conv(1, segm_inter_dim[3])])

        self.a_layers = nn.ModuleList([conv(config0.hidden_size, segm_inter_dim[0]),
                                       conv(config1.hidden_size, segm_inter_dim[1]),
                                       conv(config2.hidden_size, segm_inter_dim[2]),
                                       conv(config3.hidden_size, segm_inter_dim[3])])



        self.post_layers = nn.ModuleList([conv_no_relu(segm_inter_dim[0], 2),
                                          conv(segm_inter_dim[1], segm_inter_dim[0]),
                                          conv(segm_inter_dim[2], segm_inter_dim[1]),
                                          conv(segm_inter_dim[3], segm_inter_dim[2])])

        # Supervise every layer
        self.multi_pred = nn.ModuleList([conv_no_relu(segm_inter_dim[0], 2),
                                         conv_no_relu(segm_inter_dim[1], 2),
                                         conv_no_relu(segm_inter_dim[2], 2)])
        self.initialize_weights()

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

            Pyramid supervision for enhancing the deep representation.
        '''
        out = F.interpolate(test_dist[0], size=(feat_test_rgb[3].shape[-2], feat_test_rgb[3].shape[-1]))                 # [B, 1, 24, 24]

        out, attn_weights3 = self.process(out, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=3) # 48
        out03 = self.multi_pred[2](F.upsample(out, scale_factor=8))

        out, attn_weights2 = self.process(out, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=2) # 96
        out02 = self.multi_pred[1](F.upsample(out, scale_factor=4))

        out, attn_weights1 = self.process(out, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=1) # 192
        out01 = self.multi_pred[0](F.upsample(out, scale_factor=2))

        out, attn_weights0 = self.process(out, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=0)


        if debug:
            return (out, out01, out02, out03), (attn_weights3, attn_weights2, attn_weights1, attn_weights0)
        else:
            return (out, out01, out02, out03)

    def process(self, pre_out, test_rgb, test_d, train_rgb, train_d, mask_train, layer=0):

        f_test_rgb, f_test_d, f_train_rgb, f_train_d = test_rgb[layer], test_d[layer], train_rgb[layer], train_d[layer]
        fg_mask = F.interpolate(mask_train[0], size=(f_test_rgb.shape[-2], f_test_rgb.shape[-1]))  # Bx1x24x48
        bg_mask = 1 - fg_mask

        f_test_rgbd = self.MMF_layers[layer](f_test_rgb, f_test_d)              # BxCxHxH + BxCxHxW => BxCxHxW
        f_train_rgbd = self.MMF_layers[layer](f_train_rgb, f_train_d)

        if layer in [3, 2, 1]:
            ''' cross-attention '''
            attn_rgbd, attn_weights = self.transformers[layer](f_test_rgbd, kv=f_train_rgbd, mask=fg_mask) # [B, Patches, C=768], attn_weights, [B, heads=12, patches, headsize=64]
            n_patches = attn_rgbd.shape[1]
        else:
            ''' self-attention '''
            feat_rgbd = torch.cat((f_test_rgbd, f_train_rgbd * bg_mask), dim=2)
            attn_rgbd, attn_weights = self.transformers[layer](feat_rgbd)       # [B, Patches, C=768], attn_weights, [B, heads=12, patches, headsize=64]
            n_patches = attn_rgbd.shape[1] // 2                                 # for each image, 16 patches, 64 patches, 256 patches
            attn_rgbd = attn_rgbd[:, :n_patches, :]                             # get rid of bg features, B, 2*patches, C


        featmap_sz = int(math.sqrt(n_patches))

        attn_rgbd = attn_rgbd.view(attn_rgbd.shape[0], featmap_sz, featmap_sz, -1) # [B, H, W, C]
        attn_rgbd = attn_rgbd.permute(0, 3, 1, 2).contiguous()                     # [B, C, H, W]
        attn_rgbd = F.interpolate(attn_rgbd, size=(f_test_rgb.shape[-2], f_test_rgb.shape[-1]))  # B x 2C x 4 x 4 ->  B x 2C x 48 x 48

        out = self.post_layers[layer](F.upsample(self.a_layers[layer](attn_rgbd) + self.s_layers[layer](pre_out), scale_factor=2))

        return out, attn_weights
