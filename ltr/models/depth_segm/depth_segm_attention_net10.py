
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

        self.query = Linear(config.hidden_size, self.all_head_size) # multi heads projection
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

    def forward(self, hidden_states_q, hidden_states_k=None, hidden_states_v=None, mask=None):

        if hidden_states_k is None and hidden_states_v is None:
            hidden_states_k = hidden_states_q
            hidden_states_v = hidden_states_q

        mixed_query_layer = self.query(hidden_states_q)  # [B, Patches, C]
        mixed_key_layer = self.key(hidden_states_k)      # [B, Patches, C]
        mixed_value_layer = self.value(hidden_states_v)  # [B, Patches, C]

        query_layer = self.transpose_for_scores(mixed_query_layer) # [B, patches, C] -> [B, num_attention_heads, patches, attention_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)     # [B, head=12, patches=64, attention_head_size=64], 12 heads, 64 head size, 768 = 12 * 64
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [B, head, patches_q, patches_k]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(1), 0) # float('-inf'))

        attention_probs = self.softmax(attention_scores) # dim=-1 [B, head, P_q, P_k]


        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)     # [B,Head,P_q,P_k]*[B,Head,P_k,P_sz]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [B, Patches, heads, head_size]
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
        # self.act_fn = torch.nn.functional.gelu # pytorch 1.1.0 does not have gelu
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
    def __init__(self, config, img_size, in_channels=3, use_target_sz=False):
        super(Embeddings, self).__init__()

        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.use_target_sz = use_target_sz

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        if self.use_target_sz:
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches+1, config.hidden_size))
            self.sz_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):

        x = self.patch_embeddings(x) # [B, C, H/stride, W/stride]
        x = x.flatten(2)             # [B, C, H*W=patches]
        x = x.transpose(-1, -2)      # [B, tokens, C]

        if self.use_target_sz:
            B = x.shape[0]
            sz_tokens = self.sz_token.expand(B, -1, -1) # B, 1, C
            x = torch.cat((sz_tokens, x), dim=1)        # [B, 1+N, C]

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

class CrossBlock(nn.Module):
    def __init__(self, config, vis, patches=64):
        super(CrossBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn_cross = Attention(config, vis, patches=patches) # cross-attention for key value
        self.attn = Attention(config, vis, patches=patches)      # self-attention for query

    def forward(self, q, k, v, mask=None):
        ''' Decodder '''
        # qurey, self-attention
        h = q                        # Bx16xC
        q = self.attention_norm(q)
        q, weights_q = self.attn(q)
        q = q + h                    # Bx16xC

        h = q
        q = self.ffn_norm(q)
        q = self.ffn(q)
        q = q + h


        # q, k, v -> MultiheadAttention -> add & norm
        h = q                        # Bx16xC
        q = self.attention_norm(q)
        k = self.attention_norm(k)
        v = self.attention_norm(v)
        x, weights = self.attn_cross(q, hidden_states_k=k, hidden_states_v=v, mask=mask)
        x = x + h                    # Bx16xC

        # x -> Feed Forward -> add & norm
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

class SelfAtten(nn.Module):
    def __init__(self, config, vis, patches=64):
        super(SelfAtten, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, patches=patches)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class CrossAtten(nn.Module):
    def __init__(self, config, vis, patches=64):
        super(CrossAtten, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.decoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = CrossBlock(config, vis, patches=patches)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, query, key, value, mask=None):
        attn_weights = []
        for layer_block in self.layer:
            query, weights = layer_block(query, key, value, mask=mask)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.decoder_norm(query)
        return encoded, attn_weights

class CrossAttentionTransformer(nn.Module):
    def __init__(self, config, img_size, in_channels, vis, use_target_sz=False):
        super(CrossAttentionTransformer, self).__init__()
        self.embeddings = Embeddings(config,
                                     img_size=img_size,
                                     in_channels=in_channels,
                                     use_target_sz=use_target_sz)
        self.n_patches = self.embeddings.n_patches
        self.patch_sz = config.patches['size']
        self.use_target_sz = use_target_sz
        self.s_atten = SelfAtten(config, vis, patches=self.n_patches)
        self.c_atten = CrossAtten(config, vis, patches=self.n_patches)

    def forward(self, q_input, kv_input, mask=None):
        q_embeddings = self.embeddings(q_input)   # [B, 1+N_Patches, C], class embeddings + patches
        kv_embeddings = self.embeddings(kv_input) # [B, 1+N_Patches, C]
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=1.0/self.patch_sz[0])
            mask = mask.view(mask.shape[0], 1, -1)
            if self.use_target_sz:
                sz_mask = torch.ones(mask.shape[0], 1, 1).to('cuda')
                mask = torch.cat((sz_mask, mask), dim=-1) # B, 1, patches+1
            mask = torch.tensor(mask, dtype=torch.bool)   # B, 1, patches+1

        kv_encoded, kv_attn_weights = self.s_atten(kv_embeddings)   # encoded [B, patches+1, C]
        q_encoded, q_attn_weights = self.c_atten(q_embeddings, kv_embeddings, kv_embeddings, mask=mask)
        return q_encoded, q_attn_weights


def get_config(size=(16,16)):
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

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, dp):
        feat0 = self.conv0(dp)
        feat0 = self.maxpool0(feat0)

        feat1 = self.conv1(feat0)
        feat1 = self.maxpool1(feat1)

        feat2 = self.conv2(feat1)
        feat2 = self.maxpool2(feat2)

        feat3 = self.conv3(feat2)
        feat3 = self.maxpool3(feat3)

        return [feat0, feat1, feat2, feat3] # [4, 16, 32, 64]


class DepthSegmNetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        segm_input_dim = (64, 256, 512, 1024)
        segm_inter_dim = (4, 16, 32, 64)
        segm_dim = (64, 64)  # convolutions before cosine similarity
        feat_sz = (192, 96, 48, 24) # feature maps sizes in ResNet50 and DepthNet

        self.depth_feat_extractor = DepthNet(input_dim=1, inter_dim=segm_inter_dim)


        config0 = get_config(size=(8, 8)) # 192*192 -> 12*12 patches
        config1 = get_config(size=(8, 8)) # 96*96 -> 12*12 patches
        config2 = get_config(size=(4, 4)) # 48*48 -> 12*12 patches
        config3 = get_config(size=(2, 2)) # 24*24 -> 12*12 patches

        crossAttnTransformer0 = CrossAttentionTransformer(config0, (feat_sz[0]*2, feat_sz[0]), segm_inter_dim[0], vis=True)
        crossAttnTransformer1 = CrossAttentionTransformer(config1, (feat_sz[1]*2, feat_sz[1]), segm_inter_dim[1], vis=True)
        crossAttnTransformer2 = CrossAttentionTransformer(config2, (feat_sz[2]*2, feat_sz[2]), segm_inter_dim[2], vis=True)
        crossAttnTransformer3 = CrossAttentionTransformer(config3, (feat_sz[3]*2, feat_sz[3]), segm_inter_dim[3], vis=True)
        self.transformers = nn.ModuleList([crossAttnTransformer0, crossAttnTransformer1, crossAttnTransformer2, crossAttnTransformer3])

        # project pre-out feat
        self.s_layers = nn.ModuleList([conv(segm_inter_dim[0], segm_inter_dim[0]),
                                       conv(segm_inter_dim[1], segm_inter_dim[1]),
                                       conv(segm_inter_dim[2], segm_inter_dim[2]),
                                       conv(1, segm_inter_dim[3], kernel_size=1, padding=0)])
        # attention layers
        self.a_layers = nn.ModuleList([conv(crossAttnTransformer0.n_patches*2, segm_inter_dim[0]),
                                       conv(crossAttnTransformer1.n_patches*2, segm_inter_dim[1]),
                                       conv(crossAttnTransformer2.n_patches*2, segm_inter_dim[2]),
                                       conv(crossAttnTransformer3.n_patches*2, segm_inter_dim[3])])

        # project RGB feat
        self.f_layers = nn.ModuleList([conv(segm_input_dim[0], segm_inter_dim[0]),
                                       conv(segm_input_dim[1], segm_inter_dim[1]),
                                       conv(segm_input_dim[2], segm_inter_dim[2]),
                                       nn.Sequential(conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0),
                                                     conv_no_relu(segm_dim[0], segm_dim[1]))])

        # project D feat
        self.d_layers = nn.ModuleList([conv(segm_inter_dim[0], segm_inter_dim[0]),
                                       conv(segm_inter_dim[1], segm_inter_dim[1]),
                                       conv(segm_inter_dim[2], segm_inter_dim[2]),
                                       nn.Sequential(conv(segm_inter_dim[3], segm_inter_dim[3], kernel_size=1, padding=0),
                                                     conv_no_relu(segm_inter_dim[3], segm_inter_dim[3]))])

        # project out feat
        self.post_layers = nn.ModuleList([conv(segm_inter_dim[0], 2),           # conv_no_relu(segm_inter_dim[0], 2),
                                          conv(segm_inter_dim[1], segm_inter_dim[0]),
                                          conv(segm_inter_dim[2], segm_inter_dim[1]),
                                          conv(segm_inter_dim[3], segm_inter_dim[2])])


        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ModuleList):
                for n in m:
                    if isinstance(n, nn.Conv2d) or isinstance(n, nn.ConvTranspose2d) or isinstance(n, nn.Linear):
                        nn.init.kaiming_normal_(n.weight.data, mode='fan_in')
                        if n.bias is not None:
                            n.bias.data.zero_()

    def forward(self, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, test_dist=None, debug=False):
        ''' rgb features   : [conv1, layer1, layer2, layer3], Bx64x192x192 -> Bx256x96x96 -> Bx512x48x48 -> Bx1024x24x24
            depth features : [feat0, feat1, feat2, feat3],    Bx4x192x192  -> Bx16x96x96  -> Bx32x48x48  -> Bx64x24x24

            what happens we use attn weights maps as input??
        '''
        attn_weights3, feat_rgbd3 = self.attn_module(test_dist[0], feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=3)
        attn_weights2, feat_rgbd2 = self.attn_module(attn_weights3, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=2)
        attn_weights1, feat_rgbd1 = self.attn_module(attn_weights2, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=1)
        attn_weights0, feat_rgbd0 = self.attn_module(attn_weights1, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=0)

        # out = F.softmax(attn_weights0, dim=1)

        if debug:
            return attn_weights0, (attn_weights3, attn_weights2, attn_weights1, attn_weights0) # B, C, H, W
        else:
            return attn_weights0

    def attn_module(self, pre_attn, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=0):

        f_test_rgb, f_test_d = feat_test_rgb[layer], feat_test_d[layer]
        f_train_rgb, f_train_d = feat_train_rgb[layer], feat_train_d[layer]
        mask = F.interpolate(mask_train[0], size=(f_train_rgb.shape[-2], f_train_rgb.shape[-1])) # Bx1xHxW
        mask = torch.cat((mask, mask), dim=2)

        template = torch.cat((self.f_layers[layer](f_train_rgb), self.d_layers[layer](f_train_d)), dim=2)
        search_region = torch.cat((self.f_layers[layer](f_test_rgb), self.d_layers[layer](f_test_d)), dim=2)

        feat_rgbd, attn_weights = self.transformers[layer](template, search_region, mask=mask)  # [B, P_q, C], # [layers, B, heads, P_q, P_kv]


        ''' Use attn_weights as input , [layers, B, heads, P_q, P_kv], P_q == P_kv'''
        attn_weights = torch.stack(attn_weights)
        attn_weights = torch.mean(attn_weights, dim=2) # [layers, B, P_q, P_kv] average the attention weights across all heads
        patch_q = attn_weights.shape[-2]
        patch_kv = attn_weights.shape[-1]
        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(patch_q, m=patch_kv).to('cuda')                # [P_q, P_kv]
        aug_att_mat = attn_weights + residual_att                               # [layers, B, P_q, P_kv]
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)       # [layers, B, P_q, P_kv] / [layers, B, P_q]

        # Recursively multiply the weight matrices
        joint_attentions = aug_att_mat[0]
        for l in range(1, aug_att_mat.size(0)):
            joint_attentions = torch.matmul(aug_att_mat[l], joint_attentions)   # [B, P_q, P_kv]

        n_patches = joint_attentions.shape[1] // 2                              # for each patch, 16 patches, 64 patches, 256 patches
        featmap_sz = int(math.sqrt(n_patches))                                  # for RGB and D feat maps, 4x4, 8x8, 16x16
        attention_map = torch.cat((joint_attentions[:, :n_patches, :], joint_attentions[:, n_patches:, :]), dim=-1) # B, P_q, 2P_kv
        attention_map = attention_map.view(attention_map.shape[0], featmap_sz, featmap_sz, -1).permute(0, 3, 1, 2)  # .contiguous() # [B, 2P_kv, H, W]

        attention_map = F.interpolate(attention_map, size=(f_test_rgb.shape[-2], f_test_rgb.shape[-1]))            # B x 2P_kv x 4 x 4 ->  B x 2P_kv x 48 x 48
        pre_attn = F.interpolate(pre_attn, size=(f_test_rgb.shape[-2], f_test_rgb.shape[-1]))

        out = self.post_layers[layer](F.upsample(self.a_layers[layer](attention_map) + self.s_layers[layer](pre_attn), scale_factor=2))

        return out, feat_rgbd
