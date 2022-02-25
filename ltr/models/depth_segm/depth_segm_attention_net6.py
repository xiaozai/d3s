
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
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attention_probs = self.softmax(attention_scores) # dim=-1 [B, head, P_q, P_k]
        # it has nan values
        #Flatten:
        shape = attention_probs.shape
        tensor_reshaped = attention_probs.reshape(shape[0],-1)
        #Drop all rows containing any nan:
        tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(),dim=1)]
        #Reshape back:
        attention_probs = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:])


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
        # self.act_fn = torch.nn.functional.gelu
        self.act_fn = torch.nn.ReLU(inplace=True) # pytorch 1.1.0 does not have gelu
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

        self.use_target_sz = use_target_sz

        if self.use_target_sz:
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
            self.sz_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))


        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):

        x = self.patch_embeddings(x) # [B, C, H/stride, W/stride]
        x = x.flatten(2)             # [B, C, H*W=patches]
        x = x.transpose(-1, -2)      # [B, tokens, C]

        if self.use_target_sz:
            B = x.shape[0]
            sz_tokens = self.sz_token.expand(B, -1, -1) # B, 1, C
            x = torch.cat((sz_tokens, x), dim=1) # [B, 1+N, C]

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

class Encoder(nn.Module):
    def __init__(self, config, vis, patches=64):
        super(Encoder, self).__init__()
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

class Decoder(nn.Module):
    def __init__(self, config, vis, patches=64):
        super(Decoder, self).__init__()
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


class Transformer(nn.Module):
    def __init__(self, config, img_size, in_channels, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis, patches=self.embeddings.n_patches)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)           # [B, N_Patches, C=768], class embeddings + patches
        encoded, attn_weights = self.encoder(embedding_output)  # encoded [B, patches, C=768]
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
        self.encoder = Encoder(config, vis, patches=self.n_patches)
        self.decoder = Decoder(config, vis, patches=self.n_patches)

    def forward(self, q_input, kv_input, mask=None):
        q_embeddings = self.embeddings(q_input)   # [B, N_Patches + 1, C], class embeddings + patches
        kv_embeddings = self.embeddings(kv_input) # [B, N_Patches + 1, C]
        if mask is not None:
            # mask = self.mask_embeddings(mask) # B, 1, H//patchsize, w//size
            mask = F.interpolate(mask, scale_factor=1/self.patch_sz)
            sz_mask = torch.ones(mask.shape[0], 1, 1)
            mask = torch.cat((sz_mask, mask.view(mask.shape[0], 1, -1)), dim=-1) # B, 1, patches+1
            mask = torch.tensor(mask, dtype=torch.bool) # B, 1, patches+1

        kv_encoded, kv_attn_weights = self.encoder(kv_embeddings)   # encoded [B, patches+1, C]
        q_encoded, q_attn_weights = self.decoder(q_embeddings, kv_embeddings, kv_embeddings, mask=mask)
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


class DepthSegmNetAttention06(nn.Module):
    def __init__(self):
        super().__init__()

        segm_input_dim = (64, 256, 512, 1024)
        segm_inter_dim = (4, 16, 32, 64)
        segm_dim = (64, 64)  # convolutions before cosine similarity
        feat_sz = (192, 96, 48, 24) # feature maps sizes in ResNet50 and DepthNet

        self.depth_feat_extractor = DepthNet(input_dim=1, inter_dim=segm_inter_dim)

        # feat_test/train, layer3, Bx24x24x1024 => B, 6x6 patches * 2, C=hidden_size
        patch_sz = 4
        n_patches = (feat_sz[3] // patch_sz) ** 2 * 2      # 72 patches
        init_config = get_config(size=(patch_sz, patch_sz))
        self.cross_attn = CrossAttentionTransformer(init_config, (feat_sz[3]*2, feat_sz[3]), segm_dim[1],
                                                    vis=True, use_target_sz=True)

        self.head = nn.Linear(init_config.hidden_size, 1)



        config = get_config(size=(12, 12))
        self.rgbd_transformers = nn.ModuleList([Transformer(config, (feat_sz[0]*2, feat_sz[0]), segm_inter_dim[0], True),  # 192x192x4 -> 16x16x4 patches
                                                Transformer(config, (feat_sz[1]*2, feat_sz[1]), segm_inter_dim[1], True),  # 96x94x16  -> 8x8x4 patches
                                                Transformer(config, (feat_sz[2]*2, feat_sz[2]), segm_inter_dim[2], True)]) # 48x48x32  -> 4x4x4 patches


        # 1024， 512， 256， 64 -> 64, 32, 16, 4
        self.segment0_rgb = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
        self.segment1_rgb = conv_no_relu(segm_dim[0], segm_dim[1]) # 64

        self.segment0_d = conv(64, 64, kernel_size=1, padding=0)
        self.segment1_d = conv_no_relu(64, 64)


        # project pre-out feat
        self.s_layers = nn.ModuleList([conv(segm_inter_dim[0], segm_inter_dim[0]),
                                       conv(segm_inter_dim[1], segm_inter_dim[1]),
                                       conv(segm_inter_dim[2], segm_inter_dim[2])])

        # project attention RGBD feat
        rgbd_channels = config.hidden_size * 2 # RGB + D
        self.a_layers = nn.ModuleList([conv(rgbd_channels, segm_inter_dim[0]),
                                       conv(rgbd_channels, segm_inter_dim[1]),
                                       conv(rgbd_channels, segm_inter_dim[2]),
                                       conv(rgbd_channels, segm_inter_dim[3])])

        # project RGB feat
        self.f_layers = nn.ModuleList([conv(segm_input_dim[0], segm_inter_dim[0]),
                                       conv(segm_input_dim[1], segm_inter_dim[1]),
                                       conv(segm_input_dim[2], segm_inter_dim[2])])

        # project D feat
        self.d_layers = nn.ModuleList([conv(segm_inter_dim[0], segm_inter_dim[0]),
                                       conv(segm_inter_dim[1], segm_inter_dim[1]),
                                       conv(segm_inter_dim[2], segm_inter_dim[2])])

        # project out feat
        self.post_layers = nn.ModuleList([conv(segm_inter_dim[0], 2),           # conv_no_relu(segm_inter_dim[0], 2),
                                          conv(segm_inter_dim[1], segm_inter_dim[0]),
                                          conv(segm_inter_dim[2], segm_inter_dim[1]),
                                          conv(segm_inter_dim[3]+1, segm_inter_dim[2])])


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
        ''' rgb features   : [conv1, layer1, layer2, layer3], Bx64x192x192  -> Bx256x96x96 -> Bx512x48x48 -> Bx1024x24x24
            depth features : [feat0, feat1, feat2, feat3],    Bx4x192x192 -> Bx16x96x96 -> Bx32x48x48 -> Bx64x24x24
        '''
        out, target_sz, attn_weights3 = self.init_mask(test_dist, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=3)
        out, attn_weights2 = self.rgbd_fusion(out, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=2)
        out, attn_weights1 = self.rgbd_fusion(out, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=1)
        out, attn_weights0 = self.rgbd_fusion(out, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=0)

        out = F.softmax(out, dim=1)

        if debug:
            return out, target_sz, (attn_weights3, attn_weights2, attn_weights1, attn_weights0)
        else:
            return out

    def init_mask(self, test_dist, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=3):
        f_test_rgb = self.segment1_rgb(self.segment0_rgb(feat_test_rgb[layer]))
        f_train_rgb = self.segment1_rgb(self.segment0_rgb(feat_train_rgb[layer]))
        f_test_d = self.segment1_d(self.segment0_d(feat_test_d[layer]))
        f_train_d = self.segment1_d(self.segment0_d(feat_train_d[layer]))

        mask = F.interpolate(mask_train[0], size=(f_train_rgb.shape[-2], f_train_rgb.shape[-1])) # Bx1xHxW
        mask = torch.cat((mask, mask), dim=2)

        template = torch.cat((f_train_rgb, f_train_d), dim=2)    # Bx64x24x24 -> Bx64x(24*4)x24
        search_region = torch.cat((f_test_rgb, f_test_d), dim=2)

        out, attn_weights3 = self.cross_attn(template, search_region, mask=mask) # B x Patches x C [rgb + d ]

        target_sz = torch.sigmoid(self.head(out[:, 0])) * 2 # BxC -> Bx1


        out = out[:, 1:, :]
        n_patches = out.shape[1] // 2  # RGB + D patches
        new_feat_sz = int(math.sqrt(n_patches))
        out = torch.cat((out[:, :n_patches, :], out[:, n_patches:, :]), dim=-1)                     # BxPatchesx2C
        out = out.view(out.shape[0], new_feat_sz, new_feat_sz, -1)                                  # BxHxWx2C, hidden_size * 2,  Bx6x6x192
        out = out.permute(0, 3, 1, 2)                                                               # Bx192x6x6

        out = self.a_layers[layer](out)                                                             # Bx192x6x6
        out = F.interpolate(out, size=(f_train_rgb.shape[-2]*2, f_train_rgb.shape[-1]*2))            # Bx192x48x48
        dist = F.interpolate(test_dist[0], size=(f_train_rgb.shape[-2]*2, f_train_rgb.shape[-1]*2))  # [B, 1, 48, 48]
        out = self.post_layers[layer](torch.cat((out, dist), dim=1))                                 # [B, 32, 48, 48]

        return out, target_sz, attn_weights3[:, :, 1:, :] # attn_weights = Head, Batch, P1, P2

    def rgbd_fusion(self, pre_out, feat_test_rgb, feat_test_d, feat_train_rgb, feat_train_d, mask_train, layer=0):

        f_test_rgb, f_test_d = feat_test_rgb[layer], feat_test_d[layer]
        # f_train_rgb, f_train_d = feat_train_rgb[layer], feat_train_d[layer]

        feat_rgbd = torch.cat((self.f_layers[layer](f_test_rgb), self.d_layers[layer](f_test_d)), dim=2)
        feat_rgbd, attn_weights = self.rgbd_transformers[layer](feat_rgbd)      # [B, Patches, C], attn_weights, [B, heads=12, patches, headsize=64]
        n_patches = feat_rgbd.shape[1] // 2                                     # for each patch, 16 patches, 64 patches, 256 patches
        featmap_sz = int(math.sqrt(n_patches))                                  # for RGB and D feat maps, 4x4, 8x8, 16x16
        # Only keep test F_rgb and F_D, [B, Patches//2=32/128x512, C=768]
        feat_rgbd = torch.cat((feat_rgbd[:, :n_patches, :], feat_rgbd[:, n_patches:, :]), dim=-1) # cat(f_rgb, f_d),  B x H x W x 2C
        feat_rgbd = feat_rgbd.view(feat_rgbd.shape[0], featmap_sz, featmap_sz, -1)
        feat_rgbd = feat_rgbd.permute(0, 3, 1, 2).contiguous() # [B, 2C, H, W]
        feat_rgbd = F.interpolate(feat_rgbd, size=(f_test_rgb.shape[-2], f_test_rgb.shape[-1]))              # B x 2C x 4 x 4 ->  B x 2C x 48 x 48
        out = self.post_layers[layer](F.upsample(self.a_layers[layer](feat_rgbd) + self.s_layers[layer](pre_out), scale_factor=2))

        return out, attn_weights
