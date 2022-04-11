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

        if mask is not None:
            attention_scores = attention_scores * mask.view(mask.shape[0], 1, 1, -1) # head, layers, Pq, P_kv * [1, B, 1, 1, P_kv]

        attention_probs = self.softmax(attention_scores) # [B,Heads, Pq, Pkv]

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)                  # [B, heads, Pq, Pk] * [B, heads, Pv, HeadSize]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()              # [B, Patches, heads, head_size]
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
        h = x
        x = self.attention_norm(x)
        if kv is not None:
            kv = self.attention_norm(kv)
        x, weights = self.attn(x, kv=kv, mask=mask)
        x = x + h

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
    def __init__(self, config, img_size, in_channels, vis, mask=False):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.encoder = Encoder(config, vis, patches=self.embeddings.n_patches)

        if mask:
            self.mask_pool = nn.MaxPool2d(config.patches['size'], stride=config.patches['size'])

    def forward(self, q_ids, kv=None, mask=None):
        embedding_q = self.embeddings(q_ids)  # [B, N_Patches, C=768]
        embeddings_kv = self.embeddings(kv) if kv is not None else None
        mask = self.mask_pool(mask).view(mask.shape[0], -1) if mask is not None else None

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
