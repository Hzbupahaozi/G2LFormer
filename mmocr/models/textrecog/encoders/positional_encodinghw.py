import math

import torch
import torch.nn.functional as F
from torch import nn

from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder

import numpy as np






@ENCODERS.register_module()
class PositionEmbeddingSineHW(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, d_model,dropout=0., max_len=5000,num_pos_feats=256, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.encoder1 = build_encoder()
        # self.register_buffer('pe', pe)
    def forward(self, tensor_list):
        x = tensor_list
        # print("x:",x.shape)
        mask = torch.ones((x.shape[2],x.shape[3])).to(x.device)
        mask = mask.unsqueeze(0)
        # print("mask:",mask.shape)
        # print(mask)
        assert mask is not None
        not_mask = mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # import ipdb; ipdb.set_trace()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # print("HW:",pos_x.shape,pos_y.shape)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # print("HW:",pos.shape)
        # import ipdb; ipdb.set_trace()
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute((0,2,1))
        # print("feat:",feat.shape)
        # print("pos:",pos.shape)
        pos = pos.view(1, c, h*w).permute((0,2,1))  
        # print("pos:",pos.shape)
        # x = x + pos
        # return x
        return pos