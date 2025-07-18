import math

import torch
import torch.nn.functional as F
from torch import nn

from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder


from mmocr.models.textrecog.recognizer.DeformableDETR.models.deformable_transformer import build_deforamble_transformer,DeformableTransformer
import numpy as np


def build_encoder():
    # parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    args = None
    # print("Ar")
    
    model  = DeformableTransformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
        num_feature_levels=3,
        enc_n_points=4)
    return model


@ENCODERS.register_module()
class Featurescale(BaseEncoder):
    """ Implement the PE function. """

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(Featurescale, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.encoder = build_encoder()
        self.pos = PositionEmbeddingSine()
        self.register_buffer('pe', pe)

    def forward(self, srcs, masks,**kwargs):
        
        pos_list =  []
        for src in srcs:
            pos = self.pos(src)
            pos_list.append(pos)
        # pos_list = self.pos(srcs)   # srcs:[3,512,30,30]    pos_list:[1,512,30,30]
        out_dec, mask = self.encoder(srcs,masks,pos_list) 
        # feat = feat + self.pe[:, :feat.size(1)] # pe 1*5000*512
        # return self.dropout(feat)
        return out_dec, mask

    def init_weights(self):
        pass

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list
        mask = torch.ones((x.shape[2],x.shape[3])).to(x.device)
        mask = mask.unsqueeze(0)
        not_mask = mask
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).unsqueeze(0)  #1224
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
