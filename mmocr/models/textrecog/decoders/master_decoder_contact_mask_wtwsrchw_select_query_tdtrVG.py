import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmocr.models.builder import DECODERS
from .base_decoder import BaseDecoder
# from ..encoders.positional_encoding import PositionalEncoding

from mmocr.models.builder import DECODERS

class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
def clones(module, N):
    """ Produce N identical layers """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #tmp = self.norm(x)
        #tmp = sublayer(tmp)
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def self_attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scale Dot Product Attention'
    """
    # d_k = value.size(-1)
    d_k = key.size(-1)

    score = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    
    if mask is not None:
        # print("mask:",mask.shape)
        # score = score.masked_fill(mask == 0, -1e9) # b, h, L, L
        score = score.masked_fill(mask == 0, -6.55e4) # for fp16
     
    p_attn = F.softmax(score, dim=-1)
    # print("p_attn:",p_attn.shape)
    if dropout is not None:
        p_attn = dropout(p_attn)
    t = torch.matmul(p_attn, value)
    # print("v:",t.shape)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, headers, d_model, vdim, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % headers == 0
        self.headers = headers
        self.d_v = int(vdim / headers)
        self.linear = nn.Linear(vdim, vdim)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key= \
        #     [l(x).view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
        #      for l,x in zip(self.linears, (query, key))]
        # value  = self.linear_v(value).view(nbatches, -1, self.headers, self.d_v).transpose(1, 2)
        # 2) Apply attention on all the projected vectors in batchssss
        # print("q,k,v",query.shape,key.shape, value.shape)
    
        x, self.attn = self_attention(query, key, value, mask=mask, dropout=self.dropout)
        # print("x:",x.shape)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_v)
        # print("ok",x.shape)
    
        return self.linear(x),self.attn


class DecoderLayer(nn.Module):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout,rm_self_attn_decoder=False ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

        # Decoder Self-Attention
        # if not rm_self_attn_decoder:
        d_model = size 
        self.d_model = d_model
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        
        self.self_attn = MultiHeadAttention(headers = 8,d_model = d_model,  dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.src_attn = MultiHeadAttention(headers = 8, d_model = d_model,  dropout=dropout, vdim=d_model)
        # self.src_attn = MultiHeadAttention(headers = 8, d_model = d_model*2,  dropout=dropout, vdim=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, feature, src_mask, tgt_mask,
                pos, query_pos, query_sine_embed):
                
        # Apply projections here
        # shape: num_queries x batch_size x 256
        # tgt = x
        # print("tgt:",tgt.shape)
        headers = 8
        d_k = int(self.d_model / headers)
        nbatches = x.size(0)
        q = self.sa_qcontent_proj(x).view(nbatches, -1, headers, d_k).transpose(1, 2)      # target is the input of the first decoder layer. zero by default.
        k = self.sa_kcontent_proj(x).view(nbatches, -1, headers, d_k).transpose(1, 2)
        v = self.sa_v_proj(x).view(nbatches, -1, headers, d_k).transpose(1, 2)

        att_out,_ = self.self_attn(q, k, v, tgt_mask)
        # ========== End of Self-Attention =============
        x = x + self.dropout1(self.norm1(att_out))
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(x).view(nbatches, -1, headers, d_k).transpose(1, 2)
        k_content = self.ca_kcontent_proj(feature).view(nbatches, -1, headers, d_k).transpose(1, 2)
        v = self.ca_v_proj(feature).view(nbatches, -1, headers, d_k).transpose(1, 2)

        src_out,src_at= self.src_attn(q_content,k_content,v, src_mask)
        # ========== End of Cross-Attention =============
        x = x + self.dropout2(self.norm2(src_out))
        # print("src_at:",src_at.shape)
        # src_at = src_at.reshape(1,8,1,60,60)
        # print("x1:",x.shape)
        x = self.sublayer[2](x, self.feed_forward)
        # print("x2:",x.shape)
        return x,src_at

def visual():
    from visualize import visualize_grid_attention_v2
    import numpy as np

    img_path="/home/Dataset/huang/ch_no_3275341265_gjh_1.jpg"
    save_path="/home/Dataset/huang"
    attention_mask = np.random.randn(14, 14)
    visualize_grid_attention_v2(img_path,
                            save_path=save_path,
                            attention_mask=attention_mask,
                            save_image=True,
                            save_original_image=True,
                            quality=100)
    
class DecoderLayer1(nn.Module):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size=512,  dropout= 0.):
        super(DecoderLayer1, self).__init__()
        self.size = size
        self.feed_forward = FeedForward( d_model=512,
                d_ff=2024,
                dropout=0.)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

        # Decoder Self-Attention
        # if not rm_self_attn_decoder:
        d_model = size 
        self.d_model = d_model
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        
        self.self_attn = MultiHeadAttention(headers = 8,d_model = d_model,  dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.src_attn = MultiHeadAttention(headers = 8, d_model = d_model*2,  dropout=dropout, vdim=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, feature, src_mask, tgt_mask,
                pos, query_pos, query_sine_embed):
                
        # Apply projections here
        # shape: num_queries x batch_size x 256
        headers = 8
        d_k = int(self.d_model / headers)
        nbatches = x.size(0)
        q_content = self.sa_qcontent_proj(x)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(x)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(x).view(nbatches, -1, headers, d_k).transpose(1, 2)
        # print(q_content.shape,q_pos.shape)
        q = q_content + q_pos
        k = k_content + k_pos
        q = q.view(nbatches, -1, headers, d_k).transpose(1, 2)
        k = k.view(nbatches, -1, headers, d_k).transpose(1, 2)
        att_out,_ = self.self_attn(q, k, v, tgt_mask)
        # print("att_out:",att_out.shape)

        # ========== End of Self-Attention =============
        x = x + self.dropout1(self.norm1(att_out))

        # print("selfok")
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        feature = feature.permute(0,2,3,1)  # [3,512,30,30]变为[3,30,30,512]
        feature = feature.view(nbatches,-1,self.d_model)    # [3,900,512]
        q_content = self.ca_qcontent_proj(x)
        k_content = self.ca_kcontent_proj(feature)
        v = self.ca_v_proj(feature).view(nbatches, -1, headers, d_k).transpose(1, 2)
        _, hw, n_model= k_content.shape
        # print(nbatches, hw, n_model)
        # print("pos:",pos.shape)
        pos = pos.view(1, hw, n_model)  # [1,900,512]
        poslist = pos.repeat(nbatches,1,1)
        # poslist = pos
        # for i in range(nbatches-1):
        #     poslist = torch.cat([poslist, pos], dim=0)
        pos = poslist   # [3,900,512]
        # print("pos:",pos.shape)
        k_pos = self.ca_kpos_proj(pos)   #image_pos

        q = q_content
        k = k_content
        # print("q",q.shape)
        # print("k",k.shape)
        # q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        # print("query_sine_embed:",query_sine_embed.shape)
        q = torch.cat([q, query_sine_embed], dim=2).view(nbatches, -1, headers, d_k * 2).transpose(1, 2)
        k = torch.cat([k, k_pos], dim=2).view(nbatches,hw, headers, d_k * 2).transpose(1, 2)
        # print("k:",k.shape) 
        src_out,src_at = self.src_attn(q,k,v, src_mask)
        # src_at = src_at.reshape(1,8,499,60,60)
        # print("src_at:",src_at.shape)
        # print(src_at[0,0,0,:10,:10])
        # ========== End of Cross-Attention =============
        x = x + self.dropout2(self.norm2(src_out))
        return self.sublayer[2](x, self.feed_forward),src_at

class PositionEmbeddingSineHW(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=256, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
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
        pos = torch.cat((pos_y, pos_x), dim=3)#.permute(0, 3, 1, 2)
        # print("HW:",pos.shape)
        # import ipdb; ipdb.set_trace()

        return pos


class PositionalEncoding(nn.Module):                #1224
    """ Implement the PE function. """

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, feat, **kwargs):
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h*w) # flatten 2D feature map
            feat = feat.permute((0,2,1))
        # print("pe:",feat.shape,self.pe.shape)
        feat = feat + self.pe[:, :feat.size(1)] # pe 1*5000*512
        return self.dropout(feat)

    def init_weights(self):
        pass

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(256, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 50 ** (2 * (dim_t // 2) / 128)
    # print("posten",pos_tensor.shape)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        # print("pos_x:",pos_x.shape)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        # print("pos:",pos.shape)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
@DECODERS.register_module()
class TableMasterDecoder(BaseDecoder):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 bbox_embed_diff_each_layer = False,  #DAB
                 query_scale_type = "cond_elewise"
                 ):
        super(TableMasterDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), 1)
        self.cls_layer = clones(DecoderLayer(**decoder), 1)
        self.bbox_one = clones(DecoderLayer(**decoder), 2)
        self.num_layers = 2
        # self.bbox_layer1 = DecoderLayer1()
        # self.bbox_layer2 = DecoderLayer1()
        self.bbox_layer = clones(DecoderLayer1(), 2)
        # self.bbox_layer2 = DecoderLayer1()
        # self.bbox_layer = nn.ModuleList([self.bbox_layer1, self.bbox_layer2])
        # self.bbox_layer = nn.ModuleList([self.bbox_layer1])
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(
            nn.Linear(d_model, 4),
        )
        self.bbox_fc1 = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(decoder.size)
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.pos_target = PositionalEncoding(d_model=d_model)
        self.d_model = d_model
        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len
        #DAB-DETR
        self.query_dim = 4
        num_layers =1
        self.positional_encoding = PositionEmbeddingSineHW()
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.layers = clones(DecoderLayer(**decoder), num_layers)  
        self.ref_point_head = MLP( self.query_dim  // 2*d_model  , d_model, d_model, 4)  #2
        self.bbox_embed = MLP(d_model,d_model,4,3)
        assert  query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))
        tgt_mask = trg_pad_mask & trg_sub_mask
        # print("tgt_mask:",tgt_mask)
        return None, tgt_mask

    def decode(self, input, feat, feature, src_mask, tgt_mask, bbox_expand = None, bbox_masks =None):
        # main process of transformer decoder.
        x = self.embedding(input)
        pos = self.positional_encoding(feat)
        x = self.pos_target(x)
        # print("pos:",pos.shape)
        # print("feat:",feat.shape,feature.shape)
        # x_list = []
        cls_x_list = []
        bbox_x_list = []
        output_list = []
        for layer_id, layer in enumerate(self.layers):
            x = layer(x = x, feature=feature, tgt_mask=tgt_mask,
                           src_mask=src_mask,
                           pos=None, query_pos=None, query_sine_embed=None,
                           )
        # cls head
        cls_x = x
        for layer in self.cls_layer:
            cls_x = layer(x = cls_x, feature=feature, tgt_mask=tgt_mask,
                           src_mask=src_mask,
                           pos=None, query_pos=None, query_sine_embed=None,
                          )
            cls_x_list.append(cls_x)
        cls_x = torch.cat(cls_x_list, dim=-1)
        cls_x = self.norm(cls_x)

        # print("output",output.shape)
        
        # bbox head
        # bbox_x = x
        for layer in self.bbox_one:
            bbox_x = layer(x = x, feature=feature, tgt_mask=tgt_mask,
                           src_mask=src_mask,
                           pos=None, query_pos=None, query_sine_embed=None,
                          )
            bbox_x_list.append(bbox_x)
        bbox_x = torch.cat(bbox_x_list, dim=-1)
        bbox_x = self.norm(bbox_x)
        
        bbox_output = self.bbox_fc1(bbox_x).clamp(min=0.0, max=1.0)
        output_list.append(bbox_output)
        reference_points = bbox_output#.detach()  #1019
        ref_points = [reference_points]
        
        for layer_id, layer in enumerate(self.bbox_layer):
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 4]取前2维，作为左上锚点
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)  
            query_pos = self.ref_point_head(query_sine_embed) 
            # print("query_pos:",query_pos.shape)
            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:    
                    pos_transformation = self.query_scale(x)  #？？？？ 猜测应该为bbox
                    # print(layer_id ,pos_transformation.shape )
            # else:
            #     pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation

            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation
            # print("query_sine_embed:",query_sine_embed.shape)
            # print("bboxmask:",bbox_masks[:,:,:20,:20])
            
            # print("mask:",bbox_masks.shape, tgt_mask.shape)   
            bbox_x = layer(x = bbox_x, feature=feat, tgt_mask=tgt_mask,           #feat :feature  feature:out_enconding
                        src_mask=src_mask,
                        pos=pos, query_pos=query_pos, query_sine_embed =query_sine_embed,
                        )
            
            
        # ref_points.append(new_reference_points)
        #     update
        #update
            if self.bbox_embed  is not None:
                # if self.bbox_embed_diff_each_layer:
                #     tmp = self.bbox_fc[layer_id](output)
                # else:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                bbox_x = self.norm(bbox_x)
                bbox_output = self.bbox_embed(bbox_x )
                
            #     # import ipdb; ipdb.set_trace()
                bbox_output  = bbox_output  + reference_before_sigmoid
                bbox_output = bbox_output.sigmoid()
                # bbox_output = bbox_output.clamp(min=0.0, max=1.0)
                new_reference_points = bbox_output[..., :self.query_dim]#.sigmoid()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)
                output_list.append(bbox_output) 
                reference_points = new_reference_points.detach()
        dn_out = []
        if(bbox_expand != None):
            # print("bbox:",bbox_output.shape)
            for i in range(bbox_expand.shape[0]):
                bbox_1 = bbox_x.clone()
                # print("bbox:", bbox_expand.shape,i)
                reference_points1  = bbox_expand[i,:,:-1, :self.query_dim].sigmoid()
                
                for layer_id, layer in enumerate(self.bbox_layer):
                    obj_center = reference_points1[..., :self.query_dim]     # [num_queries, batch_size, 4]取前2维，作为左上锚点
                    # get sine embedding for the query vector
                    query_sine_embed = gen_sineembed_for_position(obj_center)  
                    query_pos = self.ref_point_head(query_sine_embed) 
                    # print("query_pos:",query_pos.shape)
                    # For the first decoder layer, we do not apply transformation over p_s
                    if self.query_scale_type != 'fix_elewise':
                        if layer_id == 0:
                            pos_transformation = 1
                        else:    
                            pos_transformation = self.query_scale(x)
                            # print(layer_id ,pos_transformation.shape )
                    else:
                        pos_transformation = self.query_scale.weight[layer_id]

                    # apply transformation

                    query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation
                    # print("query_sine_embed:",query_sine_embed.shape)
        
                    # print("mask:",bbox_masks.shape, tgt_mask.shape)   
                    bbox_1 = layer(x = bbox_1, feature=feat, tgt_mask=tgt_mask,           #feat :feature  feature:out_enconding
                                src_mask=src_mask,
                                pos=pos, query_pos=query_pos, query_sine_embed =query_sine_embed,
                                )              
            # ref_points.append(new_reference_points)
            #update
                    if self.bbox_embed  is not None:
                        # if self.bbox_embed_diff_each_layer:
                        #     tmp = self.bbox_fc[layer_id](output)
                        # else:
                        reference_before_sigmoid = inverse_sigmoid(reference_points1)
                        bbox_1 = self.norm(bbox_1)
                        bbox_output = self.bbox_embed (bbox_1)
                        bbox_output  = bbox_output  + reference_before_sigmoid
                        bbox_output = bbox_output.sigmoid()
                        # bbox_output  = bbox_output.clamp(min=0.0, max=1.0)
                        dn_out.append(bbox_output) 
                        new_reference_points = bbox_output[..., :self.query_dim]#.sigmoid() 
                        reference_points1 = new_reference_points.detach()

        # print("bboxmask:",bbox_masks.shape,tgt_mask.shape)
        # bbox_masks = bbox_masks.unsqueeze(1).unsqueeze(-1)   #test
        # # bbox_masks = bbox_masks[:,:-1].unsqueeze(1).unsqueeze(-1)
        # # print("bboxmask1:",bbox_masks.shape,tgt_mask.shape)
        # bbox_masks = bbox_masks.repeat(1,1,1,tgt_mask.shape[3]).to(feat.device)
        # # print("bboxmask2:",bbox_masks.shape,tgt_mask.shape)
        # bbox_masks = tgt_mask & bbox_masks
        # bbox_masks =tgt_mask
        # print("bboxmask3:",bbox_masks.shape)
       
            # bbox_output = bbox_output.sigmoid() 
        # bbox_output = inverse_sigmoid(bbox_output)
        # print(ref_points[0][:,:40])
        #     if self.return_intermediate:        #zhongjian jieguo
        #         intermediate.append(self.norm(output))
        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
                # intermediate.append(output)
        # reference_before_sigmoid = inverse_sigmoid(new_reference_points)
        # print(reference_before_sigmoid[:,:20])
        # print(new_reference_points[:,:20])
        # bbox_output = new_reference_points
        # bbox_output[:,:,0] = bbox_output[:,:,0] +bbox_output[:,:,2]/2
        # bbox_output[:,:,1] = bbox_output[:,:,1] +bbox_output[:,:,3]/2
        # bbox_output = None
        # pos_sine_embed = gen_sineembed_for_position(bbox_output)
        # bbox_pos = self.ref_point_head(pos_sine_embed)
        # if self.return_intermediate:
        #     if self.bbox_embed is not None:
        #         return [
        #             torch.stack(intermediate).transpose(1, 2),
        #             torch.stack(ref_points).transpose(1, 2),
        #         ]
        #     else:
        #         return [
        #             torch.stack(intermediate).transpose(1, 2), 
        #             reference_points.unsqueeze(0).transpose(1, 2)
        #         ]

        return self.cls_fc(cls_x), output_list, dn_out

    def greedy_forward(self, SOS, feat, feature, mask, text_padded_target = None):
        input = SOS
        output = None
        device = feat.device
        input= input.to(device)
        batch_size = feat.shape[0]
        bbox_list = torch.Tensor([0., 0., 0., 0.]).float().unsqueeze(0).unsqueeze(0)
        bbox_masks = torch.LongTensor([0]).unsqueeze(0)
        # text_padded_target = text_padded_target.to(device)
        sum, num = 0,0
        bbox_list = bbox_list.to(device)
        for i in range(self.max_length+1):
            # input  = text_padded_target[:,:i+1]  #gt
            _, target_mask = self.make_mask(feature, input)          
            out, bbox_output,DN = self.decode(input, feat, feature, None, target_mask,bbox_masks=bbox_masks)
            
            # print("input:",input)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            token = next_word[:, -1]
            if(token ==1 and token==3): 
                bbox_mask  = torch.LongTensor([1]).byte().unsqueeze(0)
            else:   bbox_mask  = torch.LongTensor([0]).byte().unsqueeze(0)
            bbox_masks = torch.cat([bbox_masks, bbox_mask], dim=1)
            # print("bbox_masks:",bbox_masks.shape,bbox_masks)
            # bbox_output = bbox_output + bbox_list[:-1]   #上一个点的右上角加偏移组成一个框   
            #point
            # reference_points = refpoints_unsigmoid.sigmoid()  
            # print(len(bbox_output))       
            bbox = bbox_output[2][:, -1].unsqueeze(1)
            bbox_list = torch.cat([bbox_list, bbox], dim=1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
            # if(token==34): break      #跳出测试推理
 
        # bbox_list =None
        # bbox_list = torch.cat([bbox_list, bbox], dim=1)
        return output, bbox_output
    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
      
        # D = torch.distributions.Categorical(probs=data.adj)
        #     sample = D.sample(sample_shape=[20])
        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)
        bbox_list = targets_dict["bbox"].to(device)
        scalar = 2
        known_bboxs = bbox_list.repeat(scalar,1,1,1)
        target = targets_dict["targets"]
        # print("known_bboxs:",known_bboxs.shape)
        # print("target:",target[0].shape,target)
        # print("bbox:",bbox_list[0].shape, bbox_list[:,:20])
        # print("target:",targets_dict)
        #  dn 
        box_noise_scale = 0.25
        diff = torch.zeros_like(known_bboxs)
        sp = diff.shape
        p = torch.rand(sp)
        p[p<0.2] = 0 
        # print("p:",p)
        diff[:,:,:, :2] = known_bboxs[:,:, :,2:] / 2
        diff[:,:,:, 2:] = known_bboxs[:,:, :,2:]
        diff = torch.mul(diff,p).cuda() 
        box_noise =  torch.mul((torch.rand_like(known_bboxs) * 2 - 1.0),
                                        diff).cuda() * box_noise_scale
        # length = len(box_noise[0,0])
        # print(length,len(target))
        # print("shape:",box_noise.shape)
        # print(box_noise[:,:20,:4])
        # for l in range(scalar):
        # for k in range(feat.shape[0]):
        #     for i in range(1,length):
        #         if(targets_dict['padded_targets'][k,i]!=3):
        #             box_noise[:,k,i,:1] = box_noise[:,k,i,:1]+box_noise[:,k,i-1,:1]
        # print("NOISE:",box_noise[:,:20,:4])
        bbox_expand = known_bboxs + box_noise
        # torch.mul((torch.rand_like(known_bboxs) * 2 - 1.0),                          diff).cuda() * box_noise_scale
        bbox_expand = bbox_expand.clamp(min=0.0, max=1.0)
        # bbox_expand = None
        # print("noise_bbox:",bbox_expand.shape,bbox_expand[:,:20])
      
        # print(targets_dict["bbox_masks"])
        # targets_dict["bbox_masks"]
        # batch_size = 1
        # bbox = bbox.to(device)
        # # print("bbox:",bbox_list)
        # bbox_list = torch.cat([bbox,bbox_list],dim=1)
        # print("bbox:",bbox_list)
        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])#pad_target
        # print("tgt_mask:",tgt_mask.shape)
        # print( tgt_mask[0,0,length,:50])
        # tgt_mask_pad = tgt_mask[0,0,length,:].repeat(scalar, 1)
        # tgt_mask = torch.cat([tgt_mask, tgt_mask_pad], dim=2)
        # print("tgt_mask_pad:",tgt_mask_pad)
     
        return self.decode(padded_targets[:, :-1], feat, out_enc, src_mask, tgt_mask, bbox_expand, targets_dict["bbox_masks"])#

    def forward_test(self, feat, out_enc, targets_dict, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        # text_padded_target =  targets_dict['padded_targets']
        text_padded_target = None
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output, bbox_output = self.greedy_forward(SOS, feat, out_enc, src_mask,text_padded_target )
        # print("len:",len(bbox_output))
        return output, bbox_output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, targets_dict, img_metas)
def prepare_for_dn( num_classes, labels, boxes,targets_dict,label_noise_scale=0.1,box_noise_scale= 0.25,scalar= 2):
    """
    prepare for dn components in forward function
    Args:
        num_classes: number of classes
        label_enc: label encoding embedding
        scalar  deno groups
    Returns: input_query_label, input_query_bbox, attn_mask, mask_dict
    """
    # add noise
    device = boxes.device
    known_labels = labels.repeat(scalar,1,1).view(-1)
    known_bboxs = boxes.repeat(scalar,1,1,1)
    known_labels_expaned = known_labels.clone()
    known_bbox_expand = boxes.clone()
    # noise on the label
    if label_noise_scale > 0:
        p = torch.rand_like(known_labels_expaned.float())
        chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)# usually half of bbox noise
        new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
        known_labels_expaned.scatter_(0, chosen_indice, new_label)
        known_labels_expaned = known_labels_expaned.view(scalar,labels.shape[0],labels.shape[1])
    
    # noise on the box
    if box_noise_scale > 0:
        box_noise_scale = 0.25
        diff = torch.zeros_like(known_bboxs)
        sp = diff.shape[:-1]
        p  = torch.rand(sp).to(device)
        p[p<0.25] = 0
        p[p>=0.25]= 1
        # print(p.shape)
        p = p.unsqueeze(-1)
        p = p.repeat(1,1,1,4)
        # print(p.shape)
        diff[:,:,:, :2] = known_bboxs[:,:, :,2:] / 2
        diff[:,:,:, 2:] = known_bboxs[:,:, :,2:]
        diff = torch.mul(p,diff).to(device)
        box_noise =  torch.mul((torch.rand_like(known_bboxs) * 2 - 1.0),
                                        diff).to(device) * box_noise_scale
        bbox_expand = known_bboxs + box_noise
        known_bbox_expand = bbox_expand.clamp(min=0.0, max=1.0)

    #noise on row/col    
    # avg_row,avg_col,pos = targets_dict["avg_row"],targets_dict["avg_col"],targets_dict["pos"]
    # batch  = len(avg_row)
    # for b in range(batch):
    #     diff_r = torch.zeros((scalar,max(avg_row[b].keys())+1)).cuda()
    #     diff_c = torch.zeros((scalar,max(avg_col[b].keys())+1)).cuda()
    #     cp = diff_c.shape
    #     rp = diff_r.shape
    #     pr = torch.rand(rp).cuda()
    #     pr[pr<0.25] = 0
    #     pr[pr>=0.25]= 1
    #     pc  = torch.rand(cp).cuda()
    #     pc[pc<0.25] = 0
    #     pc[pc>=0.25]= 1
    #     # print(rp,cp,avg_row[b],avg_col[b])
    #     for i in avg_row[b].keys(): diff_r[:,i] = avg_row[b][i]
    #     diff_r = torch.mul(pr,diff_r).cuda()
    #     diff_r = torch.mul((torch.rand_like(diff_r)*2-1.0),diff_r)*box_noise_scale
    #     for i in avg_col[b].keys(): diff_c[:,i] = avg_col[b][i]
    #     diff_c = torch.mul(pc,diff_c).cuda()
    #     diff_c = torch.mul((torch.rand_like(diff_c)*2-1.0),diff_c)*box_noise_scale
    #     l = len(pos[b])
    #     # print("pos:",l,pos[b])
    #     for i in range(l):
    #         # print(i, pos[b][i])
    #         try:
    #             known_bbox_expand[:,b,i,1] = known_bbox_expand[:,b,i,1]+diff_r[:,pos[b][i][0]]
    #             known_bbox_expand[:,b,i,0] = known_bbox_expand[:,b,i,0]+diff_c[:,pos[b][i][1]]
    #         except IndexError:
    #             print('pass noise on row/col')
    #             continue

    return known_labels_expaned, known_bbox_expand

@DECODERS.register_module()
class TableMasterConcatDecoder(BaseDecoder):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 bbox_embed_diff_each_layer = False,  #DAB
                 query_scale_type = "cond_elewise"
                 ):
        super(TableMasterConcatDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), 1)
        self.layers_select = clones(DecoderLayer(**decoder), 1)
        self.cls_layer = clones(DecoderLayer(**decoder), 1)
        self.bbox_one = clones(DecoderLayer(**decoder), 2)
        self.num_layers = 2
        self.visual_linear_col = nn.Linear(d_model,d_model)
        self.visual_linear_row = nn.Linear(d_model,d_model)
        self.visual_linear_tr = nn.Linear(d_model, d_model)
        # self.bbox_layer1 = DecoderLayer1()
        # self.bbox_layer2 = DecoderLayer1()
        self.bbox_layer = clones(DecoderLayer1(), 2)
        self.num_classes = num_classes
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(
            nn.Linear(d_model, 4),
        )
        self.bbox_fc1 = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(decoder.size)
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        # self.embedding_box = nn.Embedding(max_seq_len, d_model)
        self.pos_target = PositionalEncoding(d_model=d_model)
        self.d_model = d_model
        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len
        #DAB-DETR
        self.query_dim = 4
        num_layers =1
        self.positional_encoding = PositionEmbeddingSineHW(temperatureH=20,temperatureW=20,normalize=True)
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.layers = clones(DecoderLayer(**decoder), num_layers)  
        self.ref_point_head = MLP( self.query_dim  // 2*d_model  , d_model, d_model, 4)  #2
        self.bbox_embed = MLP(d_model,d_model,4,3)
        self.modulate_hw_attn  = True
        if self.modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        assert  query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        td_pad_mask = (tgt == 1) | (tgt == 3)
        tr_pad_mask = (tgt==0) | (tgt==2)

        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))
        tgt_mask = trg_pad_mask & trg_sub_mask
        ##############################################下面是local attention的设计
        tensors = tgt_mask.clone().detach()
        matrix1 = tensors[:, 0] 

        mask = torch.any(matrix1 == 1, dim=1)  # mask形状为[3, 499]

        for i in range(matrix1.shape[0]):  # 对每个张量单独操作
            for col in range(matrix1.shape[2]):
                if mask[i, col]:
                    matrix1[i, :, col] = 0
                    end_idx = min(col + 350, matrix1.shape[1])
                    matrix1[i, col:end_idx, col] = 1

        tgt_mask = tgt_mask * tensors       
        return None, tgt_mask, td_pad_mask, tr_pad_mask

    def decode(self, input, feature_select,td_masks,tr_masks, feat, feature, src_mask, src_mask_origin, tgt_mask, bbox_expand = None, bbox_masks =None):

        x = self.embedding(input)
        x = self.pos_target(x)
        pos = self.positional_encoding(feat)    
        
        # x_list = []
        src_mask = src_mask.unsqueeze(dim=1).unsqueeze(dim=1)
        src_mask = src_mask.repeat(1,1,tgt_mask.shape[2],1)
        src_mask_origin = src_mask_origin.unsqueeze(dim=1).unsqueeze(dim=1)
        src_mask_origin = src_mask_origin.repeat(1,1,tgt_mask.shape[2],1)
        cls_x_list = []
        bbox_x_list = []
        output_list = []
        att_map = []
        for layer_id, layer in enumerate(self.layers_select):
            x, _ = layer(x = x, feature=feature_select, tgt_mask=tgt_mask,
                         src_mask=None,
                         pos=None, query_pos=None, query_sine_embed=None,
                         )
        x = x

        for layer_id, layer in enumerate(self.layers):
            x,_ = layer(x = x, feature=feature, tgt_mask=tgt_mask,
                           src_mask=src_mask,
                           pos=None, query_pos=None, query_sine_embed=None,
                           )
        # cls head
        cls_x = x
        for layer in self.cls_layer:
            cls_x,_ = layer(x = cls_x, feature=feature, tgt_mask=tgt_mask,
                           src_mask=src_mask,
                           pos=None, query_pos=None, query_sine_embed=None,
                          )
            cls_x_list.append(cls_x)
        cls_x = torch.cat(cls_x_list, dim=-1)
        cls_x = self.norm(cls_x)
        td_masks = td_masks.unsqueeze(-1).expand_as(cls_x)
        tr_masks = tr_masks.unsqueeze(-1).expand_as(cls_x)

        col_query = self.visual_linear_col(cls_x)
        col_query = col_query*td_masks
        row_query = self.visual_linear_row(cls_x)
        row_query = row_query*td_masks
        tr_query = self.visual_linear_tr(cls_x)
        tr_query = tr_query*tr_masks

        if(bbox_masks==None):
            return self.cls_fc(cls_x),None,None

        bbox_masks = bbox_masks[:,1:].cuda()  #1031
        bbox_masks = bbox_masks.unsqueeze(dim=1).unsqueeze(dim=1)
        bbox_masks = bbox_masks.repeat(1,1,599,1)
        w = 350

        matrix = bbox_masks[:, 0]  

        mask = torch.all(matrix == 1, dim=1)  

        for i in range(matrix.shape[0]):  # 对每个张量单独操作
            for col in range(matrix.shape[2]):
                if mask[i, col]:
                    matrix[i, :, col] = 0
                    end_idx = min(col + w, matrix.shape[1])
                    matrix[i, col:end_idx, col] = 1

        for layer in self.bbox_one:

            x,att_map = layer(x = x, feature=feature, tgt_mask=bbox_masks,            #1105
                           src_mask=src_mask,
                           pos=None, query_pos=None, query_sine_embed=None,
                          )
        bbox_x_list.append(x)
        bbox_x = torch.cat(bbox_x_list, dim=-1)
        bbox_x = self.norm(bbox_x)
        

        dn_out,cls_dn_out = [], []
        if(bbox_expand != None):
            
            known_labels_expaned, bbox_expand = bbox_expand[0],bbox_expand[1]
            
            for i in range(known_labels_expaned.shape[0]):
                dn_x = self.embedding(known_labels_expaned[i])
                dn_x = self.pos_target(dn_x)
                for layer_id, layer in enumerate(self.layers):
                    dn_x,_ = layer(x = dn_x, feature=feature, tgt_mask=tgt_mask,
                                src_mask=src_mask,
                                pos=None, query_pos=None, query_sine_embed=None,
                                )
                # cls head
                cls_x_dn = dn_x
                cls_dn_list = []
                for layer in self.cls_layer:
                    cls_x_dn,_ = layer(x = cls_x_dn, feature=feature, tgt_mask=tgt_mask,
                                src_mask=src_mask,
                                pos=None, query_pos=None, query_sine_embed=None,
                                )
                    cls_dn_list.append(cls_x_dn)
                cls_x_dn = torch.cat(cls_dn_list, dim=-1)
                cls_x_dn = self.norm(cls_x_dn)
                cls_dn_out.append(self.cls_fc(cls_x_dn))

            for i in range(bbox_expand.shape[0]):
                bbox_1 = bbox_x.clone()
                # print("bbox:", bbox_expand.shape,i)
                reference_points1  = bbox_expand[i,:,:-1, :self.query_dim].sigmoid()    # [3,599,4]
                for layer_id, layer in enumerate(self.bbox_layer):
                    obj_center = reference_points1[..., :self.query_dim]     # [num_queries, batch_size, 4]取前2维，作为左上锚点[3,599,4]
                    # get sine embedding for the query vector
                    query_sine_embed = gen_sineembed_for_position(obj_center)   # [3,599,1024]
                    query_pos = self.ref_point_head(query_sine_embed)   # [3,599,512]

                    if self.query_scale_type != 'fix_elewise':
                        if layer_id == 0:
                            pos_transformation = 1
                        else:    
                            pos_transformation = self.query_scale(bbox_1)
                            # print(layer_id ,pos_transformation.shape )
                    else:
                        pos_transformation = self.query_scale.weight[layer_id]

                    # apply transformation

                    query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation #[3,599,512]

                    # modulated HW attentions
                    if self.modulate_hw_attn:
                        refHW_cond = self.ref_anchor_head(bbox_1).sigmoid() # nq, bs, 2
                        query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                        query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
                    # print("query_sine_embed:",query_sine_embed.shape)
        
                    # print("mask:",bbox_masks.shape, tgt_mask.shape)   
                    bbox_1,_ = layer(x = bbox_1, feature=feat, tgt_mask=bbox_masks,           #feat :feature  feature:out_enconding
                                src_mask=src_mask_origin,
                                pos=pos, query_pos=query_pos, query_sine_embed =query_sine_embed,
                                )              

                    if self.bbox_embed  is not None:
                        reference_before_sigmoid = inverse_sigmoid(reference_points1)
                        bbox_1 = self.norm(bbox_1)
                        bbox_output = self.bbox_embed (bbox_1)
                        bbox_output  = bbox_output  + reference_before_sigmoid
                        bbox_output = bbox_output.sigmoid()
                        # bbox_output  = bbox_output.clamp(min=0.0, max=1.0)
                        dn_out.append(bbox_output) 
                        new_reference_points = bbox_output[..., :self.query_dim]#.sigmoid() 
                        reference_points1 = new_reference_points.detach()


        bbox_output = self.bbox_fc1(bbox_x).clamp(min=0.0, max=1.0)
        output_list.append(bbox_output) 
        reference_points = bbox_output.detach()  #1019
        ref_points = [reference_points]
        # print("bbox_x:",bbox_x.shape)
        # bbox_x = None
        # 
        att_list = [att_map]
        for layer_id, layer in enumerate(self.bbox_layer):
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 4]取前2维，作为左上锚点
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)  
            query_pos = self.ref_point_head(query_sine_embed) 
            # print("query_pos:",quesry_pos.shape)
            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:    
                    pos_transformation = self.query_scale(bbox_x)
                    # print(layer_id ,pos_transformation.shape )
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation

            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation
            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(bbox_x).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
            
            # print("mask:",bbox_masks.shape, tgt_mask.shape)   
            bbox_x,_ = layer(x = bbox_x, feature=feat, tgt_mask=bbox_masks,           #feat :feature  feature:out_enconding
                        src_mask=src_mask_origin,
                        pos=pos, query_pos=query_pos, query_sine_embed =query_sine_embed,
                        )
            
            if self.bbox_embed  is not None:
                # if self.bbox_embed_diff_each_layer:
                #     tmp = self.bbox_fc[layer_id](output)
                # else:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                bbox_x = self.norm(bbox_x)
                bbox_output = self.bbox_embed(bbox_x )
                
            #     # import ipdb; ipdb.set_trace()
                bbox_output  = bbox_output  + reference_before_sigmoid
                bbox_output = bbox_output.sigmoid()
                # bbox_output = bbox_output.clamp(min=0.0, max=1.0)
                new_reference_points = bbox_output[..., :self.query_dim]#.sigmoid()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)
                output_list.append(bbox_output) 
                reference_points = new_reference_points.detach()
        
        
        # return self.cls_fc(cls_x), output_list, (cls_dn_out,dn_out), col_query, row_query
        return self.cls_fc(cls_x), output_list, (cls_dn_out,dn_out) # 推理的时候使用
    

    def greedy_forward(self, SOS, feature_select, feat, feature, src_mask =None, src_mask_origin=None, img_metas =None ):
       
        input = SOS
        output = None
        device = feat.device
        input= input.to(device)
        batch_size = feat.shape[0]
        bbox_list = torch.Tensor([0., 0., 0., 0.]).float().unsqueeze(0).unsqueeze(0)
        bbox_masks = torch.LongTensor([0]).unsqueeze(0)
        # text_padded_target = text_padded_target.to(device)
        sum, num = 0,0
        bbox_list = bbox_list.to(device)
        flag = 0 
        for i in range(self.max_length-1):
            # input  = text_padded_target[:,:i+1]  #gt
            _, target_mask, td_pad_mask, tr_pad_mask = self.make_mask(feature, input)          
            out, bbox_output,dn = self.decode(input, feature_select, td_pad_mask, tr_pad_mask, feat, feature, src_mask, src_mask_origin, target_mask,bbox_masks=None)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            token = next_word[:, -1]
            
            if(flag ==0):
                if(token==25): flag =1   #1128
                if(token ==1 or token==3 or token==0 or token==2) and (flag==0): 
                    bbox_mask  = torch.LongTensor([1]).byte().unsqueeze(0)
                else:   bbox_mask  = torch.LongTensor([0]).byte().unsqueeze(0)
            bbox_masks = torch.cat([bbox_masks, bbox_mask], dim=1)
                
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        # print("intput:",input)
        out, bbox_output,dn = self.decode(input[:,:-1], feature_select, td_pad_mask, tr_pad_mask, feat, feature, src_mask, src_mask_origin, target_mask,bbox_masks=bbox_masks)
        # print(att_map[0,0,0,:10,:10])
        
        return output, bbox_output
    def forward_train(self, feature_select, feat, out_enc, targets_dict, src_mask, src_mask_origin, img_metas=None):

        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)
        
        bbox_list = targets_dict["bbox"].to(device)

        target = targets_dict["targets"]

        known_labels_expaned, known_bbox_expand = prepare_for_dn(self.num_classes-3,padded_targets[:, :-1],bbox_list,targets_dict,label_noise_scale=0.1,box_noise_scale= 0.25,scalar= 2)
        bbox_expand = known_bbox_expand

        _, tgt_mask, td_masks, tr_masks = self.make_mask(out_enc, padded_targets[:,:-1])

        return self.decode(padded_targets[:, :-1], feature_select,td_masks,tr_masks, feat, out_enc, src_mask, src_mask_origin, tgt_mask, (known_labels_expaned,bbox_expand), targets_dict["bbox_masks"])#

    def forward_test(self, feature_select, feat, out_enc, targets_dict, src_mask, src_mask_origin, img_metas):
        # src_mask = None
        batch_size = out_enc.shape[0]

        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        # print("s:",src_mask)
        output, bbox_output = self.greedy_forward(SOS, feature_select, feat, out_enc, src_mask, src_mask_origin, img_metas=img_metas )

        return output, bbox_output

    def forward(self,
                feature_select,
                feat,
                out_enc,
                targets_dict=None,
                src_mask =None,
                src_mask_origin =None,
                img_metas=None,
                train_mode=True,
                ):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feature_select, feat, out_enc, targets_dict, src_mask, src_mask_origin, img_metas)

        return self.forward_test(feature_select, feat, out_enc, targets_dict,src_mask, src_mask_origin, img_metas)


@DECODERS.register_module()
class MasterDecoder(BaseDecoder):

    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 ):
        super(MasterDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), N)
        self.norm = nn.LayerNorm(decoder.size)
        self.fc = nn.Linear(d_model, num_classes)

        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))

        tgt_mask = trg_pad_mask & trg_sub_mask
        return None, tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, tgt_mask)
        x = self.norm(x)
        return self.fc(x)

    def greedy_forward(self, SOS, feature, mask):
        input = SOS
        output = None
        for i in range(self.max_length+1):
            _, target_mask = self.make_mask(feature, input)
            out = self.decode(input, feature, None, target_mask)
            #out = self.decoder(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):

        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)
        bbox_list = targets_dict["bbox"].to(device)
        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, tgt_mask, bbox_list)

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output = self.greedy_forward(SOS, out_enc, src_mask)
        return output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)  
