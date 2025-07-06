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
            # print("i:",i,x.shape)
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
    def __init__(self, feed_forward, self_attn, src_attn,size, dropout, headers =8,rm_self_attn_decoder=False ):
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
        headers =4
        self.self_attn = MultiHeadAttention(headers = headers,d_model = d_model,  dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.src_attn = MultiHeadAttention(headers =headers, d_model = d_model,  dropout=dropout, vdim=d_model)
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
        # self.positional_encoding = PositionEmbeddingSineHW()
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
        #     cls_x_list.append(cls_x)
        # cls_x = torch.cat(cls_x_list, dim=-1)
        # cls_x = self.norm(cls_x)

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
        dn_out = []
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
def prepare_for_dn( num_classes, labels, boxes):
    """
    prepare for dn components in forward function
    Args:
        dn_args: (targets, args.scalar, args.label_noise_scale,
                                                             args.box_noise_scale, args.num_patterns) from engine input
        embedweight: positional queries as anchor
        training: whether it is training or inference
        num_queries: number of queries
        num_classes: number of classes
        hidden_dim: transformer hidden dimenstion
        label_enc: label encoding embedding
        scalar  deno groups
    Returns: input_query_label, input_query_bbox, attn_mask, mask_dict
    """
    label_noise_scale =0.1 
    box_noise_scale = 0.25
    scalar = 2

    # add noise
    known_labels = labels.repeat(scalar,1,1).view(-1)
    known_bboxs = boxes.repeat(scalar,1,1,1)
    known_labels_expaned = known_labels.clone()
    known_bbox_expand = boxes.clone()
    # noise on the label
    if label_noise_scale > 0:
        p = torch.rand_like(known_labels_expaned.float())
        chosen_indice = torch.nonzero(p < (label_noise_scale)) .view(-1)# usually half of bbox noise
        new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
        known_labels_expaned.scatter_(0, chosen_indice, new_label)
        known_labels_expaned = known_labels_expaned.view(scalar,labels.shape[0],labels.shape[1])
        # print(labels[0,:50])
        # print(known_labels_expaned[0,0,:50])
        # print(known_labels_expaned[1,0,:50])
        # print(new_label.shape,known_labels_expaned.shape)
    
    # noise on the box
    if box_noise_scale > 0:
        box_noise_scale = 0.25
        diff = torch.zeros_like(known_bboxs)
        sp = diff.shape[:-1]
        p  = torch.rand(sp).cuda()
        p[p<0.25] = 0
        p[p>=0.25]= 1
        # print(p.shape)
        p = p.unsqueeze(-1)
        p = p.repeat(1,1,1,4)
        # print(p.shape)
        diff[:,:,:, :2] = known_bboxs[:,:, :,2:] / 2
        diff[:,:,:, 2:] = known_bboxs[:,:, :,2:]
        diff = torch.mul(p,diff).cuda()
        box_noise =  torch.mul((torch.rand_like(known_bboxs) * 2 - 1.0),
                                        diff).cuda() * box_noise_scale
        bbox_expand = known_bboxs + box_noise
        known_bbox_expand = bbox_expand.clamp(min=0.0, max=1.0)

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
        self.cls_layer = clones(DecoderLayer(**decoder), 4)
        self.bbox_one = clones(DecoderLayer(**decoder), 2)
        self.num_layers = 2

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=1024, dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, dropout=0.)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=1024, dropout=0.)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        # self.bbox_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # self.bbox_layer = clones(DecoderLayer1(), 2)
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
        self.embedding =nn.Embedding(num_embeddings=num_classes, embedding_dim=d_model)
        #Embeddings(d_model=d_model, vocab=num_classes)
        
        self.pos_target = PositionalEncoding(d_model=d_model)
        self.d_model = d_model
        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len
        #DAB-DETR
        self.query_dim = 4
        num_layers =1
        # self.positional_encoding = PositionEmbeddingSineHW(temperatureH=20,temperatureW=20,normalize=True)
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
        # x_list = []
        cls_x_list = []
        output_list = []
        
        feature = feature.permute(1,0,2)
        # print(feature.shape)
        enc_out = self.encoder(feature)
        src_mask = None

        # cls head
        # x = self.embedding(input)
        # x = self.pos_target(x)
        # enc_out = feature.permute(1,0,2)
        # for layer in self.cls_layer:
        #     x,_ = layer(x = x, feature=enc_out, tgt_mask=tgt_mask,
        #                    src_mask=src_mask,
        #                    pos=None, query_pos=None, query_sine_embed=None,
        #                   )
        # dec_out =  x
        # dec_out = x.permute(1,0,2)
        # enc_out = enc_out.permute(1,0,2)

        # print("done")
        input = input.permute(1,0) 
        x = self.embedding(input)
        x = self.pos_target(x)      
        padding_mask = torch.zeros(input.shape, dtype=torch.bool).to(input.device)
        padding_mask[input==self.PAD] = True
        sz = (input.shape[0])
        tgt_mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(input.device)
        # print("b:",tgt_mask.shape,padding_mask.shape)
        dec_out = self.decoder(
                tgt=x,
                memory=enc_out, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=padding_mask.permute(1,0), 
                memory_mask=None)

        # bbox_masks = bbox_masks[:,1:].cuda()  #1031
        # bbox_masks = bbox_masks
        # print("??b",bbox_masks.shape)
        # print(dec_out.shape,enc_out.shape)
        
        bbox_out, _ = self.attention(dec_out, enc_out, enc_out) # q, k, v
        bbox_out = bbox_out.permute(1,0,2)
        dec_out = dec_out.permute(1,0,2)
        
        # print("d:",dec_out.shape,bbox_out.shape)
        # bbox_out = self.bbox_decoder(
        #         tgt=cls_x,
        #         memory=enc_out, 
        #         tgt_mask=bbox_masks,
        #         tgt_key_padding_mask=None, 
        #         memory_mask=src_mask)
        # bbox_x_list.append(bbox_out)
        # bbox_x = torch.cat(bbox_x_list, dim=-1)
        # bbox_x = self.norm(bbox_x)
        # bbox_x = self.bbox_fc1(bbox_out)
        
        # print("bb:",bbox_x.shape)
        dn_out,cls_dn_out = [], []
        
        bbox_output = self.bbox_embed(bbox_out).sigmoid()
        output_list.append(bbox_output) 
        
        return self.cls_fc(dec_out), output_list, (cls_dn_out,dn_out)
        

    def greedy_forward(self, SOS, feat, feature, src_mask =None, img_metas =None ):
       
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
            _, target_mask = self.make_mask(feature, input)          
            out, bbox_output,dn = self.decode(input, feat, feature, src_mask, target_mask,bbox_masks=None)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            token = next_word[:, -1]
            
            # if(flag ==0):
            #     if(token==107): flag =1   #1128
            #     if(token ==1 or token==3) and (flag==0): 
            #         bbox_mask  = torch.LongTensor([1]).byte().unsqueeze(0)
            #     else:   bbox_mask  = torch.LongTensor([0]).byte().unsqueeze(0)
            # bbox_masks = torch.cat([bbox_masks, bbox_mask], dim=1)
                
            #point
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        # print("input:",input)
        # out, bbox_output,dn = self.decode(input[:,:-1], feat, feature, src_mask, target_mask,bbox_masks=bbox_masks)
        
        return output, bbox_output
    def forward_train(self, feat, out_enc, targets_dict, src_mask, img_metas=None):
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
        # scalar = 2
        # known_bboxs = bbox_list.repeat(scalar,1,1, 1)
        # print("img_metas",img_metas[0]["filename"])
        # print(img_metas)
        target = targets_dict["targets"]

        #  dn 
        known_labels_expaned, known_bbox_expand = prepare_for_dn(self.num_classes-3,padded_targets[:, :-1],bbox_list )
        bbox_expand = known_bbox_expand

      
        # print("bbox_masks:",targets_dict["bbox_masks"])
       
        # src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])#pad_target

        
        return self.decode(padded_targets[:, :-1], feat, out_enc, src_mask, tgt_mask, (known_labels_expaned,bbox_expand), targets_dict["bbox_masks"])#

    def forward_test(self, feat, out_enc, targets_dict, src_mask,img_metas):
        # src_mask = None
        batch_size = out_enc.shape[0]
        # print("target:",targets_dict)
        # bbox_masks = targets_dict["bbox_masks"]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        # print("s:",src_mask)
        output, bbox_output = self.greedy_forward(SOS, feat, out_enc, src_mask, img_metas=img_metas )
        # print("len:",len(bbox_output))
        # print("out:",output,bbox_output)
        return output, bbox_output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                src_mask =None,
                img_metas=None,
                train_mode=True,
                ):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, src_mask, img_metas)

        return self.forward_test(feat, out_enc, targets_dict,src_mask, img_metas)


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
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
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