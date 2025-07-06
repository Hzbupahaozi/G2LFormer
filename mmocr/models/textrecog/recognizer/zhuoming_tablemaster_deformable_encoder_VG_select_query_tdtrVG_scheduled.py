import torch
import torch.nn as nn
from torch import Tensor

import math
from mmdet.models.builder import DETECTORS, build_backbone, build_loss
import numpy as np
import math
# from mmocr.registry import MODELS
import sys
# sys.path.append('/home/chs/tablemaster-mmocr/table_recognition/PubTabNet-master/src')
# from metric import TEDS
import os
from .encode_decode_recognizer import EncodeDecodeRecognizer


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
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # print("HW:",pos.shape)
        # import ipdb; ipdb.set_trace()

        return pos


@DETECTORS.register_module()
# @MODELS.register_module()
class TABLEMASTER(EncodeDecodeRecognizer):
    # need to inherit BaseRecognizer or EncodeDecodeRecognizer in mmocr
    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 bbox_loss=None,
                 iou_loss = None,
                 span_loss = None,
                 GIOU_loss = None,
                 colrow_loss = None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None):
        super(TABLEMASTER, self).__init__(preprocessor,
                                       backbone,
                                       encoder,
                                       decoder,
                                       loss,
                                       label_convertor,
                                       train_cfg,
                                       test_cfg,
                                       max_seq_len,
                                       pretrained)
        # build bbox loss
        self.bbox_loss = build_loss(bbox_loss)
        # self.span_loss = build_loss(span_loss)
        self.colrow_loss = build_loss(colrow_loss)
        self.GIOU_loss = build_loss(GIOU_loss)
        self.iou_loss =None
        self.adaptive_pool = nn.AdaptiveAvgPool2d((15, 15))
        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((30, 30))
        self.pro = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3,stride=2, padding=1),
                    nn.GroupNorm(32, 512),
                )
        self.enc_out = nn.Linear(512,512)
        self.enc_output_norm = nn.LayerNorm(512)

        self.out_features = 37
        self.enc_out_class_embed = nn.Linear(in_features=512, out_features=self.out_features, bias=True)
        prior_prob =0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.enc_out_class_embed.bias.data = torch.ones(self.out_features) * bias_value
        if(iou_loss!=None):
            self.iou_loss = build_loss(iou_loss)
    def init_weights(self, pretrained=None):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)   # img:torch.size([3,3,480,480]) 因为samples_per_gpu=3,
        feat_origin = self.pro(feat[-2])
        feat_origin = self.adaptive_pool1(feat_origin)

        feat_mid = feat[-1]
        feat_min = self.adaptive_pool(self.pro(feat[-1]))
        feat_max = feat[-2]
        device = feat_min.device
        feats = [feat_max, feat_mid, feat_min, feat_origin]
        feats_shapes = [feat.shape for feat in feats]

        # 在 CPU 上初始化 src_masks
        src_masks = []
        for feat_shape in feats_shapes:
            b, c, h, w = feat_shape
            src_masks.append(torch.zeros((b, h, w), device='cpu'))  # 使用 CPU 进行初始化

        # 填充 src_masks
        length = len(img_metas)
        for i in range(length):
            meta = img_metas[i]
            img_shape = meta['img_shape']

            for idx, (feat_shape, src_mask) in enumerate(zip(feats_shapes, src_masks)):
                _, _, h, w = feat_shape
                h_new, w_new = self.calculate_hw(h, w, img_shape)
                src_mask[i, :h_new, :w_new] = 1  


        src_mask_origin = src_masks[-1].reshape(feats_shapes[-1][0], -1).to(device)
        srcs_masks = [src_mask.to(device) for src_mask in src_masks[:3]]  # 保留前3个 mask 并转移到 CUDA

        srcs_masksTF = []
        for mask01 in srcs_masks:
            mask_cpu = mask01.to('cpu')  # 将 mask 移到 CPU
            maskFT = torch.where(mask_cpu == 1, torch.tensor(False), torch.tensor(True))  # 在 CPU 上计算
            srcs_masksTF.append(maskFT.to(device))  # 计算完毕后转移到 GPU
                    
        srcs = [feat_max, feat_mid, feat_min]

        targets_dict = self.label_convertor.str_bbox_format(img_metas)

        feature_scale_grid_size_pairs = [(60, 60), (30, 30), (15, 15)]

        col_masks_all_max, col_masks_all_mid, col_masks_all_min = None, None, None
        row_masks_all_max, row_masks_all_mid, row_masks_all_min = None, None, None
        tr_masks_all_max, tr_masks_all_mid, tr_masks_all_min = None, None, None

        for i, (feature_scale, grid_size) in enumerate(feature_scale_grid_size_pairs):
            # 使用 CPU 计算
            col_masks_all, row_masks_all, tr_masks_all = self.generate_masks(img_metas, feature_scale, grid_size, device)
            if i == 0:
                col_masks_all_max = col_masks_all.to(device)
                row_masks_all_max = row_masks_all.to(device)
                tr_masks_all_max = tr_masks_all.to(device)
            elif i == 1:
                col_masks_all_mid = col_masks_all.to(device)
                row_masks_all_mid = row_masks_all.to(device)
                tr_masks_all_mid = tr_masks_all.to(device)

            elif i == 2:
                col_masks_all_min = col_masks_all.to(device)
                row_masks_all_min = row_masks_all.to(device)
                tr_masks_all_min = tr_masks_all.to(device)

        if self.encoder is not None:
            out_enc, mask = self.encoder(srcs, srcs_masksTF) 
        src_mask = torch.where(mask == False, torch.tensor(1).to(mask.device), torch.tensor(0).to(mask.device))
        
        memory_padding_mask = (src_mask==0)
        spatial_shapes = torch.tensor([[60,60],[30,30],[15,15]])
        input_hw = None
        out_memory,_ = self.gen_encoder_output_proposals(out_enc, memory_padding_mask, spatial_shapes, input_hw)
        out_memory = self.enc_output_norm(self.enc_out(out_memory))
        enc_outputs_class_unselected = self.enc_out_class_embed(out_memory)
        topk = 1200
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]  
        tgt_undetach = torch.gather(out_memory, 1, topk_proposals.unsqueeze(-1).repeat(1,1, 512))
        feature_select = tgt_undetach.detach()

        scheduled_prob = 0.1

        out_dec, out_bbox,dn_out, col_query, row_query, tr_query = self.decoder(
            feature_select,scheduled_prob, feat_origin, out_enc, targets_dict, src_mask, src_mask_origin, img_metas)


        split_sizes = [3600,900,225]
        out_ec_max, out_ec_mid, out_ec_min = torch.split(out_enc, split_sizes, dim=1)
        a_col_max = torch.matmul(col_query, out_ec_max.transpose(-2, -1))
        a_col_mid = torch.matmul(col_query, out_ec_mid.transpose(-2, -1))
        a_col_min = torch.matmul(col_query, out_ec_min.transpose(-2, -1))
        a_row_max = torch.matmul(row_query, out_ec_max.transpose(-2, -1))
        a_row_mid = torch.matmul(row_query, out_ec_mid.transpose(-2, -1))
        a_row_min = torch.matmul(row_query, out_ec_min.transpose(-2, -1))
        a_tr_max = torch.matmul(tr_query, out_ec_max.transpose(-2, -1))
        a_tr_mid = torch.matmul(tr_query, out_ec_mid.transpose(-2, -1))
        a_tr_min = torch.matmul(tr_query, out_ec_min.transpose(-2, -1))

        loss_inputs = (
            out_dec,    
            targets_dict,
            img_metas,
        )
        losses ={} 
        losses['loss_ce'] = 3*self.loss(*loss_inputs)['loss_ce']
        loss_cls_dn = {}
        for i in range(len(dn_out[0])):
            loss_dn_inputs = (
                dn_out[0][i],  
                targets_dict,
                img_metas,
            )
            if(loss_cls_dn!={}):
                loss_cls_dn['loss_dn'] += self.loss(*loss_dn_inputs)['loss_ce']
            else:
                loss_cls_dn['loss_dn'] = self.loss(*loss_dn_inputs)['loss_ce']
        
        losses.update(loss_cls_dn)
        colrow_loss={}
        colrow_loss['col_loss'] = (
            1*self.colrow_loss(a_col_max, col_masks_all_max)['sigmoid_focal_loss'] +
            1*self.colrow_loss(a_col_mid, col_masks_all_mid)['sigmoid_focal_loss'] +
            1*self.colrow_loss(a_col_min, col_masks_all_min)['sigmoid_focal_loss']
        )
        colrow_loss['row_loss'] = (
            1*self.colrow_loss(a_row_max, row_masks_all_max)['sigmoid_focal_loss'] +
            1*self.colrow_loss(a_row_mid, row_masks_all_mid)['sigmoid_focal_loss'] +
            1*self.colrow_loss(a_row_min, row_masks_all_min)['sigmoid_focal_loss']
        )
        colrow_loss['tr_loss'] = (
            1*self.colrow_loss(a_tr_max, tr_masks_all_max)['sigmoid_focal_loss'] +
            1*self.colrow_loss(a_tr_mid, tr_masks_all_mid)['sigmoid_focal_loss'] +
            1*self.colrow_loss(a_tr_min, tr_masks_all_min)['sigmoid_focal_loss']
        )
        losses.update(colrow_loss)
        bbox_loss_inputs = (
            out_bbox,
            dn_out[1],
            targets_dict,
            img_metas,
        )
        bbox_losses = self.bbox_loss(*bbox_loss_inputs)
        losses.update(bbox_losses)
        
        giou_loss = (
            out_bbox,
            targets_dict,
            img_metas
        )
        giou_losses = self.GIOU_loss(*giou_loss)
        losses.update(giou_losses)

        
        if(self.iou_loss!=None):
            iou_losses = self.iou_loss(*bbox_loss_inputs)
            losses.update(iou_losses)

        return losses
    
    def generate_teds(self, teds, context, gt_context):
        htmlcontext = '<html><body><table>' + context + '</table></body></html>'
        htmlgtcontext = '<html><body><table>' + gt_context + '</table></body></html>'
        score = teds.evaluate(htmlcontext, htmlgtcontext)
        return score
    
    
    def calculate_hw(self, h, w, img_shape, scale=480):
        h_new = math.ceil(h * img_shape[0] / scale)
        w_new = math.ceil(w * img_shape[1] / scale)
        return h_new, w_new
    
    def calculate_bbox_coordinates(self, bbox, scale, grid_size, img_size=480):
        x1, y1, x2, y2 = bbox
        x1 = math.floor(scale * x1 / img_size)
        y1 = math.floor(scale * y1 / img_size)
        x2 = math.ceil(scale * x2 / img_size)
        y2 = min(math.ceil(scale * y2 / img_size) + 1, grid_size)
        return x1, y1, x2, y2
    
    def generate_masks(self, img_metas, feature_scale, grid_size, device, mask_count=599):
        col_masks_all, row_masks_all, tr_masks_all = [], [], []

        for meta in img_metas:
            cell_masks_col, cell_masks_row, cell_masks_tr, origin_bboxes = meta['cell_masks']
            col_masks, row_masks, tr_masks = [], [], []            

            for b in range(cell_masks_col.shape[0]):  # 每张图片的一个 bbox
                col_mask = torch.zeros((1, 1, feature_scale, feature_scale), device=device)
                x1, y1, x2, y2 = self.calculate_bbox_coordinates(cell_masks_col[b], feature_scale, grid_size)
                col_mask[:, :, y1:y2, x1:x2] = 1
                
                row_mask = torch.zeros((1, 1, feature_scale, feature_scale), device=device)
                x1, y1, x2, y2 = self.calculate_bbox_coordinates(cell_masks_row[b], feature_scale, grid_size)
                row_mask[:, :, y1:y2, x1:x2] = 1

                tr_mask = torch.zeros((1, 1, feature_scale, feature_scale), device=device)
                x1, y1, x2, y2 = self.calculate_bbox_coordinates(cell_masks_tr[b], feature_scale, grid_size)
                tr_mask[:, :, y1:y2, x1:x2] = 1

                origin_bbox = torch.zeros((1,1,feature_scale, feature_scale), device=device)
                x1, y1, x2, y2 = self.calculate_bbox_coordinates(origin_bboxes[b], feature_scale, grid_size)
                origin_bbox[:,:,y1:y2, x1:x2] = 1
                
                ones_positions = torch.nonzero(origin_bbox[0,0] == 1, as_tuple=False)
                ones_positions = ones_positions[:, -2:].cpu().numpy()

                if len(ones_positions) > 1:
                    weight_matrix = self.create_weight_matrix(origin_bbox.shape, ones_positions, max_weight=1.0)
                    weight_matrix = torch.tensor(weight_matrix, device=col_mask.device)
                    col_mask = col_mask * weight_matrix
                    row_mask = row_mask * weight_matrix
                    tr_mask = tr_mask * weight_matrix

                col_masks.append(col_mask)
                row_masks.append(row_mask)
                tr_masks.append(tr_mask)

            # 合并 col 和 row masks
            col_masks_one_img = torch.cat(col_masks, dim=1)  # (1, num of box, feature_scale, feature_scale)
            row_masks_one_img = torch.cat(row_masks, dim=1)
            tr_masks_one_img = torch.cat(tr_masks, dim=1)
            _, num, _, _ = col_masks_one_img.shape

            if mask_count - num > 0:
                zero_tensor = torch.zeros(1, mask_count - num, feature_scale, feature_scale, device=device)
                col_masks_one_img = torch.cat((col_masks_one_img, zero_tensor), dim=1)
                row_masks_one_img = torch.cat((row_masks_one_img, zero_tensor), dim=1)
                tr_masks_one_img = torch.cat((tr_masks_one_img, zero_tensor), dim=1)
            else:
                print('输入 token 数量大于 500')

            col_masks_all.append(col_masks_one_img)
            row_masks_all.append(row_masks_one_img)
            tr_masks_all.append(tr_masks_one_img)

        col_masks_all = torch.cat(col_masks_all, dim=0).reshape(len(img_metas), mask_count, -1)
        row_masks_all = torch.cat(row_masks_all, dim=0).reshape(len(img_metas), mask_count, -1)
        tr_masks_all = torch.cat(tr_masks_all, dim=0).reshape(len(img_metas), mask_count, -1)

        return col_masks_all, row_masks_all, tr_masks_all

    def create_weight_matrix(self, tensor_shape, ones_positions, max_weight=1.0):
        rows, cols = tensor_shape[-2:]
        grid_x, grid_y = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        distances = np.min(
            [np.abs(grid_x - x) + np.abs(grid_y - y) for x, y in ones_positions], axis=0
        )
        decay = max_weight / rows
        weight_matrix = np.maximum(max_weight - decay * distances, 0)
        return weight_matrix

    # From DINO-DETR
    def gen_encoder_output_proposals(self, memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor, learnedwh=None):
        """
        Input:
            - memory: bs, \sum{hw}, d_model
            - memory_padding_mask: bs, \sum{hw}
            - spatial_shapes: nlevel, 2
            - learnedwh: 2
        Output:
            - output_memory: bs, \sum{hw}, d_model
            - output_proposals: bs, \sum{hw}, 4
        """
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

            if learnedwh is not None:
                wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
            else:
                wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

        return output_memory, output_proposals
    
    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        targets_dict = self.label_convertor.str_format(img_metas)
        feat = self.extract_feat(img)
        feat_origin = self.pro(feat[-2])
        feat_origin = self.adaptive_pool1(feat_origin)

        feat_mid = feat[-1]
        feat_min = self.adaptive_pool(self.pro(feat[-1]))
        feat_max = feat[-2]
        device = feat_min.device  
        feats = [feat_max, feat_mid, feat_min, feat_origin]
        feats_shapes = [feat.shape for feat in feats]

        # 初始化 src_mask
        src_masks = []
        for feat_shape in feats_shapes:
            b, c, h, w = feat_shape
            src_masks.append(torch.zeros((b, h, w), device=device))

        # 填充 src_mask
        length = len(img_metas)
        for i in range(length):
            meta = img_metas[i]
            img_shape = meta['img_shape']
            
            for idx, (feat_shape, src_mask) in enumerate(zip(feats_shapes, src_masks)):
                _, _, h, w = feat_shape
                h_new, w_new = self.calculate_hw(h, w, img_shape)
                src_mask[i, :h_new, :w_new] = 1

        src_mask_origin = src_masks[-1].reshape(feats_shapes[-1][0], -1)
        srcs_masks = src_masks[:3]  # 保留前3个 mask (max, mid, min)

        srcs_masksTF = []
        for mask01 in srcs_masks:
            maskFT = torch.where(mask01 == 1, torch.tensor(False).to(mask01.device), torch.tensor(True).to(mask01.device))
            srcs_masksTF.append(maskFT)
        srcs = [feat_max, feat_mid, feat_min]              
        out_enc = None
        if self.encoder is not None:
            out_enc, mask = self.encoder(srcs, srcs_masksTF)
        src_mask = torch.where(mask == False, torch.tensor(1).to(mask.device), torch.tensor(0).to(mask.device))

        # select query
        memory_padding_mask = (src_mask==0)
        spatial_shapes = torch.tensor([[60,60],[30,30],[15,15]])
        input_hw = None
        out_memory,_ = self.gen_encoder_output_proposals(out_enc, memory_padding_mask, spatial_shapes, input_hw)
        out_memory = self.enc_output_norm(self.enc_out(out_memory))
        enc_outputs_class_unselected = self.enc_out_class_embed(out_memory)
        topk = 1200
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]  
        tgt_undetach = torch.gather(out_memory, 1, topk_proposals.unsqueeze(-1).repeat(1,1,512))
        feature_select = tgt_undetach.detach()

        out_dec, out_bbox = self.decoder(
            feature_select, feat_origin, out_enc, targets_dict, src_mask, src_mask_origin, img_metas, train_mode=False)

        strings, scores, pred_bboxes = \
            self.label_convertor.output_format(out_dec, out_bbox, img_metas)

        results = []

        results.append(dict(text=strings, score=scores, bbox=pred_bboxes))

        return results