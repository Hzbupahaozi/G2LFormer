import torch
import torch.nn as nn
import math
from mmdet.models.builder import DETECTORS, build_backbone, build_loss
import numpy as np
import math
# from mmocr.registry import MODELS
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

def visual_pred_bboxes(img_metas, results):
    """
    visual after normalized bbox in results.
    :param results:
    :return:
    """
    import os
    import cv2
    import numpy as np

    for img_meta, result in zip(img_metas, results):
        img = cv2.imread(img_meta['filename'])
        bboxes = result['bbox']
        save_path = '/data_0/cache/{}_pred_bbox.jpg'. \
            format(os.path.basename(img_meta['filename']).split('.')[0])

        # x,y,w,h to x,y,x,y
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2
        new_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2
        new_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2
        new_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2
        # draw
        for new_bbox in new_bboxes:
            img = cv2.rectangle(img, (int(new_bbox[0]), int(new_bbox[1])),
                                (int(new_bbox[2]), int(new_bbox[3])), (0, 255, 0), thickness=1)
        cv2.imwrite(save_path, img)

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
        self.iou_loss =None
        encoded_image_size = 30
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.pro = nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=1),
                    nn.GroupNorm(32, 512),
                )
        if(iou_loss!=None):
            # print(iou_loss)
            self.iou_loss = build_loss(iou_loss)
        # self.positional_encoding = PositionEmbeddingSineHW()
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
        feat = self.extract_feat(img)      
        length = len(feat)
        for i in range(length):
            print("i:",feat[i].shape)
        feat = feat[-2]  #15*15
        feat = self.pro(feat)
        print(feat.shape)
        feat = self.adaptive_pool(feat) #30*30
        print(feat.shape)
        # print("mmocr")
        targets_dict = self.label_convertor.str_bbox_format(img_metas)
        # print("metas:",img_metas[0]["filename"])
        # print("img:",targets_dict)
        device = feat.device
        b, c, h, w = feat.shape
        # print("feat:",feat.shape)
        # ma = torch.ones((b, h, w), dtype=torch.bool, device=device)
        src_mask  = torch.zeros((b, h, w),  device=device)
        # mask = F.interpolate(src_mask[None].float(), size=feat[i].shape[-2:]).to(torch.bool)[0]
        length = len(img_metas)
        for i in range(length):
            meta = img_metas[i]
            img_shape = meta['img_shape']  
            h1 = math.ceil(60*img_shape[0]/480)
            w1 = math.ceil(60*img_shape[1]/480)
            # print(meta)
            # print(img_shape,h1*8,w1*8)
            src_mask[i,: h1, :w1] = 1
        src_mask = src_mask.reshape(b,-1)    

        if self.encoder is not None:
            out_enc = self.encoder(feat)
        # print ("img_meta:",img_metas[0],len(img_metas[0]["cls_bbox"]))
        # len(img_metas[0]["cls_bbox"]),img_metas[0]["cls_bbox"]
        out_dec, out_bbox,dn_out = self.decoder(
            feat, out_enc, targets_dict, src_mask, img_metas, train_mode=True)
        # out_dec, out_bbox = self.decoder(
        #     feat, out_enc, targets_dict, img_metas, train_mode=True)
        # dn_out = []
        print("dn:",dn_out)
        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        losses ={} 
        losses['loss_ce'] = 3*self.loss(*loss_inputs)['loss_ce']
        # print("dn:",dn_out[0][0].shape,dn_out[1][0].shape)
        if(dn_out[0]):
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
        bbox_loss_inputs = (
            out_bbox,
            dn_out[1],
            targets_dict,
            img_metas,
        )
        bbox_losses = self.bbox_loss(*bbox_loss_inputs)
        losses.update(bbox_losses)
        # print(targets_dict["cls_bbox"])
        # print(img_metas)
        # print("bbox:",targets_dict['bbox_masks'][:, 1:])
        # print(img_metas)
        # print("bbox:",bbox_masks)
        
        if(self.iou_loss!=None):
            iou_losses = self.iou_loss(*bbox_loss_inputs)
            losses.update(iou_losses)
        # print(losses)
        losses = None
        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        #
        targets_dict = self.label_convertor.str_format(img_metas)
        # print(img_metas)
        # targets_dict = self.label_convertor.str_bbox_format(img_metas)
        feat = self.extract_feat(img)
        feat = feat[-2]  #15*15
        
        feat = self.pro(feat)
        device = feat.device
        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        b, c, h, w = feat.shape
        src_mask  = torch.zeros((b, h, w),  device=device)
        length = len(img_metas)
        for i in range(length):
            meta = img_metas[i]
            img_shape = meta['img_shape']  
            h1 = math.ceil(60*img_shape[0]/480)
            w1 = math.ceil(60*img_shape[1]/480)
            # print(meta)
            # print(img_shape,h1*8,w1*8)
            src_mask[i,: h1, :w1] = 1
            # src_mask[i,: img_shape[1], :img_shape[2]] = 1
        src_mask = src_mask.reshape(b,-1)    


        # src_mask = None
        out_dec, out_bbox = self.decoder(
            feat, out_enc, targets_dict, src_mask, img_metas, train_mode=False)
        
        # print("r:",row_output[0].shape,len(bbox_masks))
        # for i in range(499): 
        #     masked_rowoutputs[i]= masked_rowoutputs[i]* bbox_masks[i]
        #     masked_coloutputs[i] = masked_coloutputs[i] * bbox_masks[i]
        # mask empty-bbox or non-bbox structure token's bbox.
        # print(len(bbox_masks),len(bbox_masks[0]))
        # print(masked_rowoutputs.shape)
        # print(bbox_masks[:20])

        # print("outr:",masked_rowoutputs[:20])
        # print("outc:",masked_coloutputs[:20])

        strings, scores, pred_bboxes = \
            self.label_convertor.output_format(out_dec, out_bbox, img_metas)
        # print(len(out_bbox),len(pred_bboxes))
        # print("pred:",pred_bboxes)
        # flatten batch results
        results = []
        print(strings)
            # for string, score, pred_bbox in zip(strings, scores, pred_bboxes):
        results.append(dict(text=strings, score=scores, bbox=pred_bboxes))

        # visual_pred_bboxes(img_metas, results)

        return results