import torch.nn as nn
import torch
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class DistLoss(nn.Module):
    """Implementation of loss module for table master bbox regression branch
    with Distance loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """
    def __init__(self, reduction='none'):
        super().__init__()
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        self.dist_loss = self.build_loss(reduction)

    def build_loss(self, reduction, **kwargs):
        raise NotImplementedError

    def format(self, outputs, targets_dict):
        raise NotImplementedError

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets = self.format(outputs, targets_dict)
        loss_dist = self.dist_loss(outputs, targets.to(outputs.device))
        losses = dict(loss_dist=loss_dist)
        return losses


@LOSSES.register_module()
class TableL1Loss(DistLoss):
    """Implementation of L1 loss module for table master bbox branch."""
    def __init__(self,
                 reduction='sum',
                 **kwargs):
        super().__init__(reduction)
        self.lambda_horizon = 1.    # 水平bbox损失的权重
        self.lambda_vertical = 1.   # 垂直bbox损失的权重
        self.eps = 1e-6
        # use reduction sum, and divide bbox_mask's nums, to get mean loss.
        try:
            assert reduction == 'sum'
        except AssertionError:
            raise ('Table L1 loss in bbox branch should keep reduction is sum.')

    def build_loss(self, reduction):
        return nn.L1Loss(reduction=reduction)

    def format(self, outputs, targets_dict):
        # target in calculate loss, start from idx 1.
        # print("outputs.shape:",outputs[0].shape, outputs[1].shape)
        # outputs,outputs1,outputs2 = outputs[0], outputs[1], outputs[2]       
        bboxes = targets_dict['bbox'][:, 1:, :].to(outputs[0].device)  # bxLx4  ori[3,599,4]
        bbox_masks = targets_dict['bbox_masks'][:, 1:].unsqueeze(-1).to(outputs[0].device)  # bxLx1[3,599,1]
        # print("bbox:",bboxes)
        # print("bbox_masks:",bbox_masks[:,:20])
        masked_outputs = []
        # mask empty-bbox or non-bbox structure token's bbox.
        # print("L:",len(outputs),bbox_masks.shape)
        for i in range(len(outputs)):
            masked_outputs.append(outputs[i] * bbox_masks)  # 将所有output_box乘mask
        # masked_outputs1 = outputs1 * bbox_masks
        # masked_outputs2 = outputs2 * bbox_masks
        masked_bboxes = bboxes * bbox_masks
        # print("bboxes:",bboxes)
        # return masked_outputs, masked_outputs1, masked_outputs2, masked_bboxes, bbox_masks
        return masked_outputs,masked_bboxes, bbox_masks
    

    def forward(self, outputs, dn_out, targets_dict, img_metas=None):
        # outputs, outputs1, outputs2, targets, bbox_masks = self.format(outputs, targets_dict)
        #################################################################################################在这个地方插入匈牙利匹配，找到更好的框然后交换位置
        # ins_value = self.xiongyali(outputs, targets_dict,img_metas)
        # print(ins_value)
        ##################################################################################################
        outputs, targets, bbox_masks = self.format(outputs, targets_dict)
       
        # print("outputs:",outputs[0][:,:50])
        # print("targets:",targets[:,:50])
        # print("dn:",dn_out[0][:,:50])
        # outputs = None
        # horizon loss (x and width)
        horizon_loss = 0
        vertical_loss = 0
        horizon_loss_dn = 0
        vertical_loss_dn = 0
        p1 = 2
        p2 = 1
        # print("len:",len(outputs))
        for i in range(len(outputs)):
            horizon_sum_loss = self.dist_loss(outputs[i][:, :, 0::2].contiguous(), targets[:, :, 0::2].contiguous())
            horizon_loss += p1*horizon_sum_loss / (bbox_masks.sum() + self.eps)
            # vertical loss (y and height)
            vertical_sum_loss = self.dist_loss(outputs[i][:, :, 1::2].contiguous(), targets[:, :, 1::2].contiguous())
            vertical_loss += p2*vertical_sum_loss / (bbox_masks.sum() + self.eps)
            # horizon_scale_loss = self.dist_loss(outputs[i][:, :, 0::2].contiguous(), targets[:, :, 0::2].contiguous())
            # horizon_loss += p*horizon_sum_loss / (bbox_masks.sum() + self.eps)
            # vertical loss (y and height)
            # vertical_sum_loss = self.dist_loss(outputs[i][:, :, 1::2].contiguous(), targets[:, :, 1::2].contiguous())
            # vertical_loss += p*vertical_sum_loss / (bbox_masks.sum() + self.eps)
            # p = p*4/5
        # print(dn_out)

        if(dn_out!=[]) :
            
            masked_dn = [] 
            n = len(dn_out)
            n_noise  = len(dn_out)//n
            # print(len(dn_out))
            for num in range(n_noise):
                for i in range(len(dn_out)):
                    # print(dn_out[i].shape,bbox_masks.shape)
                    masked_dn.append(dn_out[n*num+i] * bbox_masks)
            dn_out = masked_dn
            # print(n_noise)
            horizon_loss_dn,vertical_loss_dn =0,0
            for num in range(n_noise):
                for i in range(len(dn_out)):
                    horizon_sum_loss_dn= self.dist_loss(dn_out[n*num+i][:, :, 0::2].contiguous(), targets[:, :, 0::2].contiguous())
                    horizon_loss_dn += p1*horizon_sum_loss_dn / (bbox_masks.sum() + self.eps)
                    # vertical loss (y and height)
                    vertical_sum_loss_dn= self.dist_loss(dn_out[n*num+i][:, :, 1::2].contiguous(), targets[:, :, 1::2].contiguous())
                    vertical_loss_dn+= p2*vertical_sum_loss_dn / (bbox_masks.sum() + self.eps)
                    # p = p*4/5
            # print(horizon_loss_dn,vertical_loss_dn)
            horizon_loss_dn = horizon_loss_dn/len(dn_out)
            vertical_loss_dn = vertical_loss_dn/len(dn_out)

            losses = {'horizon_bbox_loss': horizon_loss, 'vertical_bbox_loss': vertical_loss,"horizon_loss_dn":horizon_loss_dn,"vertical_loss_dn":vertical_loss_dn}
        else: 
            losses = {'horizon_bbox_loss': horizon_loss, 'vertical_bbox_loss': vertical_loss}
        # losses = {}
        return losses
    def xiongyali(self, outputs, targets_dict, img_metas):
        import torch

        outputs_bbox = outputs
        bs, num_boxes = outputs_bbox[2].shape[:2]
        # 只对最后一层的输出做匈牙利匹配
        outputs_bbox2 = outputs_bbox[2].flatten(0, 1)

        padded_tgt_bbox0 = targets_dict['bbox'][:, 1:, :][0].to(outputs[0].device)
        padded_tgt_bbox1 = targets_dict['bbox'][:, 1:, :][1].to(outputs[0].device)
        padded_tgt_bbox2 = targets_dict['bbox'][:, 1:, :][2].to(outputs[0].device)
        padded_tgt_bbox_mask0 = targets_dict['bbox_masks'][:, 1:][0]
        padded_tgt_bbox_mask1 = targets_dict['bbox_masks'][:, 1:][1]
        padded_tgt_bbox_mask2 = targets_dict['bbox_masks'][:, 1:][2]
        tgt_bbox0 = padded_tgt_bbox0[padded_tgt_bbox_mask0 == 1]
        tgt_bbox1 = padded_tgt_bbox1[padded_tgt_bbox_mask1 == 1]
        tgt_bbox2 = padded_tgt_bbox2[padded_tgt_bbox_mask2 == 1]

        tgt_bboxes = torch.cat((tgt_bbox0, tgt_bbox1, tgt_bbox2),dim=0)
        cost_bbox = torch.cdist(outputs_bbox2, tgt_bboxes, p=1)
        # cost_giou = -self.generalized_box_iou(self.box_cxcywh_to_xyxy(outputs_bbox2),self.box_cxcywh_to_xyxy(tgt_bboxes))
        # C = 5 * cost_bbox + 2 * cost_giou
        C = cost_bbox.view(bs, num_boxes, -1).cpu().detach()
        # print(C.device)
        sizes = [tgt_bbox0.shape[0], tgt_bbox1.shape[0],tgt_bbox2.shape[0]]
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # 我要修改应该是outputs_bbox2[这里应该是标注的mask？还是预测结果的mask？] = outputs_bbox2[indices[0]]也就是预测框的index，这些index不一定就是标注的mask里面为1的，也有可能是为0的，也就是旁边的框可能更好
        # batchsize个indices，每一个都有两个，indices[0][0]是第一张图片499个框里面预测框的index，加噪训练好的模型可以发现其实index和mask里面为1的index是差不多的；然后indices[0][1]则是标注框的index，需要sort一下
        # indice1_np = np.array(indices[0][1])
        # indice0 = indices[0][0] 
        # # 找到 indice1 中需要交换的元素索引
        # for i in range(len(indice1_np)):
        #     for j in range(i + 1, len(indice1_np)):
        #         if indice1_np[i] > indice1_np[j]:
        #             # 交换 indice1_np 中的元素
        #             indice1_np[i], indice1_np[j] = indice1_np[j], indice1_np[i]
        #             # 交换对应位置的 indice0 中的元素
        #             indice0[i], indice0[j] = indice0[j], indice0[i]
        # gt_indices = torch.where(padded_tgt_bbox_mask0 == 1)
        # outputs_bbox[2][gt_indices] = outputs_bbox[2][indice0]
        # return outputs
        ins_value = 0
        for i in range(3):
            ins_value += sum(1 for i, num in enumerate(indices[i][1]) if num != i)
        return ins_value
    
    def generalized_box_iou(self, boxes1, boxes2):
        import torch
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format

        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = self.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area
    
    def box_iou(self, boxes1, boxes2):
        from torchvision.ops.boxes import box_area
        import torch
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union
    
    def box_cxcywh_to_xyxy(self, x):
        import torch
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

@LOSSES.register_module()
class GIOU_loss(DistLoss):
    """Implementation of L1 loss module for table master bbox branch."""
    def __init__(self,
                 reduction='sum',
                 **kwargs):
        super().__init__(reduction)
        self.eps = 1e-6
    def build_loss(self, reduction):
        return nn.L1Loss(reduction=reduction)
    def format(self, outputs, targets_dict):
        # target in calculate loss, start from idx 1.
        # print("outputs.shape:",outputs[0].shape, outputs[1].shape)
        # outputs,outputs1,outputs2 = outputs[0], outputs[1], outputs[2]       
        bboxes = targets_dict['bbox'][:, 1:, :].to(outputs[0].device)  # bxLx4  ori[3,599,4]
        bbox_masks = targets_dict['bbox_masks'][:, 1:].unsqueeze(-1).to(outputs[0].device)  # bxLx1[3,599,1]
        # print("bbox:",bboxes)
        # print("bbox_masks:",bbox_masks[:,:20])
        masked_outputs = []
        # mask empty-bbox or non-bbox structure token's bbox.
        # print("L:",len(outputs),bbox_masks.shape)
        for i in range(len(outputs)):
            masked_outputs.append(outputs[i] * bbox_masks)  # 将所有output_box乘mask
        # masked_outputs1 = outputs1 * bbox_masks
        # masked_outputs2 = outputs2 * bbox_masks
        masked_bboxes = bboxes * bbox_masks
        # print("bboxes:",bboxes)
        # return masked_outputs, masked_outputs1, masked_outputs2, masked_bboxes, bbox_masks
        return masked_outputs,masked_bboxes, bbox_masks
    

    def forward(self, outputs, targets_dict, img_metas=None):
        # outputs, outputs1, outputs2, targets, bbox_masks = self.format(outputs, targets_dict)
        #################################################################################################在这个地方插入匈牙利匹配，找到更好的框然后交换位置
        # outputs = self.xiongyali(outputs, targets_dict)
        ##################################################################################################
        outputs, targets, bbox_masks = self.format(outputs, targets_dict)
        total_giou_loss = 0
        targets = self.convert_bbox_xywh_to_x1y1x2y2(targets)
        ################# 加入GIOU loss
        for i in range(len(outputs)):
            final_preds = outputs[i]
            final_preds = self.convert_bbox_xywh_to_x1y1x2y2(final_preds)
            # 计算交集部分
            inter_xmin = torch.max(final_preds[:, :, 0], targets[:, :, 0])
            inter_ymin = torch.max(final_preds[:, :, 1], targets[:, :, 1])
            inter_xmax = torch.min(final_preds[:, :, 2], targets[:, :, 2])
            inter_ymax = torch.min(final_preds[:, :, 3], targets[:, :, 3])
            inter_area = (inter_xmax - inter_xmin).clamp(0) * (inter_ymax - inter_ymin).clamp(0)
            # 计算预测框和GT框的面积
            pred_area = (final_preds[:, :, 2] - final_preds[:, :, 0]) * (final_preds[:, :, 3] - final_preds[:, :, 1])
            target_area = (targets[:, :, 2] - targets[:, :, 0]) * (targets[:, :, 3] - targets[:, :, 1])
            # 计算IOU
            union_area = pred_area + target_area - inter_area
            iou = inter_area / (union_area + 1e-4)
            # 计算最小包围矩形 C
            enclose_xmin = torch.min(final_preds[:, :, 0], targets[:, :, 0])
            enclose_ymin = torch.min(final_preds[:, :, 1], targets[:, :, 1])
            enclose_xmax = torch.max(final_preds[:, :, 2], targets[:, :, 2])
            enclose_ymax = torch.max(final_preds[:, :, 3], targets[:, :, 3])
            
            enclose_area = (enclose_xmax - enclose_xmin) * (enclose_ymax - enclose_ymin)
            
            # 计算 GIoU
            giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
            
            # 计算 GIoU 损失
            giou_loss = (1 - giou).mean()
            total_giou_loss +=0.5 * giou_loss
        #######################GIOU loss

        losses = {'GIOU_loss':total_giou_loss}
        return losses
    
    def convert_bbox_xywh_to_x1y1x2y2(self, bbox_xywh):
        """
        将 bbox 从 [batch_size, num_bbox, 4] (xywh) 转换为 [batch_size, num_bbox, 4] (x1y1x2y2)

        参数:
        bbox_xywh: torch.Tensor形状为 (batch_size, num_bbox, 4)，格式为 (x, y, w, h)

        返回:
        bbox_x1y1x2y2: torch.Tensor形状为 (batch_size, num_bbox, 4)，格式为 (x1, y1, x2, y2)
        """
        x_c, y_c, w, h = bbox_xywh[..., 0], bbox_xywh[..., 1], bbox_xywh[..., 2], bbox_xywh[..., 3]
        x1 = x_c - 0.5*w
        y1 = y_c - 0.5 *h
        x2 = x_c + 0.5*w
        y2 = y_c + 0.5*h

        bbox_x1y1x2y2 = torch.stack([x1, y1, x2, y2], dim=-1)
        return bbox_x1y1x2y2