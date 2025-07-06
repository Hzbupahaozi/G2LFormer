import math

import mmcv
import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def iou_loss(pred, target, linear=False, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    # pred[:,]
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    if linear:
        loss = 1 - ious
    else:
        loss = -ious.log()
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def diou_loss(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ciou_loss(pred, target, eps=1e-7):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
    loss = 1 - cious
    return loss


@LOSSES.register_module()
class IoULoss1(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss instead of log scale.
            Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(IoULoss1, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def format(self, outputs, targets_dict):
        # target in calculate loss, start from idx 1.
        # print("outputs.shape:",outputs[0].shape, outputs[1].shape)    
        bboxes = targets_dict['bbox'][:, 1:, :].to(outputs[0].device)  # bxLx4  ori
        bbox_masks = targets_dict['bbox_masks'][:, 1:].unsqueeze(-1).to(outputs[0].device)  # bxLx1
     
        # mask empty-bbox or non-bbox structure token's bbox.
        masked_outputs = []
        for i in range(len(outputs)):
            masked_output = outputs[i] * bbox_masks
            masked_output[:, :, 0],masked_output[:, :, 2] = masked_output[:, :, 0]-masked_output[:, :, 2]/2,masked_output[:, :, 0] + masked_output[:, :, 2]/2
            masked_output[:, :, 1],masked_output[:, :, 3] = masked_output[:, :, 1]-masked_output[:, :, 3]/2,masked_output[:, :, 1] + masked_output[:, :, 3]/2
            masked_outputs.append(masked_output)
        masked_bboxes = bboxes * bbox_masks
        masked_bboxes[:, :, 0] = masked_bboxes[:, :, 0]-masked_bboxes[:, :, 2]/2
        masked_bboxes[:, :, 1] = masked_bboxes[:, :, 1]-masked_bboxes[:, :, 3]/2
        masked_bboxes[:, :, 2] = masked_bboxes[:, :, 0] + masked_bboxes[:, :, 2]
        masked_bboxes[:, :, 3] = masked_bboxes[:, :, 1] + masked_bboxes[:, :, 3]
        return masked_outputs,  masked_bboxes, bbox_masks
    # forward(self, outputs, dn_out, targets_dict, img_metas=None)
    def forward(self,
                pred,
                dn_out,
                targets_dict,
                img_metas = None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        outputs, target, bbox_masks = self.format(pred, targets_dict)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # print("weight:",weight)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        op = outputs[0].half() 
        target = target.half()
        # print(op.shape,target.shape)
        # print((op[0,0,0]),(target[0,0,0]))
        # print(type(outputs[0,0,0]),type(target[0,0,0]))
        loss = self.loss_weight * iou_loss(
            op,
            target,
            weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        losses = {'iou_loss': loss }
        # print(loss)
        return losses


# @LOSSES.register_module()
# class BoundedIoULoss(nn.Module):

#     def __init__(self, beta=0.2, eps=1e-3, reduction='mean', loss_weight=1.0):
#         super(BoundedIoULoss, self).__init__()
#         self.beta = beta
#         self.eps = eps
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         if weight is not None and not torch.any(weight > 0):
#             return (pred * weight).sum()  # 0
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss = self.loss_weight * bounded_iou_loss(
#             pred,
#             target,
#             weight,
#             beta=self.beta,
#             eps=self.eps,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss


@LOSSES.register_module()
class GIoULoss1(nn.Module):

    def __init__(self, eps=1e-6, reduction='sum', loss_weight=1.0):
        super(GIoULoss1, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
    def format(self, outputs, targets_dict):
        # target in calculate loss, start from idx 1.
        # print("outputs.shape:",outputs[0].shape, outputs[1].shape)
        outputs,outputs1,outputs2 = outputs[0], outputs[1], outputs[2]       
        bboxes = targets_dict['bbox'][:, 1:, :].to(outputs.device)  # bxLx4  ori
        bbox_masks = targets_dict['bbox_masks'][:, 1:].unsqueeze(-1).to(outputs.device)  # bxLx1
        # print("bbox:",bboxes)
        # print("bbox_masks:",bbox_masks[:,:20])
        # print("bboxes:",bboxes[:,:50])
        # mask empty-bbox or non-bbox structure token's bbox.
        masked_outputs = (outputs * bbox_masks).half()
        masked_outputs1 = (outputs1 * bbox_masks).half()
        masked_outputs2 = (outputs2 * bbox_masks).half()
        masked_bboxes = (bboxes * bbox_masks).half()
        # print("before:",masked_bboxes[:2,:3])
        masked_outputs[:, :, 0],masked_outputs[:, :, 2] = masked_outputs[:, :, 0] - masked_outputs[:, :, 2]/2,masked_outputs[:, :, 0] + masked_outputs[:, :, 2]/2
        masked_outputs[:, :, 1],masked_outputs[:, :, 3] = masked_outputs[:, :, 1] - masked_outputs[:, :, 3]/2,masked_outputs[:, :, 1] + masked_outputs[:, :, 3]/2
        
        masked_outputs1[:, :, 0],masked_outputs1[:, :, 2] = masked_outputs1[:, :, 0] - masked_outputs1[:, :, 2]/2,masked_outputs1[:, :, 0] + masked_outputs1[:, :, 2]/2
        masked_outputs1[:, :, 1],masked_outputs1[:, :, 3] = masked_outputs1[:, :, 1] - masked_outputs1[:, :, 3]/2,masked_outputs1[:, :, 1] + masked_outputs1[:, :, 3]/2
        masked_outputs2[:, :, 0],masked_outputs2[:, :, 2] = masked_outputs2[:, :, 0] - masked_outputs2[:, :, 2]/2,masked_outputs2[:, :, 0] + masked_outputs2[:, :, 2]/2
        masked_outputs2[:, :, 1],masked_outputs2[:, :, 3] = masked_outputs2[:, :, 1] - masked_outputs2[:, :, 3]/2,masked_outputs2[:, :, 1] + masked_outputs2[:, :, 3]/2
        masked_bboxes[:, :, 0],masked_bboxes[:, :, 2] = masked_bboxes[:, :, 0] - masked_bboxes[:, :, 2]/2,masked_bboxes[:, :, 0] + masked_bboxes[:, :, 2]/2
        masked_bboxes[:, :, 1],masked_bboxes[:, :, 3] = masked_bboxes[:, :, 1] - masked_bboxes[:, :, 3]/2,masked_bboxes[:, :, 1] + masked_bboxes[:, :, 3]/2
        # print("mask:",masked_outputs[:,:50])
        # print("mask2:",masked_outputs2[:,:50])
        # print("maskbbox:",masked_bboxes[:,:50])
        masked_bboxes = masked_bboxes.clamp(min=0.0,max=1.0)
        masked_outputs1 = masked_outputs1.clamp(min=0.0,max=1.0)
        masked_outputs2 = masked_outputs2.clamp(min=0.0,max=1.0)
        masked_outputs = masked_outputs.clamp(min=0.0,max=1.0)
        # print("after:",masked_bboxes[:2,:3])
        return masked_outputs, masked_outputs1, masked_outputs2, masked_bboxes, bbox_masks
    def forward(self,
                pred,
                dn_out,
                targets_dict,
                img_metas = None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # print(weight)
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        
        outputs, outputs1, outputs2, target, bbox_masks = self.format(pred, targets_dict)
        loss = self.loss_weight * giou_loss(
            outputs,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)/ (bbox_masks.sum() + self.eps)
        loss1 = self.loss_weight * giou_loss(
            outputs1,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)/ (bbox_masks.sum() + self.eps)
        loss2 = self.loss_weight * giou_loss(
            outputs2,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)/ (bbox_masks.sum() + self.eps)
        losses = {"iou_loss":loss+loss1+loss2}
        return losses


# @LOSSES.register_module()
# class DIoULoss(nn.Module):

#     def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
#         super(DIoULoss, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         if weight is not None and not torch.any(weight > 0):
#             return (pred * weight).sum()  # 0
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if weight is not None and weight.dim() > 1:
#             # TODO: remove this in the future
#             # reduce the weight of shape (n, 4) to (n,) to match the
#             # giou_loss of shape (n,)
#             assert weight.shape == pred.shape
#             weight = weight.mean(-1)
#         loss = self.loss_weight * diou_loss(
#             pred,
#             target,
#             weight,
#             eps=self.eps,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss


# @LOSSES.register_module()
# class CIoULoss(nn.Module):

#     def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
#         super(CIoULoss, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         if weight is not None and not torch.any(weight > 0):
#             return (pred * weight).sum()  # 0
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if weight is not None and weight.dim() > 1:
#             # TODO: remove this in the future
#             # reduce the weight of shape (n, 4) to (n,) to match the
#             # giou_loss of shape (n,)
#             assert weight.shape == pred.shape
#             weight = weight.mean(-1)
#         loss = self.loss_weight * ciou_loss(
#             pred,
#             target,
#             weight,
#             eps=self.eps,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss
