import torch.nn as nn
import torch
from mmdet.models.builder import LOSSES
import torch.nn.functional as F


@LOSSES.register_module()
class CELoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """

    def __init__(self, ignore_index=-1, reduction='none'):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']

        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']

        return outputs.permute(0, 2, 1).contiguous(), targets

    def forward(self, outputs, targets_dict, img_metas=None):
    
        outputs, targets = self.format(outputs, targets_dict)   # [1797,11][]可以看看就是input_loss和dn_loss在这里outputs和targets的区别
        loss_ce = self.loss_ce(outputs, targets.to(outputs.device)) # 这里有错误ValueError: Expected input batch_size (499) to match target batch_size (1497).
        losses = dict(loss_ce=loss_ce)
        return losses


@LOSSES.register_module()
class SARLoss(CELoss):
    """Implementation of loss module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """

    def __init__(self, ignore_index=0, reduction='mean', **kwargs):
        super().__init__(ignore_index, reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()

        return outputs, targets


@LOSSES.register_module()
class TFLoss(CELoss):
    """Implementation of loss module for transformer."""

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.flatten = flatten

    def format(self, outputs, targets_dict):
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets


@LOSSES.register_module()
class MASTERTFLoss(CELoss):
    """Implementation of loss module for transformer."""

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.flatten = flatten

    def format(self, outputs, targets_dict):
        # MASTER already cut the last in decoder.
        #outputs = outputs[:, :-1, :].contiguous()
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        # print("ce:",len(targets[0]))
        if self.flatten:    # True 
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets
    
@LOSSES.register_module()
class spanLoss(CELoss):
    """Implementation of loss module for transformer."""

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.flatten = flatten
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0,reduction='mean')

    def format(self, outputs, targets_dict):
        targets = targets_dict[:,1:].contiguous()
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        return outputs, targets

    def forward(self, outputs, targets_dict):
        pred, targets = self.format(outputs, targets_dict)   # 可以看看就是input_loss和dn_loss在这里outputs和targets的区别
        loss = self.loss_ce(pred, targets)
        # print(loss)
        losses = dict(colrow_span_loss=loss)
        return losses

@LOSSES.register_module()
class colrow_loss(CELoss):
    """Implementation of loss module for transformer."""

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.flatten = flatten
        self.gamma = 2.0
        self.alpha = 0.25
        self.reduction = reduction

    def format(self, outputs, targets_dict):
    
        targets = targets_dict
        sigmoid_a = torch.sigmoid(outputs)
        sigmoid_a = torch.clamp(sigmoid_a, min=1e-7, max=1-1e-7)

        # print('使用colrow_loss的format')
        return sigmoid_a, targets
    def forward(self, outputs, targets_dict, img_metas=None):
        
        targets_dict = targets_dict.to(outputs.device)
        # print('使用colrow的forward')
        sigmoid_a, targets = self.format(outputs, targets_dict)   # 可以看看就是input_loss和dn_loss在这里outputs和targets的区别
       
        # kimi的实现方式
        term1 = (1 - sigmoid_a) ** self.gamma * torch.log(sigmoid_a)
        term2 = sigmoid_a ** self.gamma * torch.log(1-sigmoid_a)
        loss = -self.alpha * term1 * targets - (1 - self.alpha) * term2 * (1 - targets)

        if self.reduction == 'mean':
            loss = loss.mean()
        # print('colrowloss:',loss)
        losses = dict(sigmoid_focal_loss=loss)  # 这里的名字sigmoid_focal_loss就是在计算losss的时候取的
        return losses
    
@LOSSES.register_module()
class ccm_loss(CELoss):
    """Implementation of loss module for transformer."""

    def __init__(self,
                 ignore_index=-100,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=-100, reduction=reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['num_cell']
        return outputs, targets