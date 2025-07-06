from .ce_loss import CELoss, SARLoss, TFLoss, MASTERTFLoss,spanLoss,colrow_loss,ccm_loss
from .ctc_loss import CTCLoss
from .seg_loss import SegLoss
from .dist_loss import TableL1Loss, GIOU_loss
from .iou_loss import GIoULoss1

__all__ = ['CELoss', 'SARLoss', 'CTCLoss', 'TFLoss', 'SegLoss', 'MASTERTFLoss', 'TableL1Loss','GIoULoss1','spanLoss', 'colrow_loss', 'ccm_loss','GIOU_loss']
