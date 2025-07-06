from .nrtr_modality_transformer import NRTRModalityTransform
from .resnet31_ocr import ResNet31OCR
from .very_deep_vgg import VeryDeepVgg
from .resnet_extra import ResNetExtra
from .deformable_table_resnet_extra import TableResNetExtra
# from .table_resnet_extra import TableResNetExtra
from .resnet import ResNet1

__all__ = ['ResNet31OCR', 'VeryDeepVgg', 'NRTRModalityTransform', 'ResNetExtra', 'TableResNetExtra','ResNet1']#'ResNet'