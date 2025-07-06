from .base import BaseRecognizer
from .crnn import CRNNNet
from .encode_decode_recognizer import EncodeDecodeRecognizer
from .master import MASTER
from .nrtr import NRTR
from .robust_scanner import RobustScanner
from .sar import SARNet
from .seg_recognizer import SegRecognizer
from .zhuoming_tablemaster_deformable_encoder_VG_select_query_tdtrVG_scheduled import TABLEMASTER



__all__ = [
    'BaseRecognizer', 'EncodeDecodeRecognizer', 'CRNNNet', 'SARNet', 'NRTR',
    'SegRecognizer', 'RobustScanner', 'MASTER', 'TABLEMASTER'
]
