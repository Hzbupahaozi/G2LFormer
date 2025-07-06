from mmcv.utils import Registry, build_from_cfg
# from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
RECOGNIZERS = Registry('recognizer')
CONVERTORS = Registry('convertor')
ENCODERS = Registry('encoder')
DECODERS = Registry('decoder')
PREPROCESSOR = Registry('preprocessor')
# BACKBONES = Registry('models', parent=MMDET_BACKBONES)
# BACKBONES = Registry('backbone')

# def build_backbone(cfg):
#     """Build backbone."""
    # return BACKBONES.build(cfg)

# def build_loss(cfg):
#     """Build loss."""
#     return LOSSES.build(cfg)
        
def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """Build recognizer."""
    return build_from_cfg(cfg, RECOGNIZERS,
                          dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_convertor(cfg):
    """Build label convertor for scene text recognizer."""
    return build_from_cfg(cfg, CONVERTORS)


def build_encoder(cfg):
    """Build encoder for scene text recognizer."""
    return build_from_cfg(cfg, ENCODERS)


def build_decoder(cfg):
    """Build decoder for scene text recognizer."""
    return build_from_cfg(cfg, DECODERS)


def build_preprocessor(cfg):
    """Build preprocessor for scene text recognizer."""
    return build_from_cfg(cfg, PREPROCESSOR)
