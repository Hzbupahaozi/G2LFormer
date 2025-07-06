_base_ = [
    '../../_base_/default_runtime.py'
]


# alphabet_file = '/data/zhuomingli/rongyao/structure_alphabet_total.txt'
# alphabet_file = './tools/data/alphabet/wtw_structure_alphabet.txt'
# alphabet_file = '/data/zml/mmocr_WTW_recognition/try/structure_alphabet.txt'
alphabet_file = '/home/chs/tablemaster-mmocr/tools/data/alphabet/pubtabnet_structure_alphabet.txt'

alphabet_len = len(open(alphabet_file, 'r').readlines())
max_seq_len = 600

start_end_same = False
label_convertor = dict(
            type='TableMasterConvertor',
            dict_file=alphabet_file,
            max_seq_len=max_seq_len,
            start_end_same=start_end_same,
            with_unknown=True)

if start_end_same:
    PAD = alphabet_len + 2
else:
    PAD = alphabet_len + 3

model = dict(
    type='TABLEMASTER',
    backbone=dict(
        type='TableResNetExtra',
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True, True],
        ),
        layers=[1,2,5,3,3]),
    # pretrained='torchvision://resnet50',
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch'),
    encoder=dict(
        # type='PositionalEncodingscale',
        # type = 'PositionalEncoding',
        type = 'Featurescale',
        d_model=512,
        dropout=0.2,
        max_len=5000),
    decoder=dict(
        type='TableMasterConcatDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            src_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            feed_forward=dict(
                d_model=512,
                d_ff=2024,
                dropout=0.),
            size=512,
            dropout=0.),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=PAD, reduction='mean'),
    bbox_loss=dict(type='TableL1Loss', reduction='sum'),
    # iou_loss=dict(type='GIoULoss1', loss_weight=2.0),
    colrow_loss = dict(type='colrow_loss', reduction='mean'),
    span_loss = dict(type='spanLoss', reduction='mean'),
    GIOU_loss = dict(type='GIOU_loss',reduction='mean'),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)


TRAIN_STATE = True
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomRotatePolyInstances'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'scale_factor', 'bbox', 'bbox_masks', 'pad_shape',"cls_bbox", "cell_masks", "tr_masks", "num_cell","colrow_masks"
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'bbox', 'bbox_masks', 'pad_shape'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    #dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'pad_shape','text'
        ]),
]

dataset_type = 'OCRDataset'
# train_img_prefix = '/data/zml/mmocr_WTW_recognition/new'
train_img_prefix = '/data/chs/pubtabnet/train'
# train_img_prefix = '/data/chs/wtw/part_train_img_wtw'

# train_anno_file1 = '/data/zml/mmocr_WTW_recognition/try/StructureLabelAddEmptyBbox_train'
train_anno_file1 = '/data/chs/pubtabnet/chs_pubtabnet_train_txt'
# train_anno_file1 = '/data/chs/wtw/part_train_txt_wtw'


train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

# valid_img_prefix = '/home/zhuomingli/dataset/mmocr_WTW_recognition/test/'
valid_img_prefix = '/data/chs/wtw/test'

# valid_anno_file1 = '/home/zhuomingli/dataset/mmocr_WTW_recognition/StructureLabelAddEmptyBbox_test/'
valid_anno_file1 = '/data/chs/wtw/chs_wtw_test_txt'

val = dict(
    type=dataset_type,
    img_prefix=valid_img_prefix,
    ann_file=valid_anno_file1,
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=valid_pipeline,
    dataset_info='table_master_dataset',
    test_mode=True)

# test_img_prefix = '/home/zhuomingli/dataset/huang/1/test/'
test_img_prefix = '/data/chs/wtw/test'

# test_anno_file1 = '/home/zhuomingli/dataset/mmocr_WTW_recognition/StructureLabelAddEmptyBbox_test/'
test_anno_file1 = '/data/chs/wtw/chs_wtw_test_txt'

test = dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_anno_file1,
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    dataset_info='table_master_dataset',
    test_mode=True)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train1]),
    val=dict(type='ConcatDataset', datasets=[val]),
    test=dict(type='ConcatDataset', datasets=[test]))

# optimizer
optimizer = dict(type='Ranger', lr=1e-3)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    step=[12,15])
total_epochs = 17

# evaluation
evaluation = dict(interval=1, metric='acc')

# fp16
# fp16 = dict(loss_scale='dynamic')  #半精度

# checkpoint setting
checkpoint_config = dict(interval=1)

# log_config
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')

    ])

# yapf:enable
dist_params = dict(backend='gloo')
log_level = 'INFO'
# resume_from = '/home/chs/tablemaster-mmocr/work_dir_chs_wtw0214/epoch_80.pth'
# load_from = '/home/chs/tablemaster-mmocr/work_dir_chs_wtw0204/epoch_90.pth'
# resume_from = None
workflow = [('train', 1)]

# if raise find unused_parameters, use this.
find_unused_parameters = True