_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.5
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
backend_args = None
base_lr = 0.0025
batch_shapes_cfg = None
close_mosaic_epochs = 20
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=280,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.33
default_hooks = dict(
    checkpoint=dict(
        interval=20,
        max_keep_ckpts=15,
        save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.0025,
        max_epochs=300,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
infer_type = 'one2one'
last_stage_out_channels = 1024
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
launcher = 'none'
load_from = '/home/lsw/LSW/mi/mi3/checkpoints/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'
log_interval = 300
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
loss_state_weight = 0.5
lr_factor = 0.0025
max_aspect_ratio = 100
max_epochs = 300
max_keep_ckpts = 3
metainfo = dict(
    classes=(
        'bowlor',
        'appleor',
        'mouseor',
        'keyboardor',
        'bananaor',
        'carrotor',
        'cupor',
        'orangeor',
        'chairor',
        'bookor',
        'bowlmi',
        'applemi',
        'mousemi',
        'keyboardmi',
        'bananami',
        'carrotmi',
        'cupmi',
        'orangemi',
        'chairmi',
        'bookmi',
    ),
    palette=[
        (
            0,
            47,
            167,
        ),
        (
            129,
            216,
            207,
        ),
        (
            0,
            49,
            82,
        ),
        (
            176,
            89,
            35,
        ),
        (
            230,
            0,
            0,
        ),
        (
            144,
            0,
            33,
        ),
        (
            251,
            210,
            106,
        ),
        (
            143,
            75,
            40,
        ),
        (
            1,
            132,
            127,
        ),
        (
            64,
            224,
            208,
        ),
        (
            0,
            47,
            167,
        ),
        (
            129,
            216,
            207,
        ),
        (
            0,
            49,
            82,
        ),
        (
            176,
            89,
            35,
        ),
        (
            230,
            0,
            0,
        ),
        (
            144,
            0,
            33,
        ),
        (
            251,
            210,
            106,
        ),
        (
            143,
            75,
            40,
        ),
        (
            1,
            132,
            127,
        ),
        (
            64,
            224,
            208,
        ),
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=0.33,
        last_stage_out_channels=1024,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv10Backbone',
        use_c2fcib=True,
        widen_factor=0.5),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=10,
            num_state=2,
            reg_max=16,
            type='YOLOv10HeadModule',
            widen_factor=0.5),
        infer_type='one2one',
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        loss_state=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv10Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv10PAFPN',
        widen_factor=0.5),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms_pre=30000,
        one2many_withnms=True,
        one2one_withnms=False,
        score_thr=0.001),
    train_cfg=dict(
        one2many_assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=10,
            topk=10,
            type='mi_BatchTaskAlignedAssigner',
            use_ciou=True),
        one2one_assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=10,
            topk=1,
            type='mi_BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms_pre=30000,
    one2many_withnms=True,
    one2one_withnms=False,
    score_thr=0.001)
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 10
num_det_layers = 3
num_state = 2
one2many_tal_alpha = 0.5
one2many_tal_beta = 6.0
one2many_tal_topk = 10
one2one_tal_alpha = 0.5
one2one_tal_beta = 6.0
one2one_tal_topk = 1
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=36,
        lr=0.0025,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = None
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
resume = False
save_epoch_intervals = 300
strides = [
    8,
    16,
    32,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=36,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='images/'),
        data_root='data/coco/',
        metainfo=dict(
            classes=(
                'bowlor',
                'appleor',
                'mouseor',
                'keyboardor',
                'bananaor',
                'carrotor',
                'cupor',
                'orangeor',
                'chairor',
                'bookor',
                'bowlmi',
                'applemi',
                'mousemi',
                'keyboardmi',
                'bananami',
                'carrotmi',
                'cupmi',
                'orangemi',
                'chairmi',
                'bookmi',
            ),
            palette=[
                (
                    0,
                    47,
                    167,
                ),
                (
                    129,
                    216,
                    207,
                ),
                (
                    0,
                    49,
                    82,
                ),
                (
                    176,
                    89,
                    35,
                ),
                (
                    230,
                    0,
                    0,
                ),
                (
                    144,
                    0,
                    33,
                ),
                (
                    251,
                    210,
                    106,
                ),
                (
                    143,
                    75,
                    40,
                ),
                (
                    1,
                    132,
                    127,
                ),
                (
                    64,
                    224,
                    208,
                ),
                (
                    0,
                    47,
                    167,
                ),
                (
                    129,
                    216,
                    207,
                ),
                (
                    0,
                    49,
                    82,
                ),
                (
                    176,
                    89,
                    35,
                ),
                (
                    230,
                    0,
                    0,
                ),
                (
                    144,
                    0,
                    33,
                ),
                (
                    251,
                    210,
                    106,
                ),
                (
                    143,
                    75,
                    40,
                ),
                (
                    1,
                    132,
                    127,
                ),
                (
                    64,
                    224,
                    208,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 36
train_cfg = dict(
    dynamic_intervals=[
        (
            10,
            300,
        ),
    ],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=300)
train_data_prefix = 'images/'
train_dataloader = dict(
    batch_size=36,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'bowlor',
                'appleor',
                'mouseor',
                'keyboardor',
                'bananaor',
                'carrotor',
                'cupor',
                'orangeor',
                'chairor',
                'bookor',
                'bowlmi',
                'applemi',
                'mousemi',
                'keyboardmi',
                'bananami',
                'carrotmi',
                'cupmi',
                'orangemi',
                'chairmi',
                'bookmi',
            ),
            palette=[
                (
                    0,
                    47,
                    167,
                ),
                (
                    129,
                    216,
                    207,
                ),
                (
                    0,
                    49,
                    82,
                ),
                (
                    176,
                    89,
                    35,
                ),
                (
                    230,
                    0,
                    0,
                ),
                (
                    144,
                    0,
                    33,
                ),
                (
                    251,
                    210,
                    106,
                ),
                (
                    143,
                    75,
                    40,
                ),
                (
                    1,
                    132,
                    127,
                ),
                (
                    64,
                    224,
                    208,
                ),
                (
                    0,
                    47,
                    167,
                ),
                (
                    129,
                    216,
                    207,
                ),
                (
                    0,
                    49,
                    82,
                ),
                (
                    176,
                    89,
                    35,
                ),
                (
                    230,
                    0,
                    0,
                ),
                (
                    144,
                    0,
                    33,
                ),
                (
                    251,
                    210,
                    106,
                ),
                (
                    143,
                    75,
                    40,
                ),
                (
                    1,
                    132,
                    127,
                ),
                (
                    64,
                    224,
                    208,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='Mosaic'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 12
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 36
val_cfg = dict(type='ValLoop')
val_data_prefix = 'images/'
val_dataloader = dict(
    batch_size=36,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='images/'),
        data_root='data/coco/',
        metainfo=dict(
            classes=(
                'bowlor',
                'appleor',
                'mouseor',
                'keyboardor',
                'bananaor',
                'carrotor',
                'cupor',
                'orangeor',
                'chairor',
                'bookor',
                'bowlmi',
                'applemi',
                'mousemi',
                'keyboardmi',
                'bananami',
                'carrotmi',
                'cupmi',
                'orangemi',
                'chairmi',
                'bookmi',
            ),
            palette=[
                (
                    0,
                    47,
                    167,
                ),
                (
                    129,
                    216,
                    207,
                ),
                (
                    0,
                    49,
                    82,
                ),
                (
                    176,
                    89,
                    35,
                ),
                (
                    230,
                    0,
                    0,
                ),
                (
                    144,
                    0,
                    33,
                ),
                (
                    251,
                    210,
                    106,
                ),
                (
                    143,
                    75,
                    40,
                ),
                (
                    1,
                    132,
                    127,
                ),
                (
                    64,
                    224,
                    208,
                ),
                (
                    0,
                    47,
                    167,
                ),
                (
                    129,
                    216,
                    207,
                ),
                (
                    0,
                    49,
                    82,
                ),
                (
                    176,
                    89,
                    35,
                ),
                (
                    230,
                    0,
                    0,
                ),
                (
                    144,
                    0,
                    33,
                ),
                (
                    251,
                    210,
                    106,
                ),
                (
                    143,
                    75,
                    40,
                ),
                (
                    1,
                    132,
                    127,
                ),
                (
                    64,
                    224,
                    208,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_interval_stage2 = 300
val_num_workers = 8
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.0005
widen_factor = 0.5
work_dir = 'jiyan'