comment = "(Local) Stage 2 Train with Remove.bg Toy data"
exp_name = "stage_2_local"

# model settings
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained="/mnt/raid_04/usr/tam.le/exp/ferrero_toys/semantic_seg/deeplabv3plus_r50/resnet50_v1c-2cccc1ad.pth",
    backbone=dict(
        type="ResNetV1c",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
    ),
    decode_head=dict(
        type="DepthwiseSeparableASPPHead",
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=None,
        ),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=None,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# dataset settings
dataset_type = "ToyDataset"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="BackgroundReplace",
        bg_dataset="/mnt/raid_04/usr/tam.le/data/ferrero_toys/bg_images/black_green_material/",
        prob=0.4,
    ),
    dict(type="Resize", img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

sod_data_root = (
    "/home/tam.le/raid_04/data/ferrero_toys/sod_training/v2cleancrop_white_black_green/"
)

val_data_root = "/home/tam.le/raid_04/data/ferrero_toys/84_v2_valid/val/black_bg/crop/"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=sod_data_root,
        img_dir="raw",
        ann_dir="mask",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=val_data_root,
        img_dir="raw",
        ann_dir="mask_semantic",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_dir="/home/tam.le/raid_04/data/ferrero_toys/84_v2_valid/test/crop/raw",
        pipeline=test_pipeline,
    ),
)


# optimizer
optimizer = dict(type="SGD", lr=0.01 / 4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="poly", power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric="mIoU")


# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)

# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = f"./work_dir/{exp_name}"
load_from = f"/home/tam.le/raid_04/exp/ferrero_toys/semantic_seg/deeplabv3plus_r50/run18/latest.pth"
resume_from = None
workflow = [("train", 1)]
