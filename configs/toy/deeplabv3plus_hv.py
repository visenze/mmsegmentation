comment = "V2 (Only) White & Black BG Crop & Green +  Load from DUTS-Full + Test training with HV "
exp_name = "run55"

# model settings
class_weight = None
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained="open-mmlab://resnet50_v1c",
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
            class_weight=class_weight,
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
            class_weight=class_weight,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# dataset settings
dataset_type = "ToyHVDataset"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromHV"),
    dict(type="LoadAnnotationsHV"),
    # dict(type='BackgroundReplace',
    #     background_folder='/mnt/raid_04/usr/tam.le/data/ferrero_toys/bg_images/toy84_dist_var_may18_crop/',
    # toy_list='/home/tam.le/raid_04/data/ferrero_toys/84_v2_valid/toy_list_with_black.txt',
    # prob=0.4),
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
    dict(type="LoadImageFromHV"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 512),
        # img_ratios=[0.5, 1.0, 2.0],
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

train_data_root = "/home/tam.le/raid_04/data/ferrero_toys/84_v2_valid/train_v1_v2/"
dutr_data_root = "/home/tam.le/raid_04/data/ferrero_toys/DUTS-TR/toy_train/crop_train/"
sod_data_root = (
    "/home/tam.le/raid_04/data/ferrero_toys/sod_training/v2cleancrop_white_black_green/"
)

val_data_root = "/home/tam.le/raid_04/data/ferrero_toys/84_v2_valid/val/black_bg/crop/"

test_data_root = "/home/tam.le/raid_04/data/ferrero_toys/84_v2_valid/test/"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        crop=True,
        raw_dataset="goldenlaunch_64_train_white_black_green_raw_remove_nobox",
        mask_dataset="goldenlaunch_64_train_white_black_green_mask",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        crop=True,
        raw_dataset="goldenlaunch_64_val_black_raw",
        mask_dataset="goldenlaunch_64_val_black_mask",
        pipeline=test_pipeline,
    ),
    test=dict(type=dataset_type, pipeline=test_pipeline),
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
work_dir = (
    f"/home/tam.le/raid_04/exp/ferrero_toys/semantic_seg/deeplabv3plus_r50/{exp_name}"
)
load_from = f"/home/tam.le/raid_04/exp/ferrero_toys/semantic_seg/deeplabv3plus_r50/run18/latest.pth"
# load_from = "/mnt/ssfs/usr/tam.le/code/mmsegmentation/models/deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth"
# load_from = "/mnt/ssfs/usr/tam.le/code/mmsegmentation/models/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth"
resume_from = None
workflow = [("train", 1)]
