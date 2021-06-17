import argparse
import cv2
import mmcv
import os
import torch
import warnings
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import (multi_gpu_inference, save_seg_result,
                        single_gpu_inference)
from mmseg.datasets import ToyHVDataset, build_dataloader, build_dataset
from mmseg.models import build_segmentor

IMG_NORM_CFG = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

TEST_PIPELINE_TEMPLATE = [
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **IMG_NORM_CFG),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    )
]

SUPPORT_SOURCE = ["hv", "local"]


def get_dataset_config(source):
    if source not in SUPPORT_SOURCE:
        raise ValueError(
            "Not supported source. Available are: {}".format(SUPPORT_SOURCE)
        )
    pipeline = TEST_PIPELINE_TEMPLATE.copy()
    if source == "hv":
        pipeline.insert(0, dict(type="LoadImageFromHV"))
        dataset_type = "ToyHVDataset"
    elif source == "local":
        pipeline.insert(0, dict(type="LoadImageFromFile"))
        dataset_type = "ToyDataset"

    dataset = dict(
        type=dataset_type,
        pipeline=pipeline,
        test_mode=True,
    )

    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="mmseg test (and eval) a model")
    parser.add_argument("config", help="model config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--source", choices=["hv", "local"], required=True, help="Source of dataset"
    )
    parser.add_argument(
        "--inp",
        required=True,
        help="Path to input folder in local model or Hydravision dataset name in HV mode",
    )
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument(
        "--use-box",
        action="store_true",
        help="Use provided box to crop image. Only support in Hydravision mode",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.source == "local" and args.use_box:
        raise ValueError(
            "Cropping is not supported for local data. The flag '--use-box' will have no effect"
        )

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    dataset_cfg = get_dataset_config(args.source)
    if args.source == "local":
        dataset_cfg["img_dir"] = args.inp
    elif args.source == "hv":
        dataset_cfg["raw_dataset"] = args.inp
        dataset_cfg["crop"] = args.use_box

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher)

    # build the dataloader
    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=4, dist=distributed, shuffle=False
    )

    # build the model and load checkpoint
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    test_cfg = cfg.get("test_cfg")
    model = build_segmentor(cfg.model, test_cfg=test_cfg)

    # Not supported yet
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint["meta"]["CLASSES"]
    model.PALETTE = checkpoint["meta"]["PALETTE"]

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_inference(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_inference(model, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            save_seg_result(outputs, args.out)


if __name__ == "__main__":
    main()
