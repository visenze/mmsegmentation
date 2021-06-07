from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import (multi_gpu_inference, multi_gpu_test, save_seg_result,
                   single_gpu_inference, single_gpu_test)
from .train import get_root_logger, set_random_seed, train_segmentor

__all__ = [
    "get_root_logger",
    "set_random_seed",
    "train_segmentor",
    "init_segmentor",
    "inference_segmentor",
    "multi_gpu_test",
    "single_gpu_test",
    "show_result_pyplot",
    "single_gpu_inference",
    "save_seg_result",
]
