import os.path as osp

from .builder import DATASETS
from .hv import HydraVisionDataset


@DATASETS.register_module()
class ToyHVDataset(HydraVisionDataset):
    """Toy dataset from HydraVisionDataset

    """

    CLASSES = ('background', 'toy')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
