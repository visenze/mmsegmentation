"""HydraVision Dataset."""
import data_factory.client.hydravision as hv
import mmcv
import os.path as osp
import re
from data_factory.magikarp import read_vis
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose


@DATASETS.register_module()
class HydraVisionDataset(CustomDataset):
    """HydraVisionDataset."""

    CLASSES = None
    PALETTE = None

    def __init__(
        self,
        pipeline,
        raw_dataset,
        mask_dataset=None,
        crop=False,
        test_mode=False,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
    ):

        self.pipeline = Compose(pipeline)
        self.raw_dataset = raw_dataset
        self.mask_dataset = mask_dataset
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)
        self.crop = crop

        self.img_infos = self.load_annotations(self.raw_dataset, self.mask_dataset)

    def load_annotations(self, raw_dataset, mask_dataset):
        """Load annotation from hydravision dataset."""

        raw_df = hv.HydraVisionGetDataset(dataset_name=raw_dataset).read_dataframe()

        if self.mask_dataset:
            mask_df = hv.HydraVisionGetDataset(
                dataset_name=mask_dataset
            ).read_dataframe()
            raw_mask_mapping = self._load_raw_mask_mapping(mask_df)

        img_infos = [
            dict(filename=uri, box=parse_box_str(box), crop=self.crop)
            for uri, box in zip(raw_df["vis_uri"], raw_df["box"])
        ]

        img_with_no_box = 0

        if self.mask_dataset:
            for img_info in img_infos:
                raw_uri = img_info["filename"]
                if img_info["box"] is None:
                    img_with_no_box += 1

                img_info["ann"] = dict(seg_map=raw_mask_mapping[raw_uri])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        if img_with_no_box > 0:
            print_log(
                f"{img_with_no_box} images don't have bounding box",
                logger=get_root_logger(),
            )
        return img_infos

    def _load_raw_mask_mapping(self, mask_df):
        """Create a mapping between raw and mask image uri.

        The mask dataframe should contain `labels` column in which there
        is a field `raw_uri=xxxxxx`
        """

        mask_df = mask_df.copy()

        mask_df["raw_uri"] = mask_df.labels.apply(extract_raw_uri)
        mapping = dict(zip(mask_df["raw_uri"], mask_df["vis_uri"]))
        return mapping

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results["seg_fields"] = []
        results["raw_dataset"] = self.raw_dataset
        results["mask_dataset"] = self.mask_dataset
        if self.custom_classes:
            results["label_map"] = self.label_map

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            filename = img_info["ann"]["seg_map"]
            img_bytes = read_vis(filename, silent=True)
            gt_seg_map = mmcv.imfrombytes(img_bytes, flag="unchanged", backend="pillow")
            if img_info["crop"]:
                if img_info["box"] is None:
                    raise ValueError("Cannot crop image not having bounding box")
                xmin, ymin, xmax, ymax = img_info["box"]

                gt_seg_map = gt_seg_map[ymin:ymax, xmin:xmax]

            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps


def extract_raw_uri(labels):
    """Extract the raw uri field from the label string.

    Example of label string:
    ':raw_uri=e1fb87ab3667be8f463644d64a795b17,:cam=2,:pose=1,:angle=45,:dist=32,:light=2'
    """
    pattern = "raw_uri=[a-z0-9]+"
    match = re.search(pattern, labels)
    match_str = match.group()

    raw_uri = match_str.split("=")[1]
    return raw_uri


def parse_box_str(box_str):
    """Convert box string in form "xmin, ymin, xmax, ymax" to list of integer
    coordinates."""
    if not box_str:
        return None

    return list(map(int, box_str.split(",")))
