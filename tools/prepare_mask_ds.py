"""Convert Remove.bg dataset to mask dataset for semantic segmentation
training.

This script only works for turntable toy segmentation data.
"""
import argparse
import cv2
import glob
import numpy as np
import os
import pandas as pd
import re
import tempfile
from data_factory.client.hydravision import (HydraVisionCreateDataset,
                                             HydraVisionGetDataset)
from data_factory.magikarp import read_vis
from data_factory.parser.vis_upload import VISUploader
from tqdm.auto import tqdm


def get_info_reference_ds(dataset):
    """Get concept and box infomation from the referecene dataset."""
    df = HydraVisionGetDataset(dataset).read_dataframe()

    box = dict()
    concept = dict()
    for i, row in df.iterrows():
        concept[row.vis_uri] = row.concepts
        try:
            box[row.vis_uri] = list(map(int, row.box.split(",")))
        except:
            box[row.vis_uri] = []
    return box, concept


def get_ori_vis_uri(labels):
    """Extract the original vis uri field from the label string.

    Example label:
    ':ori_vis_uri=e1fb87ab3667be8f463644d64a795b17,:cam=2,:pose=1,:angle=45,...'
    """
    pattern = "ori_vis_uri=[a-z0-9]+"
    match = re.search(pattern, labels)
    match_str = match.group()
    raw_uri = match_str.split("=")[1]
    return raw_uri


def rmbg2mask(img):
    """Convert RGBA image of Remove.BG to segmentation mask."""
    alpha = img[..., 3]
    mask = np.where(alpha >= 255 // 2, 1, 0)
    mask = mask.astype(np.uint8)
    return mask


def generate_full_mask(img, mask, xmin, ymin):
    """Generate the full image mask from the cropped mask and cropping
    location."""
    mask_h, mask_w = mask.shape[:2]
    new_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    new_mask[ymin : ymin + mask_h, xmin : xmin + mask_w] = mask
    return new_mask


def upload_hv(mask_folder, concept, mask_dataset):
    """Upload mask images to HydraVision.

    Args:
        mask_folder: folder of mask images
        concept: concept_mapping from the raw dataset
        mask_dataset: name of dataset to create
    """

    imgs = glob.glob(os.path.join(mask_folder, "*"))

    df = pd.DataFrame({"im_url": imgs})
    uploader = VISUploader(
        vis_url_column_name="vis_url", remove_download_fail=True, cache_image=False
    )
    df_vis = uploader.process_dataframes([df])[0]

    # Add the concepts based on the original vis uri
    df_vis["concepts"] = df["original_url"].apply(
        lambda x: concept[os.path.basename(x).rsplit(".")[0]]
    )

    df_vis["labels"] = df["original_url"].apply(
        lambda x: ":raw_uri={}".format(os.path.basename(x).rsplit(".")[0])
    )

    df_vis = df_vis[["im_url", "concepts", "labels"]]

    # Upload to hydravision
    new_hv_dataset = HydraVisionCreateDataset(
        dataset_name=mask_dataset, dataset_type="classification", update_existing=False
    )
    new_hv_dataset.write_dataframe(df_vis)


def convert_dataset(raw_dataset, rmbg_dataset, mask_dataset):
    """Generate mask dataset from the raw and Remove.bg dataset.

    Args:
        raw_dataset: Name of raw dataset
        rmbg_dataset: Name of dataset of Remove.bg images
        mask_dataset: Name of mask dataset to create
    """
    box, concept = get_info_reference_ds(raw_dataset)

    # Get mapping between remove.bg image and raw image
    rmbg_df = HydraVisionGetDataset(rmbg_dataset).read_dataframe()
    rmbg_df["orig_vis_uri"] = rmbg_df["labels"].apply(get_ori_vis_uri)
    rmbg_raw_mapping = dict(zip(rmbg_df["vis_uri"], rmbg_df["orig_vis_uri"]))

    with tempfile.TemporaryDirectory() as mask_dir:
        for rmbg_img_uri in tqdm(rmbg_df["vis_uri"], total=len(rmbg_df)):

            # Read image from HydraVision
            img_bytes = read_vis(rmbg_img_uri, silent=True)
            img_np = np.frombuffer(img_bytes, np.uint8)
            rmbg_img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

            mask = rmbg2mask(rmbg_img)

            # Read raw image
            raw_img_uri = rmbg_raw_mapping[rmbg_img_uri]
            img_bytes = read_vis(raw_img_uri, silent=True)
            img_np = np.frombuffer(img_bytes, np.uint8)
            raw_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            # Get box information to generate full image mask
            xmin, ymin = box[raw_img_uri][:2] if len(box[raw_img_uri]) == 4 else [0, 0]

            full_mask = generate_full_mask(raw_img, mask, xmin, ymin)

            # Write the mask down to upload later
            full_mask_path = os.path.join(mask_dir, f"{raw_img_uri}.png")
            cv2.imwrite(full_mask_path, full_mask)

        upload_hv(mask_dir, concept, mask_dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mask dataset for segmentation training from raw and Remove.bg datasets"
    )
    parser.add_argument(
        "--raw",
        required=True,
        help="Name of raw dataset",
    )

    parser.add_argument(
        "--rmbg",
        required=True,
        help="Name of Remove.bg dataset",
    )

    parser.add_argument(
        "--mask",
        required=True,
        help="Name of mask dataset",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(args.raw, args.rmbg, args.mask)
