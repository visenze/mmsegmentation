"""Run ONNX Segmentation model."""
import argparse
import cv2
import data_factory.client.hydravision as hv
import glob
import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import os.path as osp
from abc import abstractmethod
from data_factory.magikarp import read_vis
from tqdm.auto import tqdm

INPUT_SIZE = (2048, 512)
IMG_NORM_MEAN = [123.675, 116.28, 103.53]
IMG_NORM_STD = [58.395, 57.12, 57.375]


def load_img(filename, box=None):

    if osp.isfile(filename):
        img = mmcv.imread(filename)
    else:
        img_bytes = read_vis(filename, silent=True)
        img = mmcv.imfrombytes(img_bytes)

    if box is not None:
        xmin, ymin, xmax, ymax = box
        img = img[ymin:ymax, xmin:xmax]

    return img


def parse_box_str(box_str):
    """Convert box string in form "xmin, ymin, xmax, ymax" to list of integer
    coordinates."""
    if not box_str:
        return None

    return list(map(int, box_str.split(",")))


class ONNXRunner:
    def __init__(self, model, input_size, mean, std):
        self.model = model
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.sess = rt.InferenceSession(self.model)

    def _process(self, input_path, box=None):

        data = dict(filename=input_path, box=box)
        data["orig_img"] = load_img(data["filename"], box=data["box"])
        data["orig_shape"] = data["orig_img"].shape[:2]

        self._pre_process(data, self.input_size, self.mean, self.std)

        input_name = self.sess.get_inputs()[0].name
        data["result"] = self.sess.run(None, {input_name: data["img"]})[0][0]

        self._post_process(data)

        return data

    def _pre_process(self, data, input_size, mean, std):

        img = data["orig_img"]

        # 1. Rescale image keeping the ratio
        img_resize = mmcv.imrescale(img, input_size)

        # 2. Normalize Image
        norm_img = mmcv.imnormalize(img_resize, mean, std, to_rgb=True)

        # 3. HWC -> CHW
        norm_img = np.moveaxis(norm_img, -1, 0)

        # 4. Add batch dimension
        norm_img = norm_img[np.newaxis, ...]

        data["img"] = norm_img

    def _post_process(self, data):

        orig_shape = data["orig_shape"]
        result = data["result"]

        # CHW -> HWC
        result = np.moveaxis(result, 0, -1)

        # TODO: Use the palette instead. This only work for binary case
        result = result * 255

        # In the pytorch model, resizing step is performed on the logit map
        # and bilinear interpolation is used.
        # Therefore, the output is expected to be slightly different.
        post_result = mmcv.imresize(
            result.astype(np.uint8),
            (orig_shape[1], orig_shape[0]),
            interpolation="nearest",
        )

        data["post_result"] = post_result

    def process(self, input_imgs, output, box_mapping=None):

        for input_img in tqdm(input_imgs):

            if box_mapping is not None:
                box = box_mapping[input_img]
            else:
                box = None

            result = self._process(input_img, box)
            pred_mask = result["post_result"]

            self.save_output(input_img, pred_mask, output)

            # # How to define the output name
            # if osp.isdir(dataset):
            #     # Local dataset
            #     out_filename = osp.basename(input_img).rsplit(".", 1)[0] + ".png"
            # else:
            #     # HydraVisionDataset
            #     out_filename = input_img + ".png"

            # out_path = osp.join(output, out_filename)

            # # Saving output
            # mmcv.imwrite(pred_mask, out_path)

    @abstractmethod
    def save_output(self, input_img, pred_mask, output):
        raise NotImplementedError


class MissingBoxError(Exception):
    pass


class LocalONNXRunner(ONNXRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_box_mapping(self, img_paths, box_ref_dataset):

        df = hv.HydraVisionGetDataset(dataset_name=box_ref_dataset).read_dataframe()
        df["parsed_box"] = df.box.apply(parse_box_str)

        imguri_box = dict(zip(df["vis_uri"], df["parsed_box"]))
        imgpath_box = dict()

        missing_box = 0

        for img_path in img_paths:
            img_uri = osp.basename(img_path).rsplit(".", 1)[0]
            imgpath_box[img_path] = imguri_box.get(img_uri, None)

            if imgpath_box[img_path] is None:
                missing_box += 1

        if missing_box > 0:
            raise MissingBoxError(f"{missing_box} image don't have box.")

        return imgpath_box

    def process(self, folder, output, use_box):

        if osp.isdir(folder):
            input_imgs = glob.glob(osp.join(folder, "*"))
        else:
            raise ValueError(f"{folder} not exist.")

        super().process(input_imgs, output, box_mapping=None)

    def save_output(self, img_path, mask, output_folder):
        out_filename = osp.basename(img_path).rsplit(".", 1)[0] + ".png"
        out_path = osp.join(output_folder, out_filename)
        # Saving output
        mmcv.imwrite(mask, out_path)


class HVONNXRunner(ONNXRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_box_mapping(self, dataset):

        df = hv.HydraVisionGetDataset(dataset_name=dataset).read_dataframe()
        df["parsed_box"] = df.box.apply(parse_box_str)

        missing_box = df["parsed_box"]

        imguri_box = dict(zip(df["vis_uri"], df["parsed_box"]))

        missing_box = df.parsed_box.isnull().sum()

        if missing_box > 0:
            raise MissingBoxError(f"{missing_box} image don't have box.")

        return imguri_box

    def process(self, dataset, output, use_box=True):

        df = hv.HydraVisionGetDataset(dataset_name=dataset).read_dataframe()
        input_imgs = df.vis_uri.to_list()

        if use_box:
            box_mapping = self._get_box_mapping(dataset)
        else:
            box_mapping = None

        super().process(input_imgs, output, box_mapping)

    def save_output(self, img_uri, mask, output_folder):
        out_filename = img_uri + ".png"
        out_path = osp.join(output_folder, out_filename)
        # Saving output
        mmcv.imwrite(mask, out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX Segmentation model")
    parser.add_argument("model", help="ONNX model file")
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
    args = parser.parse_args()

    if args.source == "local" and args.use_box:
        raise ValueError(
            "Cropping is not supported for local data. The flag '--use-box' will have no effect"
        )

    return args


def run_onnx():
    args = parse_args()

    if args.source == "hv":
        onnx_runner = HVONNXRunner(
            model=args.model,
            input_size=INPUT_SIZE,
            mean=IMG_NORM_MEAN,
            std=IMG_NORM_STD,
        )
    elif args.source == "local":
        onnx_runner = LocalONNXRunner(
            model=args.model,
            input_size=INPUT_SIZE,
            mean=IMG_NORM_MEAN,
            std=IMG_NORM_STD,
        )

    onnx_runner.process(args.inp, args.out, use_box=args.use_box)


if __name__ == "__main__":
    run_onnx()
