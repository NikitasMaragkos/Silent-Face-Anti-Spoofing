# -*- coding: latin-1 -*-
# @Time : 22-9-22
# @Author : Nikitas Maragkos
# @Company : Nikitas Maragkos
# @File : convert_pytorch_to_tflite.py
# @Software : PyCharm

import os
import cv2
import argparse
import warnings

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from test import check_image

warnings.filterwarnings('ignore')
SAMPLE_IMAGE_PATH = "./images/sample/"


def convert(image_name, model_dir, model, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    # find all models in directory
    total_models = os.listdir(model_dir)
    # identify which model we want to use
    if model != 'all':
        if model in total_models:
            total_models = [model]
    # sum the prediction from single model's result
    for model_name in total_models:
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        model_test.convert_pytorch_to_tflite(img, os.path.join(model_dir, model_name))


if __name__ == "__main__":
    desc = "convert_pytorch_to_tflite"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--model",
        type=str,
        default='2.7_80x80_MiniFASNetV2.pth',
        help="certain model to use for conversion")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_T1.jpg",
        help="image used to convert properly our model")
    args = parser.parse_args()
    convert(args.image_name, args.model_dir, args.model, args.device_id)
