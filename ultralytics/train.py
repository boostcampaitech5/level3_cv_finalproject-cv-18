# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
# model.train(data="carplate.yaml", epochs=100, name='n_mos50_super(train)', close_mosaic=50) #, degrees=90)

import argparse
import json
import os
import shutil
import subprocess

import yaml
from ultralytics import YOLO

CONFIG_PATH = "/opt/ml/ultralytics/ultralytics/yolo/cfg/custom_configs/start"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO v8 학습 코드")

    parser.add_argument("--yaml_file", help="학습 파라미터를 저장하는 yaml 파일의 이름", required=True)
    args = parser.parse_args()
    with open(os.path.join(CONFIG_PATH, args.yaml_file), "r") as config_yaml_file:
        config = yaml.load(config_yaml_file, Loader=yaml.FullLoader)

    model = YOLO(config["model"])

    del config["model"]
    config["name"] = args.yaml_file[:-5]

    model.train(**config)