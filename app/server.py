import io
import os
import sys
import tempfile
import time
import warnings
from datetime import timedelta
from typing import Tuple

import cv2
import uvicorn
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
from PIL import Image

from ultralytics import YOLO

warnings.filterwarnings("ignore")

app = FastAPI()

sys.path.append("/opt/ml/level3_cv_finalproject-cv-18/deep-text-recognition-benchmark")

from inference import inference, load_text_recognition_model

ocr_model = load_text_recognition_model(save_model="/opt/ml/serving/best_accuracy.pth", device="cuda")

sys.path.append("/opt/ml/level3_cv_finalproject-cv-18/app")

model = YOLO("../ultralytics/object-detection/100epoch_cos_mosaic50/weights/best.pt")


# input: 이미지, output: [(x, y, x, y), (x, y, x, y)](좌표)
def detection(image):
    image = image.convert("RGB")
    results = model(image)
    plate_list = []
    boxes = results[0].boxes.cpu().numpy()
    for box in boxes:
        item = tuple([int(xy) for xy in box.xyxy[0].astype(int)])
        plate_list.append(item)
    return plate_list


# input: 단일 좌표, 이미지
# output: OCR 결과
def ocr(image, box: Tuple) -> str:
    image = image.convert("L")
    cropped_box = image.crop((*box,))
    ocr_result, confidence_score = inference(model=ocr_model, img_array=cropped_box, device="cuda")
    return ocr_result, confidence_score


def calc_avg(avg: float, val: timedelta, idx: int) -> float:
    return idx / (idx + 1) * avg + val.total_seconds() / (idx + 1)


# 이미지 바운딩박스, OCR 결과 계산
@app.post("/image")
def image(file: bytes = File(...)):
    image = Image.open(io.BytesIO(file))

    start = time.time()
    plate_list = detection(image)
    detection_time = timedelta(seconds=time.time() - start)

    output = {}
    ocr_avg_time = 0

    for i, item in enumerate(plate_list):
        start = time.time()
        ocr_result, confidence_score = ocr(image, item)
        ocr_avg_time = calc_avg(ocr_avg_time, timedelta(seconds=time.time() - start), i)

        output[f"car{i}"] = {
            "coordinate": [(item[0], item[1]), (item[2], item[3])],
            "OCR": [ocr_result, confidence_score.item()],
        }

    output["time"] = {"detection": (1, detection_time.total_seconds()), "ocr": (i + 1, ocr_avg_time)}

    return JSONResponse(content=output)


# 비디오 바운딩박스 OCR 결과 계산
@app.post("/video")
def video(file: bytes = File(...)):
    detection_avg_time = 0
    ocr_avg_time = 0
    num_frame, num_ocr = 0, 0

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file)
        temp_path = f.name

    capture = cv2.VideoCapture(temp_path)
    output_list = {}

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = {}

        if num_frame == 1:
            pass

        start = time.time()
        plate_list = detection(frame)
        detection_avg_time = calc_avg(detection_avg_time, timedelta(seconds=time.time() - start), num_frame)

        for i, item in enumerate(plate_list):
            start = time.time()
            ocr_result, confidence_score = ocr(frame, item)
            ocr_avg_time = calc_avg(ocr_avg_time, timedelta(seconds=time.time() - start), num_ocr)
            num_ocr += 1

            output[f"car{i}"] = {
                "coordinate": [(item[0], item[1]), (item[2], item[3])],
                "OCR": [ocr_result, confidence_score.item()],
            }

        output_list[f"frame{num_frame}"] = output
        num_frame += 1

    output_list["time"] = {"detection": (num_frame + 1, detection_avg_time), "ocr": (num_ocr + 1, ocr_avg_time)}

    capture.release()
    os.unlink(temp_path)

    return JSONResponse(content=output_list)


# database -> sqlquery("insert into !~~~#$@$")
# image -> GCS

if __name__ == "__main__":
    uvicorn.run(app)
