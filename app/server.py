import io
import os
import sys
import tempfile
import time
import warnings
from datetime import timedelta, datetime
from typing import Tuple
import psycopg2
from flask import request
import base64

import cv2
import uvicorn
from fastapi import FastAPI, File, Header
from fastapi.responses import JSONResponse
from PIL import Image

from ultralytics import YOLO
#####
from google.cloud import storage
from google.oauth2 import service_account

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/opt/ml/serving/visionary-database-69080ccaaf36.json"

storage_client = storage.Client()
buckets = list(storage_client.list_buckets())
bucket_name = 'file_database'
bucket = storage_client.bucket(bucket_name)
#####


warnings.filterwarnings("ignore")

app = FastAPI()

conn = psycopg2.connect(
   database="logger", user='admin', password='vision', host='127.0.0.1', port= '5432'
)
conn.autocommit = True
cursor = conn.cursor()

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
def image(file: bytes = File(...), filename = Header(None)):
    begin = time.time()
    db_ocr = []
    image = Image.open(io.BytesIO(file))
 
    filename = (base64.b64decode(filename)).decode('utf-8')

    start = time.time()
    plate_list = detection(image)
    detection_time = timedelta(seconds=time.time() - start)

    output = {}
    ocr_avg_time = 0

    for i, item in enumerate(plate_list):
        start = time.time()
        ocr_result, confidence_score = ocr(image, item)
        db_ocr.append(ocr_result)
        ocr_avg_time = calc_avg(ocr_avg_time, timedelta(seconds=time.time() - start), i)

        output[f"car{i}"] = {
            "coordinate": [(item[0], item[1]), (item[2], item[3])],
            "OCR": [ocr_result, confidence_score.item()],
        }

    output["time"] = {"detection": (1, detection_time.total_seconds()), "ocr": (i + 1, ocr_avg_time)}
    # postgresql
    output_time = timedelta(seconds=time.time() - begin)
    file_type = str("image")
    image_input_time = str(datetime.now())
    id = str(hash(image_input_time))
    ocr_ret = str(db_ocr).replace('\'', '"')
    image_bbox = str(plate_list).replace('\'', '"') 

    im1 = image.save(str(id)+".jpg")

    gcs_directory = f"image/{filename}_{id}"
    gcs_directory.replace('\'', '"')
    sql = f"insert into log (id, image_bbox, file_type, ocr_result, gcs_directory, image_input_time, input_process_time) values ('{id}', '{image_bbox}', '{file_type}', '{ocr_ret}', '{gcs_directory}', '{image_input_time}', '{output_time}');"
    print(sql)
    cursor.execute(sql)
    print("log for image input saved")
    # GCS
    source_file_name = f'/opt/ml/level3_cv_finalproject-cv-18/app/{id}.jpg'    # GCP에 업로드할 파일 절대경로
    destination_blob_name = f'image/{filename}_{id}'    # 업로드할 파일을 GCP에 저장할 때의 이름
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    os.remove(f'/opt/ml/level3_cv_finalproject-cv-18/app/{id}.jpg')
    

    return JSONResponse(content=output)


# 비디오 바운딩박스 OCR 결과 계산
@app.post("/video")
def video(file: bytes = File(...), filename = Header(None)):

    filename = (base64.b64decode(filename)).decode('utf-8')

    begin = time.time()
    db_ocr = []
    db_plate = []

    detection_avg_time = 0
    ocr_avg_time = 0
    num_frame, num_ocr = 0, 0

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file)
        temp_path = f.name

    capture = cv2.VideoCapture(temp_path)

    image_input_time = str(datetime.now())
    id = str(hash(image_input_time))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    capture = cv2.VideoCapture(temp_path)

    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{id}.mp4", fourcc, fps, (width, height))

    output_list = {}

    while capture.isOpened():
        
        temp_ocr_db = []
        ret, frame = capture.read()
        if not ret:
            break

        out.write(frame)

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = {}

        if num_frame == 1:
            pass

        start = time.time()
        plate_list = detection(frame)
        
        detection_avg_time = calc_avg(detection_avg_time, timedelta(seconds=time.time() - start), num_frame)
        db_plate.append(plate_list)
        for i, item in enumerate(plate_list):
            start = time.time()
            ocr_result, confidence_score = ocr(frame, item)
            ocr_avg_time = calc_avg(ocr_avg_time, timedelta(seconds=time.time() - start), num_ocr)
            temp_ocr_db.append(ocr_result)
            num_ocr += 1

            output[f"car{i}"] = {
                "coordinate": [(item[0], item[1]), (item[2], item[3])],
                "OCR": [ocr_result, confidence_score.item()],
            }

        output_list[f"frame{num_frame}"] = output
        num_frame += 1
        db_ocr.append(temp_ocr_db)

    out.release()

    output_list["time"] = {"detection": (num_frame + 1, detection_avg_time), "ocr": (num_ocr + 1, ocr_avg_time)}

    capture.release()
    os.unlink(temp_path)
    # postgresql
    output_time = timedelta(seconds=time.time() - begin)
    file_type = str("video")
    image_input_time = str(datetime.now())
    ocr_ret = str(db_ocr).replace('\'', '"')
    video_bbox = str(db_plate).replace('\'', '"') 
    gcs_directory = f"video/{filename}_{id}.mp4"
    gcs_directory.replace('\'', '"')
    sql = f"insert into log (id, video_bbox, file_type, ocr_result, gcs_directory, image_input_time, input_process_time) values ('{id}', '{video_bbox}', '{file_type}', '{ocr_ret}', '{gcs_directory}', '{image_input_time}', '{output_time}');"
    print(sql)
    cursor.execute(sql)
    print("log for video input saved")
    # GCS
    source_file_name = f'/opt/ml/level3_cv_finalproject-cv-18/app/{id}.mp4'    # GCP에 업로드할 파일 절대경로
    destination_blob_name = f'video/{filename}_{id}'    # 업로드할 파일을 GCP에 저장할 때의 이름
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    os.remove(f'/opt/ml/level3_cv_finalproject-cv-18/app/{id}.mp4')
    

    return JSONResponse(content=output_list)


if __name__ == "__main__":
    uvicorn.run(app)
