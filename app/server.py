import base64
import io
import os
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from typing import Tuple

import cv2
import numpy as np
import psycopg2
import uvicorn
from fastapi import FastAPI, File, Header
from google.cloud import storage
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/opt/ml/visionary-database-69080ccaaf36.json"

storage_client = storage.Client()
buckets = list(storage_client.list_buckets())
bucket_name = "file_database"
bucket = storage_client.bucket(bucket_name)


warnings.filterwarnings("ignore")

app = FastAPI()

conn = psycopg2.connect(database="logger", user="admin", password="vision", host="127.0.0.1", port="5432")
conn.autocommit = True
cursor = conn.cursor()

sys.path.append("/opt/ml/level3_cv_finalproject-cv-18/deep-text-recognition-benchmark")

from inference import inference, load_text_recognition_model

sys.path.append("/opt/ml/level3_cv_finalproject-cv-18/app")

model = YOLO("../ultralytics/object-detection/100epoch_cos_mosaic50/weights/best.pt")
ocr_model = load_text_recognition_model(
    save_model="/opt/ml/level3_cv_finalproject-cv-18_/deep-text-recognition-benchmark/best_accuracy.pth", device="cuda"
)


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


def pil_2_cv2(pil_image):
    np_image = np.array(pil_image)
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return cv2_image


def cv2_2_pil(cv2_image):
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image


def convert_image_2_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# 이미지 바운딩박스, OCR 결과 계산
@app.post("/image")
def image(file: bytes = File(...), filename=Header(None)):
    begin = time.time()
    db_ocr = []

    pil_image = Image.open(io.BytesIO(file))
    cv2_image = pil_2_cv2(pil_image)

    filename = (base64.b64decode(filename)).decode("utf-8")

    plate_list = detection(pil_image)

    response = {}
    for i, item in enumerate(plate_list):
        ocr_result, confidence_score = ocr(pil_image, item)
        db_ocr.append(ocr_result)
        crop_image = cv2_2_pil(cv2_image[item[1] : item[3], item[0] : item[2]])
        response[i] = {
            "crop_image": convert_image_2_base64(crop_image),
            "ocr_result": ocr_result,
            "ocr_confidence_score": confidence_score.item(),
        }
        cv2.rectangle(cv2_image, (item[0], item[1]), (item[2], item[3]), (255, 0, 0), 3)
    response["bbox_image"] = convert_image_2_base64(cv2_2_pil(cv2_image))

    # postgresql
    output_time = timedelta(seconds=time.time() - begin)
    file_type = str("image")
    image_input_time = str(datetime.now())
    id = str(hash(image_input_time))
    ocr_ret = str(db_ocr).replace("'", '"')
    image_bbox = str(plate_list).replace("'", '"')

    pil_image.save(str(id) + ".jpg")

    gcs_directory = f"image/{filename}_{id}"
    gcs_directory.replace("'", '"')
    sql = f"insert into log (id, image_bbox, file_type, ocr_result, gcs_directory, image_input_time, input_process_time) values ('{id}', '{image_bbox}', '{file_type}', '{ocr_ret}', '{gcs_directory}', '{image_input_time}', '{output_time}');"  # noqa: E501
    cursor.execute(sql)

    # GCS
    source_file_name = f"/opt/ml/level3_cv_finalproject-cv-18/app/{id}.jpg"  # GCP에 업로드할 파일 절대경로
    destination_blob_name = f"image/{filename}_{id}"  # 업로드할 파일을 GCP에 저장할 때의 이름
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    os.remove(f"/opt/ml/level3_cv_finalproject-cv-18/app/{id}.jpg")

    return response


# 비디오 바운딩박스 OCR 결과 계산
@app.post("/video")
def video(file: bytes = File(...), filename=Header(None)):
    image_input_time = str(datetime.now())
    id = str(abs(hash(image_input_time)))

    filename = (base64.b64decode(filename)).decode("utf-8")
    hash_filename = f"{id}.mp4"
    modified_filename = f"_{hash_filename}"

    begin = time.time()
    db_ocr = []
    db_plate = []

    num_frame, num_ocr = 0, 0

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file)
        temp_path = f.name

    capture = cv2.VideoCapture(temp_path)

    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer_local = cv2.VideoWriter(modified_filename, fourcc, fps, (width, height))
    video_writer_cloud = cv2.VideoWriter(hash_filename, fourcc, fps, (width, height))
    font = ImageFont.truetype("fonts/MaruBuri-Bold.ttf", int(height / 36))

    while capture.isOpened():
        temp_ocr_db = []
        ret, frame = capture.read()
        if not ret:
            break

        video_writer_cloud.write(frame)

        pil_frame = cv2_2_pil(frame)

        plate_list = detection(pil_frame)
        db_plate.append(plate_list)

        for item in plate_list:
            ocr_result, confidence_score = ocr(pil_frame, item)
            temp_ocr_db.append(ocr_result)
            num_ocr += 1

            if confidence_score.item() > 0.5:
                cv2.rectangle(frame, (item[0], item[1]), (item[2], item[3]), (255, 0, 0), 3)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.text(
                    (item[0], item[1] - int(height / 36) - 2),
                    f"{ocr_result}({round(confidence_score.item(), 1)})",
                    (0, 0, 255),
                    font=font,
                )
                frame = np.array(img)

        video_writer_local.write(frame)
        db_ocr.append(temp_ocr_db)
        num_frame += 1

    capture.release()
    video_writer_local.release()
    video_writer_cloud.release()

    os.system(f"ffmpeg -y -i {modified_filename} -vcodec libx264 {modified_filename.replace('.mp4', '_h264.mp4')}")
    os.unlink(temp_path)
    os.remove(modified_filename)

    # postgresql
    output_time = timedelta(seconds=time.time() - begin)
    file_type = str("video")
    image_input_time = str(datetime.now())
    ocr_ret = str(db_ocr).replace("'", '"')
    video_bbox = str(db_plate).replace("'", '"')
    gcs_directory = f"video/{filename}_{id}.mp4".replace("'", '"')
    sql = f"insert into log (id, video_bbox, file_type, ocr_result, gcs_directory, image_input_time, input_process_time) values ('{id}', '{video_bbox}', '{file_type}', '{ocr_ret}', '{gcs_directory}', '{image_input_time}', '{output_time}');"  # noqa: E501
    cursor.execute(sql)

    # GCS
    source_file_name = f"/opt/ml/level3_cv_finalproject-cv-18/app/{hash_filename}"  # GCP에 업로드할 파일 절대경로
    destination_blob_name = f"video/{filename}_{id}"  # 업로드할 파일을 GCP에 저장할 때의 이름
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    os.remove(f"/opt/ml/level3_cv_finalproject-cv-18/app/{hash_filename}")

    return modified_filename.replace(".mp4", "_h264.mp4")


if __name__ == "__main__":
    uvicorn.run(app)
