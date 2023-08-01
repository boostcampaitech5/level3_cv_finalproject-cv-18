import base64
import io
import json
import os
import re
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta

import albumentations as A
import cv2
import numpy as np
import psycopg2
import torch
import uvicorn
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, Header
from google.cloud import storage
from PIL import Image, ImageDraw, ImageFont, ImageOps

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

sys.path.append("/opt/ml/level3_cv_finalproject-cv-18/NAFNet")

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor
from basicsr.utils import tensor2img
from basicsr.utils.options import parse

sys.path.append("/opt/ml/level3_cv_finalproject-cv-18/deep-text-recognition-benchmark")

from inference import inference, load_text_recognition_model

opt_path = "/opt/ml/level3_cv_finalproject-cv-18/NAFNet/options/test/REDS/NAFNet-width64.yml"
opt = parse(opt_path, is_train=False)
opt["dist"] = False

deblur_model = create_model(opt)
model = YOLO("/opt/ml/level3_cv_finalproject-cv-18/app/weights/s_new_tot_add9299_mos90_scale05_best.pt")
ocr_model = load_text_recognition_model(
    save_model="/opt/ml/level3_cv_finalproject-cv-18/app/weights/vgg__high_best_accuracy.pth",
    device="cuda",
)
ocr_transform = A.Compose([A.Resize(32, 100), A.Normalize(mean=0, std=1), ToTensorV2()])
width_exp, height_exp = 0.3, 0.3


def img_2_tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def super_resolution(img):  # pil image
    cv2_image = pil_2_cv2(img)
    cv2.imwrite("/opt/ml/level3_cv_finalproject-cv-18/SwinIR/temp/super.png", cv2_image)
    os.system(
        "python /opt/ml/level3_cv_finalproject-cv-18/SwinIR/main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq /opt/ml/level3_cv_finalproject-cv-18/SwinIR/temp"
    )
    image = cv2.imread("/opt/ml/level3_cv_finalproject-cv-18/app/results/swinir_real_sr_x4/super_SwinIR.png", 1)
    image = cv2_2_pil(image)
    return image


def deblurring(model, image):
    model.feed_data(data={"lq": image.unsqueeze(dim=0)})
    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals["result"]])
    return sr_img


# input: 이미지, output: [(x, y, x, y), (x, y, x, y)](좌표)
def detection(image):
    output = model(image, device=0, conf=0.3)
    coordinates = list(map(lambda x: x.boxes.xyxy, output))
    return coordinates


# input: 단일 좌표, 이미지
# output: OCR 결과
def ocr(image, bbox):
    if bbox is not None:
        image = cv2.cvtColor(image[bbox[1] : bbox[3], bbox[0] : bbox[2]], cv2.COLOR_BGR2GRAY)
    input_tensor = ocr_transform(image=image)["image"]
    ocr_result, confidence_score = inference(model=ocr_model, input_tensor=input_tensor, device="cuda")

    res = p.match(ocr_result[0])
    if not (6 < len(ocr_result[0]) < 11 and res):
        ocr_result[0] = "invalid"
    return ocr_result[0], confidence_score[0]


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


p = re.compile("\D{0,5}\d{0,3}\D{1}\d{4}$")


# 이미지 바운딩박스, OCR 결과 계산
@app.post("/image")
def image(file: bytes = File(...), filename=Header(None)):
    begin = time.time()
    db_ocr = []

    pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(file)))
    cv2_image = pil_2_cv2(pil_image)

    filename = (base64.b64decode(filename)).decode("utf-8")

    coordinates = detection(pil_image)[0]
    width, height = pil_image.size
    response = {}
    for i, coordinate in enumerate(coordinates):
        mod_coordinate = coordinate.tolist()
        mod_coordinate[0] -= (
            (coordinate[2] - coordinate[0]) * width_exp
            if coordinate[0] - (coordinate[2] - coordinate[0]) * width_exp > 0
            else coordinate[0]
        )
        mod_coordinate[1] -= (
            (coordinate[3] - coordinate[1]) * height_exp
            if coordinate[1] - (coordinate[3] - coordinate[1]) * height_exp > 0
            else coordinate[1]
        )
        mod_coordinate[2] += (
            (coordinate[2] - coordinate[0]) * width_exp
            if coordinate[2] + (coordinate[2] - coordinate[0]) * width_exp < width
            else width
        )
        mod_coordinate[3] += (
            (coordinate[3] - coordinate[1]) * height_exp
            if coordinate[3] + (coordinate[3] - coordinate[1]) * height_exp < height
            else height
        )
        mod_coordinate = list(map(int, mod_coordinate))
        coordinate = list(map(lambda x: int(x.item()), coordinate))
        ocr_result, confidence_score = ocr(cv2_image, mod_coordinate)

        if confidence_score >= 0.1:
            crop_image = cv2_image[coordinate[1] : coordinate[3], coordinate[0] : coordinate[2]]

            db_ocr.append(ocr_result)

            response[i] = {
                "crop_image": convert_image_2_base64(cv2_2_pil(crop_image)),
                "ocr_result": ocr_result,
                "ocr_confidence_score": confidence_score,
            }
            cv2.rectangle(cv2_image, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (0, 0, 255), 3)
    response["bbox_image"] = convert_image_2_base64(cv2_2_pil(cv2_image))

    # postgresql
    output_time = timedelta(seconds=time.time() - begin)
    file_type = str("image")
    image_input_time = str(datetime.now())
    id = str(hash(image_input_time))
    ocr_ret = str(db_ocr).replace("'", '"')
    image_bbox = str(coordinates).replace("'", '"')

    pil_image.save(str(id) + ".png")

    gcs_directory = f"image/{filename}_{id}"
    gcs_directory.replace("'", '"')
    sql = f"insert into log (id, image_bbox, file_type, ocr_result, gcs_directory, file_input_time, input_process_time) values ('{id}', '{image_bbox}', '{file_type}', '{ocr_ret}', '{gcs_directory}', '{image_input_time}', '{output_time}');"  # noqa: E501
    cursor.execute(sql)

    # GCS
    source_file_name = f"/opt/ml/level3_cv_finalproject-cv-18/app/{id}.png"  # GCP에 업로드할 파일 절대경로
    destination_blob_name = f"image/{filename}_{id}"  # 업로드할 파일을 GCP에 저장할 때의 이름
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    os.remove(f"/opt/ml/level3_cv_finalproject-cv-18/app/{id}.png")
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

    num_frame = 0

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file)
        temp_path = f.name

    frame_dir = os.path.join("record", modified_filename.replace(".mp4", ""))
    os.makedirs(frame_dir, exist_ok=True)
    # with open(temp_path, "wb") as f:
    #     f.write(file)

    capture = cv2.VideoCapture(temp_path)

    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer_local = cv2.VideoWriter(os.path.join("temp", modified_filename), fourcc, fps, (width, height))
    video_writer_cloud = cv2.VideoWriter(os.path.join("temp", hash_filename), fourcc, fps, (width, height))
    font = ImageFont.truetype("fonts/NanumGothicBold.ttf", int(height / 36))

    os.system(f"ffmpeg -r {fps} -i {temp_path} {frame_dir}/%d.png -hide_banner &")

    batch_size = 5
    num_frame = 0
    batch = []

    output_frame = {}
    while capture.isOpened():
        one_frame = time.time()

        ret, frame = capture.read()
        num_frame += 1

        if not ret or num_frame % batch_size == 0:
            batch.append(frame)
            batch = [b for b in batch if b is not None]
            if not batch:
                break
            output = detection(batch)
            stacked_image = []
            num_plates = [len(out) for out in output]
            for frame, coordinates in zip(batch, output):
                db_plate.append(coordinates)  ###
                for coordinate in coordinates:
                    coordinate = coordinate.tolist()
                    coordinate[0] -= (
                        (coordinate[2] - coordinate[0]) * width_exp
                        if coordinate[0] - (coordinate[2] - coordinate[0]) * width_exp > 0
                        else coordinate[0]
                    )
                    coordinate[1] -= (
                        (coordinate[3] - coordinate[1]) * height_exp
                        if coordinate[1] - (coordinate[3] - coordinate[1]) * height_exp > 0
                        else coordinate[1]
                    )
                    coordinate[2] += (
                        (coordinate[2] - coordinate[0]) * width_exp
                        if coordinate[2] + (coordinate[2] - coordinate[0]) * width_exp < width
                        else width
                    )
                    coordinate[3] += (
                        (coordinate[3] - coordinate[1]) * height_exp
                        if coordinate[3] + (coordinate[3] - coordinate[1]) * height_exp < height
                        else height
                    )
                    coordinate = list(map(int, coordinate))
                    crop_frame = cv2.cvtColor(
                        frame[coordinate[1] : coordinate[3], coordinate[0] : coordinate[2]], cv2.COLOR_BGR2GRAY
                    )
                    stacked_image.append(ocr_transform(image=crop_frame)["image"])
            if stacked_image:
                output = torch.cat(output, dim=0)
                input_tensor = torch.stack(stacked_image, dim=0)
                ocr_results, ocr_confs = inference(model=ocr_model, input_tensor=input_tensor, device="cuda")
            index = 0
            for i, (frame, num_plate) in enumerate(zip(batch, num_plates)):
                temp_ocr_db = []  ####
                output_dict = {}
                if num_plate != 0:
                    image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(image)
                    for j, (ocr_result, ocr_conf, xyxy) in enumerate(
                        zip(
                            ocr_results[index : index + num_plate],
                            ocr_confs[index : index + num_plate],
                            output[index : index + num_plate],
                        )
                    ):
                        temp_ocr_db.append(ocr_result)  ###
                        if ocr_conf >= 0.3:
                            res = p.match(ocr_result)
                            if not (6 < len(ocr_result) < 11 and res):
                                ocr_result = "invalid"
                            draw.rectangle(
                                (xyxy[0], xyxy[1], xyxy[2], xyxy[3]), outline=(0, 0, 255), width=int(height / 320)
                            )
                            draw.text(
                                (xyxy[0], xyxy[1] - int(height / 36) - 2),
                                f"{ocr_result}",  # ({round(ocr_conf, 1)})
                                (0, 0, 255),
                                font=font,
                            )
                            xyxy = list(map(lambda x: int(x.item()), xyxy))
                            output_dict[f"car{j}"] = {
                                "coordinate": [(xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])],
                                "OCR": [ocr_result, ocr_conf],
                            }
                    batch[i] = np.asarray(image)
                if ret:
                    output_frame[num_frame - batch_size + i + 1] = output_dict
                else:
                    output_frame[num_frame + 1 - batch_size + i + 1] = output_dict
                index += num_plate
                db_ocr.append(temp_ocr_db)  ###
            for frame in batch:
                video_writer_local.write(frame)
            batch = []
            print(f"한 배치: {timedelta(seconds=time.time() - one_frame)}")
        else:
            batch.append(frame)

    with open(os.path.join(frame_dir, "video.json"), "w") as f:
        json.dump(output_frame, f, indent=4)

    capture.release()
    video_writer_local.release()
    video_writer_cloud.release()

    os.system(
        f"ffmpeg -y -i temp/{modified_filename} -vcodec libx264 -preset ultrafast temp/{modified_filename.replace('.mp4', '_h264.mp4')}"
    )
    os.remove(os.path.join("temp", modified_filename))

    # # postgresql
    output_time = timedelta(seconds=time.time() - begin)
    file_type = str("video")
    image_input_time = str(datetime.now())
    ocr_ret = str(db_ocr).replace("'", '"')
    video_bbox = str(db_plate).replace("'", '"')
    gcs_directory = f"video/{id}_{filename}".replace("'", '"')
    sql = f"insert into log (id, video_bbox, file_type, ocr_result, gcs_directory, file_input_time, input_process_time) values ('{id}', '{video_bbox}', '{file_type}', '{ocr_ret}', '{gcs_directory}', '{image_input_time}', '{output_time}');"  # noqa: E501
    cursor.execute(sql)

    # # GCS
    source_file_name = f"/opt/ml/level3_cv_finalproject-cv-18/app/temp/{hash_filename}"  # GCP에 업로드할 파일 절대경로
    source_file_name = temp_path
    destination_blob_name = f"video/{id}_{filename}"  # 업로드할 파일을 GCP에 저장할 때의 이름
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    os.remove(f"/opt/ml/level3_cv_finalproject-cv-18/app/temp/{hash_filename}")
    os.remove(temp_path)

    return modified_filename.replace(".mp4", "_h264.mp4")


@app.post("/deblur")
def deblur(file: bytes = File(...)):
    pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(file)))
    tensor_image = img_2_tensor(pil_2_cv2(pil_image))
    deblurred_image = cv2.cvtColor(deblurring(deblur_model, tensor_image), cv2.COLOR_BGR2RGB)

    ocr_result, confidence_score = ocr(cv2.cvtColor(deblurred_image, cv2.COLOR_RGB2GRAY), None)
    response = {
        "crop_image": convert_image_2_base64(cv2_2_pil(deblurred_image)),
        "ocr_result": ocr_result,
        "ocr_confidence_score": confidence_score,
    }
    return response


@app.post("/super")
def super(file: bytes = File(...)):
    print("super input")
    pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(file)))
    super_image = super_resolution(pil_image)
    super_image = pil_2_cv2(super_image)

    ocr_result, confidence_score = ocr(cv2.cvtColor(super_image, cv2.COLOR_RGB2GRAY), None)

    response = {
        "super_image": convert_image_2_base64(cv2_2_pil(super_image)),
        "ocr_result": ocr_result,
        "ocr_confidence_score": confidence_score,
    }

    return response


if __name__ == "__main__":
    uvicorn.run(app)
