import os
import tempfile
import time
from ast import literal_eval
from copy import deepcopy
from datetime import timedelta

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from requests_toolbelt.multipart.encoder import MultipartEncoder

# interact with FastAPI endpoint
backend_image = "http://127.0.0.1:8000/image"
backend_video = "http://127.0.0.1:8000/video"


def image_process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/png")})
    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)

    return r


def video_process(video, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", video, "video/mp4")})
    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)

    return r


# construct UI layout
st.title("블랙박스 번호판 찾기")

input = st.file_uploader("영상 또는 이미지를 업로드하여 번호판 인식")  # image upload widget
if input:
    filename = input.name

if st.button("번호판 찾기"):
    if input:
        # 입력값이 이미지일 경우
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # detection 결과를 받아서 bounding box를 쳐줌
            with st.spinner("로딩중..."):
                segments = image_process(input, backend_image)
                my_dict = literal_eval(segments.content.decode("utf-8"))
                file_bytes = np.asarray(bytearray(input.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                original_image = deepcopy(opencv_image)
                # 가능한 색 -> 하나만 해도 될거같다
                pallete = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (200, 200, 200),
                    (200, 100, 100),
                    (100, 200, 100),
                    (100, 100, 200),
                ]
                i = 0
                for key, value in my_dict.items():
                    for k, v in value.items():
                        if k == "coordinate":
                            # cv2.rectangle(opencv_image, v[0], v[1], pallete[i], 3)
                            cv2.rectangle(opencv_image, v[0], v[1], (255, 0, 0), 3)
                            i += 1
                st.image(opencv_image, use_column_width=True, channels="BGR")
            # OCR 결과
            for key, value in my_dict.items():
                with st.container():
                    col1, col2 = st.columns(2)
                    for k, v in value.items():
                        # 각 bounding box에 해당하는 영역을 crop해 출력
                        if k == "coordinate":
                            cropped_img = original_image[v[0][1] : v[1][1], v[0][0] : v[1][0]]
                            # print([v[0][0], v[1][0], v[0][1], v[1][1]], cropped_img.shape, opencv_image.shape)
                            col1.image(cropped_img, use_column_width=True, channels="BGR")
                        # 각 bounding box에 해당하는 번호판을 crop된 이미지 옆에 출력
                        elif k == "OCR":
                            col2.text(f"해당 번호판의 번호는 {round(v[1], 1) * 100}%의 신뢰도로 {v[0]}입니다.")
            print(f"detection: {my_dict['time']['detection'][0]}번 평균 시간{my_dict['time']['detection'][1]:.3f}")
            print(f"      ocr: {my_dict['time']['ocr'][0]}번 평균 시간{my_dict['time']['ocr'][1]:.3f}")
            st.text(
                f"detection은 {my_dict['time']['detection'][0]}번 수행됐고 평균 수행 시간은 {my_dict['time']['detection'][1]:.3f}초 입니다"
            )
            st.text(f"      ocr은 {my_dict['time']['ocr'][0]}번 수행됐고 평균 수행 시간은 {my_dict['time']['ocr'][1]:.3f}초 입니다")
        # 입력값이 영상일 경우
        elif filename.endswith(".avi") or filename.endswith(".mp4"):
            start = time.time()

            # 비디오 detection 및 OCR
            output = video_process(input, backend_video)
            video_dict = literal_eval(output.content.decode("utf-8"))
            output_time = timedelta(seconds=time.time() - start)

            middle = time.time()

            # 임시 파일에 동영상 쓰기
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(input.read())
                temp_path = f.name

            # 비디오를 프레임으로 변환
            capture = cv2.VideoCapture(temp_path)

            # 프레임을 비디오로 변환할 준빈
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("video.mp4", fourcc, fps, (width, height))

            # ocr 결과에 사용할 폰트 설정
            font = ImageFont.truetype("fonts/MaruBuri-Bold.ttf", int(height / 36))

            # 프레임마다 detection, ocr 결과를 합성하고 비디오로 만들기
            num_frame = 0
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                for car in video_dict[f"frame{num_frame}"].values():
                    left_top, right_bottom = car["coordinate"]
                    ocr_result, confidence_score = car["OCR"]

                    if confidence_score > 0.5:
                        cv2.rectangle(frame, left_top, right_bottom, (255, 0, 0), 3)
                        x, y = left_top[0], left_top[1]
                        img_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text(
                            (x, y - int(height / 36) - 2),
                            f"{ocr_result}({round(confidence_score, 1)})",
                            (0, 0, 255),
                            font=font,
                        )
                        frame = np.array(img_pil)
                out.write(frame)
                num_frame += 1

            # 사용한 파일 해제
            capture.release()
            out.release()
            os.unlink(temp_path)

            # 비디오를 h264 코덱으로 변경
            os.system(f"ffmpeg -y -i {'video.mp4'} -vcodec libx264 {'video_h264.mp4'}")
            os.remove("video.mp4")

            # 비디오를 불러와 streamlit으로 업로드
            video_file = open("video_h264.mp4", "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)

            # 시간 측정용 코드
            print(f"모델 출력: {output_time}")
            print(f"영상 변환: {timedelta(seconds=time.time() - middle)}")
            print(f"총 소요시간: {timedelta(seconds=time.time() - start)}")
            print(f"detection: {video_dict['time']['detection'][0]}번 평균 시간{video_dict['time']['detection'][1]:.3f}")
            print(f"      ocr: {video_dict['time']['ocr'][0]}번 평균 시간{video_dict['time']['ocr'][1]:.3f}")
            st.text(
                f"detection은 {video_dict['time']['detection'][0]}번 수행됐고 평균 수행 시간은 {video_dict['time']['detection'][1]:.3f}초 입니다"
            )
            st.text(
                f"      ocr은 {video_dict['time']['ocr'][0]}번 수행됐고 평균 수행 시간은 {video_dict['time']['ocr'][1]:.3f}초 입니다"
            )
    else:
        # handler: no image
        st.text("이미지를 업로드 한 후 번호판 찾기를 눌러주세요")
