import io
import tempfile
from ast import literal_eval
from copy import deepcopy
from datetime import datetime

import av
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
                print(my_dict)
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
                            col2.text(f"해당 번호판의 번호는 {v}입니다.")
        # 입력값이 영상일 경우
        elif filename.endswith(".avi") or filename.endswith(".mp4"):
            output = video_process(input, backend_video)
            video_dict = literal_eval(output.content.decode("utf-8"))
            frameST = st.empty()
            img_frame_list = []

            def play_video(file):
                cap = cv2.VideoCapture(file.name)
                while True:
                    ret, frame = cap.read()
                    if frame is not None:
                        img_frame_list.append(frame)
                        height, width, layers = frame.shape
                        size = (width, height)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    else:
                        break
                return size, fps, total_frame

            video = tempfile.NamedTemporaryFile(delete=False)
            video.write(input.read())
            size, fps, total_frame = play_video(video)
            output_memory_file = io.BytesIO(input.read())
            width = size[0]
            height = size[1]
            n_frmaes = int(fps)
            output = av.open(output_memory_file, "w", format="mp4")
            stream = output.add_stream("h264", str(fps))
            stream.width = width
            stream.height = height
            # stream.pix_fmt = 'yuv444p'
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "17"}
            # 적당한 폰트 크기가 얼마지?
            font = ImageFont.truetype("fonts/MaruBuri-Bold.ttf", int(height / 36))
            for i in range(total_frame):
                img = img_frame_list[i]
                for keys, values in video_dict.items():
                    if keys == f"frame{i}":
                        for key, value in values.items():
                            for k, v in value.items():
                                if k == "coordinate":
                                    # print(v[0], v[1])
                                    cv2.rectangle(img, v[0], v[1], (255, 0, 0), 3)
                                    x, y = v[0][0], v[0][1]
                                elif k == "OCR":
                                    img_pil = Image.fromarray(img)
                                    draw = ImageDraw.Draw(img_pil)
                                    draw.text((x, y - int(height / 36) - 2), v, (255, 0, 0), font=font)
                                    img = np.array(img_pil)

                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                packet = stream.encode(frame)
                output.mux(packet)
            packet = stream.encode(None)
            output.mux(packet)
            output.close()

            output_memory_file.seek(0)
            with open("output.mp4", "wb") as f:
                f.write(output_memory_file.getbuffer())
            video_file = open("output.mp4", "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)
    else:
        # handler: no image
        st.text("이미지를 업로드 한 후 번호판 찾기를 눌러주세요")
