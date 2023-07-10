import tempfile
from ast import literal_eval

import cv2
import numpy as np
import requests

# import request
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder

# interact with FastAPI endpoint
backend = "http://127.0.0.1:8000/segmentation"


def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/png")})

    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)

    return r


# construct UI layout
st.title("블랙박스 번호판 찾기")

# st.write(
#     """영상 또는 이미지를 업로드하여 번호판 인식"""
# )  # description and instructions

input = st.file_uploader("영상 또는 이미지를 업로드하여 번호판 인식")  # image upload widget
if input:
    filename = input.name

if st.button("번호판 찾기"):
    if input:
        # 입력값이 이미지일 경우
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # detection 결과를 받아서 bounding box를 쳐줌
            with st.spinner("로딩중..."):
                segments = process(input, backend)
                my_dict = literal_eval(segments.content.decode("utf-8"))
                print(my_dict)
                file_bytes = np.asarray(bytearray(input.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                # 가능한 색 -> 하나만 해도 될거같다
                # pallete = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                i = 0
                for key, value in my_dict.items():
                    for k, v in value.items():
                        if k == "coordinate":
                            # cv2.rectangle(opencv_image, v[0], v[1], pallete[i], 3)
                            cv2.rectangle(opencv_image, v[0], v[1], (255, 0, 0), 3)
                            i += 1
                st.image(opencv_image, use_column_width=True)
            # OCR 결과
            for key, value in my_dict.items():
                with st.container():
                    col1, col2 = st.columns(2)
                    for k, v in value.items():
                        # 각 bounding box에 해당하는 영역을 crop해 출력
                        if k == "coordinate":
                            cropped_img = opencv_image[v[0][1] : v[1][1], v[0][0] : v[1][0]]
                            print([v[0][0], v[1][0], v[0][1], v[1][1]], cropped_img.shape, opencv_image.shape)
                            col1.image(cropped_img, use_column_width=True)
                        # 각 bounding box에 해당하는 번호판을 crop된 이미지 옆에 출력
                        elif k == "OCR":
                            col2.text(f"해당 번호판의 번호는 {v}입니다.")
        # 입력값이 영상일 경우
        elif filename.endswith(".avi") or filename.endswith(".mp4"):
            # st.video(input.read())
            frameST = st.empty()

            def play_video(file):
                cap = cv2.VideoCapture(file.name)
                while True:
                    ret, frame = cap.read()
                    if frame is None:
                        break
                    frameST.image(frame, channels="BGR")
                cap.release()

            video = tempfile.NamedTemporaryFile(delete=False)
            video.write(input.read())

            play_video(video)

            # if st.button("재생"):
            #     play_video(video)
            # 입력
    else:
        # handler: no image
        st.text("이미지를 업로드 한 후 번호판 찾기를 눌러주세요")
