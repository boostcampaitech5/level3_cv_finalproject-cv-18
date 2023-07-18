import base64
import io
import os
from ast import literal_eval

import requests
import streamlit as st
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

# interact with FastAPI endpoint
backend_image = "http://127.0.0.1:8000/image"
backend_video = "http://127.0.0.1:8000/video"


def image_process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/png")})
    r = requests.post(
        server_url,
        data=m,
        headers={
            "Content-Type": m.content_type,
            "filename": base64.b64encode(image.name.encode()).decode("utf-8"),
        },
        timeout=8000,
    )

    return r


def video_process(video, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", video, "video/mp4")})
    r = requests.post(
        server_url,
        data=m,
        headers={
            "Content-Type": m.content_type,
            "filename": base64.b64encode(video.name.encode()).decode("utf-8"),
        },
        timeout=8000,
    )

    return r


image_extension = ["png", "jpg", "jpeg", "bmp", "tiff", "raw"]
video_extension = ["avi", "mp4", "wmv", "mpg", "mpeg", "mkv", "mov", "webm"]

# construct UI layout
st.title("블랙박스 번호판 찾기")

input = st.file_uploader("영상 또는 이미지를 업로드하여 번호판 인식")  # image upload widget
if input:
    filename = input.name

if st.button("번호판 찾기"):
    if input:
        # 입력값이 이미지일 경우
        if filename.split(".")[-1] in image_extension:
            with st.spinner("로딩중..."):
                response = image_process(input, backend_image)
                data = response.json()
                bbox_image = Image.open(io.BytesIO(base64.b64decode(data["bbox_image"])))
                st.image(bbox_image)

                for k, v in data.items():
                    if k != "bbox_image":
                        image = Image.open(io.BytesIO(base64.b64decode(v["crop_image"])))
                        st.image(image)
                        st.text(f"{v['ocr_result']}, {v['ocr_confidence_score']}")

        # 입력값이 영상일 경우
        elif filename.split(".")[-1] in video_extension:
            with st.spinner("로딩중..."):
                response = video_process(input, backend_video)
                video_path = literal_eval(response.content.decode("utf-8"))
                st.video(video_path)
                os.remove(video_path)
        else:
            st.text("이미지의 경우 jpg, png, jpeg 파일만, 영상의 경우 avi, mp4 파일만 가능합니다")
    else:
        # handler: no image
        st.text("이미지를 업로드 한 후 번호판 찾기를 눌러주세요")
