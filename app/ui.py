import base64
import csv
import io
import json
import os
import shutil
import tempfile
import time
from ast import literal_eval
from datetime import timedelta

# from io import BytesIO
import altair as alt
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

backend_image = "http://127.0.0.1:30009/image"
backend_video = "http://127.0.0.1:30009/video"
backend_deblur = "http://127.0.0.1:30009/deblur"
backend_super = "http://127.0.0.1:30009/super"

image_shape = (200, 80)

image_extension = ["png", "jpg", "jpeg", "bmp", "tiff", "raw"]
video_extension = ["avi", "mp4", "wmv", "mpg", "mpeg", "mkv", "mov", "webm"]

if "video_process" not in st.session_state:
    st.session_state["video_process"] = ""
if "image_process" not in st.session_state:
    st.session_state["image_process"] = {}
if "selected_frame_num" not in st.session_state:
    st.session_state["selected_frame_num"] = 0
if "image" not in st.session_state:
    st.session_state["image"] = []
if "video_data" not in st.session_state:
    st.session_state["video_data"] = {}
if "video" not in st.session_state:
    st.session_state["video"] = {}
if "video_component" not in st.session_state:
    st.session_state["video_component"] = {}
if "crop_image" not in st.session_state:
    st.session_state["crop_image"] = {}
st.markdown(
    """
    <style>
    .stSlider{
        padding-left: 63px;
    }
    div.stButton > button:first-child{
        margin-left: 70px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def restore_process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/png")})
    r = requests.post(
        server_url,
        data=m,
        headers={
            "Content-Type": m.content_type,
        },
        timeout=8000,
    )

    return r


def cv2_2_pil(cv2_image):
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image


def pil_2_cv2(pil_image):
    np_image = np.array(pil_image)
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return cv2_image


def load_video(path):
    return open(path, "rb").read()


# @st.cache_data
def load_image(path):
    return cv2.imread(path)


def decode_image(file):
    return Image.open(io.BytesIO(base64.b64decode(file)))


def centered_text(text):
    st.markdown(
        f"""
        <style>
            .center-text
            {{
                text-align: center;
            }}
        </style>
        <div class="center-text">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def get_fps_frame_count(file):
    with tempfile.NamedTemporaryFile() as f:
        f.write(file.read())
        temp_path = f.name
        capture = cv2.VideoCapture(temp_path)
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    return fps, frame_count


def on_change_callback(num_frame):
    st.session_state["selected_frame_num"] = num_frame


def flat_cols(df):
    df.columns = [" / ".join(x) for x in df.columns.to_flat_index()]
    return df


def cal_corr(current, target):
    cnt = 0

    for i, j in zip(current, target):
        if i == j:
            cnt += 1
    return True if cnt >= (len(current) - 1) else False


def cal_drop_idx(df):
    drop_idx = []
    for idx in range(len(df)):
        car_plate = df.iloc[idx]["plate"]
        for t_idx in range(len(df)):
            target_plate = df.iloc[t_idx]["plate"]
            if idx != t_idx:
                corr = cal_corr(car_plate, target_plate)
                c_score = df.iloc[idx]["confidence / mean"]
                t_score = df.iloc[t_idx]["confidence / mean"]
                c_cnt = df.iloc[idx]["count / sum"]
                t_cnt = df.iloc[t_idx]["count / sum"]
                if corr and c_score > t_score and c_cnt > t_cnt:
                    drop_idx.append(t_idx)
    df.drop(drop_idx, axis=0, inplace=True)
    return df


def convert_df(df):
    # return df.to_csv(index=False, encoding="euc-kr").encode("utf-8")
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


st.title("블랙박스 영상 번호판 분석 서비스")
# construct UI layout
input = st.sidebar.file_uploader("영상 또는 사진을 업로드해주세요")  # image upload widget
if input:
    filename = input.name
    if filename.split(".")[-1].lower() in image_extension:
        if not st.session_state["image_process"]:
            with st.spinner("로딩중..."):
                response = image_process(input, backend_image)
                st.session_state["image_process"] = response.json()

        bbox_image = decode_image(st.session_state["image_process"]["bbox_image"])
        st.image(bbox_image)

        if len(st.session_state["image_process"]) == 1:
            st.text("사진에서 번호판을 찾을 수 없어요 😱")
        else:
            centered_text(f"사진에서 번호판을 {len(st.session_state['image_process']) - 1}개 찾았어요!")
            centered_text(f"아래 보이는 사진은 {image_shape}로 맞춰진 크기로 우측 실제 크기와 다르다는 점 참고해주세요")
        st.markdown("""---""")

        for i, (k, v) in enumerate(st.session_state["image_process"].items()):
            with st.container():
                if k != "bbox_image":
                    with st.columns(3)[0]:
                        st.text(f"{i + 1}번째 번호판")
                    crop_col1, crop_col2 = st.columns(2)
                    crop_image = decode_image(v["crop_image"])
                    crop_col1.image(crop_image.resize(image_shape), channels="BGR")
                    if v["ocr_result"] != "invalid":
                        crop_col2.text(f"내용: {v['ocr_result']}")
                        crop_col2.text(f"내용의 신뢰도: {v['ocr_confidence_score']*100:.2f}%")
                    else:
                        crop_col2.text("내용을 판독하지 못했습니다.")
                        # crop_col2.text(f"내용: {v['ocr_result']}")
                        # crop_col2.text(f"내용의 신뢰도: {v['ocr_confidence_score']*100:.2f}%")

                    crop_col2.text(f"이미지 크기: {crop_image.size}")

                if len(st.session_state["image_process"]) - 1 != i:
                    # if not st.session_state["crop_image"]:
                    #     st.session_state["crop_image"] = crop_image
                    bio = io.BytesIO()
                    crop_image.save(bio, "PNG")
                    bio.seek(0)
                    buttons = st.columns(2)

                    if buttons[0].button("사진이 작아요", key=f"{i}resolution"):
                        super_col1, super_col2 = st.columns(2)
                        response = restore_process(bio, backend_super)
                        data = response.json()

                        image = decode_image(data["super_image"])
                        super_col1.image(image)
                        if data["ocr_result"] != "invalid":
                            super_col2.text(f"내용: {data['ocr_result']}")
                            super_col2.text(f"내용의 신뢰도: {data['ocr_confidence_score']*100:.2f}%")
                        else:
                            super_col2.text("내용을 판독하지 못했습니다.")
                        super_col2.text(f"이미지 크기: {image.size}")
                    if buttons[1].button("사진이 흐릿해요", key=f"{i}blur"):
                        blur_col1, blur_col2 = st.columns(2)
                        response = restore_process(bio, backend_deblur)
                        data = response.json()

                        image = decode_image(data["crop_image"])
                        blur_col1.image(image.resize(image_shape))
                        if data["ocr_result"] != "invalid":
                            blur_col2.text(f"내용: {data['ocr_result']}")
                            blur_col2.text(f"내용의 신뢰도: {data['ocr_confidence_score']*100:.2f}%")
                        else:
                            blur_col2.text("내용을 판독하지 못했습니다.")
                        blur_col2.text(f"이미지 크기: {image.size}")

            if len(st.session_state["image_process"]) != i + 2:
                st.markdown("""---""")

        st.sidebar.radio("평가를 남겨주세요", ("좋아요", "평범해요", "별로에요"), horizontal=True)
        temp_dict = {"좋아요": 3, "평범해요": 2, "별로에요": 1}
        if st.sidebar.button("전송"):
            st.sidebar.text("완료! 소중한 평가 감사합니다")

    # 입력값이 영상일 경우
    elif filename.split(".")[-1].lower() in video_extension:
        # start = time.time()
        fps, frame_count = get_fps_frame_count(input)

        if not st.session_state["video_process"]:
            total_processing_time_sec = frame_count / 5 * 0.3
            hours, remainder = divmod(total_processing_time_sec, 3600)
            minutes, seconds = divmod(remainder, 60)
            with st.spinner(f"예상 시간: {int(hours)} 시간 {int(minutes)} 분 {seconds:.0f} 초"):
                response = video_process(input, backend_video)
                video_path = literal_eval(response.content.decode("utf-8"))
                st.session_state["video_process"] = video_path
                # st.text(timedelta(seconds=time.time() - start))

        file_id = st.session_state["video_process"].replace("_h264.mp4", "")
        if not st.session_state["video"]:
            st.session_state["video"] = load_video(os.path.join("temp", st.session_state["video_process"]))
        video_component = st.video(st.session_state["video"])

        with open(f"record/{file_id}/video.json", "r") as video_json:
            data = json.load(video_json)
        with open("dict.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "car_num", "coord1", "coord2", "plate", "confidence", "count"])
            for key, value in data.items():
                if len(value) != 0:
                    for n_car in value:
                        writer.writerow(
                            [
                                key,
                                n_car,
                                value[n_car]["coordinate"][0],
                                value[n_car]["coordinate"][1],
                                value[n_car]["OCR"][0],
                                value[n_car]["OCR"][1],
                                1,
                            ]
                        )

        num_frame = st.sidebar.slider(
            label="움직여서 프레임을 선택해보세요",
            min_value=0,
            max_value=len(data) - 1,
            step=1,
            value=st.session_state["selected_frame_num"],
        )
        button_col = st.sidebar.columns(2)
        if button_col[0].button("이전 장면") and num_frame > 0:
            num_frame -= 1
            st.session_state["selected_frame_num"] = num_frame
        if button_col[1].button("다음 장면") and num_frame < len(data):
            num_frame += 1
            st.session_state["selected_frame_num"] = num_frame
        selected_frame = data[f"{num_frame + 1}"]
        car_np = np.array([len(v) if len(v) > 0 else np.nan for v in data.values()])
        car_chart = pd.DataFrame(car_np, columns=["번호판 수"])
        objects_per_frame = car_chart.reset_index()
        chart = (
            alt.Chart(objects_per_frame, width=350, height=150)
            .mark_point()
            .encode(alt.X("index", scale=alt.Scale(domain=[1, len(car_np) + 1])), alt.Y("%s" % "번호판 수"))
        )
        selected_frame_df = pd.DataFrame({"프레임": [num_frame]})
        vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x="프레임")
        st.sidebar.altair_chart(alt.layer(chart, vline))

        frame = cv2.imread(f"/opt/ml/level3_cv_finalproject-cv-18/app/record/{file_id}/{num_frame + 1}.png")
        start_time = timedelta(seconds=num_frame / fps)
        video_component.empty()
        video_component.video(
            st.session_state["video"],
            start_time=int(start_time.total_seconds()),
        )

        if not selected_frame:
            st.text("사진에서 번호판을 찾을 수 없어요 😱")
        else:
            centered_text(f"사진에서 번호판을 {len(selected_frame)}개 찾았어요!")
            centered_text(f"아래 보이는 사진은 {image_shape}로 맞춰진 크기로 우측 실제 크기와 다르다는 점 참고해주세요")
        st.markdown("""---""")

        for i, (key, value) in enumerate(selected_frame.items()):
            coodinate = value["coordinate"]
            ocr = value["OCR"]
            crop_image = cv2_2_pil(frame[coodinate[0][1] : coodinate[1][1], coodinate[0][0] : coodinate[1][0]])

            bio = io.BytesIO()
            crop_image.save(bio, "PNG")
            bio.seek(0)

            with st.container():
                with st.columns(3)[0]:
                    st.text(f"{i + 1}번째 번호판")
                crop_col1, crop_col2 = st.columns(2)
                crop_col1.image(crop_image.resize(image_shape), channels="BGR")

                if ocr[0] != "invalid":
                    crop_col2.text(f"내용: {ocr[0]}")
                    crop_col2.text(f"내용의 신뢰도: {ocr[1]*100:.2f}%")
                else:
                    crop_col2.text("내용을 판독하지 못했습니다.")
                crop_col2.text(f"이미지 크기: {crop_image.size}")
                buttons = st.columns(2)
                if buttons[0].button("사진이 작아요", key=f"{i}resolution"):
                    super_col1, super_col2 = st.columns(2)
                    response = restore_process(bio, backend_super)
                    data = response.json()

                    image = decode_image(data["super_image"])
                    super_col1.image(image)
                    if data["ocr_result"] != "invalid":
                        super_col2.text(f"내용: {data['ocr_result']}")
                        super_col2.text(f"내용의 신뢰도: {data['ocr_confidence_score']*100:.2f}%")
                    else:
                        super_col2.text("내용을 판독하지 못했습니다.")
                    super_col2.text(f"이미지 크기: {image.size}")
                if buttons[1].button("사진이 흐릿해요", key=f"{i}blur"):
                    blur_col1, blur_col2 = st.columns(2)
                    response = restore_process(bio, backend_deblur)
                    data = response.json()

                    image = decode_image(data["crop_image"])
                    blur_col1.image(image.resize(image_shape))
                    if data["ocr_result"] != "invalid":
                        blur_col2.text(f"내용: {data['ocr_result']}")
                        blur_col2.text(f"내용의 신뢰도: {data['ocr_confidence_score']*100:.2f}%")
                    else:
                        blur_col2.text("내용을 판독하지 못했습니다.")
                    blur_col2.text(f"이미지 크기: {image.size}")

        st.text("동영상을 요약한 결과는 다음과 같아요")
        df = pd.read_csv("dict.csv")
        df = (
            df.groupby(["plate"])
            .agg({"confidence": ["mean"], "count": ["sum"], "frame": ["min", "max"]})
            .pipe(flat_cols)
        )
        df = df.sort_values(by=["frame / min"], ascending=True)
        total_df = df.reset_index()
        total_df.columns = ["번호판", "평균 신뢰도", "탐지 빈도", "시작 프레임", "마지막 프레임"]
        df = df[df["count / sum"] > df["count / sum"].median()]
        df = df.reset_index()
        df = cal_drop_idx(df)
        df.columns = ["번호판", "평균 신뢰도", "탐지 빈도", "시작 프레임", "마지막 프레임"]
        st.dataframe(df)
        os.remove("dict.csv")
        st.text("전체 결과가 필요하다면 아래 버튼을 눌러주세요")
        csv = convert_df(total_df)
        st.download_button("다운로드", csv, "전체_요약본.csv")

        st.sidebar.radio("평가를 남겨주세요", ("좋아요", "평범해요", "별로에요"), horizontal=True)
        temp_dict = {"좋아요": 3, "평범해요": 2, "별로에요": 1}
        if st.sidebar.button("전송"):
            st.sidebar.text("완료! 소중한 평가 감사합니다")

    else:
        st.text("이미지의 경우 jpg, png, jpeg 파일만, 영상의 경우 avi, mp4 파일만 가능합니다")
elif not input:
    try:
        temp = os.path.join("temp", st.session_state["video_process"])
        record = os.path.join("record", st.session_state["video_process"].replace("_h264.mp4", ""))
        if os.path.exists(temp):
            os.remove(temp)
        if os.path.exists(record):
            shutil.rmtree(record)
    except Exception:
        pass
    finally:
        st.session_state["video_process"] = ""
        st.session_state["image_process"] = {}
        st.session_state["selected_frame_num"] = 0
        st.session_state["image"] = []
        st.session_state["video_data"] = {}
        st.session_state["video"] = {}
        st.session_state["video_component"] = {}
        st.session_state["crop_image"] = {}
