<div align=center>
    <h1>블랙박스 영상 번호판 분석 서비스</h1>
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=FastAPI&logoColor=white">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
    <img src="https://img.shields.io/badge/YOLO-00FFFF?style=flat-square&logo=yolo&logoColor=white">
    <img src="https://img.shields.io/badge/OCR-485A62?style=flat-square&logo=codereview&logoColor=white">
    <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white">
    <img src="https://img.shields.io/badge/GCS-4285F4?style=flat-square&logo=googlecloud&logoColor=white">
    <br>
</div>
<div align="center">

|        김준태        |            박재민             |             송인성             |          최홍록           |
| :----------------: | :--------------------------: | :---------------------------: | :----------------------: |
| [GitHub](https://github.com/KZunT) | [GitHub](https://github.com/jemin7709) | [GitHub](https://github.com/0523kevin) |[GitHub](https://github.com/chroki41) |

</div>


# 프로젝트 개요
"블랙박스 영상 번호판 분석 서비스"는 블랙박스 영상에서 번호판을 정보를 추출해 사용자에게 제공하는 서비스입니다.

최근 블랙박스 영상을 활용한 신고율이 높아지는 상황에서 블랙박스 영상에 등장한 차량 번호판을 탐지하고 차량 번호판의 글자를 인식하는 OCR 기술과 이미지 복원 기술을 통한 화질 개선으로 사용자가 블랙박스 영상 분석에 필요한 시간과 비용을 절약할 수 있습니다.


# 데모
### 데모영상
![video640](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-18/assets/68144124/0d32a304-e25b-41ec-a32e-e4978a094f80)

### 이미지 요청
<img width="700" alt="스크린샷 2023-07-29 오후 5 51 57" src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-18/assets/68144124/574108fe-2848-4e55-bc98-8d2f9e26932c">

### 동영상 요청
<img width="700" alt="스크린샷 2023-07-29 오후 5 52 32" src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-18/assets/68144124/e260bad8-e7d6-4675-8331-77937afa3c20">

# 사용 방법
    pip -r requirements.txt
    cd app
    uvicorn server:app
    streamlit run ui.py

# 프로젝트 구성
### 사용자 요청 흐름도
![Untitled](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-18/assets/68144124/0ed9fd8e-6859-4503-a2b8-a4c31f5be6eb)

### 프로젝트 전체 구조
![시스템 아키텍처](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-18/assets/68144124/8fac3b99-de3c-4f45-8fe0-e0020191fd5d)

**파일 구조**

    .
    ├── NAFNet
    ├── README.md
    ├── SwinIR
    ├── app
    ├── deep-text-recognition-benchmark
    └── ultralytics

# 사용 모델
### 객체 인식 모델
| 모델       | mAP50-95   | 추론 시간(ms)  |
| ----------- | ---------- | ------ |
| YOLOv8 nano | 0.8097 | **3.3** |
| YOLOv8 small | 0.8136  | 3.4 |
| YOLOv8 medium | **0.8150**  | 7.7 |

추론 시간과 성능을 고려해 YOLO v8 small 모델 사용

### 텍스트 인식 모델
Transformation: TPS, Sequence modeling: BiLSTM, Prediction: Attention

| Feature extractor   | 정확도(acc) | 추론 시간(ms) | 
| ---------- | ------ | ------ |
| ResNet | **96.646** | 55.78 |
| VGGNet  | 96.236 | 45.56 |
| MobileNet | 88.219 | **38.45** |
| EfficientNet | 93.349 | 49.23 |

정확도 대비 추론 속도가 가장 좋았던 VGGNet을 Feature extractor로 사용

# 부록
* [CV_18조_블랙박스 영상 번호판 분석 서비스_발표자료](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-18/files/12206657/CV_18._.pdf)