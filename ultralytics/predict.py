from ultralytics import YOLO

model = YOLO("/opt/ml/ultralytics/object-detection/100epoch/weights/best.pt")

source = "/opt/ml/ultralytics/data/Cars0.png"

model.predict(source, save=True, imgsz=640, conf=0.5)


