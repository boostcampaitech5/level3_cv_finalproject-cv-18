from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="carplate.yaml", epochs=100, name='100epoch_cos', cos_lr=True)
