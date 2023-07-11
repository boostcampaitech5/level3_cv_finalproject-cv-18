from ultralytics import YOLO
import cv2

model = YOLO("/opt/ml/ultralytics/object-detection/100epoch_cos_mosaic50/weights/best.pt")
results = model("/opt/ml/ultralytics/data/Cars0.png")
img = cv2.imread("/opt/ml/ultralytics/data/Cars0.png")

for result in results:
    boxes = result.boxes.cpu().numpy()
    for i, box in enumerate(boxes):
        r = box.xyxy[0].astype(int)
        
        crop = img[r[1]:r[3], r[0]:r[2]]
        cv2.imwrite(str(i) + ".jpg", crop)