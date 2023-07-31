from ultralytics import YOLO
from os import listdir
import glob

# model = YOLO("/opt/ml/ultralytics/object-detection/n_new_scale1_coslr/weights/best.pt")
# model = YOLO("/opt/ml/ultralytics/object-detection/n_new_tot_add9299_totval/weights/best.pt") #_mos90_scale05_
model = YOLO("/opt/ml/ultralytics/object-detection/n_new_tot_add9299_pseudo5103_totval/weights/best.pt")

# source = "/opt/ml/ultralytics/data/Cars0.png"
source = glob.glob("/opt/ml/ultralytics/data/*.png")
# source = glob.glob("/opt/ml/dataset/kpseudo5103/*.jpg") # 999:1998 1998:2997 2997:

model.predict(source, save=True, save_txt=True, imgsz=640, conf=0.5)
