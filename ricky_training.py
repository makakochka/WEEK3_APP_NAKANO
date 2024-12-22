from ultralytics import YOLO
from ricky_update_path import update_path

update_path()

model = YOLO('yolov8n.pt')
model.train(data="ricky_dataset.yaml", epochs=100, batch=8)
