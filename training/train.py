"""

Original file is located at
    https://colab.research.google.com/drive/1yEImcCbggEqdD_VopKn8OFx4qcCInJxK
"""

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="API")
project = rf.workspace("roboflow-100").project("construction-safety-gsnvb")
version = project.version(1)
dataset = version.download("yolov8")

!pip install ultralytics

from ultralytics import YOLO

# Yolov8n
print("--- Starting training for YOLOv8n ---")
model_n = YOLO('yolov8n.pt')
results_n = model_n.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,
    pretrained=True,
    project='runs',
    name='yolov8n'
)

# YOLOv8s
print("--- Starting training for YOLOv8s ---")
model_s = YOLO('yolov8s.pt')
results_s = model_s.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,
    pretrained=True,
    project='runs',
    name='yolov8s'
)

