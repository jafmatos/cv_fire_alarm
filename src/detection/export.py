from ultralytics import YOLO

from detection.config import IMAGE_SIZE, WEIGHTS_PATH

model = YOLO(WEIGHTS_PATH)

onnx_path = model.export(
    format="onnx",
    imgsz=IMAGE_SIZE,
    dynamic=True,
    simplify=True
)

print(f"Modelo exportado para: {onnx_path}")