import numpy as np
from ultralytics.engine.results import Masks
import cv2
from detection.config import OUTPUT_DIR
from detection.config import INPUT_DIR
from detection.config import WEIGHTS_PATH

from ultralytics import YOLO

model = YOLO(WEIGHTS_PATH)

results = model.predict(
    source=INPUT_DIR,
    verbose=False,
    conf=0.5,
    save=True,
    project=OUTPUT_DIR,
    name="infer",
    exist_ok=True,
)

for result in results:
    original_height, original_width = result.orig_shape

    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()

        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        confidence = float(box.conf[0])

        mask = result.masks[index].data[0].cpu().numpy()
        mask_scaled = cv2.resize(mask, (original_width, original_height))
        mask_binary = (mask_scaled > 0.5).astype(int)

        total_image_pixels = original_width * original_height

        coverage = np.count_nonzero(mask_binary) / total_image_pixels

        print(
            f"""
-> {class_name}
   Confiança: {confidence:.2%}
   Dimensões: {x2 - x1}x{y2 - y1}
   Coordenadas: {x1, y1, x2, y2}
   Cobertura: {coverage:.2%}
"""
        )
