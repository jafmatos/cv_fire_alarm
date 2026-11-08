from detection.config import EXIST_OK
import torch
from detection.config import OPTIMIZER
from detection.config import SAVE_PERIOD
from detection.config import DEVICE
from detection.config import NUM_WORKERS
from detection.config import PATIENCE
from ultralytics import YOLO

from detection.config import (
    BATCH_SIZE,
    DATASET_YAML,
    IMAGE_SIZE,
    NUM_EPOCHS,
    TRAIN_NAME,
    TRAIN_PROJECT,
)


def train():
    model = YOLO("yolov8s-seg.pt")

    results = model.train(
        data=DATASET_YAML,
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        workers=NUM_WORKERS,
        device=DEVICE,
        save_period=SAVE_PERIOD,
        project=str(TRAIN_PROJECT),
        name=TRAIN_NAME,
        optimizer=OPTIMIZER,
        exist_ok=EXIST_OK,
        save=True,
        verbose=True,
    )

    print("Treinamento finalizado com sucesso!")
    print(f"Melhores pesos salvos em: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    if DEVICE == 0:
        torch.cuda.set_per_process_memory_fraction(0.80, device=DEVICE)

    train()
