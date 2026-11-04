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
    model = YOLO("yolov8s.pt")

    results = model.train(
        data=DATASET_YAML,
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=15,
        workers=4,
        device=0,
        save=True,
        save_period=10,
        project=TRAIN_PROJECT,
        name=TRAIN_NAME,
        optimizer="auto",
        verbose=True,
    )

    print("Treinamento finalizado com sucesso!")
    print(f"Melhores pesos salvos em: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train()
