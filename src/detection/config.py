import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = ROOT_DIR / "runs"
TRAIN_DIRNAME = "fire_alarm_v4"

WEIGHTS_PATH = PROJECT_DIR / TRAIN_DIRNAME / "weights" / "best.pt"
DATASET_YAML = ROOT_DIR / "data.yaml"

NUM_EPOCHS = 60
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 15
NUM_WORKERS = 4
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SAVE_PERIOD = 10
OPTIMIZER = "auto"
EXIST_OK = True

INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
