from pathlib import Path

TRAIN_PROJECT = "experimentos"
TRAIN_NAME = "detector_v1-2"

ROOT_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = ROOT_DIR / "runs" / "detect" / TRAIN_PROJECT / TRAIN_NAME / "weights" / "best.pt"
DATASET_YAML = ROOT_DIR / "data.yaml"

NUM_EPOCHS = 60
IMAGE_SIZE = 640
BATCH_SIZE = 16

INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"