from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

LOG_DIR = ROOT_DIR / "logs"
ONNX_MODEL_PATH = ROOT_DIR / "runs" / "fire_alarm_v2" / "weights" / "best.onnx"

CLASS_NAMES = {0: "Fogo", 1: "Fumaça"}
TARGET_DETECTION_CLASS_INDEX = 0

VIDEO_SOURCE = "http://192.168.1.26:81/stream"
IMAGE_SIZE = 640

MINIMUM_CONFIDENCE = 0.7
MINIMUM_IOU = 0.5

DETECT_DEQUE_MAX_LENGTH = 60
DETECT_DEQUE_THRESHOLD = 0.5

TARGET_MINIMUM_COVERAGE = 0.005

RETRY_COUNT = -1
RETRY_INTERVAL_SECONDS = 5

RELAY_PORT = "/dev/ttyUSB1"
