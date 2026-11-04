import collections
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import serial
from onnxruntime import InferenceSession


class OutputRow:
    class_name: str
    class_index: int
    confidence: float
    box: list[int, int, int, int]

    def __init__(
        self,
        class_name: str,
        class_index: int,
        confidence: float,
        box: list[int, int, int, int],
    ):
        self.class_name = class_name
        self.class_index = class_index
        self.confidence = confidence
        self.box = box


class CameraSignalLostError(RuntimeError):
    def __init__(self):
        super().__init__("Sinal da câmera perdido")


class TemporalFilter:
    deque: collections.deque[bool]
    threshold: float

    def __init__(self, deque_length: int, threshold: float):
        self.deque = collections.deque(maxlen=deque_length)
        self.threshold = threshold

    def append(self, value: bool):
        self.deque.append(value)

    def validate(self):
        deque_length = len(self.deque)
        deque_sum = sum(self.deque)

        if deque_length != self.deque.maxlen:
            return False

        return deque_sum >= deque_length * self.threshold


class FrameProcessor:
    session: InferenceSession
    session_out_name: str
    session_in_name: str
    image_size: int
    min_confidence: float
    min_iou: float
    class_names: list[str]

    def __init__(
        self,
        session: InferenceSession,
        image_size: int,
        min_confidence: float,
        min_iou: float,
        class_names: list[str],
    ):
        self.session = session
        self.session_out_name = session.get_outputs()[0].name
        self.session_in_name = session.get_inputs()[0].name
        self.image_size = image_size
        self.min_confidence = min_confidence
        self.min_iou = min_iou
        self.class_names = class_names

    def process_image(self, image: np.ndarray) -> list[OutputRow]:
        original_height, original_width = image.shape[:2]

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)

        output = self.session.run(
            [self.session_out_name], {self.session_in_name: image}
        )[0]
        output: np.ndarray = np.transpose(output[0])

        scores = output[:, 4:]
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.min_confidence

        detections = []

        if np.any(mask):
            boxes = output[mask, :4]
            filtered_confidences = confidences[mask]
            class_indexes = np.argmax(scores[mask], axis=1)

            x_center, y_center, width, height = (
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2],
                boxes[:, 3],
            )

            x1 = ((x_center - (width / 2)) * (original_width / self.image_size)).astype(int)
            y1 = ((y_center - (height / 2)) * (original_height / self.image_size)).astype(int)
            scaled_width = (width * (original_width / self.image_size)).astype(int)
            scaled_height = (height * (original_height / self.image_size)).astype(int)

            boxes = np.column_stack([x1, y1, scaled_width, scaled_height]).tolist()
            class_indexes = class_indexes.tolist()
            indexes = cv2.dnn.NMSBoxes(
                boxes, filtered_confidences.tolist(), self.min_confidence, self.min_iou
            )

            detections: list[OutputRow] = []

            if len(indexes) > 0:
                for index in indexes.flatten():
                    detections.append(
                        OutputRow(
                            class_name=self.class_names[class_indexes[index]],
                            class_index=class_indexes[index],
                            confidence=filtered_confidences[index],
                            box=boxes[index],
                        )
                    )

        return detections


def load_model(model_path: Path | str) -> InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def monitor_video_capture(
    video_source,
    temporal_filter: TemporalFilter,
    validate_matches: Callable[[list[OutputRow]], bool],
    frame_processor: FrameProcessor,
    on_detected: Callable[[], None],
    on_not_detected: Callable[[], None],
    on_exception: Callable[[Exception], None],
):
    is_object_detected_current_value: bool = False
    video_capture = cv2.VideoCapture(video_source)

    if not video_capture.isOpened():
        raise CameraSignalLostError()

    try:
        while video_capture.isOpened():
            success, frame = video_capture.read()

            if not success:
                raise CameraSignalLostError()

            output = frame_processor.process_image(frame)

            frame_matches = validate_matches(output)
            temporal_filter.append(frame_matches)

            is_object_detected = temporal_filter.validate()

            if not is_object_detected_current_value and is_object_detected:
                is_object_detected_current_value = True

                on_detected()
            elif is_object_detected_current_value and not is_object_detected:
                is_object_detected_current_value = False

                on_not_detected()
    except Exception as exception:
        on_exception(exception)
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


ONNX_MODEL_PATH = "/home/jos/Desktop/Machine Learning/Deep Learning/YOLO/Detecção/runs/detect/experimentos/detector_v1-2/weights/best.onnx"

CLASS_NAMES = {0: "Fumaça", 1: "Fogo"}
TARGET_DETECTION_CLASS_INDEX = 1

VIDEO_SOURCE = 1
IMAGE_SIZE = 640

MINIMUM_CONFIDENCE = 0.5
MINIMUM_IOU = 0.5

DEQUE_MAX_LENGTH = 60
DEQUE_THRESHOLD = 0.5

RELAY_PORT = "/dev/ttyUSB0"


class RelayController:
    relay: serial.Serial

    def __init__(self, relay_port: str, baudrate: int = 9600):
        self.relay = serial.Serial(relay_port, baudrate)

    def activate(self):
        self.relay.write(b"\xa0\x01\x01\xa2")

    def deactivate(self):
        self.relay.write(b"\xa0\x01\x00\xa1")

    def close(self):
        self.relay.close()


# Registrar ativações e desativações em um arquivo de log
# Salvar registros de vídeo quando fogo for detectado


relay_controller = RelayController(relay_port=RELAY_PORT)


def on_detected():
    relay_controller.activate()


def on_not_detected():
    relay_controller.deactivate()


if __name__ == "__main__":
    model = load_model(ONNX_MODEL_PATH)
    frame_processor = FrameProcessor(
        session=model,
        image_size=IMAGE_SIZE,
        min_confidence=MINIMUM_CONFIDENCE,
        min_iou=MINIMUM_IOU,
        class_names=CLASS_NAMES,
    )
    temporal_filter = TemporalFilter(
        deque_length=DEQUE_MAX_LENGTH, threshold=DEQUE_THRESHOLD
    )

    monitor_video_capture(
        video_source=VIDEO_SOURCE,
        frame_processor=frame_processor,
        temporal_filter=temporal_filter,
        validate_matches=lambda output: any(
            d.class_index == TARGET_DETECTION_CLASS_INDEX for d in output
        ),
        on_detected=on_detected,
        on_not_detected=on_not_detected,
        on_exception=lambda exception: print(exception),
    )
