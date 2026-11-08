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
    mask: np.ndarray
    mask_coverage: float

    def __init__(
        self,
        class_name: str,
        class_index: int,
        confidence: float,
        box: list[int, int, int, int],
        mask: np.ndarray,
        mask_coverage: float,
    ):
        self.class_name = class_name
        self.class_index = class_index
        self.confidence = confidence
        self.box = box
        self.mask = mask
        self.mask_coverage = mask_coverage


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

    def clear(self):
        self.deque.clear()

    def validate(self):
        deque_length = len(self.deque)
        deque_sum = sum(self.deque)

        if deque_length != self.deque.maxlen:
            return False

        return deque_sum >= deque_length * self.threshold

    @property
    def is_full(self):
        return len(self.deque) == self.deque.maxlen


class FrameProcessor:
    session: InferenceSession
    session_out_name_0: str
    session_out_name_1: str
    session_in_name: str
    image_size: int
    min_confidence: float
    min_iou: float
    class_names: list[str]
    on_exception: Callable[[Exception], None]

    def __init__(
        self,
        session: InferenceSession,
        image_size: int,
        min_confidence: float,
        min_iou: float,
        class_names: list[str],
        on_exception: Callable[[Exception], None],
    ):
        self.session = session
        self.session_out_name_0 = session.get_outputs()[0].name
        self.session_out_name_1 = session.get_outputs()[1].name
        self.session_in_name = session.get_inputs()[0].name
        self.image_size = image_size
        self.min_confidence = min_confidence
        self.min_iou = min_iou
        self.class_names = class_names
        self.on_exception = on_exception

    def process_image(self, image: np.ndarray) -> list[OutputRow]:
        try:
            original_height, original_width = image.shape[0], image.shape[1]

            image = cv2.resize(image, (self.image_size, self.image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255
            image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
            image = np.expand_dims(image, axis=0)

            output0, output1 = self.session.run(
                [self.session_out_name_0, self.session_out_name_1],
                {self.session_in_name: image},
            )

            output0: np.ndarray = output0.squeeze()
            output0 = output0.transpose()

            confidences = output0[:, 4:6]
            max_confidences = confidences.max(axis=1)
            min_confidence_mask = max_confidences >= self.min_confidence
            valid_confidences = max_confidences[min_confidence_mask]
            class_indexes = confidences[min_confidence_mask].argmax(axis=1)

            boxes = output0[:, :4]
            valid_boxes = boxes[min_confidence_mask]

            x_center_scaled = (
                valid_boxes[:, 0] * (original_width / self.image_size)
            ).astype(int)
            y_center_scaled = (
                valid_boxes[:, 1] * (original_height / self.image_size)
            ).astype(int)
            w_scaled = (valid_boxes[:, 2] * (original_width / self.image_size)).astype(
                int
            )
            h_scaled = (valid_boxes[:, 3] * (original_height / self.image_size)).astype(
                int
            )
            x_scaled = (x_center_scaled - w_scaled / 2).astype(int)
            y_scaled = (y_center_scaled - h_scaled / 2).astype(int)
            boxes_scaled = np.column_stack([x_scaled, y_scaled, w_scaled, h_scaled])

            best_box_indexes = cv2.dnn.NMSBoxes(
                bboxes=boxes_scaled,
                scores=valid_confidences,
                score_threshold=self.min_confidence,
                nms_threshold=self.min_iou,
            )

            mask_coeficients = output0[:, 6:]
            valid_coeficients = mask_coeficients[min_confidence_mask]

            valid_masks = np.sum(
                np.multiply(np.expand_dims(valid_coeficients, axis=(2, 3)), output1),
                axis=1,
            )

            detections: list[OutputRow] = []

            for index, best_box_index in enumerate(best_box_indexes):
                x, y, w, h = boxes_scaled[best_box_index]
                box_crop = np.zeros((original_height, original_width)).astype(bool)
                box_crop[y : y + h, x : x + w] = True

                mask = valid_masks[best_box_index]
                mask_scaled = cv2.resize(mask, (original_width, original_height))
                mask_normalized = 1 / (1 + np.exp(-mask_scaled))

                mask_cropped = box_crop & (mask_normalized >= self.min_confidence)

                total_pixels = original_height * original_width
                total_pixels_mask = np.count_nonzero(mask_cropped)
                mask_coverage = total_pixels_mask / total_pixels

                detections.append(
                    OutputRow(
                        class_name=self.class_names[class_indexes[best_box_index]],
                        class_index=class_indexes[best_box_index],
                        confidence=valid_confidences[best_box_index],
                        box=boxes_scaled[best_box_index],
                        mask=mask_cropped,
                        mask_coverage=mask_coverage,
                    )
                )

            return detections
        except Exception as exception:
            self.on_exception(exception)


def load_model(model_path: Path | str) -> InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def monitor_video_capture(
    video_source,
    temporal_filter: TemporalFilter,
    validate_matches: Callable[[list[OutputRow]], bool],
    frame_processor: FrameProcessor,
    on_detected: Callable[[list[OutputRow]], None],
    on_not_detected: Callable[[], None],
    on_exception: Callable[[Exception], None],
    show=False,
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

            if show:
                for det in output:
                    x, y, w, h = det.box

                    color = (0, 0, 255) if det.class_name == "Fogo" else (255, 165, 0)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame,
                        f"{det.class_name}: {det.confidence:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color,
                        2,
                    )

                    mask_overlay = frame.copy()
                    color = (255, 0, 0)
                    mask_overlay[det.mask.astype(bool)] = color

                    alpha = 0.4
                    frame = cv2.addWeighted(mask_overlay, alpha, frame, 1 - alpha, 0)

                cv2.imshow("Câmera do Sentinela ao Vivo", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_matches = validate_matches(output)
            temporal_filter.append(frame_matches)

            if not temporal_filter.is_full:
                continue

            is_object_detected = temporal_filter.validate()

            if not is_object_detected_current_value and is_object_detected:
                is_object_detected_current_value = True

                temporal_filter.clear()
                on_detected(output)
            elif is_object_detected_current_value and not is_object_detected:
                is_object_detected_current_value = False

                temporal_filter.clear()
                on_not_detected()
    except Exception as exception:
        on_exception(exception)
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


ONNX_MODEL_PATH = "/home/jos/Desktop/Machine Learning/Deep Learning/YOLO/Detecção/runs/fire_alarm_v2/weights/best.onnx"

CLASS_NAMES = {0: "Fogo", 1: "Fumaça"}
TARGET_DETECTION_CLASS_INDEX = 0

VIDEO_SOURCE = 1
IMAGE_SIZE = 640

MINIMUM_CONFIDENCE = 0.7
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


def on_detected(output: list[OutputRow]):
    relay_controller.activate()

    total_fire_coverage = 0
    total_fire_instances = 0

    for row in output:
        if row.class_index != TARGET_DETECTION_CLASS_INDEX:
            continue
    
        total_fire_coverage += row.mask_coverage
        total_fire_instances += 1

    mean_fire_coverage = total_fire_coverage / total_fire_instances if total_fire_instances > 0 else 0

    print(f"Fogo detectado e cobrindo {mean_fire_coverage:.2%} da cobertura da imagem de vídeo")
    

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
        on_exception=lambda exception: print(f"Falha ao processar frame {exception}")
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
        on_exception=lambda exception: print(f"Falha ao monitorar vídeo: {exception}"),
        show=True,
    )
