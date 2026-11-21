from detection.sentinel.camera_video_capture import CameraVideoCapture
from detection.sentinel.config import UNDETECT_DEQUE_THRESHOLD, MAX_PROCESSED_FPS
from detection.sentinel.config import UNDETECT_DEQUE_MAX_LENGTH
from detection.sentinel.actuator_controller import ActuatorController
import traceback
from detection.sentinel.logger import logger
from detection.sentinel.video_monitoring_engine import VideoMonitoringEngine
from detection.sentinel.temporal_filter import TemporalFilter
from detection.sentinel.frame_processor import FrameProcessor, OutputRow
from detection.sentinel.config import (
    VIDEO_SOURCE,
    PREPROCESSING_FRAME_SIZE,
    TARGET_MINIMUM_COVERAGE,
    DETECT_DEQUE_MAX_LENGTH_LEVEL_1,
    DETECT_DEQUE_THRESHOLD_LEVEL_1,
    DETECT_DEQUE_MAX_LENGTH_LEVEL_2,
    DETECT_DEQUE_THRESHOLD_LEVEL_2,
    DETECT_DEQUE_MAX_LENGTH_LEVEL_3,
    DETECT_DEQUE_THRESHOLD_LEVEL_3,
    CLASS_NAMES,
    MINIMUM_IOU,
    MINIMUM_CONFIDENCE,
    ONNX_MODEL_PATH,
    TARGET_DETECTION_CLASS_INDEX,
    RETRY_COUNT,
    RETRY_INTERVAL_SECONDS,
    SIREN_ACTIVATE_URL,
    SIREN_DEACTIVATE_URL,
)

from pathlib import Path

import onnxruntime as ort
from onnxruntime import InferenceSession


actuator_controller = ActuatorController(
    actuator_activate_url=SIREN_ACTIVATE_URL,
    actuator_deactivate_url=SIREN_DEACTIVATE_URL,
    on_exception=lambda e: logger.error(traceback.format_exc()),
)


def load_model(model_path: Path | str) -> InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def on_detected(output: list[OutputRow]):
    total_fire_coverage = 0
    total_fire_instances = 0

    for row in output:
        if row.class_index != TARGET_DETECTION_CLASS_INDEX:
            continue

        total_fire_coverage += row.mask_coverage
        total_fire_instances += 1

    if total_fire_instances == 0:
        return

    mean_fire_coverage = total_fire_coverage / total_fire_instances

    logger.info(
        f"Fogo detectado e cobrindo {mean_fire_coverage:.2%} da cobertura da imagem de vídeo"
    )

    actuator_controller.activate()


def on_not_detected():
    actuator_controller.deactivate()


def init():
    model = load_model(ONNX_MODEL_PATH)

    frame_processor = FrameProcessor(
        session=model,
        image_size=PREPROCESSING_FRAME_SIZE,
        min_confidence=MINIMUM_CONFIDENCE,
        min_iou=MINIMUM_IOU,
        class_names=CLASS_NAMES,
        on_exception=lambda e: logger.error(traceback.format_exc()),
    )

    temporal_filters = [
        TemporalFilter(
            type="detect",
            deque_length=DETECT_DEQUE_MAX_LENGTH_LEVEL_1,
            threshold=DETECT_DEQUE_THRESHOLD_LEVEL_1,
        ),
        TemporalFilter(
            type="detect",
            deque_length=DETECT_DEQUE_MAX_LENGTH_LEVEL_2,
            threshold=DETECT_DEQUE_THRESHOLD_LEVEL_2,
        ),
        TemporalFilter(
            type="detect",
            deque_length=DETECT_DEQUE_MAX_LENGTH_LEVEL_3,
            threshold=DETECT_DEQUE_THRESHOLD_LEVEL_3,
        ),
        TemporalFilter(
            type="undetect",
            deque_length=UNDETECT_DEQUE_MAX_LENGTH,
            threshold=UNDETECT_DEQUE_THRESHOLD,
        ),
    ]

    video_monitoring_engine = VideoMonitoringEngine(
        video_source=VIDEO_SOURCE,
        max_processed_fps=MAX_PROCESSED_FPS,
        frame_processor=frame_processor,
        temporal_filters=temporal_filters,
        target_class_index=TARGET_DETECTION_CLASS_INDEX,
        target_minimum_coverage=TARGET_MINIMUM_COVERAGE,
        on_detected=on_detected,
        on_not_detected=on_not_detected,
        on_exception=lambda e: logger.error(traceback.format_exc()),
    )

    video_monitoring_engine.monitor(
        retry_count=RETRY_COUNT,
        retry_interval_seconds=RETRY_INTERVAL_SECONDS,
        show=True,
    )


if __name__ == "__main__":
    # Salvar registros de vídeo quando fogo for detectado
    # DONE: Resolver o delay absurdo da câmera
    # DONE: Implementar deques temporais em degrau
    # Persistir configurações da câmera no ESP32

    init()
