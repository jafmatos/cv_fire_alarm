import traceback
from detection.sentinel.config import IMAGE_SIZE
from detection.sentinel.logger import logger
from detection.sentinel.video_monitoring_engine import VideoMonitoringEngine
from detection.sentinel.temporal_filter import TemporalFilter
from detection.sentinel.frame_processor import FrameProcessor, OutputRow
from detection.sentinel.config import (
    VIDEO_SOURCE,
    TARGET_MINIMUM_COVERAGE,
    DETECT_DEQUE_THRESHOLD,
    DETECT_DEQUE_MAX_LENGTH,
    CLASS_NAMES,
    MINIMUM_IOU,
    MINIMUM_CONFIDENCE,
    ONNX_MODEL_PATH,
    TARGET_DETECTION_CLASS_INDEX,
    RELAY_PORT,
    RETRY_COUNT,
    RETRY_INTERVAL_SECONDS,
)
from detection.sentinel.relay_controller import RelayController

from pathlib import Path

import onnxruntime as ort
from onnxruntime import InferenceSession

relay_controller = RelayController(relay_port=RELAY_PORT)


def load_model(model_path: Path | str) -> InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def on_detected(output: list[OutputRow]):
    relay_controller.activate()

    total_fire_coverage = 0
    total_fire_instances = 0

    for row in output:
        if row.class_index != TARGET_DETECTION_CLASS_INDEX:
            continue

        total_fire_coverage += row.mask_coverage
        total_fire_instances += 1

    mean_fire_coverage = (
        total_fire_coverage / total_fire_instances if total_fire_instances > 0 else 0
    )

    logger.info(
        f"Fogo detectado e cobrindo {mean_fire_coverage:.2%} da cobertura da imagem de vídeo. Alarme ativado."
    )


def on_not_detected():
    relay_controller.deactivate()
    
    logger.info(
        f"Alarme desativado."
    )

def init():
    model = load_model(ONNX_MODEL_PATH)

    frame_processor = FrameProcessor(
        session=model,
        image_size=IMAGE_SIZE,
        min_confidence=MINIMUM_CONFIDENCE,
        min_iou=MINIMUM_IOU,
        class_names=CLASS_NAMES,
        on_exception=lambda e: logger.error(traceback.format_exc()),
    )

    temporal_filter = TemporalFilter(
        deque_length=DETECT_DEQUE_MAX_LENGTH, threshold=DETECT_DEQUE_THRESHOLD
    )

    video_monitoring_engine = VideoMonitoringEngine(
        frame_processor=frame_processor,
        temporal_filter=temporal_filter,
        target_class_index=TARGET_DETECTION_CLASS_INDEX,
        target_minimum_coverage=TARGET_MINIMUM_COVERAGE,
        on_detected=on_detected,
        on_not_detected=on_not_detected,
        on_exception=lambda e: logger.error(traceback.format_exc()),
    )

    video_monitoring_engine.monitor(
        video_source=VIDEO_SOURCE,
        retry_count=RETRY_COUNT,
        retry_interval_seconds=RETRY_INTERVAL_SECONDS,
        show=True,
    )


if __name__ == "__main__":
    # Salvar registros de vídeo quando fogo for detectado

    init()
