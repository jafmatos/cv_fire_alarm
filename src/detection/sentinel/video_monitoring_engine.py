from typing import Any
from detection.sentinel.camera_video_capture import CameraVideoCapture
from detection.sentinel import camera_video_capture
import time
import cv2
from collections.abc import Callable
from detection.sentinel.temporal_filter import TemporalFilter
from detection.sentinel.frame_processor import FrameProcessor, OutputRow
from detection.sentinel.logger import logger


class VideoMonitoringEngine:
    video_source: CameraVideoCapture
    max_processed_fps: int
    max_process_time_per_frame_seconds: int
    frame_processor: FrameProcessor
    target_class_index: int
    target_minimum_coverage: float
    temporal_filters: list[TemporalFilter]
    on_detected: Callable[[list[OutputRow]], None]
    on_not_detected: Callable[[], None]
    on_exception: Callable[[Exception], None]

    def __init__(
        self,
        video_source: Any,
        max_processed_fps: int,
        frame_processor: FrameProcessor,
        target_class_index: int,
        target_minimum_coverage: float,
        temporal_filters: list[TemporalFilter],
        on_detected: Callable[[list[OutputRow]], None],
        on_not_detected: Callable[[], None],
        on_exception: Callable[[Exception], None],
    ):
        self.video_source = video_source
        self.max_processed_fps = max_processed_fps
        self.max_process_time_per_frame_seconds = 1 / max_processed_fps
        self.frame_processor = frame_processor
        self.target_class_index = target_class_index
        self.target_minimum_coverage = target_minimum_coverage
        self.temporal_filters = temporal_filters
        self.on_detected = on_detected
        self.on_not_detected = on_not_detected
        self.on_exception = on_exception

    def monitor(
        self,
        show=False,
        retry_count: int = -1,
        retry_interval_seconds: int = 0,
    ):
        retries = 0
        camera_video_capture: CameraVideoCapture | None = None

        while True:
            try:
                camera_video_capture = CameraVideoCapture(
                    video_source=self.video_source
                )

                if retries > retry_count and retry_count >= 0:
                    logger.warn(
                        f"Número máximo de tentativas atingido. Interrompendo o monitoramento."
                    )

                    break

                elif retries > 0:
                    logger.info(f"Tentativa n° {retries}")

                is_object_detected_current_value: bool = False

                while camera_video_capture.is_connected:
                    process_time_start = time.perf_counter()
                    frame = camera_video_capture.read_latest()

                    output = self.frame_processor.process_image(frame)

                    if show:
                        for det in output:
                            x, y, w, h = det.box

                            color = (
                                (0, 0, 255)
                                if det.class_index == self.target_class_index
                                else (255, 165, 0)
                            )

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
                            frame = cv2.addWeighted(
                                mask_overlay, alpha, frame, 1 - alpha, 0
                            )

                        cv2.imshow("Câmera do Sentinela ao Vivo", frame)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    target_object_output_rows = list(
                        filter(
                            lambda row: row.class_index == self.target_class_index,
                            output,
                        )
                    )
                    is_object_in_frame = len(target_object_output_rows) > 0

                    total_target_object_coverage = sum(
                        row.mask_coverage for row in target_object_output_rows
                    )
                    object_has_minimum_coverage = (
                        total_target_object_coverage >= self.target_minimum_coverage
                    )

                    for temporal_filter in self.temporal_filters:
                        temporal_filter.append(
                            is_object_in_frame and object_has_minimum_coverage
                        )

                    is_object_consistently_detected = any(
                        tf.type == "detect" and tf.is_full and tf.validate()
                        for tf in self.temporal_filters
                    )

                    is_object_consistently_undetected = any(
                        tf.type == "undetect" and tf.is_full and tf.validate()
                        for tf in self.temporal_filters
                    )

                    if (
                        not is_object_detected_current_value
                        and is_object_consistently_detected
                    ):
                        is_object_detected_current_value = True

                        for tf in self.temporal_filters:
                            if tf.type == "detect":
                                continue

                            tf.clear()

                        self.on_detected(output)
                    elif (
                        is_object_detected_current_value
                        and is_object_consistently_undetected
                    ):
                        is_object_detected_current_value = False

                        for tf in self.temporal_filters:
                            if tf.type == "undetect":
                                continue

                            tf.clear()

                        self.on_not_detected()

                    process_time_end = time.perf_counter()
                    total_process_time_seconds = process_time_end - process_time_start
                    exceeding_time_seconds = (
                        self.max_process_time_per_frame_seconds
                        - total_process_time_seconds
                    )

                    if exceeding_time_seconds > 0:
                        time.sleep(exceeding_time_seconds)

            except Exception as exception:
                self.on_exception(exception)

            finally:
                camera_video_capture and camera_video_capture.release()
                cv2.destroyAllWindows()

                if retries < retry_count or retry_count < 0:
                    logger.info(f"Tentando novamente em {retry_interval_seconds}s...")

                    time.sleep(retry_interval_seconds)

                retries += 1
