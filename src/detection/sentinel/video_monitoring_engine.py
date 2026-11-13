import time
import cv2
from collections.abc import Callable
from detection.sentinel.temporal_filter import TemporalFilter
from detection.sentinel.frame_processor import FrameProcessor, OutputRow
from detection.sentinel.logger import logger


class CameraSignalLostError(RuntimeError):
    def __init__(self):
        super().__init__("Sinal da câmera perdido.")


class VideoMonitoringEngine:
    frame_processor: FrameProcessor
    target_class_index: int
    target_minimum_coverage: float
    temporal_filter: TemporalFilter
    on_detected: Callable[[list[OutputRow]], None]
    on_not_detected: Callable[[], None]
    on_exception: Callable[[Exception], None]

    def __init__(
        self,
        frame_processor: FrameProcessor,
        target_class_index: int,
        target_minimum_coverage: float,
        temporal_filter: TemporalFilter,
        on_detected: Callable[[list[OutputRow]], None],
        on_not_detected: Callable[[], None],
        on_exception: Callable[[Exception], None],
    ):
        self.frame_processor = frame_processor
        self.target_class_index = target_class_index
        self.target_minimum_coverage = target_minimum_coverage
        self.temporal_filter = temporal_filter
        self.on_detected = on_detected
        self.on_not_detected = on_not_detected
        self.on_exception = on_exception

    def monitor(
        self,
        video_source,
        show=False,
        retry_count: int = -1,
        retry_interval_seconds: int = 0,
    ):
        retries = 0

        while True:
            if retries > retry_count and retry_count >= 0:
                logger.warn(f"Número máximo de tentativas atingido. Interrompendo o monitoramento.")
                
                break

            elif retries > 0:
                logger.info(f"Tentativa n° {retries}")

            is_object_detected_current_value: bool = False
            video_capture = cv2.VideoCapture(video_source)

            try:
                if not video_capture.isOpened():
                    raise CameraSignalLostError()

                while video_capture.isOpened():
                    success, frame = video_capture.read()

                    if not success:
                        raise CameraSignalLostError()

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

                    self.temporal_filter.append(
                        is_object_in_frame and object_has_minimum_coverage
                    )

                    if not self.temporal_filter.is_full:
                        continue

                    is_object_consistently_detected = self.temporal_filter.validate()

                    if (
                        not is_object_detected_current_value
                        and is_object_consistently_detected
                    ):
                        is_object_detected_current_value = True

                        self.temporal_filter.clear()
                        self.on_detected(output)
                    elif (
                        is_object_detected_current_value
                        and not is_object_consistently_detected
                    ):
                        is_object_detected_current_value = False

                        self.temporal_filter.clear()
                        self.on_not_detected()

            except Exception as exception:
                self.on_exception(exception)

            finally:
                video_capture.release()
                cv2.destroyAllWindows()

                if retries < retry_count or retry_count < 0:
                    logger.info(f"Tentando novamente em {retry_interval_seconds}s...")

                    time.sleep(retry_interval_seconds)

                retries += 1
