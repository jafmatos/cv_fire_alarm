import cv2
from collections.abc import Callable
from onnxruntime import InferenceSession
import numpy as np


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
