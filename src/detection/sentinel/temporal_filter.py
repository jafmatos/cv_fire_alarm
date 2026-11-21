from typing import Literal
import collections


class TemporalFilter:
    type: Literal["detect", "undetect"]
    deque: collections.deque[bool]
    deque_length: int
    threshold: float

    def __init__(self, type: Literal["detect", "undetect"], deque_length: int, threshold: float):
        self.type = type
        self.deque = collections.deque(maxlen=deque_length)
        self.deque_length = deque_length
        self.threshold = threshold

    def append(self, value: bool):
        self.deque.append(value)

    def clear(self):
        self.deque.clear()

    def validate(self):
        deque_length = len(self.deque)
        deque_sum = sum(self.deque)

        if deque_length != self.deque.maxlen:
            return self.type == "undetect"

        if self.type == "detect":
            return deque_sum >= deque_length * self.threshold
        
        return deque_sum < deque_length * self.threshold

    @property
    def is_full(self) -> bool:
        return len(self.deque) == self.deque.maxlen

    @property
    def percent_true(self) -> float:
        return sum(self.deque) / len(self.deque)