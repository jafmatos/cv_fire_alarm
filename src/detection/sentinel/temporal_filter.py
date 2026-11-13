import collections


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
