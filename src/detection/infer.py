from detection.config import WEIGHTS_PATH
from collections import deque

import serial
from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

buffer = deque(maxlen=60)
has_fire_previous_value = False
has_fire_current_value = False

def verify_detection(buffer: deque):
    global has_fire_previous_value

    if sum(buffer) >= 40:
        has_fire_current_value = True
    else:
        has_fire_current_value = False

    if not has_fire_previous_value and has_fire_current_value:
        on_detected()

    elif has_fire_previous_value and not has_fire_current_value:
        on_undetected()

    has_fire_previous_value = has_fire_current_value


def on_detected():
    relay = serial.Serial("/dev/ttyUSB0", 9600)
    relay.write(b"\xA0\x01\x01\xA2")
    relay.close()


def on_undetected():
    relay = serial.Serial("/dev/ttyUSB0", 9600)
    relay.write(b"\xA0\x01\x00\xA1")
    relay.close()


for result in model.track(
    source=1,
    persist=True,
    tracker="bytetrack.yaml",
    stream=True,
    show=True,
    verbose=False,
):
    frame_has_fire = False

    print(model.predictor.trackers[0])


    for box_index, box in enumerate(result.boxes):
        if result.names[int(box.cls[0])] == "Fogo":
            frame_has_fire = True

            break

    buffer.append(frame_has_fire)

    verify_detection(buffer)
