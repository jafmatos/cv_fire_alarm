import serial

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