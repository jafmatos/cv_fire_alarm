from collections.abc import Callable
from detection.sentinel.logger import logger
import requests


class ActuatorController:
    base_url: str
    on_exception: Callable[[Exception], None]


    def __init__(self, base_url: str, on_exception: Callable[[Exception], None]):
        self.base_url = base_url
        self.on_exception = on_exception


    def activate(self):
        try:
            requests.get(f"{self.base_url}/activate")

            logger.info("Pedido de ativação enviado com sucesso")
        except Exception as exception:
            self.on_exception(exception)


    def deactivate(self):
        try:
            requests.get(f"{self.base_url}/deactivate")

            logger.info("Pedido de desativação enviado com sucesso")
        except Exception as exception:
            self.on_exception(exception)