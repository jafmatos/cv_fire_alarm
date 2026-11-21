from collections.abc import Callable
from detection.sentinel.logger import logger
import requests


class ActuatorController:
    actuator_activate_url: str
    actuator_deactivate_url: str
    on_exception: Callable[[Exception], None]


    def __init__(
        self,
        actuator_activate_url: str,
        actuator_deactivate_url: str,
        on_exception: Callable[[Exception], None],
    ):
        self.actuator_activate_url = actuator_activate_url
        self.actuator_deactivate_url = actuator_deactivate_url
        self.on_exception = on_exception


    def activate(self):
        try:
            requests.get(self.actuator_activate_url)

            logger.info("Pedido de ativação enviado com sucesso")
        except Exception as exception:
            self.on_exception(exception)


    def deactivate(self):
        try:
            requests.get(self.actuator_deactivate_url)

            logger.info("Pedido de desativação enviado com sucesso")
        except Exception as exception:
            self.on_exception(exception)