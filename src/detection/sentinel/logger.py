from detection.sentinel.config import LOG_DIR
from datetime import datetime
from pathlib import Path


class Logger:
    log_dir: Path | None

    def __init__(self, log_dir: str | Path | None = None):
        self.log_dir = Path(log_dir) if log_dir is not None else None

    def _get_log_file_path(self):
        date_str = datetime.now().strftime("%Y-%m-%d")

        return Path(f"{self.log_dir}/{date_str}.log")

    def _append_to_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as file:
            file.write(content.rstrip("\n") + "\n")

    def _generic_log(self, tag: str, message: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        content = f"{now} [{tag:<5}]: {message}\n"

        print(content)

        if self.log_dir:
            self._append_to_file(self._get_log_file_path(), content)

    def info(self, message: str):
        self._generic_log(tag="INFO", message=message)

    def warn(self, message: str):
        self._generic_log(tag="WARN", message=message)

    def error(self, message: str):
        self._generic_log(tag="ERROR", message=message)


logger = Logger(log_dir=LOG_DIR)
