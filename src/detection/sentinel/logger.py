import traceback
from detection.sentinel.config import LOGS_DIR, LOGS_MAX_AGE_DAYS
from datetime import datetime, timedelta
from pathlib import Path


class Logger:
    log_dir: Path
    max_days_age: int
    _last_cleanup_date = None

    def __init__(
        self, log_dir: str | Path, max_days_age: int
    ):
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.max_days_age = max_days_age

    def _get_log_file_path(self):
        date_str = datetime.now().strftime("%Y-%m-%d")

        return Path(f"{self.log_dir}/{date_str}.log")

    def _append_to_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as file:
            file.write(content.rstrip("\n") + "\n")

        now_date = datetime.now().date()

        if self._last_cleanup_date is not None and now_date <= self._last_cleanup_date:
            return

        try:
            self._last_cleanup_date = now_date

            x_days_ago = (now_date - timedelta(days=self.max_days_age))

            oldest_allowed_filename = f"{x_days_ago}.log"

            log_files = list(path.parent.glob("*.log"))

            for log_file in log_files:
                if log_file.name >= oldest_allowed_filename:
                    continue

                log_file.unlink(missing_ok=True)

        except Exception as exception:
            print(
                f"Erro ao verificar validade dos arquivos de logs: {traceback.format_exc()}"
            )

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


logger = Logger(log_dir=LOGS_DIR, max_days_age=LOGS_MAX_AGE_DAYS)
