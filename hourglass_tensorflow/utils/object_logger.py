from loguru import logger


class ObjectLogger:
    def __init__(self, verbose: bool = True) -> None:
        self._verbose = verbose

    def log(self, level, message) -> None:
        if self._verbose:
            logger.log(level, message)

    def info(self, message) -> None:
        self.log("INFO", message=message)

    def debug(self, message) -> None:
        self.log("DEBUG", message=message)

    def error(self, message) -> None:
        self.log("ERROR", message=message)

    def warning(self, message) -> None:
        self.log("WARNING", message=message)

    def success(self, message) -> None:
        self.log("SUCCESS", message=message)
