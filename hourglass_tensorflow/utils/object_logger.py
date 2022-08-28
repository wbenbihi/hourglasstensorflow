from loguru import logger


class ObjectLogger:
    """Helper class for inner class logging"""

    def __init__(self, verbose: bool = True) -> None:
        """see help(ObjectLogger)

        Args:
            verbose (bool, optional): Activates the logs. Defaults to True.
        """
        self._verbose = verbose

    def log(self, level: str, message: str) -> None:
        """Generic log

        Args:
            level (str): LOG LEVEL
            message (str): LOG MESSAGE
        """
        if self._verbose:
            logger.log(level, message)

    def info(self, message: str) -> None:
        """Info log

        Args:
            message (str): LOG MESSAGE
        """
        self.log("INFO", message=message)

    def debug(self, message: str) -> None:
        """Debug log

        Args:
            message (str): LOG MESSAGE
        """
        self.log("DEBUG", message=message)

    def error(self, message: str) -> None:
        """Error log

        Args:
            message (str): LOG MESSAGE
        """
        self.log("ERROR", message=message)

    def warning(self, message: str) -> None:
        """Warning log

        Args:
            message (str): LOG MESSAGE
        """
        self.log("WARNING", message=message)

    def success(self, message: str) -> None:
        """Success log

        Args:
            message (str): LOG MESSAGE
        """
        self.log("SUCCESS", message=message)
