from loguru import logger

from hourglass_tensorflow.handlers import HTFManager

CONFIG_FILE = "config/train.default.yaml"

if __name__ == "__main__":
    # Parse file as list of records
    logger.info(f"Reading HTF data at {CONFIG_FILE}")
    manager = HTFManager(filename=CONFIG_FILE, verbose=True)
    manager()
