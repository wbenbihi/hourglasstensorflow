__version__ = "0.0.0"

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "Please install tensorflow before using this package: pip install tensorflow"
    )
