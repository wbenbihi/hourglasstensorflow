__version__ = "1.0.1"

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "Please install tensorflow before using this package: pip install tensorflow"
    )
