import tensorflow as tf


def conv_2d(
    input_tensor: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    activation: str = None,
    kernel_initializer: str = "glorot_uniform",
    padding: str = "same",
    prefix: str = "",
) -> tf.Tensor:
    """[summary]

    Args:
        input_tensor (tf.Tensor): [description]
        filters (int): [description]
        kernel_size (int): [description]
        strides (int, optional): [description]. Defaults to 1.
        activation (str, optional): [description]. Defaults to None.
        kernel_initializer (str, optional): [description]. Defaults to "glorot_uniform".
        padding (str, optional): [description]. Defaults to "same".
        prefix (str, optional): [description]. Defaults to "".

    Returns:
        tf.Tensor: [description]
    """
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        name=prefix + "conv_2d",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(input_tensor)
    return x


def batch_norm(
    input_tensor: tf.Tensor,
    momentum: float = 0.9,
    epsilon: float = 1e-5,
    trainable: bool = True,
    prefix: str = "",
) -> tf.Tensor:
    """[summary]

    Args:
        input_tensor (tf.Tensor): [description]
        momentum (float, optional): [description]. Defaults to 0.9.
        epsilon (float, optional): [description]. Defaults to 1e-5.
        trainable (bool, optional): [description]. Defaults to True.
        prefix (str, optional): [description]. Defaults to "".

    Returns:
        tf.Tensor: [description]
    """
    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        trainable=trainable,
        name=prefix + "batch_norm",
    )(input_tensor)
    return x


def conv_batch_norm_relu(
    input_tensor: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    activation: str = None,
    kernel_initializer: str = "glorot_uniform",
    padding: str = "same",
    prefix: str = "",
    momentum: float = 0.9,
    epsilon: float = 1e-5,
    trainable: bool = True,
) -> tf.Tensor:
    """[summary]

    Args:
        input_tensor (tf.Tensor): [description]
        filters (int): [description]
        kernel_size (int): [description]
        strides (int, optional): [description]. Defaults to 1.
        activation (str, optional): [description]. Defaults to None.
        kernel_initializer (str, optional): [description]. Defaults to "glorot_uniform".
        padding (str, optional): [description]. Defaults to "same".
        prefix (str, optional): [description]. Defaults to "".
        momentum (float, optional): [description]. Defaults to 0.9.
        epsilon (float, optional): [description]. Defaults to 1e-5.
        trainable (bool, optional): [description]. Defaults to True.

    Returns:
        tf.Tensor: [description]
    """
    x = conv_2d(
        input_tensor=input_tensor,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        prefix=prefix + "conv_bn_relu_",
    )
    x = batch_norm(
        x, momentum=0.9, epsilon=1e-5, trainable=True, prefix=prefix + "conv_bn_relu_"
    )
    x = tf.keras.layers.ReLU(name=prefix + "relu")(x)
    return x


def conv_block(
    input_tensor: tf.Tensor, numOut: int, prefix: str = "", trainable: bool = True
) -> tf.Tensor:
    """[summary]

    Args:
        input_tensor (tf.Tensor): [description]
        numOut (int): [description]
        prefix (str, optional): [description]. Defaults to "".
        trainable (bool, optional): [description]. Defaults to True.

    Returns:
        tf.Tensor: [description]
    """
    x = batch_norm(
        input_tensor,
        momentum=0.9,
        epsilon=1e-5,
        trainable=True,
        prefix=prefix + "conv_block_1_",
    )
    x = tf.keras.layers.ReLU(name=prefix + "conv_block_1_relu")(x)
    x = conv_2d(
        x,
        filters=numOut // 2,
        kernel_size=1,
        strides=1,
        padding="same",
        prefix=prefix + "conv_block_1_",
    )

    x = batch_norm(
        x, momentum=0.9, epsilon=1e-5, trainable=True, prefix=prefix + "conv_block_2_"
    )
    x = tf.keras.layers.ReLU(name=prefix + "conv_block_2_relu")(x)
    x = conv_2d(
        x,
        filters=numOut // 2,
        kernel_size=3,
        strides=1,
        padding="same",
        prefix=prefix + "conv_block_2_",
    )

    x = batch_norm(
        x, momentum=0.9, epsilon=1e-5, trainable=True, prefix=prefix + "conv_block_3_"
    )
    x = tf.keras.layers.ReLU(name=prefix + "conv_block_3_relu")(x)
    x = conv_2d(
        x,
        filters=numOut,
        kernel_size=1,
        strides=1,
        padding="same",
        prefix=prefix + "conv_block_3_",
    )
    return x


def skip_layer(input_tensor: tf.Tensor, numOut: int, prefix: str = "") -> tf.Tensor:
    """[summary]

    Args:
        input_tensor (tf.Tensor): [description]
        numOut (int): [description]
        prefix (str, optional): [description]. Defaults to "".

    Returns:
        tf.Tensor: [description]
    """
    if input_tensor.get_shape().as_list()[-1] == numOut:
        return input_tensor
    else:
        x = conv_2d(
            input_tensor,
            filters=numOut,
            kernel_size=1,
            strides=1,
            padding="same",
            prefix=prefix + "skip_layer_",
        )
        return x


def residual(
    input_tensor: tf.Tensor, numOut: int, prefix: str = "", trainable: bool = True
) -> tf.Tensor:
    """[summary]

    Args:
        input_tensor (tf.Tensor): [description]
        numOut (int): [description]
        prefix (str, optional): [description]. Defaults to "".
        trainable (bool, optional): [description]. Defaults to True.

    Returns:
        tf.Tensor: [description]
    """
    convb = conv_block(
        input_tensor, numOut, prefix=prefix + "residual_", trainable=trainable
    )
    skipl = skip_layer(input_tensor, numOut, prefix=prefix + "residual_")
    return tf.add_n([convb, skipl], name=prefix + "residual_add")

