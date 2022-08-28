from typing import Dict

import tensorflow as tf
from keras import layers
from keras.layers import Layer


class BatchNormReluConvLayer(Layer):
    """Custom Keras Layers

    This layers apply the following Ops:
    1. Batch Normalization
    2. ReLU Activation
    3. Convolution2D

    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        activation: str = None,
        kernel_initializer: str = "glorot_uniform",
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        """see help(BatchNormReluConvLayer)

        Args:
            filters (int): Convolution filters
            kernel_size (int): Convolution kernel size
            strides (int, optional): Stride for convolution kernel. Defaults to 1.
            padding (str, optional): Padding for convolution. Defaults to "same".
            activation (str, optional): Use activation function in convolution. Defaults to None.
            kernel_initializer (str, optional): Convolution kernel initializer. Defaults to "glorot_uniform".
            momentum (float, optional): Batch Norm momentum. Defaults to 0.9.
            epsilon (float, optional): Batch Norm epsilon. Defaults to 1e-5.
            name (str, optional): Layer name. Defaults to None.
            dtype (_type_, optional): check keras.layers.Layer.dtype. Defaults to None.
            dynamic (bool, optional): check keras.layers.Layer.dynamic. Defaults to False.
            trainable (bool, optional): check keras.layers.Layer.trainable. Defaults to True.
        """
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store Config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.momentum = momentum
        self.epsilon = epsilon
        # Create Layers
        self.batch_norm = layers.BatchNormalization(
            axis=-1,
            momentum=momentum,
            epsilon=epsilon,
            trainable=trainable,
            name="BatchNorm",
        )
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name="Conv2D",
            activation=activation,
            kernel_initializer=kernel_initializer,
        )
        self.relu = layers.ReLU(
            name="ReLU",
        )

    def get_config(self) -> Dict:
        """Get the layer configuration

        Necessary for model serialization

        Returns:
            Dict: Layer configuration
        """
        return {
            **super().get_config(),
            **{
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Scripts the graph operation to perform on layer __call__

        Args:
            inputs (tf.Tensor): input tensor
            training (bool, optional): is the layer currently training. Defaults to True.

        Returns:
            tf.Tensor: the layer's output tensor
        """
        x = self.batch_norm(inputs, training=training)
        x = self.relu(x)
        x = self.conv(x)
        return x
