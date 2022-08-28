from typing import Dict

import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.skip import SkipLayer
from hourglass_tensorflow.layers.conv_block import ConvBlockLayer


class ResidualLayer(Layer):
    """Custom Keras Layers

    Add up the results of a `SkipLayer` and a `ConvBlockLayer`
    """

    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        """see help(ResidualLayer)

        Args:
            output_filters (int): The number of filters required in layer's output
            momentum (float, optional): Batch Norm momentum. Defaults to 0.9.
            epsilon (float, optional): Batch Norm epsilon. Defaults to 1e-5.
            name (str, optional): Layer name. Defaults to None.
            dtype (_type_, optional): check keras.layers.Layer.dtype. Defaults to None.
            dynamic (bool, optional): check keras.layers.Layer.dynamic. Defaults to False.
            trainable (bool, optional): check keras.layers.Layer.trainable. Defaults to True.
        """
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        self.momentum = momentum
        self.epsilon = epsilon
        # Create Layers
        self.conv_block = ConvBlockLayer(
            output_filters=output_filters,
            momentum=momentum,
            epsilon=epsilon,
            name="ConvBlock",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        self.skip = SkipLayer(
            output_filters=output_filters,
            name="Skip",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        self.add = layers.Add(name="Add")

    def get_config(self) -> Dict:
        """Get the layer configuration

        Necessary for model serialization

        Returns:
            Dict: Layer configuration
        """
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
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
        return self.add(
            [
                self.conv_block(inputs, training=training),
                self.skip(inputs, training=training),
            ]
        )
