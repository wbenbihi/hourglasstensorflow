from typing import Dict

import tensorflow as tf
from keras import layers
from keras.layers import Layer


class SkipLayer(Layer):
    """Custom Keras Layers

    Returns the input layer if input channels == `output_filters`,
    apply a convolution to force the output to have `output_filters`
    channels otherwise
    """

    def __init__(
        self,
        output_filters: int,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        """_summary_

        Args:
            output_filters (int): The number of filters required in layer's output
            name (str, optional): Layer name. Defaults to None.
            dtype (_type_, optional): check keras.layers.Layer.dtype. Defaults to None.
            dynamic (bool, optional): check keras.layers.Layer.dynamic. Defaults to False.
            trainable (bool, optional): check keras.layers.Layer.trainable. Defaults to True.
        """
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store config
        self.output_filters = output_filters
        # Create Layers
        self.conv = layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="Conv2D",
            activation=None,
            kernel_initializer="glorot_uniform",
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
                "output_filters": self.output_filters,
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
        if inputs.get_shape()[-1] == self.output_filters:
            return inputs
        else:
            return self.conv(inputs)
