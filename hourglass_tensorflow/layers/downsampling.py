import math
from typing import Dict

import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.residual import ResidualLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer


class DownSamplingLayer(Layer):
    """Custom Keras Layers

    This layers apply successive `ResidualLayer`
    based on the `input_size` and `output_size`
    parameters

    """

    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 64,
        kernel_size: int = 7,
        output_filters: int = 256,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        """see help(DownSamplingLayer)

        Args:
            input_size (int, optional): The input tensor size. Defaults to 256.
            output_size (int, optional): The output tensor expected size. Defaults to 64.
            kernel_size (int, optional): Kernel size for convolution. Defaults to 7.
            output_filters (int, optional): The number of filters on layer's output tensor.
                Defaults to 256.
            name (str, optional): Layer name. Defaults to None.
            dtype (_type_, optional): check keras.layers.Layer.dtype. Defaults to None.
            dynamic (bool, optional): check keras.layers.Layer.dynamic. Defaults to False.
            trainable (bool, optional): check keras.layers.Layer.trainable. Defaults to True.
        """
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store config
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.output_filters = output_filters
        # Init Computation
        self.downsamplings = int(math.log2(input_size // output_size) + 1)
        self.layers = []
        # Create Layers
        for i in range(self.downsamplings):
            if i == 0:
                self.layers.append(
                    ConvBatchNormReluLayer(
                        filters=(
                            output_filters // 4
                            if self.downsamplings > 1
                            else output_filters
                        ),
                        kernel_size=kernel_size,
                        strides=(2 if self.downsamplings > 1 else 1),
                        name="CNBR",
                        dtype=dtype,
                        dynamic=dynamic,
                        trainable=trainable,
                    )
                )
            elif i == self.downsamplings - 1:
                self.layers.append(
                    ResidualLayer(
                        output_filters=output_filters,
                        name=f"Residual{i}",
                        dtype=dtype,
                        dynamic=dynamic,
                        trainable=trainable,
                    )
                )
            else:
                self.layers.append(
                    ResidualLayer(
                        output_filters=output_filters // 2,
                        name=f"Residual{i}",
                        dtype=dtype,
                        dynamic=dynamic,
                        trainable=trainable,
                    )
                )
                self.layers.append(
                    layers.MaxPool2D(
                        pool_size=(2, 2), padding="same", name=f"MaxPool{i}"
                    )
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
                "input_size": self.input_size,
                "output_size": self.output_size,
                "kernel_size": self.kernel_size,
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
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        return x
