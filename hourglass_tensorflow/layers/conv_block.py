from typing import Dict

import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.batch_norm_relu_conv import BatchNormReluConvLayer


class ConvBlockLayer(Layer):
    """Custom Keras Layers

    This layers apply the following Ops:
    1. BatchNormReluConvLayer
    2. BatchNormReluConvLayer
    3. BatchNormReluConvLayer
    """

    def __init__(
        self,
        output_filters: int,
        momentum: float = 0.9,
        epsilon: float = 0.9,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        """_summary_

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
        # Create layers
        self.bnrc1 = BatchNormReluConvLayer(
            filters=output_filters // 2,
            kernel_size=1,
            name="BNRC1",
            momentum=momentum,
            epsilon=epsilon,
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        self.bnrc2 = BatchNormReluConvLayer(
            filters=output_filters // 2,
            kernel_size=3,
            name="BNRC2",
            momentum=momentum,
            epsilon=epsilon,
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        self.bnrc3 = BatchNormReluConvLayer(
            filters=output_filters,
            kernel_size=1,
            name="BNRC3",
            momentum=momentum,
            epsilon=epsilon,
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
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
        x = self.bnrc1(inputs, training=training)
        x = self.bnrc2(x, training=training)
        x = self.bnrc3(x, training=training)
        return x
