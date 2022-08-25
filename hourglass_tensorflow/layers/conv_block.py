import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.batch_norm_relu_conv import BatchNormReluConvLayer


class ConvBlockLayer(Layer):
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

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.bnrc1(inputs, training=training)
        x = self.bnrc2(x, training=training)
        x = self.bnrc3(x, training=training)
        return x
