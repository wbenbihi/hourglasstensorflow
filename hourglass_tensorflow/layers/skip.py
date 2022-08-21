import tensorflow as tf
from keras import layers
from keras.layers import Layer


class SkipLayer(Layer):
    def __init__(
        self,
        output_filters: int,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)

        self.output_filters = output_filters

        self.conv = layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="Conv2D",
            activation=None,
            kernel_initializer="glorot_uniform",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        if inputs.get_shape()[-1] == self.output_filters:
            return inputs
        else:
            return self.conv(inputs)
