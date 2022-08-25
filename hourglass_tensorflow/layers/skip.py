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

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "output_filters": self.output_filters,
            },
        }

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        if inputs.get_shape()[-1] == self.output_filters:
            return inputs
        else:
            return self.conv(inputs)
