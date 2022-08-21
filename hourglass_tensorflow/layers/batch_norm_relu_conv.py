import tensorflow as tf
from keras import layers
from keras.layers import Layer


class BatchNormReluConvLayer(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int,
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
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
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

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.batch_norm(inputs, training=training)
        x = self.relu(x)
        x = self.conv(x)
        return x
