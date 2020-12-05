import math
import tensorflow as tf
from .conv_batch_norm_relu import ConvBatchNormRelu
from .residual_layer import ResidualLayer


class DownSamplingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_size: int = 256,
        output_size: int = 64,
        kernel_size: int = 7,
        output_filters=256,
        trainable=True,
        name="",
    ):
        super(DownSamplingLayer, self).__init__(name=name)

        assert (
            input_size >= output_size
        ), "Input size must be higher than by Output size"
        assert input_size % output_size == 0, "Input size not divisible by Output size"
        assert math.log2(
            input_size // output_size
        ).is_integer(), "Input size divided by Output size should be a power of 2"
        self.iterations = int(math.log2(input_size // output_size) + 1)
        self.layers = []
        for i in range(self.iterations):
            if i == 0:
                self.layers.append(
                    ConvBatchNormRelu(
                        filters=(
                            output_filters // 4
                            if self.iterations > 1
                            else output_filters
                        ),
                        kernel_size=kernel_size,
                        strides=(2 if self.iterations > 1 else 1),
                        name=f"CBNR",
                        trainable=trainable,
                    )
                )
            elif i == self.iterations - 1:
                self.layers.append(
                    ResidualLayer(
                        output_filters=output_filters,
                        name=f"Residual{i}",
                        trainable=trainable,
                    )
                )
            else:
                self.layers.append(
                    ResidualLayer(
                        output_filters=output_filters // 2,
                        name=f"Residual{i}",
                        trainable=trainable,
                    )
                )
                self.layers.append(
                    tf.keras.layers.MaxPool2D(
                        pool_size=(2, 2), padding="same", name=f"MaxPool{i}"
                    )
                )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        return x