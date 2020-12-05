import tensorflow as tf
from .conv_block import ConvBlock
from .skip_layer import SkipLayer


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(
        self, output_filters, trainable=True, momentum=0.9, epsilon=1e-5, name=""
    ):
        super(ResidualLayer, self).__init__(name=name)
        self.layer_name = name
        self.conv_block = ConvBlock(
            output_filters=output_filters,
            trainable=trainable,
            momentum=momentum,
            epsilon=epsilon,
            name=f"ConvBlock",
        )
        self.skip = SkipLayer(output_filters=output_filters, name=f"Skip")
        self.add = tf.keras.layers.Add(name=f"Add")

    def call(self, inputs, training=False):
        return self.add(
            [
                self.conv_block(inputs, training=training),
                self.skip(inputs, training=training),
            ]
        )
