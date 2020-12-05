import tensorflow as tf
from .batch_norm_conv_relu import BatchNormReluConv


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self, output_filters, trainable=True, momentum=0.9, epsilon=1e-5, name=""
    ):
        super(ConvBlock, self).__init__(name=name)
        self.bnrc1 = BatchNormReluConv(
            filters=output_filters // 2,
            kernel_size=1,
            name=f"BNRC1",
            momentum=momentum,
            epsilon=epsilon,
        )
        self.bnrc2 = BatchNormReluConv(
            filters=output_filters // 2,
            kernel_size=3,
            name=f"BNRC2",
            momentum=momentum,
            epsilon=epsilon,
        )
        self.bnrc3 = BatchNormReluConv(
            filters=output_filters,
            kernel_size=1,
            name=f"BNRC3",
            momentum=momentum,
            epsilon=epsilon,
        )

    def call(self, inputs, training=False):
        x = self.bnrc1(inputs, training=training)
        x = self.bnrc2(x, training=training)
        x = self.bnrc3(x, training=training)
        return x