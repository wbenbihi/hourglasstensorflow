import tensorflow as tf


class SkipLayer(tf.keras.layers.Layer):
    def __init__(self, output_filters, name=""):
        super(SkipLayer, self).__init__(name=name)
        self.output_filters = output_filters
        self.layer_name = name
        self.conv = tf.keras.layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name=f"Conv2D",
            activation=None,
            kernel_initializer="glorot_uniform",
        )

    def call(self, inputs, training=False):
        if inputs.get_shape().as_list()[-1] == self.output_filters:
            return inputs
        else:
            return self.conv(inputs)