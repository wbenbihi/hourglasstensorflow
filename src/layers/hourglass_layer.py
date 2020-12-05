import tensorflow as tf
from .residual_layer import ResidualLayer
from .conv_batch_norm_relu import ConvBatchNormRelu


class HourglassLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        downsamplings: int = 4,
        feature_filters: int = 256,
        output_filters: int = 16,
        name: str = "",
        trainable: bool = True,
    ):
        super(HourglassLayer, self).__init__(name=name)
        self.downsamplings = downsamplings
        self.layers = [{} for i in range(downsamplings)]
        self._hm_output = ConvBatchNormRelu(
            filters=output_filters,
            kernel_size=1,
            name="HeatmapOutput",
            trainable=trainable,
        )
        self._transit_output = ConvBatchNormRelu(
            filters=feature_filters,
            kernel_size=1,
            name="TransitOutput",
            trainable=trainable,
        )
        for downsampling in range(downsamplings):
            self.layers[downsampling]["up_1"] = ResidualLayer(
                output_filters=feature_filters,
                trainable=trainable,
                name=f"Step{downsampling}_ResidualUp1",
            )
            self.layers[downsampling]["low_"] = tf.keras.layers.MaxPool2D(
                pool_size=(2, 2), padding="same", name=f"Step{downsampling}_MaxPool"
            )
            self.layers[downsampling]["low_1"] = ResidualLayer(
                output_filters=feature_filters,
                trainable=trainable,
                name=f"Step{downsampling}_ResidualLow1",
            )
            if downsampling == 0:
                self.layers[downsampling]["low_2"] = ResidualLayer(
                    output_filters=feature_filters,
                    trainable=trainable,
                    name=f"Step{downsampling}_ResidualLow2",
                )
            self.layers[downsampling]["low_3"] = ResidualLayer(
                output_filters=feature_filters,
                trainable=trainable,
                name=f"Step{downsampling}_ResidualLow3",
            )
            self.layers[downsampling]["up_2"] = tf.keras.layers.UpSampling2D(
                size=(2, 2),
                data_format=None,
                interpolation="nearest",
                name=f"Step{downsampling}_UpSampling2D",
            )
            self.layers[downsampling]["out"] = tf.keras.layers.Add(
                name=f"Step{downsampling}_Add"
            )

    def _recursive_call(self, input_tensor, downsampling_step, training=False):
        up_1 = self.layers[downsampling_step]["up_1"](input_tensor, training=training)
        low_ = self.layers[downsampling_step]["low_"](input_tensor, training=training)
        low_1 = self.layers[downsampling_step]["low_1"](low_, training=training)
        if downsampling_step == 0:
            low_2 = self.layers[downsampling_step]["low_2"](low_1, training=training)
        else:
            low_2 = self._recursive_call(
                low_1, downsampling_step=downsampling_step - 1, training=training
            )
        low_3 = self.layers[downsampling_step]["low_3"](low_2, training=training)
        up_2 = self.layers[downsampling_step]["up_2"](low_3, training=training)
        out = self.layers[downsampling_step]["out"]([up_1, up_2], training=training)
        return out

    def call(self, inputs, training=False):
        x = self._recursive_call(
            input_tensor=inputs,
            downsampling_step=self.downsamplings - 1,
            training=training,
        )
        intermediate = self._hm_output(x, training=training)
        out_tensor = tf.add_n(
            [inputs, self._transit_output(intermediate, training=training), x],
            name=f"{self.name}_OutputAdd",
        )
        return out_tensor, intermediate