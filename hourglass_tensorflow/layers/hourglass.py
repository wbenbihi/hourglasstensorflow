import tensorflow as tf
from keras import layers
from keras.layers import Layer

from hourglass_tensorflow.layers.residual import ResidualLayer
from hourglass_tensorflow.layers.conv_batch_norm_relu import ConvBatchNormReluLayer


class HourglassLayer(Layer):
    def __init__(
        self,
        feature_filters: int = 256,
        output_filters: int = 16,
        downsamplings: int = 4,
        name: str = None,
        dtype=None,
        dynamic=False,
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, dtype=dtype, dynamic=dynamic, trainable=trainable)
        # Store Config
        self.downsamplings = downsamplings
        self.feature_filters = feature_filters
        self.output_filters = output_filters
        # Init parameters
        self.layers = [{} for i in range(self.downsamplings)]
        # Create Layers
        self._hm_output = ConvBatchNormReluLayer(
            filters=output_filters,
            kernel_size=1,
            name="HeatmapOutput",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        self._transit_output = ConvBatchNormReluLayer(
            filters=feature_filters,
            kernel_size=1,
            name="TransitOutput",
            dtype=dtype,
            dynamic=dynamic,
            trainable=trainable,
        )
        for i, downsampling in enumerate(self.layers):
            downsampling["up_1"] = ResidualLayer(
                output_filters=feature_filters,
                name=f"Step{i}_ResidualUp1",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["low_"] = layers.MaxPool2D(
                pool_size=(2, 2),
                padding="same",
                name=f"Step{i}_MaxPool",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["low_1"] = ResidualLayer(
                output_filters=feature_filters,
                name=f"Step{i}_ResidualLow1",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            if i == 0:
                downsampling["low_2"] = ResidualLayer(
                    output_filters=feature_filters,
                    name=f"Step{i}_ResidualLow2",
                    dtype=dtype,
                    dynamic=dynamic,
                    trainable=trainable,
                )
            downsampling["low_3"] = ResidualLayer(
                output_filters=feature_filters,
                name=f"Step{i}_ResidualLow3",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["up_2"] = layers.UpSampling2D(
                size=(2, 2),
                data_format=None,
                interpolation="nearest",
                name=f"Step{i}_UpSampling2D",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
            downsampling["out"] = layers.Add(
                name=f"Step{i}_Add",
                dtype=dtype,
                dynamic=dynamic,
                trainable=trainable,
            )
        # endregion

    def get_config(self):
        return {
            **super().get_config(),
            **{
                "downsamplings": self.downsamplings,
                "feature_filters": self.feature_filters,
                "output_filters": self.output_filters,
            },
        }

    def _recursive_call(self, input_tensor, step, training=True):
        step_layers = self.layers[step]
        up_1 = step_layers["up_1"](input_tensor, training=training)
        low_ = step_layers["low_"](input_tensor, training=training)
        low_1 = step_layers["low_1"](low_, training=training)
        if step == 0:
            low_2 = step_layers["low_2"](low_1, training=training)
        else:
            low_2 = self._recursive_call(low_1, step=(step - 1), training=training)
        low_3 = step_layers["low_3"](low_2, training=training)
        up_2 = step_layers["up_2"](low_3, training=training)
        out = step_layers["out"]([up_1, up_2], training=training)
        return out

    def call(self, inputs, training=False):
        x = self._recursive_call(
            input_tensor=inputs, step=self.downsamplings - 1, training=training
        )
        intermediate = self._hm_output(x, training=training)
        out_tensor = tf.add_n(
            [inputs, self._transit_output(intermediate, training=training), x],
            name=f"{self.name}_OutputAdd",
        )
        return out_tensor, intermediate
