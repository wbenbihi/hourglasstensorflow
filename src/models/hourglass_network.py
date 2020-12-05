import tensorflow as tf
from src.layers import DownSamplingLayer, HourglassLayer


class HourglassNetwork(tf.keras.models.Model):
    def __init__(
        self,
        input_size=256,
        output_size=64,
        stages=4,
        downsampling_steps_per_stage=4,
        inner_stage_filters=256,
        output_dimensions=16,
        intermediate_supervision=True,
        trainable=True,
        name="",
    ):
        super(HourglassNetwork, self).__init__(name=name)
        self.intermediate_supervision = intermediate_supervision
        self.downsampling = DownSamplingLayer(
            input_size=input_size,
            output_size=output_size,
            kernel_size=7,
            output_filters=inner_stage_filters,
            trainable=trainable,
            name="DownSampling",
        )
        self.hourglasses = [
            HourglassLayer(
                downsamplings=downsampling_steps_per_stage,
                feature_filters=inner_stage_filters,
                output_filters=output_dimensions,
                name=f"Hourglass{i+1}",
                trainable=trainable,
            )
            for i in range(stages)
        ]

    def call(self, inputs: tf.Tensor):
        x = self.downsampling(inputs)
        outputs = []
        for layer in self.hourglasses:
            x, y = layer(x)
            if self.intermediate_supervision:
                outputs.append(y)
        if self.intermediate_supervision:
            return tf.stack(outputs, axis=1, name="NetworkStackedOutput")
        return y