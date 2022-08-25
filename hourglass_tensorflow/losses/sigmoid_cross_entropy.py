import tensorflow as tf
import keras.losses


class SigmoidCrossEntropyLoss(keras.losses.Loss):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.AUTO, name=None, *args, **kwargs
    ):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_pred,
            labels=y_true,
            name="nn.sigmoid_cross_entropy_with_logits",
        )
