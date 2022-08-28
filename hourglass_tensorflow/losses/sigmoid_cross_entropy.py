import tensorflow as tf
import keras.losses


class SigmoidCrossEntropyLoss(keras.losses.Loss):
    """Custom `keras` loss function"""

    def __init__(
        self, reduction=tf.keras.losses.Reduction.AUTO, name=None, *args, **kwargs
    ) -> None:
        """_summary_

        Args:
            reduction (_type_, optional): _description_. Defaults to tf.keras.losses.Reduction.AUTO.
            name (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(reduction, name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Generates the loss function's graph

        Args:
            y_true (tf.Tensor): Ground truth tensor
            y_pred (tf.Tensor): Prediction tensor

        Returns:
            tf.Tensor: Loss Function Tensor
        """
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_pred,
            labels=y_true,
            name="nn.sigmoid_cross_entropy_with_logits",
        )
