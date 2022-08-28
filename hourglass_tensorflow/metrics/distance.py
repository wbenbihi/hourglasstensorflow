import tensorflow as tf
import keras.metrics

from hourglass_tensorflow.utils.tf import tf_dynamic_matrix_argmax


class OverallMeanDistance(keras.metrics.Metric):
    """OverallMeanDistance metric measures the mean distance in pixel from the ground truth

    OverallMeanDistance is not an evaluation metric as performant as PCK or OKS.
    Its main use is to observe the evolution of the mean distance in pixel.
    OverallMeanDistance is a good proxy metric to observe convergence power of your model.
    The faster it decrease accross epochs, the faster the convergence is
    """

    def __init__(
        self, name=None, dtype=None, intermediate_supervision: bool = True, **kwargs
    ) -> None:
        """_summary_

        Args:
            name (str, optional): Tensor name. Defaults to None.
            dtype (tf.dtypes, optional): Tensor data type. Defaults to None.
            intermediate_supervision (bool, optional): Whether or not the intermediate supervision
            is activated.
                Defaults to True.
        """
        super().__init__(name, dtype, **kwargs)
        self.batches = self.add_weight(name="batches", initializer="zeros")
        self.distance = self.add_weight(name="distance", initializer="zeros")
        self.batch_mode = False
        self.intermediate_supervision = intermediate_supervision

    def _argmax_tensor(self, tensor):
        return tf_dynamic_matrix_argmax(
            tensor,
            intermediate_supervision=self.intermediate_supervision,
            keepdims=True,
        )

    def _internal_update(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> None:
        """Update the inner state of the metric

        This method is used to bypass coverage issue with tensorflow anc coverage.py

        Args:
            y_true (tf.Tensor): Ground Truth tensor
            y_pred (tf.Tensor): Prediction tensor
        """
        ground_truth_joints = self._argmax_tensor(y_true)
        predicted_joints = self._argmax_tensor(y_pred)
        distance = tf.cast(
            ground_truth_joints - predicted_joints, dtype=tf.dtypes.float32
        )
        mean_distance = tf.reduce_mean(tf.norm(distance, ord=2, axis=-1))
        self.distance.assign_add(mean_distance)
        self.batches.assign_add(1.0)

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs
    ) -> None:
        """Update the inner state of the metric

        Args:
            y_true (tf.Tensor): Ground Truth tensor
            y_pred (tf.Tensor): Prediction tensor
        """
        return self._internal_update(y_true, y_pred)

    def result(self) -> tf.float32:
        """Compute the current metric value

        Returns:
            tf.float32: Metric Value
        """
        return tf.math.divide_no_nan(self.distance, self.batches)

    def reset_state(self) -> None:
        """Reset the metric inner state"""
        self.batches.assign(0.0)
        self.distance.assign(0.0)
