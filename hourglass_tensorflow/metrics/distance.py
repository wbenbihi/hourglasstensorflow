import tensorflow as tf
import keras.metrics

from hourglass_tensorflow.utils.tf import tf_dynamic_matrix_argmax


class OverallMeanDistance(keras.metrics.Metric):
    def __init__(
        self, name=None, dtype=None, intermediate_supervision: bool = True, **kwargs
    ):
        super().__init__(name, dtype, **kwargs)
        self.batches = self.add_weight(name="batches", initializer="zeros")
        self.distance = self.add_weight(name="distance", initializer="zeros")
        self.batch_mode = False
        self.intermediate_supervision = intermediate_supervision

    def argmax_tensor(self, tensor):
        return tf_dynamic_matrix_argmax(
            tensor,
            intermediate_supervision=self.intermediate_supervision,
            keepdims=True,
        )

    def _internal_update(self, y_true, y_pred):
        ground_truth_joints = self.argmax_tensor(y_true)
        predicted_joints = self.argmax_tensor(y_pred)
        distance = tf.cast(
            ground_truth_joints - predicted_joints, dtype=tf.dtypes.float32
        )
        mean_distance = tf.reduce_mean(tf.norm(distance, ord=2, axis=-1))
        self.distance.assign_add(mean_distance)
        self.batches.assign_add(1.0)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        return self._internal_update(y_true, y_pred)

    def result(self):
        return tf.math.divide_no_nan(self.distance, self.batches)

    def reset_state(self) -> None:
        self.batches.assign(0.0)
        self.distance.assign(0.0)
