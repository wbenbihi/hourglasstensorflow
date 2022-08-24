import tensorflow as tf
import keras.metrics

from hourglass_tensorflow.utils.tf import tf_matrix_argmax
from hourglass_tensorflow.utils.tf import tf_batch_matrix_argmax


class OverallMeanDistance(keras.metrics.Metric):
    def __init__(
        self, name=None, dtype=None, intermediate_supervision: bool = True, **kwargs
    ):
        super().__init__(name, dtype, **kwargs)
        self.batches = self.add_weight(name="batches", initializer="zeros")
        self.distance = self.add_weight(name="distance", initializer="zeros")
        self.batch_mode = False
        self.intermediate_supervision = intermediate_supervision

    def check_batch_mode(self, tensor):
        if self.batch_mode is None:
            if self.intermediate_supervision:
                if len(tf.shape(tensor)) == 5:
                    self.batch_mode = True
                elif len(tf.shape(tensor)) == 4:
                    self.batch_mode = False
            else:
                if len(tf.shape(tensor)) == 4:
                    self.batch_mode = True
                elif len(tf.shape(tensor)) == 3:
                    self.batch_mode = False
        else:
            raise ValueError("Unknown mode for this tensor dimension tf.shape(tensor)")

    def argmax_tensor(self, tensor):
        if self.batch_mode:
            if self.intermediate_supervision:
                return tf_batch_matrix_argmax(tensor[:, -1, :, :, :])
            else:
                return tf_batch_matrix_argmax(tensor)
        else:
            if self.intermediate_supervision:
                return tf_matrix_argmax(tensor[-1])
            else:
                return tf_matrix_argmax(tensor)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        self.check_batch_mode()
        ground_truth_joints = self.argmax_tensor(y_true)
        predicted_joints = self.argmax_tensor(y_pred)
        distance = ground_truth_joints - predicted_joints
        mean_distance = tf.reduce_mean(
            tf.norm(tf.cast(distance, dtype=tf.dtypes.float32), ord=2, axis=-1)
        )
        self.distance.assign_add(mean_distance)
        self.batches.assign_add(1.0)

    def result(self, *args, **kwargs):
        return self.distance / self.batches

    def reset_states(self) -> None:
        self.batches.assign(0.0)
        self.distance.assign(0.0)
