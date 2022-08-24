import tensorflow as tf
import keras.metrics

from hourglass_tensorflow.utils.tf import tf_matrix_argmax
from hourglass_tensorflow.utils.tf import tf_batch_matrix_argmax
from hourglass_tensorflow.utils.tf import tf_dynamic_matrix_argmax


class RatioCorrectKeypoints(keras.metrics.Metric):
    """RatioCorrectKeypoints metric identifies the percentage of "true positive" keypoints detected

    This metric binarize our heatmap generation model (Regression Problem),
    with a simple statement: Is the predicted keypoint within a given distance from
    the actual value? This 0/1 modelisation allows us to consider keypoints as
    true positives (TP).

    The choice of threshold is arbitrary and should be in `range(1, sqrt(2)*HEATMAP_SIZE)`

    Args:
        threshold (int, optional): Threshold in pixel to consider the keypoint as correct.
            Defaults to 5.
        name (str, optional): Tensor name. Defaults to None.
        dtype (tf.dtypes, optional): Tensor data type. Defaults to None.
        intermediate_supervision (bool, optional): Whether or not the intermediate supervision
        is activated.
            Defaults to True.
    """

    def __init__(
        self,
        threshold: int = 5,
        name=None,
        dtype=None,
        intermediate_supervision: bool = True,
        **kwargs
    ) -> None:
        """See help(RatioCorrectKeypoints)"""
        super().__init__(name, dtype, **kwargs)
        self.threshold = threshold
        self.correct_keypoints = self.add_weight(
            name="correct_keypoints", initializer="zeros"
        )
        self.total_keypoints = self.add_weight(
            name="total_keypoints", initializer="zeros"
        )
        self.intermediate_supervision = intermediate_supervision

    def argmax_tensor(self, tensor):
        return tf_dynamic_matrix_argmax(
            tensor,
            intermediate_supervision=self.intermediate_supervision,
            keepdims=True,
        )

    def update_state(self, y_true, y_pred, *args, **kwargs):
        ground_truth_joints = self.argmax_tensor(y_true)
        predicted_joints = self.argmax_tensor(y_pred)
        distance = ground_truth_joints - predicted_joints
        norms = tf.norm(tf.cast(distance, dtype=tf.dtypes.float32), ord=2, axis=-1)
        correct_keypoints = tf.reduce_sum(
            tf.cast(norms < self.threshold, dtype=tf.dtypes.int32)
        )
        total_keypoints = tf.reduce_prod(tf.shape(norms))
        self.correct_keypoints.assign_add(correct_keypoints)
        self.total_keypoints.assign_add(total_keypoints)

    def result(self, *args, **kwargs):
        return self.correct_keypoints / self.total_keypoints

    def reset_states(self) -> None:
        self.correct_keypoints.assign(0)
        self.total_keypoints.assign(0)


class PercentageOfCorrectKeypoints(keras.metrics.Metric):
    """PercentageOfCorrectKeypoints metric measures if predicted keypoint and true joint are within a distance threshold

    PCK is used as an accuracy metric that measures if the predicted keypoint and the true joint are within
    a certain distance threshold. The PCK is usually set with respect to the scale of the subject,
    which is enclosed within the bounding box.

    PCK metric uses a dynamic threshold for each sample since the threshold is computed from the ground
    truth joints where RatioCorrectKeypoints uses a fixed threshold for every sample. Therefore, you
    need to establish a reference limb to compute this dynamic threshold.

    The threshold can either be:
        - PCKh@0.5 is when the threshold = 50% of the head bone link
        - PCK@0.2 = Distance between predicted and true joint < 0.2 * torso diameter

    Args:
        reference (tuple[int, int], optional): Joint ID tuple to consider as reference.
            Defaults to (8, 9).
        threshold (float, optional): Threshold in percentage of the considered reference limb size.
            Defaults to 0.5/50%.
        name (str, optional): Tensor name. Defaults to None.
        dtype (tf.dtypes, optional): Tensor data type. Defaults to None.
        intermediate_supervision (bool, optional): Whether or not the intermediate supervision
        is activated.
            Defaults to True.
    """

    def __init__(
        self,
        reference: tuple[int, int] = (8, 9),
        threshold: float = 0.5,
        name=None,
        dtype=None,
        intermediate_supervision: bool = True,
        **kwargs
    ) -> None:
        """See help(PercentageOfCorrectKeypoints)"""
        super().__init__(name, dtype, **kwargs)
        self.threshold = threshold
        self.reference = reference
        self.correct_keypoints = self.add_weight(
            name="correct_keypoints", initializer="zeros"
        )
        self.total_keypoints = self.add_weight(
            name="total_keypoints", initializer="zeros"
        )
        self.intermediate_supervision = intermediate_supervision

    def argmax_tensor(self, tensor):
        return tf_dynamic_matrix_argmax(
            tensor,
            intermediate_supervision=self.intermediate_supervision,
            keepdims=True,
        )

    def update_state(self, y_true, y_pred, *args, **kwargs):
        ground_truth_joints = self.argmax_tensor(y_true)
        predicted_joints = self.argmax_tensor(y_pred)
        # We compute distance between ground truth and prediction
        distance = tf.norm(
            tf.cast(ground_truth_joints - predicted_joints, dtype=tf.dtypes.float32),
            ord=2,
            axis=-1,
        )
        # We compute the norm of the reference limb from the ground truth
        reference_distance = tf.expand_dims(
            tf.norm(
                tf.cast(
                    ground_truth_joints[:, self.reference[0], :]
                    - ground_truth_joints[:, self.reference[1], :],
                    tf.dtypes.float32,
                ),
                axis=1,
            ),
            axis=-1,
        )
        # We apply the thresholding condition
        correct_keypoints = tf.reduce_sum(
            tf.cast(
                distance < (tf.expand_dims(reference_distance, -1) * self.threshold),
                dtype=tf.dtypes.int32,
            )
        )
        total_keypoints = tf.reduce_prod(tf.shape(reference_distance))
        self.correct_keypoints.assign_add(correct_keypoints)
        self.total_keypoints.assign_add(total_keypoints)

    def result(self, *args, **kwargs):
        return self.correct_keypoints / self.total_keypoints

    def reset_states(self) -> None:
        self.correct_keypoints.assign(0)
        self.total_keypoints.assign(0)
