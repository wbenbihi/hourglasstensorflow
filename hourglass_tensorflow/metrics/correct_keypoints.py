import tensorflow as tf
from keras.metrics import Metric

from hourglass_tensorflow.utils.tf import tf_dynamic_matrix_argmax


class RatioCorrectKeypoints(Metric):
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

    def _internal_update(self, y_true, y_pred):
        ground_truth_joints = self.argmax_tensor(y_true)
        predicted_joints = self.argmax_tensor(y_pred)
        distance = ground_truth_joints - predicted_joints
        norms = tf.norm(tf.cast(distance, dtype=tf.dtypes.float32), ord=2, axis=-1)
        correct_keypoints = tf.cast(
            tf.reduce_sum(tf.cast(norms < self.threshold, dtype=tf.dtypes.int32)),
            dtype=tf.dtypes.float32,
        )
        total_keypoints = tf.cast(
            tf.reduce_prod(tf.shape(norms)), dtype=tf.dtypes.float32
        )
        self.correct_keypoints.assign_add(correct_keypoints)
        self.total_keypoints.assign_add(total_keypoints)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        return self._internal_update()

    def result(self, *args, **kwargs):
        return tf.math.divide_no_nan(self.correct_keypoints, self.total_keypoints)

    def reset_state(self) -> None:
        self.correct_keypoints.assign(0.0)
        self.total_keypoints.assign(0.0)


class PercentageOfCorrectKeypoints(Metric):
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
        ratio (float, optional): Threshold in percentage of the considered reference limb size.
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
        ratio: float = 0.5,
        name=None,
        dtype=None,
        intermediate_supervision: bool = True,
        **kwargs
    ) -> None:
        """See help(PercentageOfCorrectKeypoints)"""
        super().__init__(name, dtype, **kwargs)
        self.ratio = ratio
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

    def _internal_update(self, y_true, y_pred):
        ground_truth_joints = self.argmax_tensor(y_true)
        predicted_joints = self.argmax_tensor(y_pred)
        # We compute distance between ground truth and prediction
        error = tf.cast(ground_truth_joints - predicted_joints, dtype=tf.dtypes.float32)
        distance = tf.norm(error, ord=2, axis=-1)
        # We compute the norm of the reference limb from the ground truth
        reference_limb_error = tf.cast(
            ground_truth_joints[:, self.reference[0], :]
            - ground_truth_joints[:, self.reference[1], :],
            dtype=tf.float32,
        )
        reference_distance = tf.norm(reference_limb_error, ord=2, axis=-1)
        # We apply the thresholding condition
        condition = tf.cast(
            distance < (reference_distance * self.ratio), dtype=tf.float32
        )
        correct_keypoints = tf.reduce_sum(condition)
        total_keypoints = tf.cast(
            tf.reduce_prod(tf.shape(distance)), dtype=tf.dtypes.float32
        )
        self.correct_keypoints.assign_add(correct_keypoints)
        self.total_keypoints.assign_add(total_keypoints)

    def update_state(self, y_true, y_pred, *args, **kwargs):
        return self._internal_update(y_true, y_pred)

    def result(self, *args, **kwargs):
        return tf.math.divide_no_nan(self.correct_keypoints, self.total_keypoints)

    def reset_state(self) -> None:
        self.correct_keypoints.assign(0.0)
        self.total_keypoints.assign(0.0)


class ObjectKeypointSimilarity(Metric):
    """ObjectKeypointSimilarity metric measures if predicted keypoint and true joint are within a distance threshold

    OKS is commonly used in the COCO keypoint challenge as an evaluation metric. It is calculated from
    the distance between predicted points and ground truth points normalized by the scale of the person.
    Scale and Keypoint constant needed to equalize the importance of each keypoint: neck location more precise than hip location.

    Args:
        name (str, optional): Tensor name. Defaults to None.
        dtype (tf.dtypes, optional): Tensor data type. Defaults to None.
        keypoints_constant (list, optional): Keypoints constant. Defaults to None.
        intermediate_supervision (bool, optional): Whether or not the intermediate supervision
        is activated.
            Defaults to True.
        compute_visibility_flags (bool, optional): Compute visibility flags from ground truth.
            Defaults to True.

    Notes:
        For each object, ground truth keypoints have the form [x1,y1,v1,...,xk,yk,vk],
        where x,y are the keypoint locations and v is a visibility flag defined as v=0: not labeled,
        v=1: labeled but not visible, and v=2: labeled and visible

        We define the object keypoint similarity (OKS) as: `OKS = Σi[exp(-di^2/ {2 s^2 κi^2} )δ(vi>0)] / Σi[δ(vi>0)]`

        The `di` are the Euclidean distances between each corresponding ground truth and detected keypoint
        and the `vi` are the visibility flags of the ground truth (the detector's predicted `vi` are not used).
        To compute OKS, we pass the `di` through an unnormalized Guassian with standard deviation `sκi`,
        where `s` is the object scale and `κi` is a per-keypont constant that controls falloff.
        For each keypoint this yields a keypoint similarity that ranges between 0 and 1.
        These similarities are averaged over all labeled keypoints (keypoints for which `vi>0`).
        Predicted keypoints that are not labeled (`vi=0`) do not affect the OKS. Perfect predictions will have
        OKS=1 and predictions for which all keypoints are off by more than a few standard deviations sκi will have OKS~0.
    """

    def __init__(
        self,
        name=None,
        dtype=None,
        keypoints_constants: list = None,
        intermediate_supervision: bool = True,
        compute_visibility_flags: bool = True,
        **kwargs
    ) -> None:
        """See help(ObjectKeypointSimilarity)"""
        super().__init__(name, dtype, **kwargs)
        self.oks_sum = self.add_weight(name="oks_sum", initializer="zeros")
        self.samples = self.add_weight(name="samples", initializer="zeros")
        self.intermediate_supervision = intermediate_supervision
        self.compute_visibility_flags = compute_visibility_flags
        self.keypoints_constants = keypoints_constants
        if self.keypoints_constants is None:
            # Set default value
            pass

    def reset_state(self) -> None:
        self.oks_sum.assign(0.0)
        self.samples.assign(0.0)

    def argmax_tensor(self, tensor):
        return tf_dynamic_matrix_argmax(
            tensor,
            intermediate_supervision=self.intermediate_supervision,
            keepdims=True,
        )

    def get_visibility(self, y_true):
        # TODO Implement Get Visibility Flags HERE
        raise NotImplementedError

    def get_scale(self, y_true):
        # TODO Implement Get Object Scale HERE
        raise NotImplementedError

    def oks(self, distance, visibility_flags, scale):
        # Compute the L2/Euclidean Distance
        #     distances = np.linalg.norm(y_pred - y_true, axis=-1)
        #     # Compute the exponential part of the equation
        #     exp_vector = np.exp(-(distances**2) / (2 * (SCALE**2) * (KAPPA**2)))
        #     # The numerator expression
        #     numerator = np.dot(exp_vector, visibility.astype(bool).astype(int))
        #     # The denominator expression
        #     denominator = np.sum(visibility.astype(bool).astype(int))
        #     return numerator / denominator
        raise NotImplementedError

    def update_state(self, y_true, y_pred, *args, **kwargs):
        ground_truth_joints = self.argmax_tensor(y_true)
        predicted_joints = self.argmax_tensor(y_pred)
        # We compute distance between ground truth and prediction
        distance = tf.norm(
            tf.cast(ground_truth_joints - predicted_joints, dtype=tf.dtypes.float32),
            ord=2,
            axis=-1,
        )
        # We generate visibility tensor and scale scalar
        visibility = self.get_visibility(ground_truth_joints)
        scales = self.get_scale(ground_truth_joints)
        # We compute value to add to weights
        oks = self.oks(distance, visibility_flags=visibility, scale=scales)
        total_keypoints = tf.reduce_prod(
            tf.cast(tf.shape(distance)[0], dtype=tf.dtypes.float32)
        )
        oks_sum = tf.reduce_sum(oks)
        # Add to weight
        self.oks_sum.assign_add(oks_sum)
        self.samples.assign_add(total_keypoints)
