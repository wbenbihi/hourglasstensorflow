import pytest
import tensorflow as tf
from loguru import logger

from hourglass_tensorflow.utils.tf import tf_bivariate_normal_pdf
from hourglass_tensorflow.utils.tf import tf_dynamic_matrix_argmax
from hourglass_tensorflow.metrics.distance import OverallMeanDistance
from hourglass_tensorflow.metrics.correct_keypoints import RatioCorrectKeypoints
from hourglass_tensorflow.metrics.correct_keypoints import PercentageOfCorrectKeypoints

tf.config.experimental_run_functions_eagerly(True)


# region Fixtures

SAMPLE = {
    "image": "015601864.jpg",
    "scale": "3.021046176409755",
    "bbox_tl_x": "627",
    "bbox_tl_y": "627",
    "bbox_br_x": "706",
    "bbox_br_y": "706",
    "center_x": "594",
    "center_y": "257",
    "joint_0_X": "620",
    "joint_0_Y": "394",
    "joint_0_visible": "1",
    "joint_1_X": "616",
    "joint_1_Y": "269",
    "joint_1_visible": "1",
    "joint_2_X": "573",
    "joint_2_Y": "185",
    "joint_2_visible": "1",
    "joint_3_X": "647",
    "joint_3_Y": "188",
    "joint_3_visible": "0",
    "joint_4_X": "661",
    "joint_4_Y": "221",
    "joint_4_visible": "1",
    "joint_5_X": "656",
    "joint_5_Y": "231",
    "joint_5_visible": "1",
    "joint_6_X": "610",
    "joint_6_Y": "187",
    "joint_6_visible": "0",
    "joint_7_X": "647",
    "joint_7_Y": "176",
    "joint_7_visible": "1",
    "joint_8_X": "637",
    "joint_8_Y": "189",
    "joint_8_visible": "0",
    "joint_9_X": "695",
    "joint_9_Y": "108",
    "joint_9_visible": "0",
    "joint_10_X": "606",
    "joint_10_Y": "217",
    "joint_10_visible": "1",
    "joint_11_X": "553",
    "joint_11_Y": "161",
    "joint_11_visible": "1",
    "joint_12_X": "601",
    "joint_12_Y": "167",
    "joint_12_visible": "1",
    "joint_13_X": "692",
    "joint_13_Y": "185",
    "joint_13_visible": "1",
    "joint_14_X": "693",
    "joint_14_Y": "240",
    "joint_14_visible": "1",
    "joint_15_X": "688",
    "joint_15_Y": "313",
    "joint_15_visible": "1",
    "set": "TRAIN",
}


@pytest.fixture(scope="function")
def gt_joints():
    return [
        [620, 394],
        [616, 269],
        [573, 185],
        [647, 188],
        [661, 221],
        [656, 231],
        [610, 187],
        [647, 176],
        [637, 189],
        [695, 108],
        [606, 217],
        [553, 161],
        [601, 167],
        [692, 185],
        [693, 240],
        [688, 313],
    ]


@pytest.fixture(scope="function")
def head_size(gt_joints):
    return tf.norm(
        tf.constant(gt_joints[9], dtype=tf.float32)
        - tf.constant(gt_joints[8], dtype=tf.float32),
        ord=2,
    )


@pytest.fixture(scope="function")
def torso_size(gt_joints):
    return tf.norm(
        tf.constant(gt_joints[8], dtype=tf.float32)
        - tf.constant(gt_joints[6], dtype=tf.float32),
        ord=2,
    )


@pytest.fixture(scope="function")
def error_joints():
    return [
        [30],
        [40],
        [50],
        [60],
        [70],
        [80],
        [90],
        [100],
        [100],
        [100],
        [30],
        [35],
        [40],
        [70],
        [80],
        [90],
    ]


@pytest.fixture(scope="function")
def pred_joints(gt_joints, error_joints):
    return tf.constant(gt_joints) - tf.constant(error_joints)


@pytest.fixture(scope="function")
def SHAPE():
    return tf.constant([1280, 720], dtype=tf.dtypes.int32)


@pytest.fixture(scope="function")
def STDDEV():
    return tf.constant([5.0, 5.0], dtype=tf.dtypes.float32)


@pytest.fixture(scope="function")
def gt_heatmap(gt_joints, SHAPE, STDDEV):
    return tf.transpose(
        tf.map_fn(
            fn=lambda x: tf_bivariate_normal_pdf(
                tf.cast(x, tf.float32), stddev=STDDEV, shape=SHAPE
            ),
            elems=tf.cast(tf.constant(gt_joints), dtype=tf.dtypes.int32),
            dtype=tf.float32,
        ),
        perm=[1, 2, 0],
    )


@pytest.fixture(scope="function")
def pred_heatmap(pred_joints, SHAPE, STDDEV):
    return tf.transpose(
        tf.map_fn(
            fn=lambda x: tf_bivariate_normal_pdf(
                tf.cast(x, tf.float32), stddev=STDDEV, shape=SHAPE
            ),
            elems=tf.cast(tf.constant(pred_joints), dtype=tf.dtypes.int32),
            dtype=tf.float32,
        ),
        perm=[1, 2, 0],
    )


@pytest.fixture(scope="function")
def y_true_nosup(gt_heatmap):
    return tf.expand_dims(gt_heatmap, axis=0)


@pytest.fixture(scope="function")
def y_pred_nosup(pred_heatmap):
    return tf.expand_dims(pred_heatmap, axis=0)


@pytest.fixture(scope="function")
def y_true(gt_heatmap):
    return tf.expand_dims(tf.expand_dims(gt_heatmap, axis=0), axis=0)


@pytest.fixture(scope="function")
def y_pred(pred_heatmap):
    return tf.expand_dims(tf.expand_dims(pred_heatmap, axis=0), axis=0)


# endregion


# region OverallMeanDistance


def test_overall_mean_distance_metric_supervision(y_true, y_pred):
    # errors = tf.stack([error_joints, error_joints], axis=1)[:, :, 0]
    # estimated_error = tf.reduce_mean(
    #     tf.norm(tf.cast(errors, dtype=tf.float32), ord=2, axis=-1)
    # )
    estimated_error = tf.constant(94.13359)

    metric = OverallMeanDistance(intermediate_supervision=True)

    assert metric._internal_update(y_true, y_pred) is None
    assert metric.batches == 1.0
    assert metric.distance == estimated_error
    assert (
        metric.result() == estimated_error
    ), f"Wrong result from {metric.__class__.__name__}. EXPECTED: {estimated_error} RECEIVED: {metric.result()}"

    with pytest.raises(TypeError):
        metric.update_state()

    metric.reset_state()
    assert metric.result() == 0.0


def test_overall_mean_distance_metric_no_supervision(
    y_true_nosup,
    y_pred_nosup,
):
    # errors = tf.stack([error_joints, error_joints], axis=1)[:, :, 0]
    # estimated_error = tf.reduce_mean(
    #     tf.norm(tf.cast(errors, dtype=tf.float32), ord=2, axis=-1)
    # )
    estimated_error = tf.constant(94.13359)

    metric = OverallMeanDistance(intermediate_supervision=False)

    assert metric._internal_update(y_true_nosup, y_pred_nosup) is None
    assert (
        metric.result() == estimated_error
    ), f"Wrong result from {metric.__class__.__name__}. EXPECTED: {estimated_error} RECEIVED: {metric.result()}"

    with pytest.raises(TypeError):
        metric.update_state()

    metric.reset_state()
    assert metric.result() == 0.0


# endregion


# region RatioCorrectKeypoint


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (10, 0.0),
        (50, 0.1875),
        (70, 0.3125),
        (80, 0.375),
        (90, 0.4375),
        (100, 0.5625),
        (120, 0.6875),
        (130, 0.8125),
        (150, 1.0),
    ],
)
def test_ratio_correct_keypoint_metric_supervision(y_true, y_pred, threshold, expected):

    # norm = tf.sqrt(2.0) * tf.constant(error_joints, dtype=tf.float32)
    # comparison = tf.cast(norm <= tf.constant(threshold, tf.float32), dtype=tf.float32)
    # estimated_error = tf.reduce_sum(comparison) / 16.0
    estimated_error = tf.constant(expected)

    metric = RatioCorrectKeypoints(intermediate_supervision=True, threshold=threshold)

    assert metric._internal_update(y_true, y_pred) is None
    assert (
        metric.result() == estimated_error
    ), f"Wrong result from {metric.__class__.__name__}. EXPECTED: {estimated_error} RECEIVED: {metric.result()}"

    with pytest.raises(TypeError):
        metric.update_state()

    metric.reset_state()
    assert metric.result() == 0.0


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (10, 0.0),
        (50, 0.1875),
        (70, 0.3125),
        (80, 0.375),
        (90, 0.4375),
        (100, 0.5625),
        (120, 0.6875),
        (130, 0.8125),
        (150, 1.0),
    ],
)
def test_ratio_correct_keypoint_metric_no_supervision(
    y_true_nosup, y_pred_nosup, threshold, expected
):

    # norm = tf.sqrt(2.0) * tf.constant(error_joints, dtype=tf.float32)
    # comparison = tf.cast(norm <= tf.constant(threshold, tf.float32), dtype=tf.float32)
    # estimated_error = tf.reduce_sum(comparison) / 16.0
    estimated_error = tf.constant(expected)

    metric = RatioCorrectKeypoints(intermediate_supervision=True, threshold=threshold)

    assert metric._internal_update(y_true_nosup, y_pred_nosup) is None
    assert (
        metric.result() == estimated_error
    ), f"Wrong result from {metric.__class__.__name__}. EXPECTED: {estimated_error} RECEIVED: {metric.result()}"

    with pytest.raises(TypeError):
        metric.update_state()

    metric.reset_state()
    assert metric.result() == 0.0


# endregion


# region PCKh


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0.3, 0.0),
        (0.5, 0.1875),
        (0.75, 0.375),
        (0.9, 0.4375),
        (1.0, 0.5625),
        (1.2, 0.6875),
        (1.3, 0.8125),
        (1.5, 1.0),
        (2.0, 1.0),
    ],
)
def test_pckh_supervision(y_true, y_pred, ratio, expected):

    # Error estimation
    # estimated_error = 0
    # ground_truth_joints = tf_dynamic_matrix_argmax(y_true)
    # predicted_joints = tf_dynamic_matrix_argmax(y_pred)
    # error = tf.cast(ground_truth_joints - predicted_joints, dtype=tf.float32)
    # distance = tf.norm(error, ord=2, axis=-1)
    # condition = distance < (head_size * ratio)
    # total_correct = tf.reduce_sum(tf.cast(condition, dtype=tf.float32))
    # total_keypoints = tf.reduce_prod(tf.cast(distance.shape, dtype=tf.float32))
    # estimated_error = total_correct / total_keypoints
    estimated_error = tf.constant(expected)

    # Metric computation
    metric = PercentageOfCorrectKeypoints(reference=(8, 9), ratio=ratio)
    assert metric._internal_update(y_true, y_pred) is None
    assert (
        metric.result() <= 1.0
    ), f"{metric.__class__.__name__} computes a value over 100%"
    assert (
        metric.result() == estimated_error
    ), f"Wrong result from {metric.__class__.__name__}. EXPECTED: {estimated_error} RECEIVED: {metric.result()}"

    with pytest.raises(TypeError):
        metric.update_state()

    metric.reset_state()
    assert metric.result() == 0.0


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0.3, 0.0),
        (0.5, 0.1875),
        (0.75, 0.375),
        (0.9, 0.4375),
        (1.0, 0.5625),
        (1.2, 0.6875),
        (1.3, 0.8125),
        (1.5, 1.0),
        (2.0, 1.0),
    ],
)
def test_pckh_no_supervision(y_true_nosup, y_pred_nosup, ratio, expected):

    # Error estimation
    # estimated_error = 0
    # ground_truth_joints = tf_dynamic_matrix_argmax(y_true_nosup)
    # predicted_joints = tf_dynamic_matrix_argmax(y_pred_nosup)
    # error = tf.cast(ground_truth_joints - predicted_joints, dtype=tf.float32)
    # distance = tf.norm(error, ord=2, axis=-1)
    # condition = distance < (head_size * ratio)
    # total_correct = tf.reduce_sum(tf.cast(condition, dtype=tf.float32))
    # total_keypoints = tf.reduce_prod(tf.cast(distance.shape, dtype=tf.float32))
    # estimated_error = total_correct / total_keypoints
    estimated_error = tf.constant(expected)

    # Metric computation
    metric = PercentageOfCorrectKeypoints(
        reference=(8, 9), ratio=ratio, intermediate_supervision=False
    )
    assert metric._internal_update(y_true_nosup, y_pred_nosup) is None
    assert (
        metric.result() <= 1.0
    ), f"{metric.__class__.__name__} computes a value over 100%"
    assert (
        metric.result() == estimated_error
    ), f"Wrong result from {metric.__class__.__name__}. EXPECTED: {estimated_error} RECEIVED: {metric.result()}"

    with pytest.raises(TypeError):
        metric.update_state()

    metric.reset_state()
    assert metric.result() == 0.0


# endregion


# region PCK


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (1.5, 0.0),
        (2.0, 0.1875),
        (3.0, 0.375),
        (4.0, 0.5625),
        (5.0, 0.8125),
        (6.0, 1.0),
    ],
)
def test_pck_supervision(y_true, y_pred, ratio, expected):

    # Error estimation
    # estimated_error = 0
    # ground_truth_joints = tf_dynamic_matrix_argmax(y_true)
    # predicted_joints = tf_dynamic_matrix_argmax(y_pred)
    # error = tf.cast(ground_truth_joints - predicted_joints, dtype=tf.float32)
    # distance = tf.norm(error, ord=2, axis=-1)
    # condition = distance < (torso_size * ratio)
    # total_correct = tf.reduce_sum(tf.cast(condition, dtype=tf.float32))
    # total_keypoints = tf.reduce_prod(tf.cast(distance.shape, dtype=tf.float32))
    # estimated_error = total_correct / total_keypoints
    estimated_error = tf.constant(expected)

    # Metric computation
    metric = PercentageOfCorrectKeypoints(reference=(6, 8), ratio=ratio)
    assert metric._internal_update(y_true, y_pred) is None
    assert (
        metric.result() <= 1.0
    ), f"{metric.__class__.__name__} computes a value over 100%"
    assert (
        metric.result() == estimated_error
    ), f"Wrong result from {metric.__class__.__name__}. EXPECTED: {estimated_error} RECEIVED: {metric.result()}"

    with pytest.raises(TypeError):
        metric.update_state()

    metric.reset_state()
    assert metric.result() == 0.0


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (1.5, 0.0),
        (2.0, 0.1875),
        (3.0, 0.375),
        (4.0, 0.5625),
        (5.0, 0.8125),
        (6.0, 1.0),
    ],
)
def test_pck_no_supervision(y_true_nosup, y_pred_nosup, ratio, expected):

    # Error estimation
    # estimated_error = 0
    # ground_truth_joints = tf_dynamic_matrix_argmax(y_true_nosup)
    # predicted_joints = tf_dynamic_matrix_argmax(y_pred_nosup)
    # error = tf.cast(ground_truth_joints - predicted_joints, dtype=tf.float32)
    # distance = tf.norm(error, ord=2, axis=-1)
    # condition = distance < (torso_size * ratio)
    # total_correct = tf.reduce_sum(tf.cast(condition, dtype=tf.float32))
    # total_keypoints = tf.reduce_prod(tf.cast(distance.shape, dtype=tf.float32))
    # estimated_error = total_correct / total_keypoints
    estimated_error = tf.constant(expected)

    # Metric computation
    metric = PercentageOfCorrectKeypoints(
        reference=(6, 8), ratio=ratio, intermediate_supervision=False
    )
    assert metric._internal_update(y_true_nosup, y_pred_nosup) is None
    assert (
        metric.result() <= 1.0
    ), f"{metric.__class__.__name__} computes a value over 100%"
    assert (
        metric.result() == estimated_error
    ), f"Wrong result from {metric.__class__.__name__}. EXPECTED: {estimated_error} RECEIVED: {metric.result()}"

    with pytest.raises(TypeError):
        metric.update_state()

    metric.reset_state()
    assert metric.result() == 0.0


# endregion
