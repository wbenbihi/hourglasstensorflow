import pytest
import tensorflow as tf
from loguru import logger

from hourglass_tensorflow.losses import SigmoidCrossEntropyLoss
from hourglass_tensorflow.utils.tf import tf_bivariate_normal_pdf

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


def test_sigmoid_cross_entropy_loss(y_true, y_pred):
    epsilon = tf.constant(0.00001)
    estimated_error = tf.constant(0.6931466)

    loss = SigmoidCrossEntropyLoss()
    loss.call(y_true, y_pred)

    assert (
        tf.abs(loss(y_true, y_pred) - estimated_error) <= epsilon
    ), f"{loss.__class__.__name__} estimated error is incorrect. EXPECTED {estimated_error}. RECEIVED: {loss(y_true, y_pred)}"
