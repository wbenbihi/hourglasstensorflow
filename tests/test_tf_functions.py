import pytest
import tensorflow as tf
from loguru import logger

from hourglass_tensorflow.utils.tf import tf_stack
from hourglass_tensorflow.utils.tf import tf_load_image
from hourglass_tensorflow.utils.tf import tf_expand_bbox
from hourglass_tensorflow.utils.tf import tf_compute_bbox
from hourglass_tensorflow.utils.tf import tf_matrix_argmax
from hourglass_tensorflow.utils.tf import tf_reshape_slice
from hourglass_tensorflow.utils.tf import tf_resize_tensor
from hourglass_tensorflow.utils.tf import tf_batch_matrix_argmax
from hourglass_tensorflow.utils.tf import tf_bivariate_normal_pdf
from hourglass_tensorflow.utils.tf import tf_dynamic_matrix_argmax
from hourglass_tensorflow.utils.tf import tf_generate_padding_tensor
from hourglass_tensorflow.utils.tf import tf_compute_padding_from_bbox

IMAGE_FILE = "data/test.image.jpg"

tf.config.run_functions_eagerly(True)


@pytest.fixture(scope="function")
def img():
    image = tf_load_image(filename=IMAGE_FILE, channels=3)
    return image


def test_tf_load_image():
    image = tf_load_image(filename=IMAGE_FILE, channels=3)
    assert image.dtype == tf.dtypes.uint8, "Image has not the expected dtype"
    assert image.shape == [1080, 1920, 3], "Image has not the expected shape"


def test_tf_stack(img):
    source_rank = tf.rank(img)
    stacked_tensor = tf_stack(img, stacks=3)
    output_rank = tf.rank(stacked_tensor)
    assert source_rank + 1 == output_rank, "Tensor rank is wrong"
    assert tf.reduce_all(
        tf.equal(stacked_tensor.shape[1:], img.shape)
    ), "Tensor shape has been altered"
    assert all(
        [tf.math.reduce_all(tf.equal(stacked_tensor[i], img)) for i in range(3)]
    ), "Tensor data has been altered"


def test_tf_reshape_slice():
    data = [([0, 1, 2, 3, 4, 5, 6, 7], 2), ([0, 1, 2, 3, 4, 5, 6, 7, 8], 3)]
    for slice, shape in data:
        reshaped_slice = tf_reshape_slice(tf.constant(slice), shape=shape)
        assert len(reshaped_slice) == len(slice) / shape, "Slice Length does not match"
        assert tf.reduce_all(
            tf.equal(reshaped_slice[0], slice[:shape])
        ), "Slice data does not match"

    # Check if raising error
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf_reshape_slice(tf.constant([0, 1, 2, 3, 4, 5, 6, 7]), shape=3)


def test_resize_tensor(img):
    resized_img = tf_resize_tensor(img, size=100)
    assert tf.rank(resized_img) == tf.rank(img), "Rank has been altered"
    assert tf.reduce_all(
        tf.equal(
            tf.shape(resized_img),
            [
                100,
                100,
                3,
            ],
        )
    ), "The new shape is not the one expected"


def test_tf_compute_padding_from_bbox():
    a_x = 20
    a_y = 30
    b_x = 40
    b_y = 60

    # Valid Bounding box for X Padding
    bbox = tf.constant([[a_x, a_y], [b_x, b_y]])
    padding = tf_compute_padding_from_bbox(bbox)
    assert padding.dtype == tf.dtypes.int32, "Bad return type"
    assert tf.math.reduce_all(
        tf.equal(padding, [5, 0])
    ), "The Padding is not as expected [5, 0]"

    # Valid Bounding box for Y Padding
    bbox = tf.constant([[a_y, a_x], [b_y, b_x]])
    padding = tf_compute_padding_from_bbox(bbox)
    assert tf.math.reduce_all(
        tf.equal(padding, [0, 5])
    ), "The Padding is not as expected [0, 5]"

    # Invalid Bounding box for Y Padding
    bbox = tf.constant([[b_x, b_y], [a_x, a_y]])
    padding = tf_compute_padding_from_bbox(bbox)
    assert tf.math.reduce_all(
        tf.equal(padding, [0, 5])
    ), "The Padding is not as expected [0, 5]"


def test_tf_generate_padding_tensor():
    a_x = 20
    a_y = 30
    b_x = 40
    b_y = 60
    # Valid Bounding box for X Padding
    bbox = tf.constant([[a_x, a_y], [b_x, b_y]])
    padding = tf_compute_padding_from_bbox(bbox)
    assert tf.reduce_all(
        tf.equal(
            tf_generate_padding_tensor(padding),
            [[padding[1], padding[1]], [padding[0], padding[0]], [0, 0]],
        )
    )


def test_tf_compute_bbox():
    coordinates = [[10, 20], [30, 6], [40, 100], [25, 80]]
    bbox = tf_compute_bbox(tf.constant(coordinates))

    assert tf.rank(bbox) == 2, "Bad rank for bounding box tensor"
    assert bbox.dtype == tf.dtypes.int32, "Bad type for bounding box tensor"
    assert tf.reduce_all(
        tf.equal(tf.shape(bbox), (2, 2))
    ), "Bad shape for bounding box tensor"
    assert tf.reduce_all(
        tf.equal(bbox, tf.constant([[10, 6], [40, 100]], dtype=tf.dtypes.int32))
    ), "Bad values for bounding box tensor"


TEST_EXPAND_BBOX = [
    (1, [[100, 100], [200, 200]]),
    (1.5, [[50, 50], [250, 250]]),
    (1.75, [[25, 25], [275, 275]]),
    (2, [[0, 0], [300, 300]]),
    (3, [[0, 0], [400, 400]]),
    (8, [[0, 0], [900, 900]]),
    (11, [[0, 0], [999, 999]]),
]


@pytest.mark.parametrize("factor, expected", TEST_EXPAND_BBOX)
def test_tf_expand_bbox(factor, expected):
    expanded_bbox = tf_expand_bbox(
        [[100, 100], [200, 200]], [1000, 1000, 3], bbox_factor=factor
    )
    assert tf.reduce_all(
        tf.equal(expanded_bbox, expected)
    ), "Expanded BBox is not as expected"


@pytest.mark.parametrize(
    "params",
    [
        {
            "mean": [120, 140],
            "stddev": [5.0, 5.0],
            "shape": [500, 500],
        },
        {
            "mean": [450, 340],
            "stddev": [25.0, 25.0],
            "shape": [600, 500],
        },
        {
            "mean": [0, 0],
            "stddev": [5.0, 25.0],
            "shape": [100, 300],
        },
    ],
)
def test_tf_bivariate_normal_pdf(params):

    heatmap = tf_bivariate_normal_pdf(**params)
    hm_1d = tf.reshape(heatmap, (-1))
    max_index = tf.argmax(hm_1d)
    argmax_y = max_index // params["shape"][0]
    argmax_x = max_index % params["shape"][0]

    assert tf.rank(heatmap) == 2, "The generated heatmap is not a matrix"
    assert tf.reduce_all(
        tf.equal(tf.shape(heatmap), params["shape"][::-1])
    ), "The generated heatmap has not the expected shape"
    assert tf.shape(hm_1d) == tf.reduce_prod(
        tf.shape(heatmap)
    ), "The generated heatmap has not the expected shape"
    assert tf.reduce_all(
        tf.equal([argmax_x, argmax_y], params["mean"])
    ), "The Argmax is not equal to the Mean"


@pytest.mark.parametrize(
    "params",
    [
        {
            "mean": [120, 140],
            "stddev": [5.0, 5.0],
            "shape": [500, 500],
        },
        {
            "mean": [450, 340],
            "stddev": [25.0, 25.0],
            "shape": [600, 500],
        },
        {
            "mean": [0, 0],
            "stddev": [5.0, 25.0],
            "shape": [100, 300],
        },
    ],
)
def test_tf_matrix_argmax(params):
    heatmap = tf.expand_dims(tf_bivariate_normal_pdf(**params), axis=-1)
    argmax = tf_matrix_argmax(heatmap)
    assert tf.reduce_all(
        tf.equal(argmax, params["mean"][::-1])
    ), "The Argmax is not equal to the Mean"


@pytest.mark.parametrize(
    "params",
    [
        {
            "mean": [120, 140],
            "stddev": [5.0, 5.0],
            "shape": [500, 500],
        },
        {
            "mean": [450, 340],
            "stddev": [25.0, 25.0],
            "shape": [600, 500],
        },
        {
            "mean": [0, 0],
            "stddev": [5.0, 25.0],
            "shape": [100, 300],
        },
    ],
)
def test_tf_matrix_argmax(params):
    hm = tf_bivariate_normal_pdf(**params)
    # 1 Channel
    # requires a 3D tensor HWC
    heatmap = tf.expand_dims(hm, -1)
    argmax = tf_matrix_argmax(heatmap)
    assert tf.reduce_all(
        tf.equal(argmax, params["mean"][::-1])
    ), "SingleChannel - The Argmax is not equal to the Mean"
    # Multiple Channels
    heatmap = tf.stack([hm, hm, hm], axis=-1)
    argmax = tf_matrix_argmax(heatmap)
    assert tf.reduce_all(
        [tf.equal(argmax[i], params["mean"][::-1]) for i in range(3)]
    ), "MultipleChannel - The Argmax is not equal to the Mean"


def test_tf_batch_matrix_argmax():
    COMMON_SIZE = [500, 700]
    STACK_PARAMS = [
        {
            "mean": [120, 140],
            "stddev": [5.0, 5.0],
            "shape": COMMON_SIZE,
        },
        {
            "mean": [330, 40],
            "stddev": [25.0, 5.0],
            "shape": COMMON_SIZE,
        },
        {
            "mean": [0, 300],
            "stddev": [15.0, 30.0],
            "shape": COMMON_SIZE,
        },
    ]

    # N = 3, H = 700 W = 500 C = 1
    stacks = tf.stack(  # Stack to generate a batch NHWC
        [
            # Expand Dim to generate channels HWC
            tf.expand_dims(tf_bivariate_normal_pdf(**params), axis=-1)
            for params in STACK_PARAMS
        ],
        axis=0,
    )

    argmax = tf_batch_matrix_argmax(stacks)
    assert tf.rank(argmax) == 3, "The Argmax tensor is not of rank 3"
    assert tf.reduce_all(
        tf.equal(tf.shape(argmax), [3, 1, 2])
    ), "Argmax tensor has the wrong shape"
    assert all(
        [
            tf.reduce_all(tf.equal(argmax[nid, cid], STACK_PARAMS[nid]["mean"][::-1]))
            for nid in range(len(stacks))
            for cid in range(tf.shape(stacks[nid])[-1])
        ]
    ), "Argmax values are not the ones specified in Means"

    # N = 3, H = 700 W = 500 C = 4
    stacks = tf.stack(  # Stack to generate a batch NHWC
        [
            # Expand Dim to generate channels HWC
            tf.stack([tf_bivariate_normal_pdf(**params)] * 4, axis=-1)
            for params in STACK_PARAMS
        ],
        axis=0,
    )

    argmax = tf_batch_matrix_argmax(stacks)
    assert tf.rank(argmax) == 3, "The Argmax tensor is not of rank 3"
    assert tf.reduce_all(
        tf.equal(tf.shape(argmax), [3, 4, 2])
    ), "Argmax tensor has the wrong shape"
    assert all(
        [
            tf.reduce_all(tf.equal(argmax[nid, cid], STACK_PARAMS[nid]["mean"][::-1]))
            for nid in range(len(stacks))
            for cid in range(tf.shape(stacks[nid])[-1])
        ]
    ), "Argmax values are not the ones specified in Means"


@pytest.mark.parametrize(
    "shape, params, expected_rank",
    [
        ([64, 64], dict(intermediate_supervision=False, keepdims=True), 2),
        ([64, 64], dict(intermediate_supervision=False, keepdims=False), 1),
        ([64, 64], dict(intermediate_supervision=True, keepdims=True), 2),
        ([64, 64], dict(intermediate_supervision=True, keepdims=False), 1),
        ([64, 64, 16], dict(intermediate_supervision=True, keepdims=True), 3),
        ([64, 64, 16], dict(intermediate_supervision=True, keepdims=False), 2),
        ([64, 64, 16], dict(intermediate_supervision=False, keepdims=True), 3),
        ([64, 64, 16], dict(intermediate_supervision=False, keepdims=False), 2),
        ([4, 64, 64, 16], dict(intermediate_supervision=True, keepdims=True), 3),
        ([4, 64, 64, 16], dict(intermediate_supervision=True, keepdims=False), 2),
        ([4, 64, 64, 16], dict(intermediate_supervision=False, keepdims=True), 3),
        ([4, 64, 64, 16], dict(intermediate_supervision=False, keepdims=False), 3),
        ([10, 4, 64, 64, 16], dict(intermediate_supervision=True, keepdims=True), 3),
        ([10, 4, 64, 64, 16], dict(intermediate_supervision=True, keepdims=False), 3),
        ([10, 4, 64, 64, 16], dict(intermediate_supervision=False, keepdims=True), 3),
        ([10, 4, 64, 64, 16], dict(intermediate_supervision=False, keepdims=False), 3),
    ],
)
def test_tf_dynamic_matrix_argmax(shape, params, expected_rank):
    # For [64, 64] TT (1, 2)
    # For [64, 64, 16] TT (1, 16, 2)
    # For [4, 64, 64, 16] TT (1, 16, 2)
    # For [10, 4, 64, 64, 16] TT (10, 16, 2)
    # For [64, 64] FT (2,)
    # For [64, 64, 16] FT (16, 2)
    # For [4, 64, 64, 16] FT (16, 2)
    # For [10, 4, 64, 64, 16] FT (10, 16, 2)
    # For [64, 64] TF (1, 2)
    # For [64, 64, 16] TF (1, 16, 2)
    # For [4, 64, 64, 16] TF (4, 16, 2)
    # For [10, 4, 64, 64, 16] TF (10, 16, 2)
    # For [64, 64] FF (2,)
    # For [64, 64, 16] FF (16, 2)
    # For [4, 64, 64, 16] FF (4, 16, 2)
    # For [10, 4, 64, 64, 16] FF (10, 16, 2)

    # For [4, 64, 64, 16] TT (1, 16, 2)
    # For [4, 64, 64, 16] FT (16, 2)
    # For [4, 64, 64, 16] TF (4, 16, 2)
    # For [4, 64, 64, 16] FF (4, 16, 2)

    tensor = tf.random.uniform(shape)
    argmax = tf_dynamic_matrix_argmax(tensor, **params)
    assert (
        tf.rank(argmax) == expected_rank
    ), f"Expected rank is {expected_rank}, received {tf.rank(argmax)}"
    if len(shape) == 4 and len(tf.shape(argmax)) == 3:
        if params["intermediate_supervision"]:
            assert tf.shape(argmax)[0] == 1
        else:
            assert tf.shape(argmax)[0] == shape[0]


@pytest.mark.parametrize(
    "intermediate_supervision, keepdims",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_tf_dynamic_matrix_argmax_with_unsupported_rank(
    intermediate_supervision, keepdims
):
    tensor = tf.random.uniform(shape=[12, 10, 4, 64, 64, 16])
    with pytest.raises(ValueError):
        _ = tf_dynamic_matrix_argmax(
            tensor, intermediate_supervision=intermediate_supervision, keepdims=keepdims
        )
