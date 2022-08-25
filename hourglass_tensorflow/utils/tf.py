import math as m

import tensorflow as tf


@tf.function
def tf_load_image(filename: tf.Tensor, channels: int = 3, **kwargs) -> tf.Tensor:
    """Load image from filename

    Args:
        filename (tf.Tensor): string tensor containing the image path to read
        channels (int, optional): Number of image channels. Defaults to 3.

    Returns:
        tf.Tensor: Image Tensor of shape [HEIGHT, WIDTH, channels]
    """
    return tf.io.decode_image(tf.io.read_file(filename), channels=channels)


@tf.function
def tf_stack(tensor: tf.Tensor, stacks: int = 1) -> tf.Tensor:
    """Stack copies of tensor on each other

    Notes:
        This function will augment the dimensionality of the tensor.
        For a 2D Tensor of shape [HEIGHT, WIDTH] the output will
        be a tensor of 3D tensor of shape [stacks, HEIGHT, WIDTH]

    Args:
        tensor (tf.Tensor): Tensor to stacks
        stacks (int, optional): Number of stacks to use. Defaults to 1.

    Returns:
        tf.Tensor: Stacked tensor of DIM = DIM(tensor) + 1
    """
    return tf.stack([tensor] * stacks, axis=0)


@tf.function
def tf_reshape_slice(tensor: tf.Tensor, shape=3, **kwargs) -> tf.Tensor:
    """Reshape 1D to 2D tensor

    Args:
        tensor (tf.Tensor): tensor to reshape
        shape (int, optional): Shape of the 2nd dimension. Defaults to 3.

    Returns:
        tf.Tensor: 2D reshaped tensor
    """
    return tf.reshape(tensor, shape=(-1, shape))


@tf.function
def tf_resize_tensor(tensor: tf.Tensor, size: int) -> tf.Tensor:
    """Apply tensor square resizing with nearest neighbor image interpolation

    Args:
        tensor (tf.Tensor): tensor to reshape
        size (int): output size

    Returns:
        tf.Tensor: resized tensor
    """
    return tf.image.resize(tensor, size=[size, size], method="nearest")


@tf.function
def tf_compute_padding_from_bbox(bbox: tf.Tensor) -> tf.Tensor:
    """Given a bounding box tensor compute the padding needed to make a square bbox

    Notes:
        `bbox` is a 2x2 Tensor [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]

    Args:
        bbox (tf.Tensor): bounding box tensor

    Returns:
        tf.Tensor: tensor of shape (2,) containing [width paddding, height padding]
    """
    size = bbox[1] - bbox[0]
    width, height = size[0], size[1]
    # Compute Padding
    height_padding = tf.math.maximum(tf.constant(0, dtype=tf.int32), width - height)
    width_padding = tf.math.maximum(tf.constant(0, dtype=tf.int32), height - width)
    return tf.reshape([width_padding // 2, height_padding // 2], shape=(2,))


@tf.function
def tf_generate_padding_tensor(padding: tf.Tensor) -> tf.Tensor:
    """Given a Width X Height padding compute a tensor to apply `tf.pad` function

    Notes:
        `padding` argument must be consistent with `tf_compute_padding_from_bbox` output

    Args:
        padding (tf.Tensor): padding tensor

    Returns:
        tf.Tensor: tensor ready to be used with `tf.pad`
    """
    width_padding, height_padding = padding[0], padding[1]
    padding_tensor = [
        [height_padding, height_padding],
        [width_padding, width_padding],
        [0, 0],
    ]
    return padding_tensor


@tf.function
def tf_compute_bbox(coordinates: tf.Tensor, **kwargs) -> tf.Tensor:
    """From a 2D coordinates tensor compute the bounding box

    Args:
        coordinates (tf.Tensor): Joint coordinates 2D tensor

    Returns:
        tf.Tensor: Bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
    """
    Xs = coordinates[:, 0]
    Ys = coordinates[:, 1]
    maxx, minx = tf.reduce_max(Xs), tf.reduce_min(Xs)
    maxy, miny = tf.reduce_max(Ys), tf.reduce_min(Ys)
    return tf_reshape_slice([minx, miny, maxx, maxy], shape=2, **kwargs)


@tf.function
def tf_expand_bbox(
    bbox: tf.Tensor, image_shape: tf.Tensor, bbox_factor: float = 1.0, **kwargs
) -> tf.Tensor:
    """Expand a bounding box area by a given factor

    Args:
        bbox (tf.Tensor): Bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
        image_shape (tf.Tensor): Image shape Tensor as [Height, Width, Channels]
        bbox_factor (float, optional): Expansion factor. Defaults to 1.0.

    Returns:
        tf.Tensor: Expanded bounding box 2x2 tensor as [[TopLeftX, TopLeftY], [BottomRightX, BottomRightY]]
    """
    # Unpack BBox
    top_left = bbox[0]
    top_left_x, top_left_y = tf.cast(top_left[0], dtype=tf.float64), tf.cast(
        top_left[1], dtype=tf.float64
    )
    bottom_right = bbox[1]
    bottom_right_x, bottom_right_y = tf.cast(
        bottom_right[0], dtype=tf.float64
    ), tf.cast(bottom_right[1], dtype=tf.float64)
    # Compute Bbox H/W
    height, width = bottom_right_y - top_left_y, bottom_right_x - top_left_x
    # Increase BBox Size
    new_tl_x = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), top_left_x - width * (bbox_factor - 1.0)
    )
    new_tl_y = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), top_left_y - height * (bbox_factor - 1.0)
    )
    new_br_x = tf.math.minimum(
        tf.cast(image_shape[1] - 1, dtype=tf.float64),
        bottom_right_x + width * (bbox_factor - 1.0),
    )
    new_br_y = tf.math.minimum(
        tf.cast(image_shape[0] - 1, dtype=tf.float64),
        bottom_right_y + height * (bbox_factor - 1.0),
    )
    return tf.cast(
        tf_reshape_slice([new_tl_x, new_tl_y, new_br_x, new_br_y], shape=2, **kwargs),
        dtype=tf.int32,
    )


@tf.function
def tf_bivariate_normal_pdf(
    mean: tf.Tensor, stddev: tf.Tensor, shape: tf.Tensor, precision=tf.dtypes.float32
) -> tf.Tensor:
    """Produce a heatmap given a Bivariate normal propability density function

    Args:
        mean (tf.Tensor): Mean Tensor(tf.dtype.float*) as [m_x, m_y]
        stddev (tf.Tensor): Standard Deviation Tensor(tf.dtype.float*) as [stdd_x, stdd_y]
        shape (tf.Tensor): Heatmap shape Tensor(tf.dtype.float*) as [width, height]
        precision (tf.dtypes, optional): Precision of the output tensor. Defaults to tf.dtypes.float32.

    Returns:
        tf.Tensor: Heatmap
    """
    # Compute Grid
    X, Y = tf.meshgrid(
        tf.range(
            start=0.0, limit=tf.cast(shape[0], precision), delta=1.0, dtype=precision
        ),
        tf.range(
            start=0.0, limit=tf.cast(shape[1], precision), delta=1.0, dtype=precision
        ),
    )
    R = tf.sqrt(((X - mean[0]) ** 2 / (stddev[0])) + ((Y - mean[1]) ** 2 / (stddev[1])))
    factor = tf.cast(1.0 / (2.0 * m.pi * tf.reduce_prod(stddev)), precision)
    Z = factor * tf.exp(-0.5 * R)
    return Z


@tf.function
def tf_matrix_argmax(tensor: tf.Tensor) -> tf.Tensor:
    """Apply a 2D argmax to a tensor

    Args:
        tensor (tf.Tensor): 3D Tensor with data format HWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension Cx2
    """
    flat_tensor = tf.reshape(tensor, (-1, tf.shape(tensor)[-1]))
    argmax = tf.cast(tf.argmax(flat_tensor, axis=0), tf.int32)
    argmax_x = argmax // tf.shape(tensor)[1]
    argmax_y = argmax % tf.shape(tensor)[1]
    # stack and return 2D coordinates
    return tf.transpose(tf.stack((argmax_x, argmax_y), axis=0), [1, 0])


@tf.function
def tf_batch_matrix_argmax(tensor: tf.Tensor) -> tf.Tensor:
    """Apply 2D argmax along a batch

    Args:
        tensor (tf.Tensor): 4D Tensor with data format NHWC

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension NxCx2
    """
    return tf.map_fn(
        fn=tf_matrix_argmax, elems=tensor, fn_output_signature=tf.dtypes.int32
    )


@tf.function
def tf_dynamic_matrix_argmax(
    tensor: tf.Tensor, keepdims: bool = True, intermediate_supervision: bool = True
) -> tf.Tensor:
    """Apply 2D argmax for 5D, 4D, 3D, 2D tensors

    This function consider the following dimension cases:
        * `2D tensor` A single joint heatmap.
            Function returns a tensor of `dim=2`.
        * `3D tensor` A multiple joint heatmap.
            Function returns a tensor of `dim=2`.
        * `4D tensor` A multiple joints heatmap with intermediate supervision.
            Function returns a tensor of `dim=2`.
            2D Argmax will only be applied on last stage.
        * `5D tensor` A batch of multiple joints heatmap with intermediate supervision.
            Function returns a tensor of `dim=3`.
            2D Argmax will only be applied on last stage.

    Notes:
        For a batch of heatmap with no intermediate supervision, you need to apply
        a dimension expansion before using this function.
        >>> batch_tensor_no_supervision.shape
        [4, 64, 64, 16]
        >>> tf_dynamic_matrix_argmax(batch_tensor_no_supervision).shape
        [16, 2] # Considered as a single heatmap with intermediate supervision

        >>> expanded_batch = tf.expand_dims(batch_tensor_no_supervision, 1)
        >>> expanded_batch.shape
        [4, 1, 64, 64, 16]
        >>> tf_dynamic_matrix_argmax(batch_tensor_no_supervision).shape
        [4, 16, 2] # Considered as a batch of 4 image


    Args:
        tensor (tf.Tensor): Tensor to apply argmax
        keepdims (bool, optional): Force return tensor to be 3D.
            Defaults to True.
        intermediate_supervision (bool, optional): Modify function behavior if tensor rank is 4.
            Defaults to True.

    Returns:
        tf.Tensor: tf.dtypes.int32 Tensor of dimension NxCx2

    Raises:
        ValueError: If the input `tensor` rank not in [2 - 6]
    """
    if len(tf.shape(tensor)) == 2:
        # Single Joint
        argmax = tf_matrix_argmax(tf.expand_dims(tensor, -1))
        return argmax if keepdims else argmax[0, :]
    elif len(tf.shape(tensor)) == 3:
        # Multiple Joint Heatmaps
        argmax = tf_matrix_argmax(tensor)
        return tf.expand_dims(argmax, 0) if keepdims else argmax
    elif len(tf.shape(tensor)) == 4 and intermediate_supervision:
        # Multiple Joint Heatmaps with Intermediate supervision
        argmax = tf_matrix_argmax(tensor[-1, :, :, :])
        return tf.expand_dims(argmax, 0) if keepdims else argmax
    elif len(tf.shape(tensor)) == 4 and not intermediate_supervision:
        # Batch of multiple Joint Heatmaps without Intermediate supervision
        argmax = tf_batch_matrix_argmax(tensor)
        return argmax
    elif len(tf.shape(tensor)) == 5:
        # Batch of multiple Joint Heatmaps with Intermediate supervision
        argmax = tf_batch_matrix_argmax(tensor[:, -1, :, :, :])
        return argmax
    else:
        raise ValueError(
            f"No argmax operation available for {len(tf.shape(tensor))}D tensor"
        )
