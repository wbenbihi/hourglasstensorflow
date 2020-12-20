import math as m
import tensorflow as tf
import tensorflow_addons as tfa

# Image Reader
@tf.function
def tf_read_image(filename):
    image = tf.io.decode_jpeg(tf.io.read_file(filename), channels=3)
    return image


# Bivariate(2D) Gaussian
@tf.function
def tf_bivariate_normal_pdf(height, width, mx, my, sx, sy):
    X = tf.range(start=0.0, limit=tf.cast(width, tf.float32), delta=1.0)
    Y = tf.range(start=0.0, limit=tf.cast(height, tf.float32), delta=1.0)
    X, Y = tf.meshgrid(X, Y)
    R = tf.sqrt(((X - mx) ** 2 / (sx ** 2)) + ((Y - my) ** 2) / (sy ** 2))
    Z = (1.0 / (2 * tf.constant(m.pi)) * tf.sqrt(sx * sy)) * tf.exp(-0.5 * R ** 2)
    return Z


# Bivariate(2D) Gaussian
@tf.function
def tf_bivariate_normal_pdf_parallel(mx, my, height, width, sx, sy):
    X = tf.range(
        start=0.0, limit=tf.cast(width, tf.float64), delta=1.0, dtype=tf.float64
    )
    Y = tf.range(
        start=0.0, limit=tf.cast(height, tf.float64), delta=1.0, dtype=tf.float64
    )
    X, Y = tf.meshgrid(X, Y)
    R = tf.cast(
        tf.sqrt(((X - mx) ** 2 / (sx ** 2)) + ((Y - my) ** 2) / (sy ** 2)),
        dtype=tf.float64,
    )
    Z = (
        1.0
        / (
            tf.constant(2.0 * m.pi, dtype=tf.float64)
            * tf.cast(tf.sqrt(sx * sy), dtype=tf.float64)
        )
    ) * tf.exp(-0.5 * R ** 2)
    return Z


# Single Heatmap generation
@tf.function
def tf_generate_heatmap(coord, image):
    shape = tf.shape(image, out_type=tf.dtypes.int32)
    height = shape[0]
    width = shape[1]
    if tf.reduce_all(tf.math.is_finite(coord)):
        x = coord[0]
        y = coord[1]
        return tf_bivariate_normal_pdf(height, width, x, y, sx=10.0, sy=10.0)
    else:
        return tf.zeros([height, width])


# Heatmaps generation iterator
@tf.function
def tf_heatmaps(coords, image):
    return tf.transpose(
        tf.map_fn(lambda x: tf_generate_heatmap(x, image), tf.cast(coords, tf.float32)),
        [1, 2, 0],
    )


@tf.function
def tf_parse_dataset(filename, coords):
    images = tf_read_image(filename)
    heatmaps = tf_heatmaps(coords, images)
    return images, heatmaps


@tf.function
def tf_autopad(tensor):
    shape = tf.shape(tensor, out_type=tf.dtypes.int32)
    pad_x = tf.clip_by_value(shape[1] - shape[0], 0, 10000, name=None)
    pad_y = tf.clip_by_value(shape[0] - shape[1], 0, 10000, name=None)
    padded_tensor = tf.pad(
        tensor,
        [
            [pad_x // 2, pad_x // 2],
            [pad_y // 2, pad_y // 2],
            [0, 0],
        ],
    )
    return padded_tensor


@tf.function
def tf_resize(tensor, size):
    return tf.image.resize(tensor, [size, size])


@tf.function
def tf_preprocess(images, heatmaps, input_size, output_size):
    images = tf_autopad(images)
    heatmaps = tf_autopad(heatmaps)
    images = tf_resize(images, input_size)
    heatmaps = tf_resize(heatmaps, output_size)
    return images, heatmaps


@tf.function
def tf_stacker(images, heatmaps, stacks):
    return images, tf.stack([heatmaps] * stacks, axis=0)


@tf.function
def tf_random_rotation(images, heatmaps, rotation_range):
    rand = tf.random.uniform(
        [],
        minval=-1 * rotation_range,
        maxval=rotation_range,
        dtype=tf.dtypes.float32,
        seed=None,
        name=None,
    )
    rotated_images = tfa.image.rotate(images, rand)
    rotated_heatmaps = tfa.image.rotate(heatmaps, rand)
    return rotated_images, rotated_heatmaps


@tf.function
def tf_normalize_by_255(tensor):
    # We assume a Tensor with NHWC format
    return tensor / tf.constant(255.0, tf.float64)


@tf.function
def tf_normalize_minmax(tensor):
    # We assume a Tensor with NHWC format
    min_values = tf.reduce_min(tensor, axis=[-1, -2, -3], keepdims=True)
    max_values = tf.reduce_max(tensor, axis=[-1, -2, -3], keepdims=True)
    normalized_tensor = (tensor - min_values) / (max_values - min_values)
    return normalized_tensor


@tf.function
def tf_normalize_stddev(tensor):
    # We assume a Tensor with NHWC format
    mean_values = tf.reduce_mean(tensor, axis=[-1, -2, -3], keepdims=True)
    std_values = tf.math.reduce_std(tensor, axis=[-1, -2, -3], keepdims=True)
    normalized_tensor = (tensor - mean_values) / tf.sqrt(std_values)
    return normalized_tensor


@tf.function
def tf_get_bbox_coordinates(coords):
    # Get Min-Max X/Y
    xs = coords[:, 0]
    ys = coords[:, 1]
    maxx, minx = tf.reduce_max(xs), tf.reduce_min(xs)
    maxy, miny = tf.reduce_max(ys), tf.reduce_min(ys)

    return minx, maxx, miny, maxy


@tf.function
def tf_increase_bbox_area(bbox, img_shape, bbox_factor=0.5):
    # Unpack coordinates
    minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    # Compute Bbow Width-Height
    width, height = maxx - minx, maxy - miny
    minx = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), minx - width * bbox_factor
    )
    miny = tf.math.maximum(
        tf.constant(0.0, dtype=tf.float64), miny - height * bbox_factor
    )
    maxx = tf.math.minimum(
        tf.cast(img_shape[1], dtype=tf.float64), maxx + width * bbox_factor
    )
    maxy = tf.math.minimum(
        tf.cast(img_shape[0], dtype=tf.float64), maxy + height * bbox_factor
    )

    return minx, maxx, miny, maxy


@tf.function
def tf_compute_padding(bbox):
    # Unpack coordinates
    minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    # Compute Bbow Width-Height
    width, height = maxx - minx, maxy - miny

    # Compute Padding
    height_padding = tf.math.maximum(tf.constant(0, dtype=tf.int64), width - height)
    width_padding = tf.math.maximum(tf.constant(0, dtype=tf.int64), height - width)
    padding = [
        [height_padding // 2, height_padding // 2],
        [width_padding // 2, width_padding // 2],
        [0, 0],
    ]

    return padding


@tf.function
def tf_load_images(filenames, coords):
    images = tf_read_image(filenames)
    return images, coords

def tf_get_heatmaps(coords, height, width, sx, sy):
    heatmaps = tf.map_fn(
        fn=lambda x: tf_bivariate_normal_pdf_parallel(*x, height, width, sx, sy) if tf.reduce_all(tf.math.is_finite(x)) else tf.zeros([height, width], dtype=tf.float64),
        elems=coords,
        dtype=tf.float64
    )
    return heatmaps


@tf.function
def tf_compute_coordinates(images, coords, bbox_factor, resize_output=None):
    # Get Image Shape
    img_shape = tf.shape(images, out_type=tf.dtypes.int64)
    # Get Person Bbox
    minx, maxx, miny, maxy = tf_get_bbox_coordinates(coords)
    # Augment Bbox
    minx, maxx, miny, maxy = tf_increase_bbox_area(
        bbox=tf.stack([minx, maxx, miny, maxy]),
        img_shape=img_shape,
        bbox_factor=bbox_factor,
    )
    # Compute Padding
    bbox = tf.cast(tf.stack([minx, maxx, miny, maxy]), dtype=tf.int64)
    padding = tf_compute_padding(bbox=bbox)
    minx, maxx, miny, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    # Crop Image
    cropped_image = images[miny:maxy, minx:maxx, :]
    padded_image = tf.pad(cropped_image, paddings=padding)
    # Recompute coordinates
    coords = coords - (
        tf.cast([minx, miny], dtype=tf.float64)
        - tf.cast([padding[1][0], padding[0][0]], dtype=tf.float64)
    )
    if resize_output is not None:
        img_shape = tf.shape(padded_image, out_type=tf.dtypes.int64)
        coords = (
            coords
            * tf.cast(resize_output, dtype=tf.float64)
            / tf.cast(tf.reduce_max(img_shape[:-1]), dtype=tf.float64)
        )
    return padded_image, coords