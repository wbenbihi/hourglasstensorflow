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