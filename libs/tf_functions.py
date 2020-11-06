from numpy import pi
import tensorflow as tf
from config import CFG

# Image Reader
@tf.function
def tf_read_image(filename):
    image = tf.io.decode_jpeg(tf.io.read_file(filename), channels=3)
    return image

# Bivariate(2D) Gaussian
@tf.function
def tf_bivariate_normal_pdf(height, width, mx, my, sx, sy):
    X = tf.range(start=0., limit=tf.cast(width, tf.float32), delta=1.)
    Y = tf.range(start=0., limit=tf.cast(height, tf.float32), delta=1.)
    X, Y = tf.meshgrid(X, Y)
    R = tf.sqrt(((X-mx)**2/(sx**2)) + ((Y-my)**2)/(sy**2))
    Z = ((1. / (2 * pi) * tf.sqrt(sx*sy)) * tf.exp(-.5*R**2))
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
        return tf_bivariate_normal_pdf(height, width, x, y, sx=10., sy=10.)
    else:
        return tf.zeros([height, width])

# Heatmaps generation iterator
@tf.function
def tf_heatmaps(coords, image):
    return tf.transpose(tf.map_fn(lambda x: tf_generate_heatmap(x, image), tf.cast(coords, tf.float32)), [1, 2, 0])

@tf.function
def tf_parse_dataset(filename, coords):
    images = tf_read_image(filename)
    heatmaps = tf_heatmaps(coords, images)
    return images, heatmaps

@tf.function
def tf_autopad(tensor):
    shape = tf.shape(tensor,out_type=tf.dtypes.int32)
    pad_x = tf.clip_by_value(
        shape[1] - shape[0],
        0,
        10000,
        name=None
    )
    pad_y = tf.clip_by_value(
        shape[0] - shape[1],
        0,
        10000,
        name=None
    )
    padded_tensor = tf.pad(
        tensor,
        [
            [pad_x // 2, pad_x //2],
            [pad_y // 2, pad_y //2],
            [0, 0],
        ]
    )
    return padded_tensor

@tf.function
def tf_resize(tensor, size):
    return tf.image.resize(tensor, [size, size])

@tf.function
def tf_resize_input(tensor):
    return tf.image.resize(tensor, [CFG.default.HOURGLASS.inputsize, CFG.default.HOURGLASS.inputsize])

@tf.function
def tf_resize_output(tensor):
    return tf.image.resize(tensor, [CFG.default.HOURGLASS.outputsize, CFG.default.HOURGLASS.outputsize])
    
@tf.function
def tf_preprocess(images, heatmaps):
    images = tf_autopad(images)
    heatmaps = tf_autopad(heatmaps)
    images = tf_resize(images, CFG.default.HOURGLASS.inputsize)
    heatmaps = tf_resize(heatmaps, CFG.default.HOURGLASS.outputsize)
    return images, heatmaps