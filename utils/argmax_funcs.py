import tensorflow as tf
# TensorFlow implementation of ArgMax2D
# Define a function to get the location of a predicted joint
# since argmax works on a single dimension we use the following protocol
# 1. Reshape the Tensor to have only 1 spatial dimension 
#    From 
#       Batch X Stage X Height X Width X Joint
#    To
#       Batch X Stage X Spatial(=Height.Width) X Joint
# 2. Find the Argmax along this Spatial Axis
# 3. Compute the Column and Row indices with euclidean division and modulo (resp.)
# 4. Average those indices along the Stage Axis
# 5. Stack and Reshape the indices to get the coordinates per image per joint
#       Batch X Joint X 2 (X,Y coordinates)

# TODO (wben): Parametrize the shape with regard to config files

def argmax2D_with_batch(tensor):
    indices = tf.argmax(tf.reshape(tensor, (tensor.shape[0], STAGES, 4096, 14)), axis=-2)
    col_indices = tf.reduce_mean(indices // 64, axis=-2)
    row_indices = tf.reduce_mean(indices % 64, axis=-2)
    final_indices = tf.transpose(tf.stack([col_indices, row_indices]), perm=[1,2, 0])
    return final_indices

def argmax2D_without_batch(tensor):
    indices = tf.argmax(tf.reshape(tensor, (STAGES, 4096, 14)), axis=-2)
    col_indices = tf.reduce_mean(indices // 64, axis=-2)
    row_indices = tf.reduce_mean(indices % 64, axis=-2)
    final_indices = tf.transpose(tf.stack([col_indices, row_indices]))
    return tf.cast(final_indices, tf.float32)