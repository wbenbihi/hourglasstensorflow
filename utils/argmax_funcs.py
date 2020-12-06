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

def argmax2D_no_mean(tensor, stages, size, channels):
    # Reshape as STAGES X (Spatial) X Channels
    indices = tf.argmax(tf.reshape(tensor, (stages, size**2, channels)), axis=1)
    constant = tf.cast(tf.transpose(size), tf.int64)
    col_indices = indices // constant
    row_indices = indices % constant
    argmax = tf.transpose(tf.stack([col_indices, row_indices]), [1, 2, 0])
    reshaped = tf.reshape(argmax, (stages*channels, 2))
    # Final Shape [Channels*Stages] X 2 - Every Stage is Taken Into Account
    return reshaped

def argmax2D_mean(tensor, stages, size, channels):
    # Reshape as STAGES X (Spatial) X Channels
    indices = tf.argmax(tf.reshape(tensor, (stages, size**2, channels)), axis=1)
    constant = tf.cast(tf.transpose(size), tf.int64)
    col_indices = tf.reduce_mean(indices // constant, axis=0)
    row_indices = tf.reduce_mean(indices % constant, axis=0)
    argmax = tf.transpose(tf.stack([col_indices, row_indices]))
    reshaped = tf.reshape(argmax, (stages*channels, 2))
    # Final Shape Channels X 2 - Every Stage is Taken Into Account and has the same weight
    return reshaped

def argmax2D_last_stage(tensor, stages, size, channels):
    # Reshape as (Spatial) X Channels
    indices = tf.argmax(tf.reshape(tensor[-1], (size**2, channels)), axis=0)
    constant = tf.cast(tf.transpose(size), tf.int64)
    col_indices = indices // constant
    row_indices = indices % constant
    argmax = tf.transpose(tf.stack([col_indices, row_indices]))
    # Final Shape Stages X 2 - Only Last Stage is Taken into account
    return argmax
