import tensorflow as tf
from utils import argmax2D_with_batch, argmax2D_without_batch

class AmountCorrectKeypointsMetric(tf.keras.metrics.Metric):

    def __init__(self, name='amount_correct_keypoints', threshold=5,**kwargs):
        super(AmountCorrectKeypointsMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct_keypoints = self.add_weight(name='correct_keypoints', initializer='zeros')
        self.total_keypoints = self.add_weight(name='total_keypoints', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        idx_true = tf.map_fn(argmax2D_without_batch, y_true)
        idx_pred = tf.map_fn(argmax2D_without_batch, y_pred)
        distance = tf.norm(tf.cast(idx_pred - idx_true, tf.float32), axis=-1)
        is_near = tf.cast(distance < self.threshold, tf.float32)
        is_item = tf.cast(distance > 0.0, tf.float32)
        self.correct_keypoints.assign_add(tf.reduce_sum(is_near))
        self.total_keypoints.assign_add(tf.reduce_sum(is_item))
    
    def result(self):
        return self.correct_keypoints / self.total_keypoints
    
    def reset_states(self):
        self.correct_keypoints.assign(0.)
        self.total_keypoints.assign(0.)