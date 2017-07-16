# -*- coding: utf-8 -*-
"""
Created on Sun May 28 03:32:16 2017

@author: Walid Benbihi
@mail : w.benbihi (at) gmail.com
"""

import tensorflow as tf
import numpy as np
from layers import conv2d, convBnrelu, residual

################################################
#	             Hourglass Model
################################################

class HourglassModel():
	
	def __init__(self, nbStacks = 8, nFeat = 256, nModules = 1, outDim = 16, nLow = 3, training = True, name = 'stacked_hourglass'):
		"""
			args:
				nbStack 	: (int) number of (single) hourlgass modules stacked together
				name 	 	: (str) Name of the layer (TensorFlow useful)
				nModules 	: (int)
				outDim 	: (int) number of output dimension (body joints)
				train 	 	: (bool) trigger Dropout layers (avoid overfitting)
		"""
		self.nbStack = nbStacks
		self.name = name
		self.nFeat = nFeat
		self.nModules = nModules
		self.outDim = outDim
		self.train = training
		self.nLow = nLow
		
	def __call__(self, inputs):
		with tf.name_scope(self.name):
			with tf.name_scope('preprocessing'):
				pad_1 = tf.pad(inputs, np.array([[0,0],[2,2],[2,2],[0,0]]))
				conv_1 = conv2d(pad_1, 64, kernel_size=6, strides = 2, name = '256to128')
				res_1 = residual(conv_1, 128)
				pool_1 = tf.contrib.layers.max_pool2d(res_1, [2,2], [2,2], padding= 'VALID')
				res_2 = residual(pool_1, 128)
				res_3 = residual(res_2, self.nFeat)
			# Supervision Table
			hg = [None] * self.nbStack
			ll = [None] * self.nbStack
			ll_ = [None] * self.nbStack
			drop = [None] * self.nbStack
			out = [None] * self.nbStack
			out_ = [None] * self.nbStack
			sum_ = [None] * self.nbStack
			with tf.name_scope('stacks'):
				with tf.name_scope('hourglass.1'):
					hg[0] = self.hourglass(res_3, self.nLow, self.nFeat, 'hourglass')
					ll[0] = convBnrelu(hg[0], self.nFeat, name= 'conv_1')
					ll_[0] = conv2d(ll[0],self.nFeat,1,1,'VALID','ll')
					drop[0] = tf.layers.dropout(ll_[0], rate = 0.1, training = self.train)
					out[0] = conv2d(ll[0],self.outDim,1,1,'VALID','out')
					out_[0] = conv2d(out[0],self.nFeat,1,1,'VALID','out_')
					sum_[0] = tf.add_n([drop[0], out_[0], res_3])
				for i in range(1, self.nbStack-1):
					with tf.name_scope('hourglass.' + str(i+1)):
						hg[i] = self.hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
						ll[i] = convBnrelu(hg[i], self.nFeat, name='conv_1')
						ll_[i] = conv2d(ll[i],self.nFeat,1,1,'VALID','ll')
						drop[i] = tf.layers.dropout(ll_[i],rate=0.1, training = self.train)
						out[i] = conv2d(ll[i],self.outDim,1,1,'VALID','out')
						out_[i] = conv2d(out[i],self.nFeat,1,1,'VALID','out_')
						sum_[i] = tf.add_n([drop[i], out_[i], sum_[i-1]])
				with tf.name_scope('hourglass.' + str(self.nbStack)):
					hg[self.nbStack-1] = self.hourglass(sum_[self.nbStack - 2], self.nLow, self.nFeat, 'hourglass')
					ll[self.nbStack-1] = convBnrelu(hg[self.nbStack - 1], self.nFeat, name='conv_1')
					drop[self.nbStack-1] = tf.layers.dropout(ll[self.nbStack-1], rate=0.1, training = self.train)
					out[self.nbStack-1] = conv2d(drop[self.nbStack-1],self.outDim,1,1,'VALID', 'out')
			return tf.stack(out, name = 'output')
		
	def hourglass(self, inputs, n, numOut, name = 'hourglass'):
		with tf.name_scope(name):
			up_1 = residual(inputs, numOut, name = 'up1')
			low_ = tf.contrib.layers.max_pool2d(inputs, [2,2],[2,2], 'VALID')
			low_1 = residual(low_, numOut, name = 'low1')
			if n > 0:
				low_2 = self.hourglass(low_1, n-1, numOut, name='low2')
			else:
				low_2 = residual(low_1, numOut, name='low2')
			low_3 = residual(low_2, numOut, name = 'low3')
			up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name= 'upsampling')
			return tf.add([up_1, up_2])
