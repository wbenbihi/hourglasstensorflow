# -*- coding: utf-8 -*-
"""
Created on Sun May 28 04:21:16 2017

@author: Walid Benbihi
@mail : w.benbihi (at) gmail.com
"""
import tensorflow as tf
import numpy as np


"""
	TensorBoard Local Run Windows : http://192.168.56.1:6006/ 
	
	DOCUMENTATION:
		Calculus for Convolution Dimension (W-K+2*P)/S + 1:
			W: input size, K: kernel size, P: padding, S: stride
"""

def conv2d(inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = None):
	"""
		Create a Convolutional Layer
		args :
			inputs 	: (tensor) input Tensor
			filters 	: (int) number of filters
			kernel_size : (int) size of the kernel
			strides 	: (int) Value of stride
			pad 	 	: ('VALID'/'SAME')
		return :
			tf.Tensor
	"""
	with tf.name_scope(name):
		kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
		conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
		with tf.device('/cpu:0'):
			tf.summary.histogram('weights_summary', kernel, collections = ['train'])
		return conv
	
def convBnrelu(inputs, filters, kernel_size = 1, strides = 1, name = None):
	"""
		Create a Convolutional Layer + Batch Normalization + ReLU Activation 
		args :
			inputs 	: (tf.Tensor) input Tensor
			filters 	: (int) number of filters
			kernel_size : (int) size of the kernel
			strides 	: (int) Value of stride
			pad 	 	: ('VALID'/'SAME')
		return :
			tf.Tensor
	"""
	with tf.name_scope(name):
		kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
		conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='VALID', data_format='NHWC')
		norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, scope = '_bn_relu')
		with tf.device('/cpu:0'):
			tf.summary.histogram('weights_summary', kernel, collections = ['train'])
		return norm
	
def convBlock(inputs, numOut, name = 'convBlock'):
	"""
		Create a Convolutional Block Layer for Residual Units
		args:
			inputs : (tf.Tensor)
			numOut : (int) number of output channels
		return :
			tf.Tensor
	"""
	# DIMENSION CONSERVED
	with tf.name_scope(name):
		norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
		conv_1 = conv2d(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID')
		norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
		pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]))
		conv_2 = conv2d(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID')
		norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu)
		conv_3 = conv2d(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID')
		return conv_3
	
def skipLayer(inputs, numOut, name = 'skipLayer'):
	"""
		Create a skip layer : Identity if number of input channel = numOut, convolution else
		args :
			inputs : (tf.Tensor)
			numOut : (int)
		return :
			tf.Tensor
	"""
	# DIMENSION CONSERVED
	with tf.name_scope(name):
		if inputs.get_shape().as_list()[3] == numOut:
			return inputs
		else:
			conv = conv2d(inputs, numOut, kernel_size=1, strides=1, name= 'skipLayer-conv')
			return conv
	
def residual(inputs, numOut, name = 'residual'):
	# DIMENSION CONSERVED
	with tf.name_scope(name):
		convb = convBlock(inputs, numOut)
		skip = skipLayer(inputs,numOut)
		return tf.add_n([convb,skip])
