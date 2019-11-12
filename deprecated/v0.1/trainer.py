# -*- coding: utf-8 -*-
"""
Created on Sun May 28 05:08:22 2017

@author: Walid Benbihi
@mail : w.benbihi (at) gmail.com
"""

import tensorflow as tf
import numpy as np
import os
from hourglassModel import HourglassModel
from time import time, strftime
from random import shuffle, choice
import params
import process
from tools import fullTestSet, modifyOutput, rotatehm
import sys
from skimage.transform import rotate




GPU = params.gpu
CPU = params.cpu

date = strftime('%Y.%m.%d')
lr = params.learning_rate
nepochs = params.nEpochs
epochiter = params.iter_by_epoch
_,_,name,_ = process.toTrainList()
weights = np.array(np.genfromtxt(process.path + 'weight_joint.csv', delimiter=','), np.uint8)
imageOnDisk = process.cleanList(os.listdir(process.arraypath))
shuffle(imageOnDisk)
trainingData = imageOnDisk[:params.limit_train_test]
testingData = imageOnDisk[params.limit_train_test:]
testingData = fullTestSet(testingData, name, weights)
batchSize = params.batch_size


if __name__ == '__main__':
	with tf.device(GPU):
		print('Creating Model')
		t_start = time()
		with tf.name_scope('inputs'):
			x = tf.placeholder(tf.float32, [None, 256,256,3], name = 'x_train')
			y = tf.placeholder(tf.float32, [params.nbStacks,None, 64,64, params.outDim], name= 'y_train')
		print('--Inputs : Done')
		with tf.name_scope('model'):
			output = HourglassModel(params.nbStacks, params.nFeat, params.nModule, params.outDim, params.nLow, True, name = 'stacked_hourglass')
		print('--Model : Done')
		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels= y), name = 'cross_entropy_loss') *4096 * params.nbStacks * params.batch_size
		print('--Loss : Done')
		with tf.name_scope('rmsprop_optimizer'):
			rmsprop = tf.train.RMSPropOptimizer(lr)
		print('--Optim : Done')
	with tf.name_scope('steps'):
		train_steps = tf.Variable(0, trainable=False)
	with tf.device(GPU):
		with tf.name_scope('minimize'):
			train_rmsprop = rmsprop.minimize(loss, train_steps)
		print('--Minimizer : Done')
	init = tf.global_variables_initializer()
	print('--Init : Done')
	print('Model generation: ' + str(time()- t_start))
	with tf.name_scope('loss'):
		tf.summary.scalar('loss', loss, collections = ['train'])
	merged_summary_op = tf.summary.merge_all('train')
	with tf.name_scope('Session'):
		with tf.device(GPU):
			sess = tf.Session()
			sess.run(init)
			print('Session initilized')
		with tf.device(CPU):
			saver = tf.train.Saver()
		with tf.device(GPU):
			summary_train = tf.summary.FileWriter(process.summarytrain , tf.get_default_graph())
			t_train_start = time()
			print('Start training')
			with tf.name_scope('training'):
				for epoch in range(nepochs):
					t_epoch_start = time()
					avg_cost = 0.
					print('========Training Epoch: ', (epoch + 1))
					with tf.name_scope('epoch_' + str(epoch)):
						for i in range(epochiter):
							percent = ((i+1)/epochiter )*100
							num = np.int(20*percent/100)
							sys.stdout.write("\r Train: {0}>".format("="*num) + "{0}>".format(" "*(20-num)) + "||" + str(percent)[:3] + "%" + ' time :' + str(time()- t_epoch_start)[:5] + 'sec  loss =' + str(avg_cost)[:6])
							sys.stdout.flush()
							y_batch = np.zeros((params.nbStacks, batchSize, 64,64,16))
							x_batch = np.zeros((batchSize,256,256,3))
							with tf.name_scope('batch_train'):
								for k in range(batchSize):
									item = choice(trainingData)
									path_img = process.arraypath + item + 'img.npy'
									path_hm = process.arraypath + item + 'hm.npy'
									r_angle = np.random.randint(params.random_angle_min,params.random_angle_max)
									img = np.load(path_img)
									hm = np.load(path_hm)
									img = rotate(img, r_angle)
									hm =  rotatehm(hm, r_angle)
									hm_f = modifyOutput(hm, params.nbStacks)
									x_batch[k,:,:,:] = img / 255
									y_batch[:,k,:,:,:] = hm_f[:,0,:,:,:]
								_,c,summary = sess.run([train_rmsprop, loss, merged_summary_op], feed_dict={x: x_batch, y: y_batch})
								avg_cost += c / (epochiter*batchSize)
								if (epoch * epochiter + i) % params.step_to_save == 0:
									summary_train.add_summary(summary, epoch * epochiter + i)
									summary_train.flush()
					t_epoch_finish = time()
					print("Epoch:", (epoch + 1), '  avg_cost= ', "{:.9f}".format(avg_cost),' time_epoch=', str(t_epoch_finish-t_epoch_start))
		t_end = time()
		print('Training Done : ' + str(t_end - t_start))
		
		
		
		
		
