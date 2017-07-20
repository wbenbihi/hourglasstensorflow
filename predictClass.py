# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 17 15:50:43 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com

Abstract:
	This python code creates a Stacked Hourglass Model
	(Credits : A.Newell et al.)
	(Paper : https://arxiv.org/abs/1603.06937)
	
	Code translated from 'anewell' github
	Torch7(LUA) --> TensorFlow(PYTHON)
	(Code : https://github.com/anewell/pose-hg-train)
	
	Modification are made and explained in the report
	Goal : Achieve Real Time detection (Webcam)
	----- Modifications made to obtain faster results (trade off speed/accuracy)
	
	This work is free of use, please cite the author if you use it!

@author: Walid
"""

import sys
sys.path.append('./')

from hourglass_tiny_backup import HourglassModel
from time import time
import numpy as np
import tensorflow as tf
from train_launcher import process_config
import cv2
from yolo_tiny_net import YoloTinyNet
import copy


class PredictProcessor():
	"""
	PredictProcessor class: Give the tools to open and use a trained model for
	prediction.
	Dependency: OpenCV or PIL (OpenCV prefered)
	
	Comments:
		Every CAPITAL LETTER methods are to be modified with regard to your needs and dataset
	"""
	#-------------------------INITIALIZATION METHODS---------------------------
	def __init__(self, config_dict):
		""" Initializer
		Args:
			config_dict	: config_dict
		"""
		self.params = config_dict
		self.HG = HourglassModel(nFeat= self.params['nfeats'], nStack = self.params['nstacks'], 
						nModules = self.params['nmodules'], nLow = self.params['nlow'], outputDim = self.params['num_joints'], 
						batch_size = self.params['batch_size'], drop_rate = self.params['dropout_rate'], lear_rate = self.params['learning_rate'],
						decay = self.params['learning_rate_decay'], decay_step = self.params['decay_step'], dataset = None, training = False,
						w_summary = True, logdir_test = self.params['log_dir_test'],
						logdir_train = self.params['log_dir_test'], tiny = self.params['tiny'], 
						modif = False, name = self.params['name'])
		
	def color_palette(self):
		""" Creates a color palette dictionnary
		Drawing Purposes
		You don't need to modify this function.
		In case you need other colors, add BGR color code to the color list
		and make sure to give it a name in the color_name list
		/!\ Make sure those 2 lists have the same size
		"""
		#BGR COLOR CODE
		self.color = [(241,242,224), (196,203,128), (136,150,0), (64,77,0), 
				(201,230,200), (132,199,129), (71,160,67), (32,94,27),
				(130,224,255), (7,193,255), (0,160,255), (0,111,255),
				(220,216,207), (174,164,144), (139,125,96), (100,90,69),
				(252,229,179), (247,195,79), (229,155,3), (155,87,1),
				(231,190,225), (200,104,186), (176,39,156), (162,31,123),
				(210,205,255), (115,115,229), (80,83,239), (40,40,198)]
		# Color Names
		self.color_name = ['teal01', 'teal02', 'teal03', 'teal04',
				'green01', 'green02', 'green03', 'green04',
				'amber01', 'amber02', 'amber03', 'amber04',
				'bluegrey01', 'bluegrey02', 'bluegrey03', 'bluegrey04',
				'lightblue01', 'lightblue02', 'lightblue03', 'lightblue04',
				'purple01', 'purple02', 'purple03', 'purple04',
				'red01', 'red02', 'red03', 'red04']
		self.classes_name =  ["aeroplane", "bicycle", "bird",
				"boat", "bottle", "bus", "car", "cat", "chair",
				"cow", "diningtable", "dog", "horse", "motorbike",
				"person", "pottedplant", "sheep",
				"sofa", "train","tvmonitor"]
		self.color_class = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
		self.palette = {}
		for i, name in enumerate(self.color_name):
			self.palette[name] = self.color[i]
			
	def LINKS_JOINTS(self):
		""" Defines links to be joined for visualization
		Drawing Purposes
		You may need to modify this function
		"""
		self.links = {}
		# Edit Links with your needed skeleton
		LINKS = [(0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,8),(8,9),(8,13),(13,14),(14,15),(8,12),(12,11),(11,10)]
		color_id = [1,2,3,3,2,1,5,7,27,26,25,27,26,25]
		# 10 lines version
		# LINKS = [(0,1),(1,2),(3,4),(4,5),(6,8),(8,9),(13,14),(14,15),(12,11),(11,10)]
		for i in range(len(LINKS)):
			self.links[i] = {'link' : LINKS[i], 'color' : self.palette[self.color_name[color_id[i]]]}
	
	# ----------------------------TOOLS----------------------------------------
	def col2RGB(self, col):
		""" 
		Args:
			col 	: (int-tuple) Color code in BGR MODE
		Returns
			out 	: (int-tuple) Color code in RGB MODE
		"""
		return col[::-1]
	
	def givePixel(self, link, joints):
		""" Returns the pixel corresponding to a link
		Args:
			link 	: (int-tuple) Tuple of linked joints
			joints	: (array) Array of joints position shape = outDim x 2
		Returns:
			out		: (tuple) Tuple of joints position
		"""
		return (joints[link[0]].astype(np.int), joints[link[1]].astype(np.int))
	
	# ---------------------------MODEL METHODS---------------------------------
	def model_init(self):
		""" Initialize the Hourglass Model
		"""
		t = time()
		self.HG.generate_model()
		print('Graph Generated in ', int(time() - t), ' sec.')
	
	def load_model(self, load = None):
		""" Load pretrained weights (See README)
		"""
		self.HG.restore(load)
		
	def _create_joint_tensor(self, tensor, name = 'joint_tensor',debug = False):
		""" TensorFlow Computation of Joint Position
		Args:
			tensor		: Prediction Tensor Shape [nbStack x 64 x 64 x outDim] or [64 x 64 x outDim]
			name		: name of the tensor
		Returns:
			out			: Tensor of joints position
		
		Comment:
			Genuinely Agreeing this tensor is UGLY. If you don't trust me, look at
			'prediction' node in TensorBoard.
			In my defence, I implement it to compare computation times with numpy.
		"""
		with tf.name_scope(name):
			shape = tensor.get_shape().as_list()
			if debug:
				print(shape)
			if len(shape) == 3:
				resh = tf.reshape(tensor[:,:,0], [-1])
			elif len(shape) == 4:
				resh = tf.reshape(tensor[-1,:,:,0], [-1])
			if debug:
				print(resh)
			arg = tf.arg_max(resh,0)
			if debug:
				print(arg, arg.get_shape(), arg.get_shape().as_list())
			joints = tf.expand_dims(tf.stack([arg // tf.to_int64(shape[1]), arg % tf.to_int64(shape[1])], axis = -1), axis = 0)
			for i in range(1, shape[-1]):
				if len(shape) == 3:
					resh = tf.reshape(tensor[:,:,i], [-1])
				elif len(shape) == 4:
					resh = tf.reshape(tensor[-1,:,:,i], [-1])
				arg = tf.arg_max(resh,0)
				j = tf.expand_dims(tf.stack([arg // tf.to_int64(shape[1]), arg % tf.to_int64(shape[1])], axis = -1), axis = 0)
				joints = tf.concat([joints, j], axis = 0)
			return tf.identity(joints, name = 'joints')
			
	def _create_prediction_tensor(self):
		""" Create Tensor for prediction purposes
		"""
		with tf.name_scope('prediction'):
			self.HG.pred_sigmoid = tf.nn.sigmoid(self.HG.output[:,self.HG.nStack - 1], name= 'sigmoid_final_prediction')
			self.HG.pred_final = self.HG.output[:,self.HG.nStack - 1]
			self.HG.joint_tensor = self._create_joint_tensor(self.HG.output[0], name = 'joint_tensor')
			self.HG.joint_tensor_final = self._create_joint_tensor(self.HG.output[0,-1] , name = 'joint_tensor_final')
		print('Prediction Tensors Ready!')
	
	
	
	#----------------------------PREDICTION METHODS----------------------------
	def predict_coarse(self, img, debug = False):
		""" Given a 256 x 256 image, Returns prediction Tensor
		This prediction method returns a non processed Output
		Values not Bounded
		Args:
			img		: Image -Shape (256 x256 x 3) -Type : float32
			debug	: (bool) True to output prediction time
		Returns:
			out		: Array -Shape (nbStacks x 64 x 64 x outputDim) -Type : float32
		"""
		if debug:
			t = time()
		if img.shape == (256,256,3):
			out = self.HG.Session.run(self.HG.output, feed_dict={self.HG.img : np.expand_dims(img, axis = 0)})
		else:
			print('Image Size does not match placeholder shape')
			raise Exception
		if debug:
			print('Pred: ', time() - t, ' sec.')
		return out
	
	def pred(self, img, debug = False):
		""" Given a 256 x 256 image, Returns prediction Tensor
		This prediction method returns values in [0,1]
		Use this method for inference
		Args:
			img		: Image -Shape (256 x256 x 3) -Type : float32
			debug	: (bool) True to output prediction time
		Returns:
			out		: Array -Shape (64 x 64 x outputDim) -Type : float32
		"""
		if debug:
			t = time()
		if img.shape == (256,256,3):
			out = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict={self.HG.img : np.expand_dims(img, axis = 0)})
		else:
			print('Image Size does not match placeholder shape')
			raise Exception
		if debug:
			print('Pred: ', time() - t, ' sec.')
		return out
	
	def joints_pred(self, img, coord = 'hm', debug = False):
		""" Given an Image, Returns an array with joints position
		Args:
			img		: Image -Shape (256 x 256 x 3) -Type : float32 
			coord	: 'hm'/'img' Give pixel coordinates relative to heatMap('hm') or Image('img')
			debug	: (bool) True to output prediction time
		Returns
			out		: Array -Shape(num_joints x 2) -Type : int
		"""
		if debug:
			t = time()
			j1 = self.HG.Session.run(self.HG.joint_tensor, feed_dict = {self.HG.img: img})
			print('JT:', time() - t)
			t = time()
			j2 = self.HG.Session.run(self.HG.joint_tensor_final, feed_dict = {self.HG.img: img})
			print('JTF:', time() - t)
			if coord == 'hm':
				return j1, j2
			elif coord == 'img':
				return j1 * self.params['img_size'] / self.params['hm_size'], j2 *self.params['img_size'] / self.params['hm_size']
			else:
				print("Error: 'coord' argument different of ['hm','img']")
		else:
			j = self.HG.Session.run(self.HG.joint_tensor_final, feed_dict = {self.HG.img: img})
			if coord == 'hm':
				return j
			elif coord == 'img':
				return j * self.params['img_size'] / self.params['hm_size']
			else:
				print("Error: 'coord' argument different of ['hm','img']")
				
	def joints_pred_numpy(self, img, coord = 'hm', thresh = 0.2):
		"""
		"""
		hm = self.HG.Session.run(self.HG.pred_sigmoid , feed_dict = {self.HG.img: img})
		joints = -1*np.ones(shape = (self.params['num_joints'], 2))
		for i in range(self.params['num_joints']):
			index = np.unravel_index(hm[0,:,:,i].argmax(), (self.params['hm_size'],self.params['hm_size']))
			if hm[0,index[0], index[1],i] > thresh:
				if coord == 'hm':
					joints[i] = np.array(index)
				elif coord == 'img':
					joints[i] = np.array(index) * self.params['img_size'] / self.params['hm_size']
		return joints
			
	def batch_pred(self, batch, debug = False):
		""" Given a 256 x 256 images, Returns prediction Tensor
		This prediction method returns values in [0,1]
		Use this method for inference
		Args:
			batch	: Batch -Shape (batchSize x 256 x 256 x 3) -Type : float32
			debug	: (bool) True to output prediction time
		Returns:
			out		: Array -Shape (batchSize x 64 x 64 x outputDim) -Type : float32
		"""
		if debug:
			t = time()
		if batch[0].shape == (256,256,3):
			out = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict={self.HG.img : batch})
		else:
			print('Image Size does not match placeholder shape')
			raise Exception
		if debug:
			print('Pred: ', time() - t, ' sec.')
		return out
	
	
	
	#-------------------------------PLOT FUNCTION------------------------------
	def plt_skeleton(self, img, copy = True, debug = False):
		""" Plot skeleton on Image
		Args:
			img 	: Image shape = (256,256,3)
			copy 	: (bool) False to write on input image / True to return a new array
		"""
		joints = self.joints_pred(np.expand_dims(img, axis = 0), coord = 'img', debug = False)
		if copy:
			img = np.copy(img)
		for i in range(len(self.links)):
			position = self.givePixel(self.links[i]['link'],joints)
			cv2.line(img, tuple(position[0])[::-1], tuple(position[1])[::-1], self.links[i]['color'], thickness = 2)
		if copy:
			return img
		
	def plt_skeleton_numpy(self, img, copy = True, thresh = 0.2):
		"""
		"""
		joints = self.joints_pred_numpy(np.expand_dims(img, axis = 0), coord = 'img', thresh = thresh)
		if copy:
			img = np.copy(img)
		for i in range(len(self.links)):
			l = self.links[i]['link']
			good_link = True
			for p in l:
				if np.array_equal(joints[p], [-1,-1]):
					good_link = False
			if good_link:
				position = self.givePixel(self.links[i]['link'],joints)
				cv2.line(img, tuple(position[0])[::-1], tuple(position[1])[::-1], self.links[i]['color'], thickness = 2)
		if copy:
			return img
			
		
	def streamPredict(self, src = 0, mirror = True, numpy = True, thresh = 0.2):
		""" Runs prediction on Webcam
		Args:
			src	 : Stream Source (0 if single webcam)
		"""
		cam = cv2.VideoCapture(0)
		while True:
			i = 0
			ret_val, img = cam.read()
			img = cv2.resize(img, (256,256))
			if numpy:
				self.plt_skeleton_numpy(img, copy = False, thresh = thresh)
			else :
				self.plt_skeleton(img, copy = False)
			img= cv2.resize(img, (800,800))
			if mirror: 
				img = cv2.flip(img, 1)
			cv2.imshow('stream', img)
			i = i + 1
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				break
		cv2.destroyAllWindows()
		cam.release()
	
	
	# -------------------------OBJECT DETECTOR---------------------------------
	def od_init(self):
		"""
		"""
		print('Adding YOLO Graph to Main Graph')
		t = time() 
		self.common_params = {'image_size': 448, 'num_classes': 20, 'batch_size':8}
		self.net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}
		self.net = YoloTinyNet(self.common_params, self.net_params, test = True)
		self.image_od = tf.placeholder(tf.float32, (1,448,448,3))
		self.predicts_od = self.net.inference(self.image_od)
		print('YOLO created: ', time() - t, ' sec.')
		
	def restore_od(self, load = 'yolo_tiny.ckpt'):
		"""
		"""
		print('Loading YOLO...')
		t = time()
		self.HG.saver.restore(self.HG.Session, load)
		print('Trained YOLO Loaded: ', time() - t, ' sec.')
		
	def largestind(self, array, n):
		flat = array.flatten()
		indices = np.argpartition(flat, -n)[-n:]
		indices = indices[np.argsort(-flat[indices])]
		return  np.transpose(np.asarray(np.unravel_index(indices, array.shape)))
	
	def non_max_suppression_fast(self, boxes, overlapThresh):
		# if there are no boxes, return an empty list
		if len(boxes) == 0:
			return []
	 
		# if the bounding boxes integers, convert them to floats --
		# this is important since we'll be doing a bunch of divisions
		if boxes.dtype.kind == "i":
			boxes = boxes.astype("float")
	 
		# initialize the list of picked indexes	
		pick = []
	 
		# grab the coordinates of the bounding boxes
		x1 = boxes[:,0]
		y1 = boxes[:,1]
		x2 = boxes[:,2]
		y2 = boxes[:,3]
	 
		# compute the area of the bounding boxes and sort the bounding
		# boxes by the bottom-right y-coordinate of the bounding box
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)
	 
		# keep looping while some indexes still remain in the indexes
		# list
		while len(idxs) > 0:
			# grab the last index in the indexes list and add the
			# index value to the list of picked indexes
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
	 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = np.maximum(x1[i], x1[idxs[:last]])
			yy1 = np.maximum(y1[i], y1[idxs[:last]])
			xx2 = np.minimum(x2[i], x2[idxs[:last]])
			yy2 = np.minimum(y2[i], y2[idxs[:last]])
	 
			# compute the width and height of the bounding box
			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)
	 
			# compute the ratio of overlap
			overlap = (w * h) / area[idxs[:last]]
	 
			# delete all indexes from the index list that have
			idxs = np.delete(idxs, np.concatenate(([last],
				np.where(overlap > overlapThresh)[0])))
	 
		# return only the bounding boxes that were picked using the
		# integer data type
		return boxes[pick].astype("int")
		
	def processboxes(self, predicts, num_objects = 30, num_classes = 2, class_thresh = 0.10,nms_thresh = 0.5, debug = False):
		p_classes = predicts[0, :, :, 0:num_classes]
		C = predicts[0, :, :, num_classes:num_classes+2]
		coordinate = predicts[0, :, :, num_classes+2:]
		
		p_classes = np.reshape(p_classes, (7,7,1,num_classes))
		C = np.reshape(C, (7,7,2,1))
		P = C * p_classes
		coordinate = np.reshape(coordinate, (7, 7, 2, 4))
		
		indices = self.largestind(P, num_objects)
		boxes = [[]] * num_classes
		for i in range(num_objects):
			index = indices[i]
			prob = P[index[0], index[1], index[2], index[3]]
			max_coordinate = coordinate[index[0], index[1], index[2], :]
			class_num = index[3]
			xcenter = max_coordinate[0]
			ycenter = max_coordinate[1]
		
			w = max_coordinate[2]
			h = max_coordinate[3]
		
			xcenter = (index[1] + xcenter) * (448/7.0)
			ycenter = (index[0] + ycenter) * (448/7.0)
		
			w = w * 448
			h = h * 448
		
			xmin = xcenter - w/2.0
			ymin = xcenter - h/2.0
			xmax = xmin + w
			ymax = ymin + h
			if prob > class_thresh:
				class_box = copy.deepcopy(boxes[class_num])
				class_box.append([xmin, ymin , xmax, ymax])
				boxes[class_num] = class_box
		for i in range(num_classes):
			boxes[i] = self.non_max_suppression_fast(np.asarray(boxes[i]), overlapThresh = nms_thresh)
		if debug:
			return boxes, P
		else:
			return boxes

		
	def camYolo(self, src = 0, mirror = True, cbox = 0.0, cls_thresh = 0.12, nms_thresh = 0.6):
		"""
		"""
		cam = cv2.VideoCapture(src)
		i = 0
		while True:
			t_start = time()
			# Getting Source Image
			ret_val, img = cam.read()
			t_read = time()
			# Mirror Image
			if mirror:
				img = cv2.flip(img, 1)
			# Resize Image to fit placeholder
			resized_img = cv2.resize(img, (448,448))
			# Color Mode conversion
			np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
			np_img = np_img.astype(np.float32)
			# Fit to [-1,1]
			np_img = (np_img - 127.0) / 128.0 
			# Expand Dim
			np_img = np.reshape(np_img, (1,448,448,3))
			# Prediction
			t_preprocess = time()
			np_predict = self.HG.Session.run(self.predicts_od, feed_dict={self.image_od : np_img})
			t_predict = time()
			# PostProcessing
			results, P = self.processboxes(np_predict, num_classes= self.common_params['num_classes'], class_thresh = cls_thresh, nms_thresh = nms_thresh, debug = True)
			# DRAWING BOX
			for class_num in range(self.common_params['num_classes']):
				class_name = self.classes_name[class_num]
				for xmin, ymin , xmax, ymax in results[class_num]:
					bbox = (max(0,int(xmin - cbox*(xmax-xmin))),max(0,int(ymin - cbox*(ymax-ymin))), min(447,int(xmax + cbox*(xmax-xmin))), min(447,int(ymax + cbox*(ymax-ymin))))
					cv2.rectangle(resized_img, (bbox[0],bbox[1]), (bbox[2], bbox[3]), self.color[class_num], thickness = 2)
					cv2.rectangle(resized_img, (bbox[0], bbox[1]), (bbox[0] + 100, bbox[1] + 30), self.color[class_num], thickness = -1)
					cv2.putText(resized_img, class_name, (bbox[0] + 15, bbox[1] + 15), 2, 0.6, (0,0,0), thickness = 2)
			t_postprocess = time()
			resized_img = cv2.resize(resized_img, (800,800))
			cv2.imshow('stream', resized_img)
			if i % 25 == 0:
				print('------Frame-----')
				print('--Read:\t\t ' + str(t_read - t_start) + '\n--Preproc:\t ' + str(t_preprocess - t_read) + '\n--Predict:\t ' + str(t_predict-t_preprocess) + '\n--Postproc:\t ' + str(t_postprocess-t_predict) + '\n--Full:\t\t ' + str(t_postprocess - t_start))
			i += 1
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				break
		cv2.destroyAllWindows()
		cam.release()
	
if __name__ == '__main__':
	params = process_config('config.cfg')
	predict = PredictProcessor(params)
	predict.color_palette()
	predict.LINKS_JOINTS()
	predict.model_init()
	predict.od_init()
	predict.load_model(load = 'hourglass_tiny_modif_90')
	predict.restore_od(load = 'yolo_tiny.ckpt')
	predict._create_prediction_tensor()