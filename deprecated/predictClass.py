# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 17 15:50:43 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

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

"""

import sys
sys.path.append('./')

from hourglass_tiny import HourglassModel
from time import time, clock, sleep
import numpy as np
import tensorflow as tf
import scipy.io
from train_launcher import process_config
import cv2
#from yolo_tiny_net import YoloTinyNet
from yolo_net import YOLONet
from datagen import DataGenerator
import config as cfg
import threading

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
						modif = False, name = self.params['name'], attention = self.params['mcam'], w_loss=self.params['weighted_loss'] , joints= self.params['joint_list'])
		self.graph = tf.Graph()
		self.src = 0
		self.cam_res = (480,640)
		
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
		# Person ID = 14
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
		LINKS = [(0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,8),(8,13),(13,14),(14,15),(8,12),(12,11),(11,10)]
		self.LINKS_ACP = [(0,1),(1,2),(3,4),(4,5),(7,8),(8,9),(10,11),(11,12)]
		color_id = [1,2,3,3,2,1,5,27,26,25,27,26,25]
		self.color_id_acp = [8,9,9,8,19,20,20,19]
		# 13 joints version
		#LINKS = [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(6,11),(11,12),(12,13),(6,11),(10,9),(9,8)]
		#color_id = [1,2,3,2,1,0,27,26,25,27,26,25]
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
		with self.graph.as_default():
			self.HG.generate_model()
		print('Graph Generated in ', int(time() - t), ' sec.')
	
	def load_model(self, load = None):
		""" Load pretrained weights (See README)
		Args:
			load : File to load
		"""
		with self.graph.as_default():
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
		with self.graph.as_default():
			with tf.name_scope('prediction'):
				self.HG.pred_sigmoid = tf.nn.sigmoid(self.HG.output[:,self.HG.nStack - 1], name= 'sigmoid_final_prediction')
				self.HG.pred_final = self.HG.output[:,self.HG.nStack - 1]
				self.HG.joint_tensor = self._create_joint_tensor(self.HG.output[0], name = 'joint_tensor')
				self.HG.joint_tensor_final = self._create_joint_tensor(self.HG.output[0,-1] , name = 'joint_tensor_final')
		print('Prediction Tensors Ready!')
	
	
	
	#----------------------------PREDICTION METHODS----------------------------
	def predict_coarse(self, img, debug = False, sess = None):
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
			if sess is None:
				out = self.HG.Session.run(self.HG.output, feed_dict={self.HG.img : np.expand_dims(img, axis = 0)})
			else:
				out = sess.run(self.HG.output, feed_dict={self.HG.img : np.expand_dims(img, axis = 0)})
		else:
			print('Image Size does not match placeholder shape')
			raise Exception
		if debug:
			print('Pred: ', time() - t, ' sec.')
		return out
	
	def pred(self, img, debug = False, sess = None):
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
			if sess is None:
				out = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict={self.HG.img : np.expand_dims(img, axis = 0)})
			else:
				out = sess.run(self.HG.pred_sigmoid, feed_dict={self.HG.img : np.expand_dims(img, axis = 0)})
		else:
			print('Image Size does not match placeholder shape')
			raise Exception
		if debug:
			print('Pred: ', time() - t, ' sec.')
		return out
	
	def joints_pred(self, img, coord = 'hm', debug = False, sess = None):
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
			if sess is None:
				j1 = self.HG.Session.run(self.HG.joint_tensor, feed_dict = {self.HG.img: img})
			else:
				j1 = sess.run(self.HG.joint_tensor, feed_dict = {self.HG.img: img})
			print('JT:', time() - t)
			t = time()
			if sess is None:
				j2 = self.HG.Session.run(self.HG.joint_tensor_final, feed_dict = {self.HG.img: img})
			else:
				j2 = sess.run(self.HG.joint_tensor_final, feed_dict = {self.HG.img: img})
			print('JTF:', time() - t)
			if coord == 'hm':
				return j1, j2
			elif coord == 'img':
				return j1 * self.params['img_size'] / self.params['hm_size'], j2 *self.params['img_size'] / self.params['hm_size']
			else:
				print("Error: 'coord' argument different of ['hm','img']")
		else:
			if sess is None:
				j = self.HG.Session.run(self.HG.joint_tensor_final, feed_dict = {self.HG.img: img})
			else:
				j = sess.run(self.HG.joint_tensor_final, feed_dict = {self.HG.img: img})
			if coord == 'hm':
				return j
			elif coord == 'img':
				return j * self.params['img_size'] / self.params['hm_size']
			else:
				print("Error: 'coord' argument different of ['hm','img']")
				
	def joints_pred_numpy(self, img, coord = 'hm', thresh = 0.2, sess = None):
		""" Create Tensor for joint position prediction
		NON TRAINABLE
		TO CALL AFTER GENERATING GRAPH
		Notes:
			Not more efficient than Numpy, prefer Numpy for such operation!
		"""
		if sess is None:
			hm = self.HG.Session.run(self.HG.pred_sigmoid , feed_dict = {self.HG.img: img})
		else:
			hm = sess.run(self.HG.pred_sigmoid , feed_dict = {self.HG.img: img})
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
	def plt_skeleton(self, img, tocopy = True, debug = False, sess = None):
		""" Given an Image, returns Image with plotted limbs (TF VERSION)
		Args:
			img 	: Source Image shape = (256,256,3)
			tocopy 	: (bool) False to write on source image / True to return a new array
			debug	: (bool) for testing puposes
			sess	: KEEP NONE
		"""
		joints = self.joints_pred(np.expand_dims(img, axis = 0), coord = 'img', debug = False, sess = sess)
		if tocopy:
			img = np.copy(img)
		for i in range(len(self.links)):
			position = self.givePixel(self.links[i]['link'],joints)
			cv2.line(img, tuple(position[0])[::-1], tuple(position[1])[::-1], self.links[i]['color'][::-1], thickness = 2)
		if tocopy:
			return img
		
	def plt_skeleton_numpy(self, img, tocopy = True, thresh = 0.2, sess = None, joint_plt = True):
		""" Given an Image, returns Image with plotted limbs (NUMPY VERSION)
		Args:
			img			: Source Image shape = (256,256,3)
			tocopy		: (bool) False to write on source image / True to return a new array
			thresh		: Joint Threshold
			sess		: KEEP NONE
			joint_plt	: (bool) True to plot joints (as circles)
		"""
		joints = self.joints_pred_numpy(np.expand_dims(img, axis = 0), coord = 'img', thresh = thresh, sess = sess)
		if tocopy:
			img = np.copy(img)*255
		for i in range(len(self.links)):
			l = self.links[i]['link']
			good_link = True
			for p in l:
				if np.array_equal(joints[p], [-1,-1]):
					good_link = False
			if good_link:
				position = self.givePixel(self.links[i]['link'],joints)
				cv2.line(img, tuple(position[0])[::-1], tuple(position[1])[::-1], self.links[i]['color'][::-1], thickness = 2)
		if joint_plt:
			for p in range(len(joints)):
				if not(np.array_equal(joints[p], [-1,-1])):
					cv2.circle(img, (int(joints[p,1]), int(joints[p,0])), radius = 3, color = self.color[p][::-1], thickness = -1)
		if tocopy:
			return img
		
	def pltSkeleton(self, img, thresh = 0.2, pltJ = True, pltL = True, tocopy = True, norm = True):
		""" Plot skeleton on Image (Single Detection)
		Args:
			img			: Input Image ( RGB IMAGE 256x256x3)
			thresh		: Joints Threshold
			pltL		: (bool) Plot Limbs
			tocopy		: (bool) Plot on imput image or return a copy
			norm		: (bool) Normalize input Image (DON'T MODIFY)
		Returns:
			img			: Copy of input image if 'tocopy'
		"""
		if tocopy:
			img = np.copy(img)
		if norm:
			img_hg = img / 255
		hg = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict = {self.HG.img: np.expand_dims(img_hg, axis = 0)})
		j = np.ones(shape = (self.params['num_joints'],2)) * -1
		for i in range(len(j)):
			idx = np.unravel_index( hg[0,:,:,i].argmax(), (64,64))
			if hg[0, idx[0], idx[1], i] > thresh:
					j[i] = np.asarray(idx) * 256 / 64
					if pltJ:
						cv2.circle(img, center = tuple(j[i].astype(np.int))[::-1], radius= 5, color= self.color[i][::-1], thickness= -1)
		if pltL:
			for i in range(len(self.links)):
					l = self.links[i]['link']
					good_link = True
					for p in l:
						if np.array_equal(j[p], [-1,-1]):
							good_link = False
					if good_link:
						pos = self.givePixel(l, j)
						cv2.line(img, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.links[i]['color'][::-1], thickness = 5)
		if tocopy:
			return img
		
	#------------------------Visualiazing Methods------------------------------
	def jointsToMat(self, joints):
		""" Given a 16 Joints Matrix, returns a 13 joints Matrix
		MPII Formalism to PennAction DeepDragon Formalism
		(Neck, Torso, Pelvis) erased
		Args:
			joints : Joints of shape (16,2) (HAS TO FOLLOW MPII FORMALISM)
		Returns:
			position : Joints of shape (13,2)
		"""
		position = np.zeros((13,2))
		position[0:6,:] = joints[0:6,:]
		position[6,:] = joints[9,:]
		position[7:,:] = joints[10:,:]
		position = np.reshape(position,(26,1),order = 'F')
		return position
	
	def computeErr(self, history, frame = 3):
		""" Given frames, compute error vector to project into new basis
		Args:
			history	: List of Joints detected in previous frames
			frame		: Number of previous frames to consider
		Returns:
			err			: Error vector
		"""
		err = np.zeros((13*2*(frame-1),1))
		rsho = 9
		lhip = 3
		eps = 0.00000001
		for i in range(frame-1):
			jf,js = history[-i-1], history[-i-2]
			xf,yf = jf[:13], jf[13:]
			xs,ys = js[:13], js[13:]
			x = xf / abs(xf[rsho] - xf[lhip] + eps) - xs / abs(xs[rsho] - xs[lhip] + eps)
			y = yf / abs(yf[rsho] - yf[lhip] + eps) - ys / abs(ys[rsho] - ys[lhip] + eps)
			err[26*(i):26*(i)+26,:] = np.concatenate([x,y],axis = 0)
		return err
	
	def errToJoints(self, err, frame, hFrame):
		""" Given an error vector and the frame considered, compute the joint's 
		location from the error
		Args:
			err			: Error Vector
			frame		: Current Frame Detected Joints
			hFrame		: Previous Frames Detected Joints
		Returns:
			joints		: Joints computed from error
		"""
		# Don't change Matrix computed with those joints to normalize
		lhip = 3
		rsho = 9
		# Epsilon to avoir ZeroDivision Exception
		eps = 0.00000001
		# Compute X and Y coordinates
		x = err[:,0:13]
		y = err[:,13:26]
		x = np.transpose(x)
		y = np.transpose(y)
		xh,yh = hFrame[0:13,:], hFrame[13:26,:]
		xf,yf = frame[0:13,:], frame[13:26,:]
		x = (x + (xh /np.abs(xh[rsho,:] - xh[lhip,:] + eps))) * np.abs(xf[rsho,:] - xf[lhip,:] + eps)
		y = (y + (yh /np.abs(yh[rsho,:] - yh[lhip,:] + eps))) * np.abs(yf[rsho,:] - yf[lhip,:] + eps)
		joints = np.concatenate([x,y], axis = 1).astype(np.int64)
		return joints
	
	def reconstructACPVideo(self, load = 'F:/Cours/DHPE/DHPE/hourglass_tiny/withyolo/p4frames.mat',n = 5):
		""" Single Person Detection with Principle Componenent Analysis (PCA)
		This method reconstructs joints given an error matrix (computed on MATLAB and available on GitHub)
		and try to use temporal information from previous frames to improve detection and reduce False Positive
		/!\Work In Progress on this model
		/!\Our PCA considers the current frames and the last 3 frames
		/!\WORKS ONLY ON 13 JOINTS FOR 16 JOINTS TRAINED MODELS
		Args:
			load	: MATLAB file to load (the eigenvectors matrix should be called P for convenience)
			n		: Numbers of Dimensions to consider
		"""
		# OpenCv Video Capture : 0 is Webcam
		cam = cv2.VideoCapture(self.src)
		frame = 0
		#/!\NOT USED YET
		#rsho = 9
		#lhip = 3
		# Keep History of previous frames
		jHistory = []
		# Load Eigenvectors MATLAB matrix
		P = scipy.io.loadmat(load)['P']
		while True:
			# Detection As Usual (See hpeWebcam)
			t = time()
			rec = np.zeros((500,500,3)).astype(np.uint8)
			plt = np.zeros((500,500,3)).astype(np.uint8)
			ret_val, img = cam.read()
			img = cv2.flip(img,1)
			img[:, self.cam_res[1]//2 - self.cam_res[0]//2:self.cam_res[1]//2 + self.cam_res[0]//2]
			img_hg = cv2.resize(img, (256,256))
			img_res = cv2.resize(img, (500,500))
			img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
			j = self.HG.Session.run(self.HG.joint_tensor_final, feed_dict = {self.HG.img: np.expand_dims(img_hg/255, axis = 0)})
			j = np.asarray(j * 500 / 64).astype(np.int)
			joints = self.jointsToMat(j)
			jHistory.append(joints)
			# When enough frames are available
			if frame > 4:
				#Compute Error
				err = np.transpose(self.computeErr(jHistory, frame = 4))
				# Dismiss useless frame
				jHistory.pop(0)
				# Project Error into New Basis (PCA new basis)
				projectedErr = np.dot(err, P)
				nComp = n
				# Reconstruct Error by dimensionality reduction
				recErr = np.dot(projectedErr[:,:nComp], np.transpose(P[:,:nComp]))
				# Compute joints position from reconstructed error
				newJ = self.errToJoints(recErr, jHistory[-1], jHistory[-2])
				for i in range(8):
				#for i in [4,5,6,7]:
					pos = self.givePixel(self.LINKS_ACP[i], newJ)
					cv2.line(img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.color[self.color_id_acp[i]][::-1], thickness = 8)
					cv2.line(rec, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.color[self.color_id_acp[i]][::-1], thickness = 8)
			for i in range(13):
			#for i in [8,9,11,12]:
					l = self.links[i]['link']
					pos = self.givePixel(l, j)
					cv2.line(img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.links[i]['color'][::-1], thickness = 5)
					cv2.line(plt, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.links[i]['color'][::-1], thickness = 5)
			fps = 1/(time()-t)
			cv2.putText(img_res, 'FPS: ' + str(fps)[:4], (60, 40), 2, 2, (0,0,0), thickness = 2)
			toplot = np.concatenate([rec, img_res, plt], axis = 1)
			cv2.imshow('stream', toplot)
			frame += 1
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				break
		cv2.destroyAllWindows()
		cam.release()

	
	
	def _singleDetection(self, plt_j = True, plt_l = True):
		""" /!\/!\DO NOT USE THIS FUNCTION/!\/!\
		/!\/!\METHOD FOR TEST PURPOSES ONLY/!\/!\
		PREFER HPE WEBCAM METHOD
		"""
		cam = cv2.VideoCapture(self.src)
		frame = 0
		position = np.zeros((32,1))
		while True:
			t = time()
			ret_val, img = cam.read()
			img = cv2.flip(img, 1)
			img[:, self.cam_res[1]//2 - self.cam_res[0]//2:self.cam_res[1]//2 + self.cam_res[0]//2]
			img_hg = cv2.resize(img, (256,256))
			img_res = cv2.resize(img, (400,400))
			cv2.imwrite('F:/Cours/DHPE/photos/acp/%04d.png' % (frame,), img_res)
			img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
			j = self.HG.Session.run(self.HG.joint_tensor_final, feed_dict = {self.HG.img: np.expand_dims(img_hg/255, axis = 0)})
			j = np.asarray(j * 400 / 64).astype(np.int)
			joints = self.jointsToMat(j)
			X = j.reshape((32,1),order = 'F')
			position = np.hstack((position, X))
			if plt_j:
				for i in range(len(j)):
					cv2.circle(img_res, center = tuple(j[i].astype(np.int))[::-1], radius= 5, color= self.color[i][::-1], thickness= -1)
			if plt_l:
				for i in range(len(self.links)):
					l = self.links[i]['link']
					pos = self.givePixel(l, j)
					cv2.line(img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.links[i]['color'][::-1], thickness = 5)
			fps = 1/(time()-t)
			cv2.putText(img_res, 'FPS: ' + str(fps)[:4], (60, 40), 2, 2, (0,0,0), thickness = 2)
			cv2.imshow('stream', img_res)
			frame += 1
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				scipy.io.savemat('acpTest2.mat', dict(history = position))
				break
		cv2.destroyAllWindows()
		cam.release()
			
			
	def hpeWebcam(self, thresh = 0.6, plt_j = True, plt_l = True, plt_hm = False, debug = True):
		""" Single Person Detector
		Args:
			thresh		: Threshold for joint plotting
			plt_j		: (bool) To plot joints (as circles)
			plt_l		: (bool) To plot links/limbs (as lines)
			plt_hm		: (bool) To plot heat map
		"""
		cam = cv2.VideoCapture(self.src)
		if debug:
			framerate = []
		while True:
			t = time()
			ret_val, img = cam.read()
			img = cv2.flip(img, 1)
			img[:, self.cam_res[1]//2 - self.cam_res[0]//2:self.cam_res[1]//2 + self.cam_res[0]//2]
			img_hg = cv2.resize(img, (256,256))
			img_res = cv2.resize(img, (800,800))
			#img_copy = np.copy(img_res)
			img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
			hg = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict = {self.HG.img: np.expand_dims(img_hg/255, axis = 0)})
			j = np.ones(shape = (self.params['num_joints'],2)) * -1
			if plt_hm:
				hm = np.sum(hg[0], axis = 2)
				hm = np.repeat(np.expand_dims(hm, axis = 2), 3, axis = 2)
				hm = cv2.resize(hm, (800,800))
				img_res = img_res / 255 + hm
			for i in range(len(j)):
				idx = np.unravel_index( hg[0,:,:,i].argmax(), (64,64))
				if hg[0, idx[0], idx[1], i] > thresh:
					j[i] = np.asarray(idx) * 800 / 64
					if plt_j:
						cv2.circle(img_res, center = tuple(j[i].astype(np.int))[::-1], radius= 5, color= self.color[i][::-1], thickness= -1)
			if plt_l:
				for i in range(len(self.links)):
					l = self.links[i]['link']
					good_link = True
					for p in l:
						if np.array_equal(j[p], [-1,-1]):
							good_link = False
					if good_link:
						pos = self.givePixel(l, j)
						cv2.line(img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.links[i]['color'][::-1], thickness = 5)
			fps = 1/(time()-t)
			if debug:
				framerate.append(fps)
			cv2.putText(img_res, 'FPS: ' + str(fps)[:4], (60, 40), 2, 2, (0,0,0), thickness = 2)
			cv2.imshow('stream', img_res)
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				break
		cv2.destroyAllWindows()
		cam.release()
		if debug:
			return framerate
	
	def mpe(self, j_thresh = 0.5, nms_thresh = 0.5, plt_l = True, plt_j = True, plt_b = True, img_size = 800,  skeleton = False):
		""" Multiple Person Estimation (WebCam usage)
		Args:
			j_thresh		: Joint Threshold
			nms_thresh	: Non Maxima Suppression Threshold
			plt_l			: (bool) Plot Limbs
			plt_j			: (bool) Plot Joints
			plt_b			: (bool) Plot Bounding Boxes
			img_size		: Resolution of Output Image
			skeleton		: (bool) Plot Skeleton alone next to image
		"""
		cam = cv2.VideoCapture(self.src)
		res = img_size
		fpsl = []
		while True:
			t = time()
			if skeleton:
				skel = np.zeros((img_size,img_size,3)).astype(np.uint8)
			ret_val, img = cam.read()
			img = cv2.flip(img,1)
			img[:, self.cam_res[1]//2 - self.cam_res[0]//2:self.cam_res[1]//2 + self.cam_res[0]//2]
			img_res = cv2.resize(img, (res,res))
			img_yolo = np.copy(img_res)
			img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
			results = self.detect(img_yolo)
			results_person = []
			for i in range(len(results)):
				if results[i][0] == 'person':
					results_person.append(results[i])
			results_person = self.nms(results_person, nms_thresh)
			for box in results_person:
				class_name = box[0]
				x = int(box[1])
				y = int(box[2])
				w = int(box[3] / 2)
				h = int(box[4] / 2)
				prob = box[5]
				bbox = np.asarray((max(0,x-w), max(0, y-h), min(img_size-1, x+w), min(img_size-1, y+h)))
				if plt_b:
					cv2.rectangle(img_res, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
					cv2.rectangle(img_res, (bbox[0], bbox[1] - 20),(bbox[2], bbox[1]), (125, 125, 125), -1)
					cv2.putText(img_res, class_name + ' : %.2f' % prob, (bbox[0] + 5, bbox[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
				maxl = np.max([w+0,h+0])
				lenghtarray = np.array([maxl/h, maxl/w])
				nbox = np.array([x-maxl, y-maxl, x+maxl, y+maxl])
				padding = np.abs(nbox-bbox).astype(np.int)
				img_person = np.copy(img_yolo[bbox[1]:bbox[3],bbox[0]:bbox[2],:])
				padd = np.array([[padding[1],padding[3]],[padding[0],padding[2]],[0,0]])
				img_person = np.pad(img_person, padd, mode = 'constant')
				img_person = cv2.resize(img_person, (256,256))
				hm = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict={self.HG.img: np.expand_dims(img_person/255, axis= 0)})
				j = -1*np.ones(shape = (self.params['num_joints'],2))
				joint = -1*np.ones(shape = (self.params['num_joints'],2))
				for i in range(self.params['num_joints']):
					idx = np.unravel_index(hm[0,:,:,i].argmax(), (64,64))
					if hm[0, idx[0], idx[1],i] > j_thresh:
						j[i] = idx
						joint[i] = np.asarray(np.array([y,x]) + ((j[i]-32)/32 * np.array([h,w])* lenghtarray )).astype(np.int)
						if plt_j:
							cv2.circle(img_res, tuple(joint[i].astype(np.int))[::-1], radius = 5, color = self.color[i][::-1], thickness = -1)
				if plt_l:
					for k in range(len(self.links)):
						l = self.links[k]['link']
						good_link = True
						for p in l:
							if np.array_equal(joint[p], [-1,-1]):
								good_link = False
						if good_link:
							cv2.line(img_res, tuple(joint[l[0]][::-1].astype(np.int)), tuple(joint[l[1]][::-1].astype(np.int)), self.links[k]['color'][::-1], thickness = 3)
							if skeleton:
								cv2.line(skel, tuple(joint[l[0]][::-1].astype(np.int)), tuple(joint[l[1]][::-1].astype(np.int)), self.links[k]['color'][::-1], thickness = 3)
			t_f = time()
			cv2.putText(img_res, 'FPS: ' + str(1/(t_f-t))[:4], (60, 40), 2, 2, (0,0,0), thickness = 2)
			fpsl.append(1/(t_f-t))
			if skeleton:
				img_res = np.concatenate([img_res,skel], axis = 1)
			cv2.imshow('stream', img_res)
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				return fpsl
				break
		cv2.destroyAllWindows()
		cam.release()
	
	
	
	
	
	# ---------------------------------MPE MULTITHREAD--------------------------
	def threadProcessing(self, box, img_size, j_thresh, plt_l, plt_j, plt_b):
		#if not coord.should_stop():
		class_name = box[0]
		x = int(box[1])
		y = int(box[2])
		w = int(box[3] / 2)
		h = int(box[4] / 2)
		prob = box[5]
		bbox = np.asarray((max(0,x-w), max(0, y-h), min(img_size-1, x+w), min(img_size-1, y+h)))
		if plt_b:
			cv2.rectangle(self.img_res, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
			cv2.rectangle(self.img_res, (bbox[0], bbox[1] - 20),(bbox[2], bbox[1]), (125, 125, 125), -1)
			cv2.putText(self.img_res, class_name + ' : %.2f' % prob, (bbox[0] + 5, bbox[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		maxl = np.max([w+0,h+0])
		lenghtarray = np.array([maxl/h, maxl/w])
		nbox = np.array([x-maxl, y-maxl, x+maxl, y+maxl])
		padding = np.abs(nbox-bbox).astype(np.int)
		img_person = np.copy(self.img_yolo[bbox[1]:bbox[3],bbox[0]:bbox[2],:])
		padd = np.array([[padding[1],padding[3]],[padding[0],padding[2]],[0,0]])
		img_person = np.pad(img_person, padd, mode = 'constant')
		img_person = cv2.resize(img_person, (256,256))
		hm = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict={self.HG.img: np.expand_dims(img_person/255, axis= 0)})
		j = -1*np.ones(shape = (self.params['num_joints'],2))
		joint = -1*np.ones(shape = (self.params['num_joints'],2))
		for i in range(self.params['num_joints']):
			idx = np.unravel_index(hm[0,:,:,i].argmax(), (64,64))
			if hm[0, idx[0], idx[1],i] > j_thresh:
				j[i] = idx
				joint[i] = np.asarray(np.array([y,x]) + ((j[i]-32)/32 * np.array([h,w])* lenghtarray )).astype(np.int)
				if plt_j:
					cv2.circle(self.img_res, tuple(joint[i].astype(np.int))[::-1], radius = 5, color = self.color[i][::-1], thickness = -1)
		if plt_l:
			for k in range(len(self.links)):
				l = self.links[k]['link']
				good_link = True
				for p in l:
					if np.array_equal(joint[p], [-1,-1]):
						good_link = False
				if good_link:
					cv2.line(self.img_res, tuple(joint[l[0]][::-1].astype(np.int)), tuple(joint[l[1]][::-1].astype(np.int)), self.links[k]['color'][::-1], thickness = 3)
	#	self.pprocessed += 1
	#	if self.pprocessed == self.pnum:
	#		coord.request_stop()
	
	
	def imgPrepare(self, img, res):
		img = cv2.flip(img,1)
		img[:, self.cam_res[1]//2 - self.cam_res[0]//2:self.cam_res[1]//2 + self.cam_res[0]//2]
		self.img_res = cv2.resize(img, (res,res))
		self.img_yolo = np.copy(self.img_res)
		self.img_yolo = cv2.cvtColor(self.img_yolo, cv2.COLOR_BGR2RGB)
	
	def yoloPrepare(self, nms):
		results = self.detect(self.img_yolo)
		results_person = []
		for i in range(len(results)):
			if results[i][0] == 'person':
				results_person.append(results[i])
		results_person = self.nms(results_person, nms)
		return results_person
	
	def mpeThread(self, jThresh = 0.2, nms = 0.5, plt_l = True, plt_j = True, plt_b = True, img_size = 800, wait = 0.07):
		cam = cv2.VideoCapture(self.src)
		res = img_size
		while True:
			#coord = tf.train.Coordinator()
			t = time()
			ret_val, img = cam.read()
			self.img_res, self.img_yolo = self.imgPrepare(img, res)
			results_person = self.yoloPrepare(nms)
			self.pnum = len(results_person)
			self.pprocessed = 0
			#workers = []
			for box in results_person:
				Thr = threading.Thread(target = self.threadProcessing, args = (box,img_size, jThresh, plt_l, plt_j, plt_b))
				Thr.start()
				#workers.append(Thr)
			sleep(wait)
			#coord.join(workers)
			#for i in workers:
			#	i.join()
			t_f = time()
			cv2.putText(self.img_res, 'FPS: ' + str(1/(t_f-t))[:4], (60, 40), 2, 2, (0,0,0), thickness = 2)
			cv2.imshow('stream', self.img_res)
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				break
		cv2.destroyAllWindows()
		cam.release()
	
	
	
	#---------------------------Conversion Method------------------------------
	
	def videoDetection(self, src = None, outName = None, codec = 'DIVX', j_thresh = 0.5, nms_thresh = 0.5, show = True, plt_j = True, plt_l = True, plt_b = True):
		""" Process Video with Pose Estimation
		Args:
			src				: Source (video path or 0 for webcam)
			outName		: outName (set name of output file, set to None if you don't want to save)
			codec			: Codec to use for video compression (see OpenCV documentation)
			j_thresh		: Joint Threshold
			nms_thresh	: Non Maxima Suppression Threshold
			show			: (bool) True to show the video
			plt_j			: (bool) Plot Body Joints as circles
			plt_l			: (bool) Plot Limbs
			plt_b			: (bool) Plot Bounding Boxes
		"""
		cam = cv2.VideoCapture(src)
		shape = np.asarray((cam.get(cv2.CAP_PROP_FRAME_HEIGHT),cam.get(cv2.CAP_PROP_FRAME_WIDTH))).astype(np.int)
		frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
		fps = cam.get(cv2.CAP_PROP_FPS)
		if outName != None:
			fourcc = cv2.VideoWriter_fourcc(*codec)
			outVid = cv2.VideoWriter( outName, fourcc, fps, tuple(shape.astype(np.int))[::-1], 1)
		cur_frame = 0
		startT = time()
		while (cur_frame < frames or frames == -1) and outVid.isOpened():
			RECONSTRUCT_IMG = np.zeros((shape[0].astype(np.int),shape[1].astype(np.int),3))
			ret_val, IMG_BASE = cam.read()
			WIDTH = shape[1].astype(np.int)
			HEIGHT = shape[0].astype(np.int)
			XC = WIDTH // 2
			YC = HEIGHT // 2
			if WIDTH > HEIGHT:
				top = np.copy(IMG_BASE[:,:XC - HEIGHT //2])
				bottom = np.copy(IMG_BASE[:,XC + HEIGHT //2:])
				img_square = np.copy(IMG_BASE[:,XC - HEIGHT //2:XC + HEIGHT //2])
			elif HEIGHT > WIDTH:
				top = np.copy(IMG_BASE[:YC - WIDTH //2])
				bottom = np.copy(IMG_BASE[YC + WIDTH //2:])
				img_square = np.copy(IMG_BASE[YC - WIDTH //2:YC + WIDTH //2])
			else:
				img_square = np.copy(IMG_BASE)
			img_od = cv2.cvtColor(np.copy(img_square), cv2.COLOR_BGR2RGB)
			shapeOd = img_od.shape
			results = self.detect(img_od)
			results_person = []
			for i in range(len(results)):
				if results[i][0] == 'person':
					results_person.append(results[i])
			results_person = self.nms(results_person, nms_thresh)
			for box in results_person:
				class_name = box[0]
				x = int(box[1])
				y = int(box[2])
				w = int(box[3] / 2)
				h = int(box[4] / 2)
				prob = box[5]
				bbox = np.asarray((max(0,x-w), max(0, y-h), min(shapeOd[1]-1, x+w), min(shapeOd[0]-1, y+h)))
				if plt_b:
					cv2.rectangle(img_square, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
					cv2.rectangle(img_square, (bbox[0], bbox[1] - 20),(bbox[2], bbox[1]), (125, 125, 125), -1)
					cv2.putText(img_square, class_name + ' : %.2f' % prob, (bbox[0] + 5, bbox[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
				maxl = np.max([w+0,h+0])
				lenghtarray = np.array([maxl/h, maxl/w])
				nbox = np.array([x-maxl, y-maxl, x+maxl, y+maxl])
				padding = np.abs(nbox-bbox).astype(np.int)
				img_person = np.copy(img_od[bbox[1]:bbox[3],bbox[0]:bbox[2],:])
				padd = np.array([[padding[1],padding[3]],[padding[0],padding[2]],[0,0]])
				img_person = np.pad(img_person, padd, mode = 'constant')
				img_person = cv2.resize(img_person, (256,256))
				hm = self.HG.Session.run(self.HG.pred_sigmoid, feed_dict={self.HG.img: np.expand_dims(img_person/255, axis= 0)})
				j = -1*np.ones(shape = (self.params['num_joints'],2))
				joint = -1*np.ones(shape = (self.params['num_joints'],2))
				for i in range(self.params['num_joints']):
					idx = np.unravel_index(hm[0,:,:,i].argmax(), (64,64))
					if hm[0, idx[0], idx[1],i] > j_thresh:
						j[i] = idx
						joint[i] = np.asarray(np.array([y,x]) + ((j[i]-32)/32 * np.array([h,w])* lenghtarray )).astype(np.int)
						if plt_j:
							cv2.circle(img_square, tuple(joint[i].astype(np.int))[::-1], radius = 5, color = self.color[i][::-1], thickness = -1)
				if plt_l:
					for k in range(len(self.links)):
						l = self.links[k]['link']
						good_link = True
						for p in l:
							if np.array_equal(joint[p], [-1,-1]):
								good_link = False
						if good_link:
							cv2.line(img_square, tuple(joint[l[0]][::-1].astype(np.int)), tuple(joint[l[1]][::-1].astype(np.int)), self.links[k]['color'][::-1], thickness = 3)
			if WIDTH > HEIGHT:
				RECONSTRUCT_IMG[:,:XC - HEIGHT //2] = top
				RECONSTRUCT_IMG[:,XC + HEIGHT //2:] = bottom
				RECONSTRUCT_IMG[:,XC - HEIGHT //2: XC + HEIGHT //2] = img_square
			elif HEIGHT < WIDTH:
				RECONSTRUCT_IMG[:YC - WIDTH //2] = top
				RECONSTRUCT_IMG[YC + WIDTH //2:] = bottom
				RECONSTRUCT_IMG[YC - WIDTH //2:YC + WIDTH //2] = img_square
			else:
				RECONSTRUCT_IMG = img_square.astype(np.uint8)
			RECONSTRUCT_IMG = RECONSTRUCT_IMG.astype(np.uint8)
			outVid.write(np.uint8(RECONSTRUCT_IMG))
			cur_frame = cur_frame + 1		
			if frames != -1:
				percent = ((cur_frame+1)/frames) * 100
				num = np.int(20*percent/100)
				tToEpoch = int((time() - startT) * (100 - percent)/(percent))
				sys.stdout.write('\r Processing: {0}>'.format("="*num) + "{0}>".format(" "*(20-num)) + '||' + str(percent)[:4] + '%' + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
				sys.stdout.flush()
			if show:
				cv2.imshow('stream', RECONSTRUCT_IMG)
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				cam.release()
				print(time() - startT)
				#if outName != None:
					#outVid.release()
		cv2.destroyAllWindows()
		cam.release()
		if outName != None:
			print(outVid.isOpened())
			outVid.release()
			print(outVid.isOpened())
		print(time() - startT)	

	
	 #-------------------------Benchmark Methods (PCK)-------------------------
	
	def pcki(self, joint_id, gtJ, prJ, idlh = 3, idrs = 12):
		""" Compute PCK accuracy on a given joint
		Args:
			joint_id	: Index of the joint considered
			gtJ			: Ground Truth Joint
			prJ			: Predicted Joint
			idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
			idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
		Returns:
			(float) NORMALIZED L2 ERROR
		"""
		return np.linalg.norm(gtJ[joint_id]-prJ[joint_id][::-1]) / np.linalg.norm(gtJ[idlh]-gtJ[idrs])
		
	def pck(self, weight, gtJ, prJ, gtJFull, boxL, idlh = 3, idrs = 12):
		""" Compute PCK accuracy for a sample
		Args:
			weight		: Index of the joint considered
			gtJFull	: Ground Truth (sampled on whole image)
			gtJ			: Ground Truth (sampled on reduced image)
			prJ			: Prediction
			boxL		: Box Lenght
			idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
			idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
		"""
		for i in range(len(weight)):
			if weight[i] == 1:
				self.ratio_pck.append(self.pcki(i, gtJ, prJ, idlh=idlh, idrs = idrs))
				self.ratio_pck_full.append(self.pcki(i, gtJFull, np.asarray(prJ / 255 * boxL)))
				self.pck_id.append(i)
	
	def compute_pck(self, datagen, idlh = 3, idrs = 12, testSet = None):
		""" Compute PCK on dataset
		Args:
			datagen	: (DataGenerator)
			idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
			idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
		"""
		datagen.pck_ready(idlh = idlh, idrs = idrs, testSet = testSet)
		self.ratio_pck = []
		self.ratio_pck_full = []
		self.pck_id = []
		samples = len(datagen.pck_samples)
		startT = time()
		for idx, sample in enumerate(datagen.pck_samples):
			percent = ((idx+1)/samples) * 100
			num = np.int(20*percent/100)
			tToEpoch = int((time() - startT) * (100 - percent)/(percent))
			sys.stdout.write('\r PCK : {0}>'.format("="*num) + "{0}>".format(" "*(20-num)) + '||' + str(percent)[:4] + '%' + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
			sys.stdout.flush()
			res = datagen.getSample(sample)
			if res != False:
				img, gtJoints, w, gtJFull, boxL = res
				prJoints = self.joints_pred_numpy(np.expand_dims(img/255, axis = 0), coord = 'img', thresh = 0)
				self.pck(w, gtJoints, prJoints, gtJFull, boxL, idlh=idlh, idrs = idrs)
		print('Done in ', int(time() - startT), 'sec.')
			
	#-------------------------Object Detector (YOLO)-------------------------
	
	# YOLO MODEL
	# Source : https://github.com/hizhangp/yolo_tensorflow
	# Author : Peng Zhang (https://github.com/hizhangp/)
	# yolo_init, iou, detect, detect_from_cvmat, interpret_output are methods
	# to the credit of Peng Zhang
	def yolo_init(self):
		"""YOLO Initializer
		Initialize the YOLO Model
		"""
		t = time()
		self.classes = cfg.CLASSES
		self.num_class = len(self.classes)
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.boxes_per_cell = cfg.BOXES_PER_CELL
		self.threshold = cfg.THRESHOLD
		self.iou_threshold = cfg.IOU_THRESHOLD
		self.boundary1 = self.cell_size * self.cell_size * self.num_class
		self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell
		with self.graph.as_default():
			self.net = YOLONet(is_training=False)
		print('YOLO created: ', time() - t, ' sec.')
	
	def restore_yolo(self, load = 'yolo_small.ckpt'):
		""" Restore Weights
		Args:
			load : File to load
		"""
		print('Loading YOLO...')
		print('Restoring weights from: ' + load)
		t = time()
		with self.graph.as_default():
			self.saver = tf.train.Saver(tf.contrib.framework.get_trainable_variables(scope='yolo'))
			self.saver.restore(self.HG.Session, load)
		print('Trained YOLO Loaded: ', time() - t, ' sec.')
	
	
	def iou(self, box1, box2):
		""" Intersection over Union (IoU)
		Args:
			box1 : Bounding Box
			box2 : Bounding Box
		Returns:
			IoU
		"""
		tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
		lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
		if tb < 0 or lr < 0:
			intersection = 0
		else:
			intersection = tb * lr
		return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
	
	def detect(self, img):
		""" Method for Object Detection
		Args:
			img			: Input Image (BGR Image)
		Returns:
			result		: List of Bounding Boxes
		"""
		img_h, img_w, _ = img.shape
		inputs = cv2.resize(img, (self.image_size, self.image_size))
		inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
		inputs = (inputs / 255.0) * 2.0 - 1.0
		inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))
		result = self.detect_from_cvmat(inputs)[0]
		for i in range(len(result)):
			result[i][1] *= (1.0 * img_w / self.image_size)
			result[i][2] *= (1.0 * img_h / self.image_size)
			result[i][3] *= (1.0 * img_w / self.image_size)
			result[i][4] *= (1.0 * img_h / self.image_size)
		return result
	
	def detect_from_cvmat(self, inputs):
		""" Runs detection on Session (TENSORFLOW RELATED)
		"""
		net_output = self.HG.Session.run(self.net.logits,feed_dict={self.net.images: inputs})
		results = []
		for i in range(net_output.shape[0]):
			results.append(self.interpret_output(net_output[i]))
		return results
	
	def interpret_output(self, output):
		""" Post Process the Output of the network
		Args:
			output : Network Prediction (Tensor)
		"""
		probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class))
		class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
		scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
		boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
		offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),[self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
		boxes[:, :, :, 0] += offset
		boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
		boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
		boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
		boxes *= self.image_size
		for i in range(self.boxes_per_cell):
			for j in range(self.num_class):
				probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])
		filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
		filter_mat_boxes = np.nonzero(filter_mat_probs)
		boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1], filter_mat_boxes[2]]
		probs_filtered = probs[filter_mat_probs]
		classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
		argsort = np.array(np.argsort(probs_filtered))[::-1]
		boxes_filtered = boxes_filtered[argsort]
		probs_filtered = probs_filtered[argsort]
		classes_num_filtered = classes_num_filtered[argsort]
		for i in range(len(boxes_filtered)):
			if probs_filtered[i] == 0:
				continue
			for j in range(i + 1, len(boxes_filtered)):
				if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
					probs_filtered[j] = 0.0
		filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
		boxes_filtered = boxes_filtered[filter_iou]
		probs_filtered = probs_filtered[filter_iou]
		classes_num_filtered = classes_num_filtered[filter_iou]
		result = []
		for i in range(len(boxes_filtered)):
			result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])
		return result
	
	def nms(self, boxes, overlapThresh):
		""" Non Maxima Suppression
		Args:
			boxes					: List of Bounding Boxes
			overlapThreshold	: Non Maxima Suppression Threshold
		Returns:
			ret						: List of processed Bounding Boxes
		"""
		if len(boxes) == 0:
			return []
		array = []
		for i in range(len(boxes)):
			array.append(boxes[i][1:5])
		array = np.array(array)
		pick = []
		x = array[:,0]
		y = array[:,1]
		w = array[:,2]
		h = array[:,3]
		x1 = x - w / 2
		x2 = x + w / 2
		y1 = y - h / 2
		y2 = y + h / 2
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)
		while len(idxs) > 0:
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
			xx1 = np.maximum(x1[i], x1[idxs[:last]])
			yy1 = np.maximum(y1[i], y1[idxs[:last]])
			xx2 = np.minimum(x2[i], x2[idxs[:last]])
			yy2 = np.minimum(y2[i], y2[idxs[:last]])
			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)
			overlap = (w * h) / area[idxs[:last]]
			idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
		ret = []
		for i in pick:
			ret.append(boxes[i])
		return ret
	
	def camera_detector(self, cap, wait=10, mirror = True):
		""" YOLO Webcam Detector
		Args:
			cap			: Video Capture (OpenCv Related)
			wait		: Time between frames
			mirror		: Apply mirror Effect
		"""
		while True:
			t = time()
			ret, frame = cap.read()
			if mirror:
				frame = cv2.flip(frame, 1)
			result = self.detect(frame)
			shapeOd = frame.shape
			for box in result:
				class_name = box[0]
				x = int(box[1])
				y = int(box[2])
				w = int(box[3] / 2)
				h = int(box[4] / 2)
				prob = box[5]
				bbox = np.asarray((max(0,x-w), max(0, y-h), min(shapeOd[1]-1, x+w), min(shapeOd[0]-1, y+h)))
				cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
				cv2.rectangle(frame, (bbox[0], bbox[1] - 20),(bbox[2], bbox[1]), (125, 125, 125), -1)
				cv2.putText(frame, class_name + ' : %.2f' % prob, (bbox[0] + 5, bbox[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
			fps = 1/(time() - t)
			cv2.putText(frame, str(fps)[:4] + ' fps', (20, 20), 2, 1, (255,255,255), thickness = 2)
			cv2.imshow('Camera', frame)
			ret, frame = cap.read()
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				cap.release()
		cv2.destroyAllWindows()
		cap.release()
		
	def person_detector(self, wait=10, mirror = True, plot = True):
		""" YOLO Webcam Detector
		Args:
			cap			: Video Capture (OpenCv Related)
			wait		: Time between frames
			mirror		: Apply mirror Effect
		"""
		cap = cv2.VideoCapture(0)
		while True:
			t = time()
			ret, frame = cap.read()
			if mirror:
				frame = cv2.flip(frame, 1)
			result_all = self.detect(frame)
			result = []
			for i in range(len(result_all)):
				if result_all[i][0] == 'person':
					result.append(result_all[i])
			shapeOd = frame.shape
			if plot:
				for box in result:
					class_name = box[0]
					x = int(box[1])
					y = int(box[2])
					w = int(box[3] / 2)
					h = int(box[4] / 2)
					prob = box[5]
					bbox = np.asarray((max(0,x-w), max(0, y-h), min(shapeOd[1]-1, x+w), min(shapeOd[0]-1, y+h)))
					cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
					cv2.rectangle(frame, (bbox[0], bbox[1] - 20),(bbox[2], bbox[1]), (125, 125, 125), -1)
					cv2.putText(frame, class_name + ' : %.2f' % prob, (bbox[0] + 5, bbox[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
			fps = 1/(time() - t)
			cv2.putText(frame, str(fps)[:4] + ' fps' + ' PinS:' + str(len(result)), (20, 20), 2, 1, (0,0,0), thickness = 2)
			cv2.imshow('Camera', frame)
			ret, frame = cap.read()
			if cv2.waitKey(1) == 27:
				print('Stream Ended')
				cv2.destroyAllWindows()
				cap.release()
		cv2.destroyAllWindows()
		cap.release()
		
if __name__ == '__main__':
	t = time()
	params = process_config('configTiny.cfg')
	predict = PredictProcessor(params)
	predict.color_palette()
	predict.LINKS_JOINTS()
	predict.model_init()
	predict.load_model(load = 'hg_refined_tiny_200')
	predict.yolo_init()
	predict.restore_yolo(load = 'YOLO_small.ckpt')
	predict._create_prediction_tensor()
	print('Done: ', time() - t, ' sec.')