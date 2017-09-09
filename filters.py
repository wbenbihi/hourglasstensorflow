# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation
 
Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Sep  4 20:54:25 2017
 
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
import numpy as np
import cv2
class VideoFilters():
	"""
	Work In Progress
	"""
	
	def __init__(self):
		""" Initialize Filters
		"""
		self._sayan_params()
		self.num_filters = 1
		self.existing_filters = ['isSayan']
		self.activated_filters = [0]
		self.filter_func = ['plotSayan']
		
	def _sayan_params(self):
		"""
		Initialize Sayan parameters
		"""
		self.sayan_avg = np.array([40, 53, 167, 59, 53, 122, 29, 136, 39, 128])
		self.sayan_std = np.array([44, 16, 13, 15, 43, 18, 6, 17, 20, 18])
	
	
	def joint2Vect(self, pt1,pt2):
		""" Given 2 Joints (Points), returns the associated Vector
		"""
		vect = pt1 - pt2
		d = np.linalg.norm(vect)
		return vect/d
	
	def vect2angle(self, u,v):
		""" Given 2 vectors, returns the Angle between Vectors
		"""
		return abs(np.arccos(np.dot(u,v)))

	def angleAdir(self, joints):
		""" Given a list of Joints, returns Vectors and Angles of body
		"""
		j = joints.reshape((16,2), order = 'F')
		links = [(0,1),(1,2),(2,6),(3,6),(4,3),(5,4),(10,11),(11,12),(12,8),(13,8),(14,13),(15,14)]
		angles_l = [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(10,11)]
		vects = []
		angles = []
		for i in range(len(links)):
			vects.append(self.joint2Vect( j[links[i][0]], j[links[i][1]]))
		for i in range(len(angles_l)):
			angles.append(self.vect2angle(vects[angles_l[i][0]], vects[angles_l[i][1]]))
		return vects, np.degrees(angles)

	def isSayan(self, angles):
		""" Given an angle list, returns a boolean to state if the Sayan pose is detected
		"""
		say = True
		for i in range(10):
			if not(self.sayan_avg[i] - 1.5*self.sayan_std[i] < angles[i] < self.sayan_avg[i] + 1.5*self.sayan_std[i] ):
				say = False
		return say
	
	def plotSayan(self, img, j):
		"""
		WORK IN PROGRESS
		FUNCTION MAY CRASH
		"""
		hair = cv2.imread('./hair.png')
		ratio = hair.shape[1]/ hair.shape[0]
		mask = cv2.imread('./maskhair.png') /255
		dist_h_n = np.linalg.norm(j[9]-j[8])
		h = int(hair.shape[0] *20 / dist_h_n) 
		w =  int(h * ratio)
		hair = cv2.resize(hair, (w,h))
		mask = cv2.resize(mask, (w,h))
		padd = [[0,0],[0,0],[0,0]]
		if h / 2 > j[9][0]:
			padd[0][0] = int(h/2 - j[9][0])
		if h / 2 + j[9][0] > img.shape[0]:
			padd[0][1] = int(h/2 + j[9][0] - img.shape[0])
		if w / 2 > j[9][1]:
			padd[1][0] = int(w/2 - j[9][1])	
		if w / 2 + j[9][1] > img.shape[1]:
			padd[1][1] = int(w/2 + j[9][1] - img.shape[1])
		print('Frame')
		shape = img[int(j[9][0]) - int(np.ceil(h/2)) + padd[0][0]:int(j[9][0]) + int(np.ceil(h/2)) - padd[0][1] ,int(j[9][1]) - int(np.ceil(w/2)) + padd[1][0]:int(j[9][1]) + int(np.ceil(w/2)) - padd[1][1],:].shape
		print(shape)
		print(hair[padd[0][0]:mask.shape[0] -padd[0][1],padd[1][0]:mask.shape[1] -padd[1][1],:].shape)
		print(mask[padd[0][0]:mask.shape[0] -padd[0][1],padd[1][0]:mask.shape[1] -padd[1][1],:].shape)
		mask = mask[padd[0][0]:mask.shape[0] -padd[0][1],padd[1][0]:mask.shape[1] -padd[1][1],:]
		hair = hair[padd[0][0]:mask.shape[0] -padd[0][1],padd[1][0]:mask.shape[1] -padd[1][1],:]
		hair = cv2.resize(hair, (shape[1],shape[0]))
		mask = cv2.resize(mask, (shape[1],shape[0]))
		mask[mask != 1] = 0
		reco = img[int(j[9][0]) - int(np.ceil(h/2)) + padd[0][0]:int(j[9][0]) + int(np.ceil(h/2)) - padd[0][1] ,int(j[9][1]) - int(np.ceil(w/2)) + padd[1][0]:int(j[9][1]) + int(np.ceil(w/2)) - padd[1][1],:] * mask+ hair
		img[int(j[9][0]) - int(np.ceil(h/2)) + padd[0][0]:int(j[9][0]) + int(np.ceil(h/2)) - padd[0][1] ,int(j[9][1]) - int(np.ceil(w/2)) + padd[1][0]:int(j[9][1]) + int(np.ceil(w/2)) - padd[1][1],:] = reco
		img = img.astype(np.uint8)
		return img
		
		
		
		
	
	
	