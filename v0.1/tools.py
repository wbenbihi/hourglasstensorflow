# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:59:37 2017

@author: Walid Benbihi
@mail: w.benbihi (at) gmail.com
"""
import numpy as np
from skimage import transform

def fullTestSet(test, name, weight):
	"""
		Filter the test list to salect only the test images with no missing values
		args :
			test 	 : (list) List of testing set
			name 	 : (list) List of all images on the dataset
			weight : (list) List of weight (0 if missing value else 1)
	"""
	final = []
	for i in range(len(test)):
		if np.array_equal(weight[name.index([test[i]+'.jpg'])], np.ones((16,),np.uint8)):
			final.append(test[i])
	return final

def rotatehm(hm, angle):
	"""
		Given a heatMap, returns a rotated heatMap
		args : 
			hm 	 	: (numpy.array) heatMap
			angle : (int) Angle
	"""
	rot_hm = np.zeros((16,64,64))
	for i in range(16):
		rot_hm[i] = transform.rotate(hm[i],angle)
	return rot_hm


def modifyOutput(hm, stack):
	"""
		Given a heatMap, returns repeated stacked heatMaps as many time as 'stack'
		args :
			hm 	 	 : (numpy.array) heatMap
			stacks : (int) number of stacks
	"""
	hm_rolled = np.rollaxis(hm, 0, 3)
	hm_full = np.zeros((stack,1,64,64,16))
	for i in range(stack):
		hm_full[i,0,:,:,:] = hm_rolled
	return hm_full
