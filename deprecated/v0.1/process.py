# -*- coding: utf-8 -*-
"""
Created on Sun May 28 03:28:07 2017

@author: Walid Benbihi
@mail : w.benbihi (at) gmail.com
"""

###########################################
#              PROCESS DATA
###########################################

# Import
import csv
import os
import numpy as np
import params
import tools



folder = params.img_dir
path = params.data_dir
summarytrain = params.train_dir
summarytest = params.test_dir
arraypath = params.processed_dir


def openCsv(path, filename):
	"""
		Open a .csv file and convert it as a numpy array
		WARNING: csv file should contain only numbers
		DATASET : MPII only
		args : 
			path : (str) path to folder
			filename: (str) name of .csv file
		return : 
			np.array
	"""
	return np.array(np.genfromtxt(path + filename, delimiter = ','), np.float32)[0:]


# Creation of Joints Position Arrays:

joint00 = openCsv(path, '00_rAnckle.csv')
joint01 = openCsv(path, '01_rKnee.csv')
joint02 = openCsv(path, '02_rHip.csv')
joint03 = openCsv(path, '03_lAnckle.csv')
joint04 = openCsv(path, '04_lKnee.csv')
joint05 = openCsv(path, '05_lHip.csv')
joint06 = openCsv(path, '06_pelvis.csv')
joint07 = openCsv(path, '07_thorax.csv')
joint08 = openCsv(path, '08_neck.csv')
joint09 = openCsv(path, '09_head.csv')
joint10 = openCsv(path, '10_rWrist.csv')
joint11 = openCsv(path, '11_rElbow.csv')
joint12 = openCsv(path, '12_rShoulder.csv')
joint13 = openCsv(path, '13_lWrist.csv')
joint14 = openCsv(path, '14_lElbow.csv')
joint15 = openCsv(path, '15_lShoulder.csv')

def getPosition(index):
	"""
		Given an Image Index, returns an array of body joints position
		DATASET : MPII only
		args :
			index : (str) index of the image
		return :
			np.array : size 16x2 (body joints x coordinates)
	"""
	pos = np.zeros((16,2))
	pos[0,:] = joint00[index,:]
	pos[1,:] = joint01[index,:]
	pos[2,:] = joint02[index,:]
	pos[3,:] = joint03[index,:]
	pos[4,:] = joint04[index,:]
	pos[5,:] = joint05[index,:]
	pos[6,:] = joint06[index,:]
	pos[7,:] = joint07[index,:]
	pos[8,:] = joint08[index,:]
	pos[9,:] = joint09[index,:]
	pos[10,:] = joint10[index,:]
	pos[11,:] = joint11[index,:]
	pos[12,:] = joint12[index,:]
	pos[13,:] = joint13[index,:]
	pos[14,:] = joint14[index,:]
	pos[15,:] = joint15[index,:]
	return pos


def cleanList(L):
	"""
		Given a list L returns a list without duplicates
		DATASET : MPII only
		args:
			L : (list(str))
		return:
			list
	"""
	clean = []
	for i in range(len(L)):
		if not(L[i][:9] in clean):
			clean.append(L[i][9])
	return clean


def toTrainList():
	"""
		Procedure to generate intel about dataset
		DATASET : MPII only
		return :
			labels  : (list) list of labels (0/1) between testing and training data
			imList  : (list) list of all numpy arrays in the 'arraypath'
			name    : (list) list of all images indexed on the dataset
			toTrain : (list) list of all images' index with label 1 and in the 'arraypath' directoty			
	"""
	labels = np.array(np.genfromtxt( path + 'tset.csv', delimiter = ','), np.uint8)[1:]
	with open(path + 'name.csv', 'r') as file:
		reader = csv.reader(file)
		name = list(reader)[1:]
		imList = os.listdir(arraypath)
		toTrain = []
		for i in range(len(imList)):
			if labels[name.index(imList[i])]:
				toTrain.append(imList[i])
		return labels, imList, name, toTrain
	
	
