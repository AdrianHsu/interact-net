# coding: utf-8

import os
import numpy as np
import glob
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
import cv2
import matplotlib.pyplot as plt
import random
# import vgg

LEARNING_RATE = 1e-3
EPOCHES = 100
CROP_SIZE = 224
NUM_CHANNELS = 3 # RGB
BATCH_SIZE = 100 # take 100 photos
VGG_MEAN = np.array([104., 117., 123.], dtype='float32')

ROOT_PATH = '/Users/adrianhsu/Desktop/interact-net/'

images = open(ROOT_PATH + 'train.txt', 'r')
for img_path in images:
	ALL_COLOR = 1 # 0 is grey scale
	img_path = img_path.split('\n')[0] # remove '\n'
	frame = cv2.imread(img_path, ALL_COLOR)
	x = frame.shape[0]
	y = frame.shape[1]
	# print('original shape: ' + str(x) +  ', ' + str(y))
	dmin = min(x, y)
	ratio = 256.0/dmin
	frame = cv2.resize(frame, None, fx=ratio, fy=ratio)

	x = frame.shape[0]
	y = frame.shape[1]
	# print('modified shape: ' + str(frame.shape[0]) +  ', ' + str(frame.shape[1]))

	x = round((x - CROP_SIZE) / 2)
	y = round((y - CROP_SIZE) / 2)

	crop = frame[x : x + CROP_SIZE, y : y + CROP_SIZE, :]
	# cv2.imshow('image', crop)
	# cv2.waitKey()
	crop -= VGG_MEAN
	crop = crop.flatten()
