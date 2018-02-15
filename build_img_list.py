# coding: utf-8

import os
import numpy as np
import glob


ROOT_PATH = '/Users/adrianhsu/Desktop/v-coco/'
DATA_PATH = ROOT_PATH + 'coco/images/'
TRAIN_PATH = DATA_PATH + 'train2017/'
TEST_PATH = DATA_PATH + 'test2017/'
VAL_PATH = DATA_PATH + 'val2017/'

IDS_PATH = ROOT_PATH + 'data/splits/'
ids = open(IDS_PATH + 'vcoco_all.ids', 'r')
id_train_list = []
for i in ids.readlines():
	id_train_list.append(int(i))

ids = open(IDS_PATH + 'vcoco_test.ids', 'r')
id_test_list = []
for i in ids.readlines():
	id_test_list.append(int(i))

ids = open(IDS_PATH + 'vcoco_val.ids', 'r')
id_val_list = []
for i in ids.readlines():
	id_val_list.append(int(i))

print(len(id_train_list)) # 10346
print(len(id_test_list)) # 4946
print(len(id_val_list)) # 2867


file_train = open('./train.txt', 'w')
file_test = open('./test.txt', 'w')
file_val = open('./val.txt', 'w')


mylist = glob.glob(TRAIN_PATH + '*.jpg')
mylist += glob.glob(TEST_PATH + '*.jpg')
mylist += glob.glob(VAL_PATH + '*.jpg')
print(len(mylist)) #163957

for e in mylist:
	# print(e)
	arr = e.split('/')
	id = int(arr[-1].split('.')[0])
	if id in id_train_list:
		file_train.write(e + '\n')
	if id in id_test_list:
		file_test.write(e + '\n')
	if id in id_val_list:
		file_val.write(e + '\n')


