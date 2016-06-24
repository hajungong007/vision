#!/usr/bin/env python
import os
import sys
import copy
import rospy
import rospkg
import roslib
import random

rospack = rospkg.RosPack()   
prefix = rospack.get_path('deep_cnn_object_detection')
dir_path = prefix + '/data/images/'

objects = sorted(os.listdir(dir_path))
i = 1
file_train = open(dir_path + 'train_list.txt', 'w')
file_test = open(dir_path + 'test_list.txt', 'w')
file_words = open(dir_path + 'words_list.txt', 'w')
for object in objects:
    object_path = dir_path + object + '/'    
    if (os.path.isdir(object_path)):
	file_words.write(str(i) + ' ' + object + '\n')
	images = sorted(os.listdir(object_path))
	train_images = random.sample(images, len(images) * 7 / 10)
	test_images = [x for x in images if x not in train_images]
	for image in train_images:
	    image_path = object_path + image
	    file_train.write(image_path + ' ' + str(i) + '\n')
	for image in test_images:
	    image_path = object_path + image
	    file_test.write(image_path + ' ' + str(i) + '\n')
	i += 1
