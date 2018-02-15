# coding: utf-8

import os
import numpy as np
import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import matplotlib.pyplot as plt
import random


LEARNING_RATE = 1e-3
EPOCHES = 100
CROP_SIZE = 224
NUM_CHANNELS = 3 # RGB
BATCH_SIZE = 100 # take 100 photos
# VGG_MEAN = [103.939, 116.779, 123.68] # rgb
VGG_MEAN = [123.68, 116.78, 103.94] # bgr

ROOT_PATH = '/Users/adrianhsu/Desktop/interact-net/'


def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image = tf.cast(image_decoded, tf.float32)

  smallest_side = 256.0
  height, width = tf.shape(image)[0], tf.shape(image)[1]
  height = tf.to_float(height)
  width = tf.to_float(width)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)

  resized_image = tf.image.resize_images(image, [new_height, new_width]) 
  return resized_image, label
def training_preprocess(image, label):
  crop_image = tf.random_crop(image, [224, 224, 3])
  flip_image = tf.image.random_flip_left_right(crop_image)

  means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
  centered_image = flip_image - means

  return centered_image, label

train_log_dir = 'ROOT_PATH' + 'logs'
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

images = []
labels = []
img_path = '/Users/adrianhsu/Desktop/v-coco/coco/images/train2017/000000000165.jpg'
train_filenames = [img_path, img_path]
train_labels = [1, 2]

train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.map(_parse_function,
    num_threads=4, output_buffer_size=32)
train_dataset = train_dataset.map(training_preprocess,
            num_threads=4, output_buffer_size=32)
train_dataset = train_dataset.shuffle(buffer_size=10000)

# vgg = slim.nets.vgg.vgg_16(images, is_training=True)