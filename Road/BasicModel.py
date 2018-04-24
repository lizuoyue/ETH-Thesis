import numpy as np
import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf

def VGG19(img, scope, reuse = None):
	"""
		img: [batch_size, height, width, num_channels]
	"""
	with tf.variable_scope(scope, reuse = reuse):
		conv1_1 = tf.layers.conv2d       (inputs = img    , filters =  64, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) # 256
		conv1_2 = tf.layers.conv2d       (inputs = conv1_1, filters =  64, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) # 256
		pool1   = tf.layers.max_pooling2d(inputs = conv1_2, pool_size = (2, 2), strides = 2)												# 128
		conv2_1 = tf.layers.conv2d       (inputs = pool1  , filters = 128, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) # 128
		conv2_2 = tf.layers.conv2d       (inputs = conv2_1, filters = 128, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) # 128
		pool2   = tf.layers.max_pooling2d(inputs = conv2_2, pool_size = (2, 2), strides = 2)												#  64
		conv3_1 = tf.layers.conv2d       (inputs = pool2  , filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  64
		conv3_2 = tf.layers.conv2d       (inputs = conv3_1, filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  64
		conv3_3 = tf.layers.conv2d       (inputs = conv3_2, filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  64
		conv3_4 = tf.layers.conv2d       (inputs = conv3_3, filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  64
		pool3   = tf.layers.max_pooling2d(inputs = conv3_4, pool_size = (2, 2), strides = 2)												#  32
		conv4_1 = tf.layers.conv2d       (inputs = pool3  , filters = 512, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  32
		conv4_2 = tf.layers.conv2d       (inputs = conv4_1, filters = 512, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  32
		conv4_3 = tf.layers.conv2d       (inputs = conv4_2, filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  32
		conv4_4 = tf.layers.conv2d       (inputs = conv4_3, filters = 128, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  32
		return conv4_4

def FirstStageBranch(feature, num, scope, last_active = None, reuse = None):
	"""
		feature: [batch_size, height, width, num_channels]
	"""
	with tf.variable_scope(scope, reuse = reuse):
		conv1   = tf.layers.conv2d       (inputs = feature, filters = 128, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu)
		conv2   = tf.layers.conv2d       (inputs = conv1  , filters = 128, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu)
		conv3   = tf.layers.conv2d       (inputs = conv2  , filters = 128, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu)
		conv4   = tf.layers.conv2d       (inputs = conv3  , filters = 512, kernel_size = (1, 1), padding = 'valid', activation = last_active)
		conv5   = tf.layers.conv2d       (inputs = conv4  , filters = num, kernel_size = (1, 1), padding = 'valid', activation = last_active)
		return conv5

def StageBranch(feature, num, scope, last_active = None, reuse = None):
	"""
		feature: [batch_size, height, width, num_channels]
	"""
	with tf.variable_scope(scope, reuse = reuse):
		conv1   = tf.layers.conv2d       (inputs = feature, filters = 128, kernel_size = (7, 7), padding = 'same', activation = tf.nn.relu)
		conv2   = tf.layers.conv2d       (inputs = conv1  , filters = 128, kernel_size = (7, 7), padding = 'same', activation = tf.nn.relu)
		conv3   = tf.layers.conv2d       (inputs = conv2  , filters = 128, kernel_size = (7, 7), padding = 'same', activation = tf.nn.relu)
		conv4   = tf.layers.conv2d       (inputs = conv3  , filters = 128, kernel_size = (7, 7), padding = 'same', activation = tf.nn.relu)
		conv5   = tf.layers.conv2d       (inputs = conv4  , filters = 128, kernel_size = (7, 7), padding = 'same', activation = tf.nn.relu)
		conv6   = tf.layers.conv2d       (inputs = conv5  , filters = 128, kernel_size = (1, 1), padding = 'valid', activation = last_active)
		conv7   = tf.layers.conv2d       (inputs = conv6  , filters = num, kernel_size = (1, 1), padding = 'valid', activation = last_active)
		return conv7

def LossL2(pred_map, gt_map, mask):
	loss = tf.multiply(tf.reduce_sum(tf.square(pred_map - gt_map), axis = -1), mask)
	return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def Model(mode, img, gt = None, msk = None, num_stage = 6):
	if mode == 'train':
		reuse = None
	else:
		reuse = True
	feature = VGG19(img, 'VGG19', reuse = reuse)
	s = [FirstStageBranch(feature, 1, 's1', last_active = tf.sigmoid, reuse = reuse)]
	l = [FirstStageBranch(feature, 2, 'l1', reuse = reuse)]
	for i in range(1, num_stage):
		stage_input = tf.concat([feature, s[-1], l[-1]], axis = -1)
		s.append(StageBranch(stage_input, 1, 's%d' % (i + 1), last_active = tf.sigmoid, reuse = reuse))
		l.append(StageBranch(stage_input, 2, 'l%d' % (i + 1), reuse = reuse))
	if mode == 'train':
		loss_s, loss_l = 0, 0
		gt_s, gt_l = gt[..., 0: 1], gt[..., 1: 3]
		for ss in s:
			loss_s += LossL2(ss, gt_s, msk)
		for ll in l:
			loss_l += LossL2(ll, gt_l, msk)
		return loss_s, loss_l
	else:
		return s[-1], l[-1]


