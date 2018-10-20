import numpy as np
import tensorflow as tf
from Config import *
config = Config()

import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf

def VGG19(scope, img, reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		conv1_1 = tf.layers.conv2d       (inputs = img    , filters =  64, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv1_1') # 224
		conv1_2 = tf.layers.conv2d       (inputs = conv1_1, filters =  64, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv1_2') # 224
		pool1   = tf.layers.max_pooling2d(inputs = conv1_2, pool_size = 2, strides = 2)																	 # 112
		conv2_1 = tf.layers.conv2d       (inputs = pool1  , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv2_1') # 112
		conv2_2 = tf.layers.conv2d       (inputs = conv2_1, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv2_2') # 112
		pool2   = tf.layers.max_pooling2d(inputs = conv2_2, pool_size = 2, strides = 2)																	 #  56
		conv3_1 = tf.layers.conv2d       (inputs = pool2  , filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv3_1') #  56
		conv3_2 = tf.layers.conv2d       (inputs = conv3_1, filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv3_2') #  56
		conv3_3 = tf.layers.conv2d       (inputs = conv3_2, filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv3_3') #  56
		conv3_4 = tf.layers.conv2d       (inputs = conv3_3, filters = 256, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv3_4') #  56
		pool3   = tf.layers.max_pooling2d(inputs = conv3_4, pool_size = 2, strides = 2)																	 #  28
		conv4_1 = tf.layers.conv2d       (inputs = pool3  , filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv4_1') #  28
		conv4_2 = tf.layers.conv2d       (inputs = conv4_1, filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv4_2') #  28
		conv4_3 = tf.layers.conv2d       (inputs = conv4_2, filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv4_3') #  28
		conv4_4 = tf.layers.conv2d       (inputs = conv4_3, filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv4_4') #  28
		pool4   = tf.layers.max_pooling2d(inputs = conv4_4, pool_size = 2, strides = 2)																	 #  14
		conv5_1 = tf.layers.conv2d       (inputs = pool4  , filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv5_1') #  14
		conv5_2 = tf.layers.conv2d       (inputs = conv5_1, filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv5_2') #  14
		conv5_3 = tf.layers.conv2d       (inputs = conv5_2, filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv5_3') #  14
		conv5_4 = tf.layers.conv2d       (inputs = conv5_3, filters = 512, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'conv5_4') #  14
		return pool2, pool3, conv3_4, conv4_4, conv5_4

def SkipFeature(scope, vgg_result, crop_info, reuse = None):
	"""
		vgg_result: see the return of VGG16 function defined above
	"""
	pool2, pool3, _, conv4_4, conv5_4 = vgg_result
	idx = tf.cast(crop_info[:, 0], tf.int32)
	pool2   = tf.image.crop_and_resize(pool2  , crop_info[:, 1: 5], idx, config.PATCH_SIZE_4 )
	pool3   = tf.image.crop_and_resize(pool3  , crop_info[:, 1: 5], idx, config.PATCH_SIZE_8 )
	conv4_4 = tf.image.crop_and_resize(conv4_4, crop_info[:, 1: 5], idx, config.PATCH_SIZE_8 )
	conv5_4 = tf.image.crop_and_resize(conv5_4, crop_info[:, 1: 5], idx, config.PATCH_SIZE_16)
	with tf.variable_scope(scope, reuse = reuse):
		inter1  = tf.layers.max_pooling2d(inputs = pool2  , pool_size = 2, strides = 2)																	 #  28
		part1   = tf.layers.conv2d       (inputs = inter1 , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv1' ) #  28
		part2   = tf.layers.conv2d       (inputs = pool3  , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv2' ) #  28
		part3   = tf.layers.conv2d       (inputs = conv4_4, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv3' ) #  28
		inter4  = tf.layers.conv2d       (inputs = conv5_4, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv4' ) #  14
		part4   = tf.image.resize_images (images = inter4 , size = [inter4.shape[1] * 2, inter4.shape[2] * 2])											 #  28
		inter_f = tf.concat([part1, part2, part3, part4], axis = 3)																						 #  28
		feature = tf.layers.conv2d       (inputs = inter_f, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'SconvF' ) #  28
		return feature

def Mask(scope, feature, reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		bconv1  = tf.layers.conv2d       (inputs = feature, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Bconv1' )
		bconv2  = tf.layers.conv2d       (inputs = bconv1 , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Bconv2' )
		bconv3  = tf.layers.conv2d       (inputs = bconv2 , filters = 512, kernel_size = 1, padding = 'same', activation = tf.nn.relu, name = 'Bconv3' )
		bconv4  = tf.layers.conv2d       (inputs = bconv3 , filters =   2, kernel_size = 1, padding = 'same', activation = None      , name = 'Bconv4' )

		combine = tf.concat([feature, bconv4], -1)

		vconv1  = tf.layers.conv2d       (inputs = combine, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Vconv1' )
		vconv2  = tf.layers.conv2d       (inputs = vconv1 , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Vconv2' )
		vconv3  = tf.layers.conv2d       (inputs = vconv2 , filters = 512, kernel_size = 1, padding = 'same', activation = tf.nn.relu, name = 'Vconv3' )
		vconv4  = tf.layers.conv2d       (inputs = vconv3 , filters =   2, kernel_size = 1, padding = 'same', activation = None      , name = 'Vconv4' )

	return bconv4, vconv4

def PyramidAnchorFeature(vgg_result, reuse = None):
	"""
		vgg_result: see the return of VGG16 function defined above
	"""
	_, _, conv3_4, conv4_4, conv5_4 = vgg_result
	with tf.variable_scope('PyramidAnchorFeature', reuse = reuse):
		c5      = tf.layers.conv2d       (inputs = conv5_4, filters = 256, kernel_size = (1, 1), padding = 'same', activation = tf.nn.relu) #  16
		c4      = tf.layers.conv2d       (inputs = conv4_4, filters = 256, kernel_size = (1, 1), padding = 'same', activation = tf.nn.relu) #  32
		c4     += tf.image.resize_images (images = c5, size = [c5.shape[1] * 2, c5.shape[2] * 2])											#  32
		c3      = tf.layers.conv2d       (inputs = conv3_4, filters = 256, kernel_size = (1, 1), padding = 'same', activation = tf.nn.relu) #  64
		c3     += tf.image.resize_images (images = c4, size = [c4.shape[1] * 2, c4.shape[2] * 2])											#  64
		p3      = tf.layers.conv2d       (inputs = c3     , filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  64
		p4      = tf.layers.conv2d       (inputs = c4     , filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  32
		p5      = tf.layers.conv2d       (inputs = c5     , filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  16
		p6      = tf.layers.max_pooling2d(inputs = p5     , pool_size = (2, 2), strides = 2)												#   8
	return p3, p4, p5, p6

def SingleLayerFPN(feature, anchors_per_pixel, reuse = None):
	"""
		feature: [batch_size, height, width, num_channels]
		anchors_per_pixel: scalar
	"""
	a = anchors_per_pixel
	num_anchors = feature.shape[1] * feature.shape[2] * a
	with tf.variable_scope('SingleLayerFPN', reuse = reuse):
		inter   = tf.layers.conv2d       (inputs = feature, filters = 512, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #   ?
		logit   = tf.layers.conv2d       (inputs = inter  , filters = 2*a, kernel_size = (1, 1), padding = 'valid', activation = None)		#   ?
		delta   = tf.layers.conv2d       (inputs = inter  , filters = 4*a, kernel_size = (1, 1), padding = 'valid', activation = None)		#   ?
	return tf.reshape(logit, [-1, num_anchors, 2]), tf.reshape(delta,  [-1, num_anchors, 4])






def BottleneckV1(scope, img, out_ch, down, dilation, reuse = None):
	"""
		img   : tensor with shape [batch_size, height, width, num_channels]
		out_ch: number of output channels
		down  : bool
	"""
	assert(out_ch % 4 == 0)
	out_ch_4 = int(out_ch / 4)
	shortcut = (img.get_shape().as_list()[-1] != out_ch)
	stride = 1 + 1 * down
	with tf.variable_scope(scope, reuse = reuse):
		if shortcut:
			sc = tf.layers.conv2d(inputs = img   , filters = out_ch  , kernel_size = 1, padding = 'same', use_bias = False, strides = stride)
			sc = tf.layers.batch_normalization(sc)
		else:
			sc = img
		conv_1 = tf.layers.conv2d(inputs = img   , filters = out_ch_4, kernel_size = 1, padding = 'same', use_bias = False, strides = 1)
		conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1))
		conv_2 = tf.layers.conv2d(inputs = conv_1, filters = out_ch_4, kernel_size = 3, padding = 'same', use_bias = False, strides = stride, dilation_rate = dilation)
		conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2))
		conv_3 = tf.layers.conv2d(inputs = conv_2, filters = out_ch  , kernel_size = 1, padding = 'same', use_bias = False, strides = 1)
		conv_3 = tf.layers.batch_normalization(conv_3)
	return tf.nn.relu(conv_3 + sc)

def ResNetV1(scope, img, li_nb, reuse = None):
	"""
		img: tensor with shape [batch_size, height, width, num_channels]
		li_nb: list of numbers of bottlenecks for each of the 4 blocks
	"""
	assert(type(li_nb) == list and len(li_nb) == 4)	
	with tf.variable_scope(scope, reuse = reuse):
		with tf.variable_scope('Conv1_1', reuse = reuse):
			conv1 = tf.layers.conv2d(inputs = img, filters =  64, kernel_size = 7, strides = 2, padding = 'same', use_bias = False)
			conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))		
		conv2 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 3, strides = 2, padding = 'same')
		for i in range(li_nb[0]):
			conv2 = BottleneckV1('Conv2_%d' % (i + 1), conv2, 256 , False, 1, reuse) 
		conv3 = conv2
		for i in range(li_nb[1]):
			conv3 = BottleneckV1('Conv3_%d' % (i + 1), conv3, 512 , i == 0, 1, reuse)
		conv4 = conv3
		for i in range(li_nb[2]):
			conv4 = BottleneckV1('Conv4_%d' % (i + 1), conv4, 1024, i == 0, 1, reuse)
		conv5 = conv4
		for i in range(li_nb[3]):
			conv5 = BottleneckV1('Conv5_%d' % (i + 1), conv5, 2048, i == 0, 1, reuse)
	return conv1, conv2, conv3, conv4, conv5

def ResNetV1_50(scope, img, reuse = None):
	return ResNetV1(scope, img, [3, 4,  6, 3], reuse)

def ResNetV1_101(scope, img, reuse = None):
	return ResNetV1(scope, img, [3, 4, 23, 3], reuse)

def ResNetV1_152(scope, img, reuse = None):
	return ResNetV1(scope, img, [3, 8, 36, 3], reuse)

def PyramidAnchorFeatureResNet(scope, resnet_result, reuse = None):
	_, conv2, conv3, conv4, conv5 = resnet_result
	with tf.variable_scope(scope, reuse = reuse):
		c5  = tf.layers.conv2d      (inputs = conv5, filters = 256, kernel_size = (1, 1), padding = 'same', activation = tf.nn.relu) #   7
		c4  = tf.layers.conv2d      (inputs = conv4, filters = 256, kernel_size = (1, 1), padding = 'same', activation = tf.nn.relu) #  14
		c4 += tf.image.resize_images(images = c5, size = [c5.shape[1] * 2, c5.shape[2] * 2])										 #  14
		c3  = tf.layers.conv2d      (inputs = conv3, filters = 256, kernel_size = (1, 1), padding = 'same', activation = tf.nn.relu) #  28
		c3 += tf.image.resize_images(images = c4, size = [c4.shape[1] * 2, c4.shape[2] * 2])										 #  28
		c2  = tf.layers.conv2d      (inputs = conv2, filters = 256, kernel_size = (1, 1), padding = 'same', activation = tf.nn.relu) #  56
		c2 += tf.image.resize_images(images = c3, size = [c3.shape[1] * 2, c3.shape[2] * 2])										 #  56
		p2  = tf.layers.conv2d      (inputs = c2   , filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  56
		p3  = tf.layers.conv2d      (inputs = c3   , filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  28
		p4  = tf.layers.conv2d      (inputs = c4   , filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #  14
		p5  = tf.layers.conv2d      (inputs = c5   , filters = 256, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu) #   7
	return p2, p3, p4, p5

def SkipFeatureResNet(scope, resnet_result, crop_info, reuse = None):
	_, conv2, conv3, conv4, conv5 = resnet_result
	idx = tf.cast(crop_info[:, 0], tf.int32)
	conv2 = tf.image.crop_and_resize(conv2, crop_info[:, 1: 5], idx, config.PATCH_SIZE_4 )
	conv3 = tf.image.crop_and_resize(conv3, crop_info[:, 1: 5], idx, config.PATCH_SIZE_8 )
	conv4 = tf.image.crop_and_resize(conv4, crop_info[:, 1: 5], idx, config.PATCH_SIZE_16)
	conv5 = tf.image.crop_and_resize(conv5, crop_info[:, 1: 5], idx, config.PATCH_SIZE_32)
	with tf.variable_scope(scope, reuse = reuse):
		inter1 = tf.layers.max_pooling2d(inputs = conv2 , pool_size = 2, strides = 2)																   #  28
		part1  = tf.layers.conv2d       (inputs = inter1, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv1' ) #  28
		
		part2  = tf.layers.conv2d       (inputs = conv3 , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv2' ) #  28
		
		inter3 = tf.image.resize_images (images = conv4 , size = [conv4.shape[1] * 2, conv4.shape[2] * 2])
		part3  = tf.layers.conv2d       (inputs = inter3, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv3' ) #  28

		inter4 = tf.image.resize_images (images = conv5 , size = [conv5.shape[1] * 4, conv5.shape[2] * 4])	
		part4  = tf.layers.conv2d       (inputs = inter4, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv4' ) #  14

		inter  = tf.concat([part1, part2, part3, part4], axis = 3)																						 #  28
		feature = tf.layers.conv2d      (inputs = inter , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'SconvF' ) #  28
		return feature

