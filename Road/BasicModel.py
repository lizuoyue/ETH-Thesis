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
		return pool2, pool3, conv4_4, conv5_4

def SkipFeature(scope, vgg_result, reuse = None):
	"""
		vgg_result: see the return of VGG16 function defined above
	"""
	pool2, pool3, conv4_3, conv5_3 = vgg_result
	with tf.variable_scope(scope, reuse = reuse):
		inter1  = tf.layers.max_pooling2d(inputs = pool2  , pool_size = 2, strides = 2)																	 #  28
		part1   = tf.layers.conv2d       (inputs = inter1 , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv1' ) #  28
		part2   = tf.layers.conv2d       (inputs = pool3  , filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv2' ) #  28
		part3   = tf.layers.conv2d       (inputs = conv4_3, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv3' ) #  28
		inter4  = tf.layers.conv2d       (inputs = conv5_3, filters = 128, kernel_size = 3, padding = 'same', activation = tf.nn.relu, name = 'Sconv4' ) #  14
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




