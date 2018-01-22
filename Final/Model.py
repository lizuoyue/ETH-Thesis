import tensorflow as tf

def VGG16(img, reuse = None):
	"""
		img: [batch_size, height, width, num_channels]
	"""
	with tf.variable_scope('VGG16', reuse = reuse):
		conv1_1 = tf.layers.conv2d(
			inputs = img,
			filters = 64,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		conv1_2 = tf.layers.conv2d(
			inputs = conv1_1,
			filters = 64,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		pool1 = tf.layers.max_pooling2d(
			inputs = conv1_2,
			pool_size = (2, 2),
			strides = 2
		)
		conv2_1 = tf.layers.conv2d(
			inputs = pool1,
			filters = 128,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		conv2_2 = tf.layers.conv2d(
			inputs = conv2_1,
			filters = 128,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		pool2 = tf.layers.max_pooling2d(
			inputs = conv2_2,
			pool_size = (2, 2),
			strides = 2
		)
		conv3_1 = tf.layers.conv2d(
			inputs = pool2,
			filters = 256,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		conv3_2 = tf.layers.conv2d(
			inputs = conv3_1,
			filters = 256,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		conv3_3 = tf.layers.conv2d(
			inputs = conv3_2,
			filters = 256,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		pool3 = tf.layers.max_pooling2d(
			inputs = conv3_3,
			pool_size = (2, 2),
			strides = 2
		)
		conv4_1 = tf.layers.conv2d(
			inputs = pool3,
			filters = 512,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		conv4_2 = tf.layers.conv2d(
			inputs = conv4_1,
			filters = 512,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		conv4_3 = tf.layers.conv2d(
			inputs = conv4_2,
			filters = 512,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		pool4 = tf.layers.max_pooling2d(
			inputs = conv4_3,
			pool_size = (2, 2),
			strides = 2
		)
		conv5_1 = tf.layers.conv2d(
			inputs = pool4,
			filters = 512,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		conv5_2 = tf.layers.conv2d(
			inputs = conv5_1,
			filters = 512,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		conv5_3 = tf.layers.conv2d(
			inputs = conv5_2,
			filters = 512,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		return pool2, pool3, conv3_3, conv4_3, conv5_3


def PolygonFeature(vgg_result, reuse = None):
	pool2, pool3, _, conv4_3, conv5_3 = vgg_result
	with tf.variable_scope('PolygonFeature', reuse = reuse):
		part1_pool = tf.layers.max_pooling2d(
			inputs = pool2,
			pool_size = (2, 2),
			strides = 2
		)
		part1 = tf.layers.conv2d(
			inputs = part1_pool,
			filters = 128,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		part2 = tf.layers.conv2d(
			inputs = pool3,
			filters = 128,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		part3 = tf.layers.conv2d(
			inputs = conv4_3,
			filters = 128,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		part4_conv = tf.layers.conv2d(
			inputs = conv5_3,
			filters = 128,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		part4 = tf.image.resize_images(
			images = part4_conv,
			size = [part4_conv.shape[1] * 2, part4_conv.shape[2] * 2]
		)
		feature = tf.layers.conv2d(
			inputs = tf.concat([part1, part2, part3, part4], 3),
			filters = 128,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		return feature


def PyramidFeature(vgg_result, reuse = None):
	_, _, c2, c3, c4 = vgg_result
	with tf.variable_scope('PyramidFeature', reuse = reuse):
		p4 = tf.layers.conv2d(
			inputs = c4,
			filters = 256,
			kernel_size = (1, 1),
			padding = 'same',
			activation = tf.nn.relu
		)
		p3 = tf.layers.conv2d(
			inputs = c3,
			filters = 256,
			kernel_size = (1, 1),
			padding = 'same',
			activation = tf.nn.relu) + tf.image.resize_images(
			images = p4,
			size = [p4.shape[1] * 2, p4.shape[2] * 2]
		)
		p2 = tf.layers.conv2d(
			inputs = c2,
			filters = 256,
			kernel_size = (1, 1),
			padding = 'same',
			activation = tf.nn.relu) + tf.image.resize_images(
			images = p3,
			size = [p3.shape[1] * 2, p3.shape[2] * 2]
		)
		p2_conv = tf.layers.conv2d(
			inputs = p2,
			filters = 256,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		p3_conv = tf.layers.conv2d(
			inputs = p3,
			filters = 256,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		p4_conv = tf.layers.conv2d(
			inputs = p4,
			filters = 256,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		p5_pool = tf.layers.max_pooling2d(
			inputs = p4_conv,
			pool_size = (2, 2),
			strides = 2
		)
	return [p2_conv, p3_conv, p4_conv, p5_pool]


def RPNLayer(feature, anchors_per_pixel, reuse = None):
	"""
		feature: [batch_size, height, width, num_channels]
		anchors_per_pixel: scalar
	"""
	num_anchors = feature.shape[1] * feature.shape[2] * anchors_per_pixel
	with tf.variable_scope('RPNLayer', reuse = reuse):
		rpn_conv = tf.layers.conv2d(
			inputs = feature,
			filters = 512,
			kernel_size = (3, 3),
			padding = 'same',
			activation = tf.nn.relu
		)
		bbox_logit = tf.layers.conv2d(
			inputs = rpn_conv,
			filters = 2 * anchors_per_pixel,
			kernel_size = (1, 1),
			padding = 'valid',
			activation = None
		)
		bbox_delta = tf.layers.conv2d(
			inputs = rpn_conv,
			filters = 4 * anchors_per_pixel,
			kernel_size = (1, 1),
			padding = 'valid',
			activation = None
		)
	return (
		tf.reshape(bbox_logit, [-1, num_anchors, 2]),
		tf.reshape(bbox_delta,  [-1, num_anchors, 4]),
	)

