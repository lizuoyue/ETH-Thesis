import random
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image, ImageDraw
import utility

def model(x, b, v, y12, y1, y2):
	input_layer = tf.reshape(x, [-1, 224, 224, 3])
	boundary_true = tf.reshape(b, [-1, 28, 28, 1])
	vertices_true = tf.reshape(v, [-1, 28, 28, 1])
	y12_true = tf.reshape(y12, [-1, 28, 28, 1])
	y1_true = tf.reshape(y1, [-1, 28, 28, 1])
	y2_true = tf.reshape(y2, [-1, 28, 28, 1])

	conv1_1 = tf.layers.conv2d(
		inputs = input_layer,
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
	pool4 = tf.layers.max_pooling2d(
		inputs = conv4_3,
		pool_size = (2, 2),
		strides = 2
	)
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
		size = [28, 28]
	)
	part = tf.concat([part1, part2, part3, part4], 3)
	feature = tf.layers.conv2d(
		inputs = part,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	###########################################
	boundary = tf.layers.conv2d(
		inputs = feature,
		filters = 1,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.sigmoid
	)
	vertices = tf.layers.conv2d(
		inputs = tf.concat([feature, boundary], 3),
		filters = 1,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.sigmoid
	)
	first_two = tf.layers.conv2d(
		inputs = tf.concat([feature, boundary, vertices], 3),
		filters = 1,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.sigmoid
	)
	first = tf.layers.conv2d(
		inputs = tf.concat([feature, boundary, vertices, first_two], 3),
		filters = 1,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.sigmoid
	)
	second = tf.layers.conv2d(
		inputs = tf.concat([feature, boundary, vertices, first_two], 3),
		filters = 1,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.sigmoid
	)

	loss = 0.0
	loss += tf.losses.log_loss(labels = boundary_true, predictions = boundary , weights = (boundary_true * 686 + 49))
	loss += tf.losses.log_loss(labels = vertices_true, predictions = vertices , weights = (vertices_true * 772 + 6 ))
	loss += tf.losses.log_loss(labels = y12_true     , predictions = first_two, weights = (y12_true      * 804 + 10))
	loss += tf.losses.log_loss(labels = y1_true      , predictions = first    , weights = (y1_true       * 782 + 1 ))
	loss += tf.losses.log_loss(labels = y2_true      , predictions = second   , weights = (y2_true       * 782 + 1 ))
	loss /= (5 * 784 / 100.0)
	return loss, boundary, vertices, first_two, first, second

if __name__ == '__main__':
	f = open('a.out', 'w')
	random.seed(3142857)
	x = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.int32)
	v = tf.placeholder(tf.int32)
	y12 = tf.placeholder(tf.int32)
	y1 = tf.placeholder(tf.int32)
	y2 = tf.placeholder(tf.int32)
	result = model(x, b, v, y12, y1, y2)
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)
	train = optimizer.minimize(result[0])
	init = tf.global_variables_initializer()
	batch = 16
	n_iter = 10000
	with tf.Session() as sess:
		sess.run(init)
		for i in range(n_iter):
			data = [utility.plotPolygon() for j in range(batch)]
			polygon = [item[0] for item in data]
			img = [item[1] for item in data]
			boundary_true = [item[2] for item in data]
			vertices_true = [item[3] for item in data]
			y12_true = [item[4] for item in data]
			y1_true = [item[5] for item in data]
			y2_true = [item[6] for item in data]

			feed_dict = {x: img, b: boundary_true, v: vertices_true, y12: y12_true, y1: y1_true, y2: y2_true}
			sess.run(train, feed_dict)
			loss, boundary, vertices, y12_p, y1_p, y2_p = sess.run(result, feed_dict)
			for j in range(batch):
				Image.fromarray(np.array(img[j] * 255.0, dtype = np.uint8)).save('./res/%d-a.png' % j)
				Image.fromarray(np.array(boundary[j,...,0] * 255.0, dtype = np.uint8)).resize((224,224),PIL.Image.BILINEAR).save('./res/%d-b.png' % j)
				Image.fromarray(np.array(vertices[j,...,0] * 255.0, dtype = np.uint8)).resize((224,224),PIL.Image.BILINEAR).save('./res/%d-c.png' % j)
				Image.fromarray(np.array(y12_p[j,...,0] * 255.0, dtype = np.uint8)).resize((224,224),PIL.Image.BILINEAR).save('./res/%d-d.png' % j)
				Image.fromarray(np.array(y1_p[j,...,0] * 255.0, dtype = np.uint8)).resize((224,224),PIL.Image.BILINEAR).save('./res/%d-e.png' % j)
				Image.fromarray(np.array(y2_p[j,...,0] * 255.0, dtype = np.uint8)).resize((224,224),PIL.Image.BILINEAR).save('./res/%d-f.png' % j)
			f.write('%.6lf\n' % loss)
			print('%.6lf' % loss)
			f.flush()

