import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import utility

def model(image, boundary_true, vertices_true):
	input_layer = tf.reshape(x, [-1, 224, 224, 3])
	label = tf.reshape(y, [-1])

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
	#######
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
	loss = 0.0
	loss += tf.losses.log_loss(labels = boundary_true, prediction = boundary)
	loss += tf.losses.log_loss(labels = vertices_true, prediction = vertices)
	loss /= 2
	return loss, boundary, vertices

if __name__ == '__main__':
	f = open('a.out', 'w')
	random.seed(3142857)
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.int32)
	res = model(x, y, 3)
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)
	train = optimizer.minimize(res[2])
	init = tf.global_variables_initializer()
	batch = 30
	n_iter = 1000
	with tf.Session() as sess:
		sess.run(init)
		for i in range(n_iter):
			xy_train = []
			for j in range(int(batch / 3)):
				xy_train.append((generatePolygon(polygon_type = 'tri'), 0))
				xy_train.append((generatePolygon(polygon_type = 'qua'), 1))
				xy_train.append((generatePolygon(polygon_type = 'ell'), 2))
			random.shuffle(xy_train)
			x_train = [item[0] for item in xy_train]
			y_train = [item[1] for item in xy_train]
			# for j in range(batch):
			# 	plt.imshow(x_train[j])
			# 	plt.show()
			f.write(str(np.array(y_train)) + '\n')
			f.flush()
			feed_dict = {x: x_train, y: y_train}
			sess.run(train, feed_dict)
			pred_class, pred_prob, loss = sess.run(res, feed_dict)
			f.write(str(pred_class) + '\n')
			f.flush()
			acc = sum(pred_class == y_train) / float(batch)
			f.write('%d, %.8lf, %.2lf\n' % (i, loss, acc))
			f.flush()
