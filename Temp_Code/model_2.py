import random
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image, ImageDraw
import BuildingImage

def vgg16(x):
	conv1_1 = tf.layers.conv2d(
		inputs = x,
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
	comb = tf.concat([part1, part2, part3, part4], 3)
	feature = tf.layers.conv2d(
		inputs = comb,
		filters = 128,
		kernel_size = (3, 3),
		padding = 'same',
		activation = tf.nn.relu
	)
	return feature

BATCH_SIZE = 8

def conv_lstm_cell():
	return tf.contrib.rnn.ConvLSTMCell(
		conv_ndims = 2,
		input_shape = [28, 28, 131],
		output_channels = 8,
		kernel_shape = (3, 3)
	)

def combine(y, end):
	y_re = tf.reshape(y, [-1, 28 * 28])
	end_re = tf.reshape(end, [-1, 1])
	return tf.concat([y_re, end_re], 1)

def model(x, b, v, y, end):
	img = tf.reshape(x, [-1, 224, 224, 3])
	boundary_true = tf.reshape(b, [-1, 28, 28, 1])
	vertices_true = tf.reshape(v, [-1, 28, 28, 1])
	y_true = tf.reshape(y, [-1, 28, 28, 1, 5])
	end_true = tf.reshape(end, [-1, 1, 1, 1, 5])
	feature = vgg16(img) # batch_size 28 28 128

	loss1 = 0.0
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
	loss1 += tf.losses.log_loss(labels = boundary_true, predictions = boundary , weights = (boundary_true * 686 + 49))
	loss1 += tf.losses.log_loss(labels = vertices_true, predictions = vertices , weights = (vertices_true * 772 + 6 ))
	loss1 /= (2 * 784 / 100)

	stacked_lstm = tf.contrib.rnn.MultiRNNCell([conv_lstm_cell() for _ in range(2)])
	initial_state = state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
	pred = []
	loss2 = 0.0
	t = []
	for i in range(2, 5):
		comb = tf.concat([feature, y_true[..., i - 1], y_true[..., i - 2], y_true[..., 0]], 3)
		output, state = stacked_lstm(comb, state)
		y_pred = tf.layers.conv2d(
			inputs = output,
			filters = 1,
			kernel_size = (3, 3),
			padding = 'same',
			# activation = tf.sigmoid
		)
		end_pred = tf.layers.conv2d(
			inputs = output,
			filters = 1,
			kernel_size = (28, 28),
			# activation = tf.sigmoid
		)
		y_end_true = combine(y_true[..., i], end_true[..., i])
		y_end_pred = combine(y_pred, end_pred)
		temp = tf.nn.softmax(y_end_pred)
		pred.append(tf.reshape(temp[..., 0: 784], [-1, 28, 28]))
		t.append(temp[..., 784])
		loss2 += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_end_true, logits = y_end_pred))
	loss2 /= 3
	return (loss1, loss2), pred, boundary, vertices, t

def trans(arr):
	ma = np.max(arr)
	mi = np.min(arr)
	return (arr - mi) / (ma - mi)

if __name__ == '__main__':
	f = open('a.out', 'w')
	random.seed(31415926)
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)
	v = tf.placeholder(tf.float32)
	end = tf.placeholder(tf.float32)
	result = model(x, b, v, y, end)
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)
	train = optimizer.minimize(result[0][0] + result[0][1])
	init = tf.global_variables_initializer()
	n_iter = 10000
	obj = BuildingImage.BuildingListConstructor(range_vertices = (4, 4), filename = './buildingList.npy')
	with tf.Session() as sess:
		sess.run(init)
		for i in range(n_iter):
			data = obj.getImage(BATCH_SIZE)
			img = [np.array(item[0])[...,0:3]/255.0 for item in data] # batch_size 224 224 3
			single = [item[5] for item in data] # batch_size num_vertices+1 28 28
			single_true = np.transpose(np.array(single), axes = [0, 2, 3, 1]) # batch_size 28 28 num_vertices+1
			end_true = np.array([[0,0,0,0,1] for i in range(BATCH_SIZE)]) # batch_size num_vertices+1
			boundary_true = [item[3] for item in data]
			vertices_true = [item[4] for item in data]
			feed_dict = {x: img, y: single_true, b: boundary_true, v: vertices_true, end: end_true}
			sess.run(train, feed_dict)
			loss, pred, boundary, vertices, t = sess.run(result, feed_dict)
			# print(np.sum(np.array(pred), axis = (2, 3))) # 7 8 28 28
			# print(np.array(t)) # 7 8
			for j in range(BATCH_SIZE):
				Image.fromarray(np.array(img[j] * 255.0, dtype = np.uint8)).save('./res/%d-a.png' % j)
				Image.fromarray(np.array(boundary[j,...,0] * 255.0, dtype = np.uint8)).save('./res/%d-b.png' % j)
				Image.fromarray(np.array(vertices[j,...,0] * 255.0, dtype = np.uint8)).save('./res/%d-c.png' % j)
				Image.fromarray(np.array(single[j][0] * 255.0, dtype = np.uint8)).save('./res/%d-p1.png' % j)
				Image.fromarray(np.array(single[j][1] * 255.0, dtype = np.uint8)).save('./res/%d-p2.png' % j)
				for k in range(3):
					Image.fromarray(np.array(pred[k][j, ...] * 255.0, dtype = np.uint8)).save('./res/%d-p%d.png' % (j, k + 3))
					Image.fromarray(np.array(trans(pred[k][j, ...]) * 255.0, dtype = np.uint8)).save('./res/%d-z%d.png' % (j, k + 3))
			f.write('%.6lf, %.6lf, %.6lf\n' % (loss[0], loss[1], sum(loss)))
			print('%.6lf, %.6lf, %.6lf' % (loss[0], loss[1], sum(loss)))
			f.flush()

