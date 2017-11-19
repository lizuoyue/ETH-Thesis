import os, glob, random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

def modifiedVGG16(x):
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

BATCH_SIZE = 4
MAX_SEQ_LEN = 20
LSTM_OUT_CHANNEL = [16, 8]

def conv_lstm_cell(output_channels):
	return tf.contrib.rnn.ConvLSTMCell(
		conv_ndims = 2,
		input_shape = [28, 28, 131],
		output_channels = output_channels,
		kernel_shape = (3, 3)
	)

def combine(y, end):
	y_re = tf.reshape(y, [-1, 28 * 28])
	end_re = tf.reshape(end, [-1, 1])
	return tf.concat([y_re, end_re], 1)

def polyRNN(x, b, v, y, e, l):
	# Reshape
	img           = tf.reshape(x, [-1, 224, 224, 3])
	boundary_true = tf.reshape(b, [-1, 28, 28, 1])
	vertices_true = tf.reshape(v, [-1, 28, 28, 1])
	y_true        = tf.reshape(y, [-1, MAX_SEQ_LEN, 28, 28, 1])
	end_true      = tf.reshape(e, [-1, MAX_SEQ_LEN, 1, 1, 1])
	seq_len       = tf.reshape(l, [-1])

	# CNN Part
	feature = modifiedVGG16(img) # batch_size 28 28 128

	loss_1 = 0.0
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
	n_b = tf.reduce_sum(boundary_true) / BATCH_SIZE
	n_v = tf.reduce_sum(vertices_true) / BATCH_SIZE
	loss_1 += tf.losses.log_loss(labels = boundary_true, predictions = boundary, weights = (boundary_true * (784 - 2 * n_b) + n_b))
	loss_1 += tf.losses.log_loss(labels = vertices_true, predictions = vertices, weights = (vertices_true * (784 - 2 * n_v) + n_v))
	loss_1 /= (2 * 784 / 100)

	# RNN Part
	feature_rep = tf.tile(tf.reshape(feature, [-1, 1, 28, 28, 128]), [1, MAX_SEQ_LEN, 1, 1, 1]) # batch_size max_len 28 28 128
	y_true_1 = tf.stack([y_true[:, 0, ...]] + tf.unstack(y_true, axis = 1)[0: -1], axis = 1)
	y_true_2 = tf.stack([y_true[:, 0, ...], y_true[:, 0, ...]] + tf.unstack(y_true, axis = 1)[0: -2], axis = 1)
	rnn_input = tf.concat([feature_rep, y_true, y_true_1, y_true_2], axis = 4)

	stacked_lstm = tf.contrib.rnn.MultiRNNCell([conv_lstm_cell(out) for out in LSTM_OUT_CHANNEL])
	initial_state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
	outputs, state = tf.nn.dynamic_rnn(
		cell = stacked_lstm,
		inputs = rnn_input,
		sequence_length = seq_len,
		initial_state = initial_state,
		dtype = tf.float32
	)
	return outputs
	# print(outputs.shape)
	# print(np.array(seq_len).shape)

	# for i in range(BATCH_SIZE):
	# 	for j in range(np.array(seq_len)):
	# 		pass

	# return

	# pred = []
	# loss2 = 0.0
	# t = []
	# for i in range(2, 5):
	# 	comb = tf.concat([feature, y_true[..., i - 1], y_true[..., i - 2], y_true[..., 0]], 3)
	# 	output, state = stacked_lstm(comb, state)
	# 	y_pred = tf.layers.conv2d(
	# 		inputs = output,
	# 		filters = 1,
	# 		kernel_size = (3, 3),
	# 		padding = 'same',
	# 		# activation = tf.sigmoid
	# 	)
	# 	end_pred = tf.layers.conv2d(
	# 		inputs = output,
	# 		filters = 1,
	# 		kernel_size = (28, 28),
	# 		# activation = tf.sigmoid
	# 	)
	# 	y_end_true = combine(y_true[..., i], end_true[..., i])
	# 	y_end_pred = combine(y_pred, end_pred)
	# 	temp = tf.nn.softmax(y_end_pred)
	# 	pred.append(tf.reshape(temp[..., 0: 784], [-1, 28, 28]))
	# 	t.append(temp[..., 784])
	# 	loss2 += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_end_true, logits = y_end_pred))
	# loss2 /= 3
	# return (loss1, loss2), pred, boundary, vertices, t

def trans(array):
	ma = np.max(array)
	mi = np.min(array)
	return (array - mi) / (ma - mi)

class DataGenerator(object):

	def __init__(self, data_path):
		self.data_path = data_path
		self.id_list = os.listdir(data_path)
		if '.DS_Store' in self.id_list:
			self.id_list.remove('.DS_Store')

	def getDataSingle(self, building_id):
		if type(building_id) == int:
			building_id = str(building_id)
		path = self.data_path + '/' + building_id
		img = np.array(Image.open(glob.glob(path + '/' + '0-img.png')[0]))[..., 0: 3] / 255.0
		boundary = np.array(Image.open(glob.glob(path + '/' + '3-b.png')[0])) / 255.0
		vertices = np.array(Image.open(glob.glob(path + '/' + '4-v.png')[0])) / 255.0
		vertex = [np.array(Image.open(item)) / 255.0 for item in glob.glob(path + '/' + '5-v*.png')]
		seq_len = len(vertex)
		while len(vertex) < MAX_SEQ_LEN:
			vertex.append(np.zeros((28, 28), dtype = np.float32))
		vertex = np.array(vertex)
		end = [0.0 for i in range(MAX_SEQ_LEN)]
		end[seq_len] = 1.0
		end = np.array(end)
		return img, boundary, vertices, vertex, end, seq_len

	def getDataBatch(self, batch_size):
		res = []
		sel = np.random.choice(len(self.id_list), batch_size, replace = False)
		for i in sel:
			res.append(self.getDataSingle(self.id_list[i]))
		return ([item[i] for item in res] for i in range(6))

if __name__ == '__main__':
	obj = DataGenerator('../Dataset')
	# f = open('a.out', 'w')
	random.seed(31415926)
	x = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)
	v = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)
	e = tf.placeholder(tf.float32)
	l = tf.placeholder(tf.uint8)
	result = polyRNN(x, b, v, y, e, l)
	# optimizer = tf.train.AdamOptimizer(learning_rate = 0.0004)
	# train = optimizer.minimize(result)
	init = tf.global_variables_initializer()
	n_iter = 100000
	with tf.Session() as sess:
		sess.run(init)
		for i in range(n_iter):
			d1,d2,d3,d4,d5,d6 = obj.getDataBatch(4)
			# data = obj.getImage(BATCH_SIZE)
			# img = [np.array(item[0])[...,0:3]/255.0 for item in data] # batch_size 224 224 3
			# single = [item[5] for item in data] # batch_size num_vertices+1 28 28
			# single_true = np.transpose(np.array(single), axes = [0, 2, 3, 1]) # batch_size 28 28 num_vertices+1
			# end_true = np.array([[0,0,0,0,1] for i in range(BATCH_SIZE)]) # batch_size num_vertices+1
			# boundary_true = [item[3] for item in data]
			# vertices_true = [item[4] for item in data]
			feed_dict = {x: d1, b: d2, v: d3, y: d4, e: d5, l:d6}
			a = sess.run(result, feed_dict)
			a = (a - np.min(a))/(np.max(a) - np.min(a))
			print(d6)
			import matplotlib.pyplot as plt
			for i in range(a.shape[0]):
				for j in range(a.shape[1]):
					plt.imshow(a[i,j,...,0])
					plt.show()
			break
			# loss, pred, boundary, vertices, t = sess.run(result, feed_dict)
			# # print(np.sum(np.array(pred), axis = (2, 3))) # 7 8 28 28
			# # print(np.array(t)) # 7 8
			# for j in range(BATCH_SIZE):
			# 	Image.fromarray(np.array(img[j] * 255.0, dtype = np.uint8)).save('./res/%d-a.png' % j)
			# 	Image.fromarray(np.array(boundary[j,...,0] * 255.0, dtype = np.uint8)).save('./res/%d-b.png' % j)
			# 	Image.fromarray(np.array(vertices[j,...,0] * 255.0, dtype = np.uint8)).save('./res/%d-c.png' % j)
			# 	Image.fromarray(np.array(single[j][0] * 255.0, dtype = np.uint8)).save('./res/%d-p1.png' % j)
			# 	Image.fromarray(np.array(single[j][1] * 255.0, dtype = np.uint8)).save('./res/%d-p2.png' % j)
			# 	for k in range(3):
			# 		Image.fromarray(np.array(pred[k][j, ...] * 255.0, dtype = np.uint8)).save('./res/%d-p%d.png' % (j, k + 3))
			# 		Image.fromarray(np.array(trans(pred[k][j, ...]) * 255.0, dtype = np.uint8)).save('./res/%d-z%d.png' % (j, k + 3))
			# f.write('%.6lf, %.6lf, %.6lf\n' % (loss[0], loss[1], sum(loss)))
			# print('%.6lf, %.6lf, %.6lf' % (loss[0], loss[1], sum(loss)))
			# f.flush()

