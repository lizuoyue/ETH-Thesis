import os, glob, random, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from wxpy import *
plt.switch_backend('agg')
ut = __import__('utility')

BATCH_SIZE = 8
MAX_SEQ_LEN = 12
LSTM_OUT_CHANNEL = [16, 8]
SET_WECHAT = False

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

def conv_lstm_cell(output_channels):
	return tf.contrib.rnn.ConvLSTMCell(
		conv_ndims = 2,
		input_shape = [28, 28, 133],
		output_channels = output_channels,
		kernel_shape = (3, 3)
	)

def polyRNN(xx, bb, vv, yy, ee, ll):
	# Reshape
	img           = tf.reshape(xx, [-1, 224, 224, 3])
	boundary_true = tf.reshape(bb, [-1, 28, 28, 1])
	vertices_true = tf.reshape(vv, [-1, 28, 28, 1])
	y_true        = tf.reshape(yy, [-1, MAX_SEQ_LEN, 28, 28, 1])
	end_true      = tf.reshape(ee, [-1, MAX_SEQ_LEN, 1, 1, 1])
	seq_len       = tf.reshape(ll, [-1])
	y_re          = tf.reshape(yy, [-1, MAX_SEQ_LEN, 28 * 28])
	e_re          = tf.reshape(ee, [-1, MAX_SEQ_LEN, 1])
	y_end_true    = tf.concat([y_re, e_re], 2)

	# CNN part
	feature = modifiedVGG16(img) # batch_size 28 28 128
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
	loss_1 = 0.0
	loss_1 += tf.losses.log_loss(labels = boundary_true, predictions = boundary, weights = (boundary_true * (784 - 2 * n_b) + n_b))
	loss_1 += tf.losses.log_loss(labels = vertices_true, predictions = vertices, weights = (vertices_true * (784 - 2 * n_v) + n_v))
	loss_1 /= (2 * 784 / 200)

	# RNN part
	feature_new = tf.concat([feature, boundary, vertices], 3)
	feature_rep = tf.tile(tf.reshape(feature_new, [-1, 1, 28, 28, 130]), [1, MAX_SEQ_LEN, 1, 1, 1]) # batch_size max_len 28 28 130
	y_true_1 = tf.stack([y_true[:, 0, ...]] + tf.unstack(y_true, axis = 1)[0: -1], axis = 1)
	y_true_2 = tf.stack([y_true[:, 0, ...], y_true[:, 0, ...]] + tf.unstack(y_true, axis = 1)[0: -2], axis = 1)
	rnn_input = tf.concat([feature_rep, y_true, y_true_1, y_true_2], axis = 4) # batch_size max_len 28 28 133

	stacked_lstm = tf.contrib.rnn.MultiRNNCell([conv_lstm_cell(out) for out in LSTM_OUT_CHANNEL])
	initial_state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
	outputs, state = tf.nn.dynamic_rnn(
		cell = stacked_lstm,
		inputs = rnn_input,
		sequence_length = seq_len,
		initial_state = initial_state,
		dtype = tf.float32
	)
	outputs_reshape = tf.reshape(outputs, [-1, MAX_SEQ_LEN, 28 * 28 * LSTM_OUT_CHANNEL[1]])

	# FC part
	logits = tf.layers.dense( # batch_size max_len 785
		inputs = outputs_reshape,
		units = 28 * 28 + 1,
		activation = None
	)
	loss_2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = y_end_true, logits = logits)) / tf.reduce_sum(seq_len)

	# Return
	y_end_pred = tf.nn.softmax(logits)
	y_pred = tf.reshape(y_end_pred[..., 0: 28 * 28], [-1, MAX_SEQ_LEN, 28, 28])
	end_pred = tf.reshape(y_end_pred[..., 28 * 28], [-1, MAX_SEQ_LEN, 1])
	return loss_1, loss_2, boundary, vertices, y_pred, end_pred

class DataGenerator(object):

	def __init__(self, data_path):
		self.data_path = data_path
		self.id_list = os.listdir(data_path)
		if '.DS_Store' in self.id_list:
			self.id_list.remove('.DS_Store')

	def getDataSingle(self, building_id):
		# Set path
		if type(building_id) == int:
			building_id = str(building_id)
		path = self.data_path + '/' + building_id

		# Get images
		img = np.array(Image.open(glob.glob(path + '/' + '0-img.png')[0]))[..., 0: 3] / 255.0
		boundary = np.array(Image.open(glob.glob(path + '/' + '3-b.png')[0])) / 255.0
		vertices = np.array(Image.open(glob.glob(path + '/' + '4-v.png')[0])) / 255.0
		vertex_file_list = glob.glob(path + '/' + '5-v*.png')
		vertex_file_list.sort()
		vertex = [np.array(Image.open(item)) / 255.0 for item in vertex_file_list]
		seq_len = len(vertex)
		while len(vertex) < MAX_SEQ_LEN:
			vertex.append(np.zeros((28, 28), dtype = np.float32))
		vertex = np.array(vertex)
		end = [0.0 for i in range(MAX_SEQ_LEN)]
		end[seq_len] = 1.0
		end = np.array(end)

		# Return
		return img, boundary, vertices, vertex, end, seq_len

	def getDataBatch(self, batch_size):
		res = []
		sel = np.random.choice(len(self.id_list), batch_size, replace = False)
		for i in sel:
			res.append(self.getDataSingle(self.id_list[i]))
		return (np.array([item[i] for item in res]) for i in range(6))

	def getToyDataBatch(self, batch_size):
		res = []
		num_v = np.random.choice(6, batch_size, replace = True) + 4
		for n in num_v:
			img, b, v, vertex_list = ut.plotPolygon(num_vertices = n)
			while len(vertex_list) < MAX_SEQ_LEN:
				vertex_list.append(np.zeros((28, 28), dtype = np.float32))
			vertex_list = np.array(vertex_list)
			end = [0.0 for i in range(MAX_SEQ_LEN)]
			end[n] = 1.0
			end = np.array(end)
			res.append((img, b, v, vertex_list, end, n))
		return (np.array([item[i] for item in res]) for i in range(6))

def norm(array):
	ma = np.amax(array)
	mi = np.amin(array)
	if ma == mi:
		return np.zeros(array.shape)
	else:
		return (array - mi) / (ma - mi)

if __name__ == '__main__':
	# Set WeChat
	if SET_WECHAT:
		bot = Bot()
		friend = bot.friends().search('李作越')[0]

	# Set parameters
	random.seed(31415926)
	lr = 0.0005
	n_iter = 200000
	f = open('polyRNN.out', 'w')
	obj = DataGenerator('../Dataset')

	# Define graph
	xx = tf.placeholder(tf.float32)
	bb = tf.placeholder(tf.float32)
	vv = tf.placeholder(tf.float32)
	yy = tf.placeholder(tf.float32)
	ee = tf.placeholder(tf.float32)
	ll = tf.placeholder(tf.float32)
	result = polyRNN(xx, bb, vv, yy, ee, ll)
	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	train = optimizer.minimize(result[0] + result[1])
	saver = tf.train.Saver(max_to_keep = 8)
	init = tf.global_variables_initializer()

	# Launch graph
	with tf.Session() as sess:
		sess.run(init)
		for i in range(n_iter):
			# Get batch data and create feed dictionary
			img, boundary, vertices, vertex, end, seq_len = obj.getToyDataBatch(BATCH_SIZE)
			feed_dict = {xx: img, bb: boundary, vv: vertices, yy: vertex, ee: end, ll: seq_len}

			# Training and get result
			sess.run(train, feed_dict)
			loss_1, loss_2, b_pred, v_pred, y_pred, end_pred = sess.run(result, feed_dict)

			# Save model
			if i % 200 == 0:
				saver.save(sess, './tmp/model-%d.ckpt' % i)

			# Write loss to file
			print('%d, %.6lf, %.6lf, %.6lf' % (i, loss_1, loss_2, loss_1 + loss_2))
			f.write('%d, %.6lf, %.6lf, %.6lf\n' % (i, loss_1, loss_2, loss_1 + loss_2))
			f.flush()

			# Send to mobile
			if SET_WECHAT and int(time.time()) % 7200 < 60:
				friend.send('%d, %.6lf, %.6lf, %.6lf' % (i, loss_1, loss_2, loss_1 + loss_2))

			# Clear last files
			for item in glob.glob('./res/*'):
				os.remove(item)

			# Visualize prediction
			for j in range(BATCH_SIZE):
				org = Image.fromarray(np.array(img[j, ...] * 255.0, dtype = np.uint8)).convert('RGBA')
				org.save('./res/%d-0-img.png' % j)
				Image.fromarray(np.array(b_pred[j, ..., 0] * 255.0, dtype = np.uint8)).save('./res/%d-1-b.png' % j)
				Image.fromarray(np.array(boundary[j] * 255.0, dtype = np.uint8)).save('./res/%d-1-b-t.png' % j)
				Image.fromarray(np.array(v_pred[j, ..., 0] * 255.0, dtype = np.uint8)).save('./res/%d-2-v.png' % j)
				Image.fromarray(np.array(vertices[j] * 255.0, dtype = np.uint8)).save('./res/%d-2-v-t.png' % j)
				Image.fromarray(np.array(vertex[j, 0, ...] * 255.0, dtype = np.uint8)).save('./res/%d-3-v00.png' % j)
				for k in range(1, seq_len[j] + 1):
					Image.fromarray(np.array(y_pred[j, k, ...] * 255.0, dtype = np.uint8)).save('./res/%d-3-v%s.png' % (j, str(k).zfill(2)))
					Image.fromarray(np.array(norm(y_pred[j, k, ...]) * 255.0, dtype = np.uint8)).save('./res/%d-4-p%s.png' % (j, str(k).zfill(2)))
					alpha = np.array(norm(y_pred[j, k, ...]) * 128.0, dtype = np.uint8)
					alpha = np.concatenate((np.ones((28, 28, 1)) * 255.0, np.zeros((28, 28, 2)), np.reshape(alpha, (28, 28, 1))), axis = 2)
					alpha = Image.fromarray(np.array(alpha, dtype = np.uint8), mode = 'RGBA')
					alpha = alpha.resize((224, 224), resample = Image.BILINEAR)
					merge = Image.alpha_composite(org, alpha)
					merge.save('./res/%d-5-m%s.png' % (j, str(k).zfill(2)))
				plt.plot(end_pred[j, 1: seq_len[j] + 1])
				plt.savefig('./res/%d-5-end.pdf' % j)
				plt.gcf().clear()
	f.close()



