import io, os, sys, glob
import time, random, tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from wxpy import *
plt.switch_backend('agg')
ut = __import__('Utility')

BATCH_SIZE = 8
MAX_SEQ_LEN = 24
LSTM_OUT_CHANNEL = [32, 12]
LSTM_IN_CHANNEL = [133, 32]
SET_WECHAT = False
DATA_PATH = '../Chicago'

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

def conv_lstm_cell(intput_channels, output_channels):
	return tf.contrib.rnn.ConvLSTMCell(
		conv_ndims = 2,
		input_shape = [28, 28, intput_channels],
		output_channels = output_channels,
		kernel_shape = [3, 3]
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
	loss_1 /= (2 * 784 / 50)

	# RNN part
	feature_new = tf.concat([feature, boundary, vertices], 3)
	feature_rep = tf.tile(tf.reshape(feature_new, [-1, 1, 28, 28, 130]), [1, MAX_SEQ_LEN, 1, 1, 1]) # batch_size max_len 28 28 130
	y_true_1 = tf.stack([y_true[:, 0, ...]] + tf.unstack(y_true, axis = 1)[0: -1], axis = 1)
	y_true_2 = tf.stack([y_true[:, 0, ...], y_true[:, 0, ...]] + tf.unstack(y_true, axis = 1)[0: -2], axis = 1)
	rnn_input = tf.concat([feature_rep, y_true, y_true_1, y_true_2], axis = 4) # batch_size max_len 28 28 133

	stacked_lstm = tf.contrib.rnn.MultiRNNCell([conv_lstm_cell(in_c, out_c) for in_c, out_c in zip(LSTM_IN_CHANNEL, LSTM_OUT_CHANNEL)])
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

def overlay(img, mask):
	org = Image.fromarray(np.array(img * 255.0, dtype = np.uint8)).convert('RGBA')
	alpha = np.array(mask * 128.0, dtype = np.uint8)
	alpha = np.concatenate((np.ones((28, 28, 1)) * 255.0, np.zeros((28, 28, 2)), np.reshape(alpha, (28, 28, 1))), axis = 2)
	alpha = Image.fromarray(np.array(alpha, dtype = np.uint8), mode = 'RGBA')
	alpha = alpha.resize((224, 224), resample = Image.BILINEAR)
	merge = Image.alpha_composite(org, alpha)
	return merge

def visualize(path, img, boundary, vertices, vertex, b_pred, v_pred, y_pred, end_pred):
	# Clear last files
	for item in glob.glob(path + '/*'):
		os.remove(item)
	for j in range(img.shape[0]):
		org = Image.fromarray(np.array(img[j, ...] * 255.0, dtype = np.uint8)).convert('RGBA')
		org.save(path + '/%d-0-img.png' % j)
		overlay(img[j, ...], b_pred[j, ..., 0]).save(path + '/%d-1-b-p.png' % j)
		overlay(img[j, ...], boundary[j]).save(path + '/%d-1-b-t.png' % j)
		overlay(img[j, ...], v_pred[j, ..., 0]).save(path + '/%d-2-v-p.png' % j)
		overlay(img[j, ...], vertices[j]).save(path + '/%d-2-v-t.png' % j)
		overlay(img[j, ...], vertex[j, 0, ...]).save(path + '/%d-3-v00.png' % j)
		for k in range(1, seq_len[j] + 1):
			overlay(img[j, ...], y_pred[j, k, ...]).save(path + '/%d-3-v%s.png' % (j, str(k).zfill(2)))
		plt.plot(end_pred[j, 1: seq_len[j] + 1])
		plt.savefig(path + '/%d-5-end.pdf' % j)
		plt.gcf().clear()
	return

if __name__ == '__main__':
	# Create new folder
	if not os.path.exists('./tmp/'):
		os.makedirs('./tmp/')
	if not os.path.exists('./res/'):
		os.makedirs('./res/')
	if not os.path.exists('./val/'):
		os.makedirs('./val/')

	# Set WeChat
	if SET_WECHAT:
		bot = Bot()
		friend = bot.friends().search('李作越')[0]

	# Set parameters
	lr = 0.0005
	n_iter = 2000000
	f = open('PolyRNN.out', 'w')
	obj = ut.DataGenerator(DATA_PATH, train_prob = 0.5)

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
	saver = tf.train.Saver(max_to_keep = 20)
	init = tf.global_variables_initializer()

	# Launch graph
	with tf.Session() as sess:
		if len(sys.argv) > 1 and sys.argv[1] != None:
			saver.restore(sess, './tmp/model-%s.ckpt' % sys.argv[1])
			iter_obj = range(int(sys.argv[1]) + 1, n_iter)
		else:
			sess.run(init)
			iter_obj = range(n_iter)
		for i in iter_obj:
			# Get batch data and create feed dictionary
			img, boundary, vertices, vertex, end, seq_len = obj.getDataBatch(BATCH_SIZE)
			feed_dict = {xx: img, bb: boundary, vv: vertices, yy: vertex, ee: end, ll: seq_len}

			# Training and get result
			sess.run(train, feed_dict)
			loss_1, loss_2, b_pred, v_pred, y_pred, end_pred = sess.run(result, feed_dict)
			print(b_pred.shape)
			print(v_pred.shape)
			print(y_pred.shape)
			print(end_pred.shape)

			# Write loss to file
			print('Train Iter %d, %.6lf, %.6lf, %.6lf' % (i, loss_1, loss_2, loss_1 + loss_2))
			f.write('Train Iter %d, %.6lf, %.6lf, %.6lf\n' % (i, loss_1, loss_2, loss_1 + loss_2))
			f.flush()

			# Send to mobile
			if SET_WECHAT and int(time.time()) % 7200 < 60:
				friend.send('%d, %.6lf, %.6lf, %.6lf' % (i, loss_1, loss_2, loss_1 + loss_2))

			# Visualize
			visualize('./res', img, boundary, vertices, vertex, b_pred, v_pred, y_pred, end_pred)

			# Save model and validate
			if i % 200 == 0:
				saver.save(sess, './tmp/model-%d.ckpt' % i)
				img, boundary, vertices, vertex, end, seq_len = obj.getDataBatch(BATCH_SIZE, mode = 'valid')
				feed_dict = {xx: img, bb: boundary, vv: vertices, yy: vertex, ee: end, ll: seq_len}
				loss_1, loss_2, b_pred, v_pred, y_pred, end_pred = sess.run(result, feed_dict)
				print('Valid Iter %d, %.6lf, %.6lf, %.6lf' % (i, loss_1, loss_2, loss_1 + loss_2))
				f.write('Valid Iter %d, %.6lf, %.6lf, %.6lf\n' % (i, loss_1, loss_2, loss_1 + loss_2))
				f.flush()
				visualize('./val', img, boundary, vertices, vertex, b_pred, v_pred, y_pred, end_pred)

	f.close()

