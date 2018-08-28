import numpy as np
import tensorflow as tf
from Config import *
from BasicModel import *

config = Config()

class Model(object):
	def __init__(self, max_num_vertices, lstm_out_channel, v_out_res):
		"""
			max_num_vertices  : scalar
			lstm_out_channel  : list of int numbers
			v_out_res         : [n_row, n_col]
		"""
		# RolygonRNN parameters
		self.max_num_vertices = max_num_vertices
		self.lstm_out_channel = lstm_out_channel
		self.lstm_in_channel  = [133] + lstm_out_channel[: -1]
		self.v_out_res        = v_out_res
		self.v_out_nrow       = self.v_out_res[0]
		self.v_out_ncol       = self.v_out_res[1]
		self.res_num          = self.v_out_nrow * self.v_out_ncol

		# Multi-layer LSTM and inital state
		self.stacked_lstm     = tf.contrib.rnn.MultiRNNCell(
			[self.ConvLSTMCell(in_c, out_c) for in_c, out_c in zip(self.lstm_in_channel, self.lstm_out_channel)]
		)
		self.lstm_init_state  = [
			tf.get_variable('ConvLSTM_Cell_%d_State' % i, [2, self.v_out_nrow, self.v_out_ncol, c_out])
			for i, c_out in enumerate(lstm_out_channel)
		]

		# Vertex pool for prediction
		self.vertex_pool = []
		for i in range(self.v_out_nrow):
			for j in range(self.v_out_ncol):
				self.vertex_pool.append(np.zeros(self.v_out_res, dtype = np.float32))
				self.vertex_pool[-1][i, j] = 1.0
		self.vertex_pool.append(np.zeros(self.v_out_res, dtype = np.float32))
		self.vertex_pool = np.array(self.vertex_pool)
		return

	def ConvLSTMCell(self, num_in, num_out):
		"""
			input_channels    : scalar
			output_channels   : scalar
		"""
		return tf.contrib.rnn.ConvLSTMCell(
			conv_ndims = 2,
			input_shape = [self.v_out_nrow, self.v_out_ncol, num_in],
			output_channels = num_out,
			kernel_shape = [3, 3]
		)

	def smoothL1Loss(self, labels, predictions):
		diff = tf.abs(predictions - labels)
		val = tf.where(tf.less(diff, 1), 0.5 * tf.square(diff), diff - 0.5)
		return tf.reduce_mean(val)

	def L2LossWeighted(self, gt, pred):
		# return 3 * self.smoothL1Loss(gt, pred)
		batch_size = tf.cast(tf.shape(gt)[0], tf.float32)
		nt, nf = tf.reduce_sum(gt) / batch_size, tf.reduce_sum(1 - gt) / batch_size
		nt = tf.maximum(tf.minimum(nt, 0.999999), 0.000001)
		nf = tf.maximum(tf.minimum(nf, 0.999999), 0.000001)
		weights = gt * 0.5 / nt + (1 - gt) * 0.5 / nf
		weights = weights / tf.reduce_sum(weights)
		return tf.reduce_sum(tf.multiply(tf.square(gt - pred) * 33, weights))

	def weightedLogLoss(self, gt, pred):
		num = tf.reduce_sum(tf.ones(gt.shape))
		n_pos = tf.reduce_sum(gt)
		n_neg = tf.reduce_sum(1 - gt)
		n_pos = tf.maximum(tf.minimum(n_pos, num - 1), 0)
		n_neg = tf.maximum(tf.minimum(n_neg, num - 1), 0)
		num /= 2
		w = gt * num / n_pos + (1 - gt) * num / n_neg
		return tf.losses.log_loss(gt, pred, w)

	def CNN(self, img, gt_boundary = None, gt_vertices = None, reuse = None):
		"""
			gt_boundary       : [batch_size, height, width, 1]
			gt_vertices       : [batch_size, height, width, 1]
			combine           : [batch_size, height, width, 130]
			v_first           : 
			prob              :
		"""
		if not reuse:
			feature = PolygonRNNFeature(VGG16(img))
		else:
			feature = PolygonRNNFeature(VGG16(img, True), True)
		with tf.variable_scope('CNN', reuse = reuse):
			boundary = tf.layers.conv2d(inputs = feature, filters = 2, kernel_size = 1, padding = 'valid', activation = None)
			combine  = tf.concat([feature, boundary], 3)
			vertices = tf.layers.conv2d(inputs = combine, filters = 2, kernel_size = 1, padding = 'valid', activation = None)
		boundary_prob = tf.nn.softmax(boundary)[..., 0]
		vertices_prob = tf.nn.softmax(vertices)[..., 0]
		new_combine = tf.concat([feature, boundary, vertices], 3)
		if not reuse:
			loss  = self.weightedLogLoss(gt_boundary, boundary_prob)
			loss += self.weightedLogLoss(gt_vertices, vertices_prob)
			return new_combine, boundary_prob, vertices_prob, loss
		else:
			return new_combine, boundary_prob, vertices_prob

	def FC(self, rnn_output, gt_rnn_out = None, gt_seq_len = None, reuse = None):
		""" 
			rnn_output
			gt_rnn_out
			gt_seq_len
		"""
		if not reuse:
			output_reshape = tf.reshape(rnn_output, [-1, self.max_num_vertices, self.res_num * self.lstm_out_channel[-1]])	
		else:
			output_reshape = tf.reshape(rnn_output, [-1, 1, self.res_num * self.lstm_out_channel[-1]])
		with tf.variable_scope('FC', reuse = reuse):
			logits = tf.layers.dense(inputs = output_reshape, units = self.res_num * 8, activation = None)
			logits_pos = tf.layers.dense(inputs = logits, units = self.res_num, activation = None)
			logits_neg = tf.layers.dense(inputs = logits, units = self.res_num, activation = None)
			logits_pos = tf.expand_dims(logits_pos, -1)
			logits_neg = tf.expand_dims(logits_neg, -1)
			logits = tf.concat([logits_pos, logits_neg], -1)
			prob = tf.nn.softmax(logits)[..., 0]
		if not reuse:
			loss = self.weightedLogLosstf(gt_rnn_out, prob)
			return prob, loss
		else:
			return prob

	def RNN(self, feature, v_in = None, gt_rnn_out = None, gt_seq_len = None, reuse = None):
		if not reuse:
			batch_size = [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, 1, 1, 1]
			initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
				c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
				h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
			) for i in range(len(self.lstm_out_channel))])
			feature_rep = tf.reshape(tf.tile(tf.reshape(feature,
				[config.AREA_TRAIN_BATCH, 1, 1, self.v_out_nrow, self.v_out_ncol, 132]),
				[1, config.TRAIN_NUM_PATH, self.max_num_vertices, 1, 1, 1]),
				[config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 132]
			)
			rnn_input = tf.concat([feature_rep, v_in], axis = 4)
			outputs, state = tf.nn.dynamic_rnn(cell = self.stacked_lstm, inputs = rnn_input,
				sequence_length = gt_seq_len, initial_state = initial_state, dtype = tf.float32
			)

			return self.FC(outputs, gt_rnn_out, gt_seq_len)
		else:
			batch_size = tf.concat([[tf.shape(v_in)[0]], [1, 1, 1]], 0)
			initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
				c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
				h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
			) for i in range(len(self.lstm_out_channel))])
			feature_rep = tf.tile(feature, batch_size)
			rnn_input = tf.concat([feature_rep, v_in], axis = 3)
			outputs, _ = self.stacked_lstm(rnn_input, initial_state)
			return self.FC(outputs, reuse = True)

	def train(self, aa, bb, vv, ii, oo, ll):
		#
		img          = tf.reshape(aa, [config.AREA_TRAIN_BATCH, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		gt_boundary  = tf.reshape(bb, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol])
		gt_vertices  = tf.reshape(vv, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol])
		gt_v_in      = tf.reshape(ii, [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 1])
		gt_v_out     = tf.reshape(oo, [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, self.max_num_vertices, self.res_num])
		gt_seq_len   = tf.reshape(ll, [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH])

		# PolygonRNN part
		feature, pred_boundary, pred_vertices, loss_CNN = self.CNN(img, gt_boundary, gt_vertices)
		pred_rnn, loss_RNN = self.RNN(feature, gt_v_in, gt_v_out, gt_seq_len)

		# 
		pred_v_out    = tf.reshape(pred_rnn, [-1, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol])

		return loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out

	def predict_mask(self, aa):
		img = tf.reshape(aa, [1, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		feature, pred_boundary, pred_vertices = self.CNN(img, reuse = True)
		return feature, pred_boundary, pred_vertices

	def predict_path(self, ff, ii):
		#
		feature  = tf.reshape(ff, [1, self.v_out_nrow, self.v_out_ncol, 132])
		v_in = tf.reshape(ii, [-1, self.v_out_nrow, self.v_out_ncol, 1])

		#
		pred_rnn = self.RNN(feature, v_in, reuse = True)
		pred_v_out = tf.reshape(pred_rnn, [-1, self.v_out_nrow, self.v_out_ncol])

		return pred_v_out

class Logger(object):
	def __init__(self, log_dir):
		self.writer = tf.summary.FileWriter(log_dir)
		return

	def log_scalar(self, tag, value, step):
		summary = tf.Summary(value = [tf.Summary.Value(tag = tag, simple_value = value)])
		self.writer.add_summary(summary, step)
		return

	def close(self):
		self.writer.close()
		return
