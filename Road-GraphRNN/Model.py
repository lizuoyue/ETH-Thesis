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
		self.lstm_in_channel  = [131] + lstm_out_channel[: -1]
		self.v_out_res        = v_out_res
		self.v_out_nrow       = self.v_out_res[0]
		self.v_out_ncol       = self.v_out_res[1]
		self.res_num          = self.v_out_nrow * self.v_out_ncol
		self.num_stage        = 3

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

	def weightedLogLoss(self, gt, pred):
		num = tf.reduce_sum(tf.ones(tf.shape(gt)))
		n_pos = tf.reduce_sum(gt)
		n_neg = tf.reduce_sum(1 - gt)
		n_pos = tf.maximum(tf.minimum(n_pos, num - 1), 1)
		n_neg = tf.maximum(tf.minimum(n_neg, num - 1), 1)
		w = gt * num / n_pos + (1 - gt) * num / n_neg
		return tf.losses.log_loss(gt, pred, w / 2)

	def CNN(self, img, gt_boundary = None, gt_vertices = None, reuse = None):
		"""
			gt_boundary       : [batch_size, height, width, 1]
			gt_vertices       : [batch_size, height, width, 1]
		"""
		vgg_result = VGG19('VGG19', img, reuse = reuse)
		skip_feature = SkipFeature('SkipFeature', vgg_result, reuse = reuse)
		bb, vv = Mask('Mask_1', skip_feature, reuse = reuse)
		b_prob = [tf.nn.softmax(bb)[..., 0: 1]]
		v_prob = [tf.nn.softmax(bb)[..., 0: 1]]
		for i in range(2, self.num_stage + 1):
			stage_input = tf.concat([skip_feature, bb, vv], axis = -1)
			bb, vv = Mask('Mask_%d' % i, stage_input, reuse = reuse)
			b_prob.append(tf.nn.softmax(bb)[..., 0: 1])
			v_prob.append(tf.nn.softmax(vv)[..., 0: 1])
		if not reuse:
			loss_B, loss_V = 0, 0
			for item in b_prob:
				loss_B += self.weightedLogLoss(gt_boundary, item)
			for item in v_prob:
				loss_V += self.weightedLogLoss(gt_vertices, item)
			return skip_feature, b_prob[-1], v_prob[-1], loss_B + loss_V
		else:
			return skip_feature, b_prob[-1], v_prob[-1]

	def FC(self, rnn_output, gt_rnn_out = None, gt_v_mask = None, gt_seq_len = None, reuse = None):
		if not reuse:
			output_reshape = tf.reshape(rnn_output, [-1, self.max_num_vertices, self.res_num * self.lstm_out_channel[-1]])	
		else:
			output_reshape = tf.reshape(rnn_output, [-1, 1, self.res_num * self.lstm_out_channel[-1]])
		with tf.variable_scope('FC', reuse = reuse):
			logits = tf.layers.dense(inputs = output_reshape, units = 4096, activation = tf.nn.relu)
			logits = tf.layers.dense(inputs = logits, units = 4096, activation = tf.nn.relu)
			logits = tf.layers.dense(inputs = logits, units = self.res_num * 2, activation = None)
			logits_reshape = tf.reshape(logits, [config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 2])
		if not reuse:
			prob = tf.nn.softmax(logits_reshape)[..., 0: 1]
			loss = self.num_stage * 2 * tf.losses.log_loss(gt_rnn_out, prob, gt_v_mask)
			return prob, loss
		else:
			prob = tf.nn.softmax(logits)
			val, idx = tf.nn.top_k(prob[:, 0, :], k = config.BEAM_WIDTH)
			return tf.log(val), tf.gather(self.vertex_pool, idx, axis = 0)

	def RNN(self, feature, v_in = None, gt_v_out = None, gt_v_mask = None, gt_seq_len = None, gt_idx = None, reuse = None):
		batch_size = tf.concat([[tf.shape(v_in)[0]], [1, 1, 1]], 0)
		initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
			c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
			h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
		) for i in range(len(self.lstm_out_channel))])
		if not reuse:
			feature_rep = tf.gather(feature, gt_idx)
			feature_rep = tf.tile(tf.expand_dims(feature_rep, axis = 1), [1, self.max_num_vertices, 1, 1, 1])
			rnn_input = tf.concat([feature_rep, v_in], axis = 4)
			outputs, state = tf.nn.dynamic_rnn(cell = self.stacked_lstm, inputs = rnn_input,
				sequence_length = gt_seq_len, initial_state = initial_state, dtype = tf.float32
			)
			return self.FC(outputs, gt_v_out, gt_v_mask, gt_seq_len)
		else:
			pass

	def train(self, aa, bb, vv, ii, oo, mm, ll, dd):
		#
		img          = tf.reshape(aa, [config.AREA_TRAIN_BATCH, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		gt_boundary  = tf.reshape(bb, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol, 1])
		gt_vertices  = tf.reshape(vv, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol, 1])

		gt_v_in      = tf.reshape(ii, [config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 1])
		gt_v_out     = tf.reshape(oo, [config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 1])
		gt_v_mask    = tf.reshape(mm, [config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 1])
		gt_seq_len   = tf.reshape(ll, [config.TRAIN_NUM_PATH])
		gt_idx       = tf.reshape(dd, [config.TRAIN_NUM_PATH])

		# PolygonRNN part
		feature, pred_boundary, pred_vertices, loss_CNN = self.CNN(img, gt_boundary, gt_vertices)
		feature_RNN = tf.concat([feature, gt_boundary, gt_vertices], axis = -1)
		prob, loss_RNN = self.RNN(feature_RNN, gt_v_in, gt_v_out, gt_v_mask, gt_seq_len, gt_idx)

		return loss_CNN, loss_RNN, pred_boundary, pred_vertices, prob

	def predict_mask(self, aa):
		img = tf.reshape(aa, [1, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		feature, pred_boundary, pred_vertices = self.CNN(img, reuse = True)
		return feature, pred_boundary, pred_vertices

	def predict_path(self, ff, tt):
		#
		feature  = tf.reshape(ff, [1, self.v_out_nrow, self.v_out_ncol, 130])
		terminal = tf.reshape(tt, [1, 2, self.v_out_nrow, self.v_out_ncol, 1])

		#
		pred_v_out = self.RNN(feature, terminal, reuse = True)
		pred_v_out = tf.transpose(pred_v_out, [1, 0, 4, 2, 3])
		# print(pred_v_out.shape) # config.BEAM_WIDTH ? 6 24 24
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
