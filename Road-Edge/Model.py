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
		self.lstm_in_channel  = [136] + lstm_out_channel[: -1]
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
			loss  = tf.losses.log_loss(gt_boundary, boundary_prob)#self.L2LossWeighted(gt_boundary, boundary_prob)
			loss += tf.losses.log_loss(gt_vertices, vertices_prob)#self.L2LossWeighted(gt_vertices, vertices_prob)
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
			logits = tf.layers.dense(inputs = output_reshape, units = self.res_num + 1, activation = None)
		if not reuse:
			loss = tf.reduce_sum(
				tf.nn.softmax_cross_entropy_with_logits_v2(labels = gt_rnn_out, logits = logits)
			) / tf.reduce_sum(gt_seq_len)
			return logits, loss
		else:
			prob = tf.nn.softmax(logits)
			val, idx = tf.nn.top_k(prob[:, 0, :], k = config.BEAM_WIDTH)
			return tf.log(val), tf.gather(self.vertex_pool, idx, axis = 0)

	def RNN(self, feature, terminal, v_in = None, gt_rnn_out = None, gt_seq_len = None, reuse = None):
		batch_size = tf.concat([[tf.shape(terminal)[0]], [1, 1, 1]], 0)
		initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
			c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
			h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
		) for i in range(len(self.lstm_out_channel))])
		if not reuse:
			feature_rep = tf.reshape(tf.tile(tf.reshape(feature,
				[config.AREA_TRAIN_BATCH, 1, 1, self.v_out_nrow, self.v_out_ncol, 132]),
				[1, config.TRAIN_NUM_PATH, self.max_num_vertices, 1, 1, 1]),
				[config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 132]
			)
			v_in_0 = tf.tile(terminal[:, 1: 2, ...], [1, self.max_num_vertices, 1, 1, 1])
			v_in_e = tf.tile(terminal[:, 1: 2, ...], [1, self.max_num_vertices, 1, 1, 1])
			v_in_1 = v_in
			v_in_2 = tf.stack([v_in[:, 0, ...]] + tf.unstack(v_in, axis = 1)[: -1], axis = 1)
			rnn_input = tf.concat([feature_rep, v_in_0, v_in_1, v_in_2, v_in_e], axis = 4)
			# v_in_0:   0 0 0 0 0 ... 0
			# v_in_1:   0 1 2 3 4 ... N - 1
			# v_in_2:   0 0 1 2 3 ... N - 2
			# rnn_out:  1 2 3 4 5 ... N
			outputs, state = tf.nn.dynamic_rnn(cell = self.stacked_lstm, inputs = rnn_input,
				sequence_length = gt_seq_len, initial_state = initial_state, dtype = tf.float32
			)
			return self.FC(outputs, gt_rnn_out, gt_seq_len)
		else:
			# current prob, time line, current state
			rnn_prob = [tf.zeros([1]) for _ in range(config.BEAM_WIDTH)]
			rnn_time = [terminal[:, 0, ...] for _ in range(config.BEAM_WIDTH)]
			rnn_stat = [initial_state for _ in range(config.BEAM_WIDTH)]

			# beam search
			for i in range(1, self.max_num_vertices):
				prob, time, cell = [], [], [[[], []] for item in self.lstm_out_channel]
				for j in range(config.BEAM_WIDTH):
					prob_last = tf.tile(tf.expand_dims(rnn_prob[j], 1), [1, config.BEAM_WIDTH])
					v_first = terminal[:, 1, ...] # rnn_time[j][..., 0: 1]
					v_last_ = rnn_time[j][..., i - 1: i]
					v__last = rnn_time[j][..., max(i - 2, 0): max(i - 2, 0) + 1]
					inputs = tf.concat([feature, v_first, v_last_, v__last, terminal[:, 1, ...]], 3)
					outputs, states = self.stacked_lstm(inputs = inputs, state = rnn_stat[j])
					prob_new, time_new = self.FC(rnn_output = outputs, reuse = True)
					time_new = tf.transpose(time_new, [0, 2, 3, 1])
					prob.append(prob_last + prob_new)
					for k, item in enumerate(states):
						for l in [0, 1]:
							cell[k][l].append(tf.tile(tf.expand_dims(item[l], 1), [1, config.BEAM_WIDTH, 1, 1, 1]))
					for k in range(config.BEAM_WIDTH):
						time.append(tf.concat([rnn_time[j], time_new[..., k: k + 1]], 3))
				prob = tf.concat(prob, 1)
				val, idx = tf.nn.top_k(prob, k = config.BEAM_WIDTH)
				idx = tf.stack([tf.tile(tf.expand_dims(tf.range(tf.shape(prob)[0]), 1), [1, config.BEAM_WIDTH]), idx], 2)
				time = tf.stack(time, 1)
				ret = tf.gather_nd(time, idx)
				for k, item in enumerate(states):
					for l in [0, 1]:
						cell[k][l] = tf.gather_nd(tf.concat(cell[k][l], 1), idx)

				# Update every timeline
				for j in range(config.BEAM_WIDTH):
					rnn_prob[j] = val[..., j]
					rnn_time[j] = ret[:, j, ...]
					rnn_stat[j] = tuple([tf.contrib.rnn.LSTMStateTuple(c = item[0][:, j], h = item[1][:, j]) for item in cell])
			return tf.stack(rnn_time, 1)

	def train(self, aa, bb, vv, ii, oo, tt, ee, ll):
		#
		img          = tf.reshape(aa, [config.AREA_TRAIN_BATCH, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		gt_boundary  = tf.reshape(bb, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol])
		gt_vertices  = tf.reshape(vv, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol])
		gt_v_in      = tf.reshape(ii, [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 1])
		gt_v_out     = tf.reshape(oo, [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, self.max_num_vertices, self.res_num])
		gt_terminal  = tf.reshape(tt, [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, 2, self.v_out_nrow, self.v_out_ncol, 1])
		gt_end       = tf.reshape(ee, [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH, self.max_num_vertices, 1])
		gt_seq_len   = tf.reshape(ll, [config.AREA_TRAIN_BATCH * config.TRAIN_NUM_PATH])
		gt_rnn_out   = tf.concat([gt_v_out, gt_end], 2)

		# PolygonRNN part
		feature, pred_boundary, pred_vertices, loss_CNN = self.CNN(img, gt_boundary, gt_vertices)
		logits , loss_RNN = self.RNN(feature, gt_terminal, gt_v_in, gt_rnn_out, gt_seq_len)

		# 
		pred_rnn      = tf.nn.softmax(logits)
		pred_v_out    = tf.reshape(pred_rnn[..., 0: self.res_num], [-1, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol])
		pred_v_out    = tf.concat([gt_terminal[:, 0: 1, :, :, 0], pred_v_out[:, :-1, ...]], axis = 1)
		pred_end      = tf.reshape(pred_rnn[..., self.res_num], [-1, self.max_num_vertices])

		return loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end

	def predict_mask(self, aa):
		img = tf.reshape(aa, [1, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		feature, pred_boundary, pred_vertices = self.CNN(img, reuse = True)
		return feature, pred_boundary, pred_vertices

	def predict_path(self, ff, tt):
		#
		feature  = tf.reshape(ff, [1, self.v_out_nrow, self.v_out_ncol, 132])
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
