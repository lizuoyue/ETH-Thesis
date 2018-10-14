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
		self.lstm_in_channel  = [134] + lstm_out_channel[: -1]
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

	def FC(self, rnn_output, gt_rnn_out = None, gt_seq_len = None, gt_vertices = None, reuse = None):
		""" 
			rnn_output
			gt_rnn_out
			gt_seq_len
		"""
		if not reuse:
			output_reshape = tf.reshape(rnn_output, [config.TRAIN_NUM_PATH, self.max_num_vertices, self.res_num * self.lstm_out_channel[-1]])	
		else:
			output_reshape = tf.reshape(rnn_output, [1, 1, self.res_num * self.lstm_out_channel[-1]])
		with tf.variable_scope('FC', reuse = reuse):
			logits = tf.layers.dense(inputs = output_reshape, units = 4096, activation = tf.nn.relu)
			logits = tf.layers.dense(inputs = logits, units = 1024, activation = tf.nn.relu)
			logits = tf.layers.dense(inputs = logits, units = self.res_num + 1, activation = None)
		if not reuse:
			# gt_vertices = tf.reshape(gt_vertices, [config.TRAIN_NUM_PATH, self.max_num_vertices, self.res_num])
			# gt_vertices = tf.concat([gt_vertices, tf.ones([config.TRAIN_NUM_PATH, self.max_num_vertices, 1])], axis = -1)
			# logits -= 1e10 * (1 - gt_vertices)
			loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = gt_rnn_out, logits = logits)
			loss = tf.reduce_sum(loss) / tf.reduce_sum(gt_seq_len)
			return logits, loss
		else:
			prob = tf.nn.softmax(logits)
			val, idx = tf.nn.top_k(prob[0, 0, :], k = config.BEAM_WIDTH)
			return tf.log(val), tf.expand_dims(tf.gather(self.vertex_pool, idx, axis = 0), axis = 3), prob[0, 0, :]

	def RNN(self, feature, terminal, v_in = None, gt_rnn_out = None, gt_seq_len = None, gt_idx = None, reuse = None):
		batch_size = tf.concat([[tf.shape(terminal)[0]], [1, 1, 1]], 0)
		initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
			c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
			h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
		) for i in range(len(self.lstm_out_channel))])
		if not reuse:
			feature_rep = tf.gather(feature, gt_idx)
			feature_rep = tf.tile(tf.expand_dims(feature_rep, axis = 1), [1, self.max_num_vertices, 1, 1, 1])
			v_in_0 = tf.tile(terminal[:, 0: 1, ...], [1, self.max_num_vertices, 1, 1, 1])
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
			return self.FC(outputs, gt_rnn_out, gt_seq_len, feature_rep[..., -1])
		else:
			# current prob, time line, current state
			rnn_prob = [tf.zeros([1])] + [tf.ones([1]) * -99999999 for _ in range(config.BEAM_WIDTH - 1)]
			rnn_tmln = [terminal[:, 0, ...] for _ in range(config.BEAM_WIDTH)]
			rnn_stat = [initial_state for _ in range(config.BEAM_WIDTH)]
			rnn_hmap = [tf.zeros([785, 1]) for _ in range(config.BEAM_WIDTH)]

			# beam search
			for i in range(1, self.max_num_vertices + 1):
				prob, tmln, stat, hmap = [], [], [[[], []] for item in self.lstm_out_channel], []
				for j in range(config.BEAM_WIDTH):
					prob_last = tf.tile(rnn_prob[j], [config.BEAM_WIDTH])
					v_in_0 = terminal[:, 0, ...]
					v_in_e = terminal[:, 1, ...]
					v_in_1 = rnn_tmln[j][..., i - 1: i]
					v_in_2 = rnn_tmln[j][..., max(i - 2, 0): max(i - 2, 0) + 1]
					inputs = tf.concat([feature, v_in_0, v_in_1, v_in_2, v_in_e], 3)
					print(v_in_1.shape)
					outputs, states = self.stacked_lstm(inputs = inputs, state = rnn_stat[j])
					prob_new, time_new, prob_hmap = self.FC(rnn_output = outputs, reuse = True)
					print(prob_new.shape)
					print(time_new.shape)
					print(prob_hmap.shape)
					quit()
					prob.append(prob_last + prob_new)
					### deal with each state
					for k, item in enumerate(states):
						for l in range(2):
							stat[k][l].append(tf.tile(tf.expand_dims(item[l], 0), [config.BEAM_WIDTH, 1, 1, 1, 1]))
					########################
					for k in range(config.BEAM_WIDTH):
						tmln.append(tf.concat([rnn_tmln[j], time_new[k: k + 1]], 3))
					for k in range(config.BEAM_WIDTH):
						hmap.append(tf.concat([rnn_hmap[j], tf.expand_dims(prob_hmap, 1)], 1))
				prob = tf.concat(prob, 0)
				val, idx = tf.nn.top_k(prob, k = config.BEAM_WIDTH)
				tmln = tf.gather(tf.stack(tmln, 0), idx)
				hmap = tf.gather(tf.stack(hmap, 0), idx)
				### deal with each state
				for k, item in enumerate(states):
					for l in range(2):
						stat[k][l] = tf.gather(tf.concat(stat[k][l], 0), idx)
				########################
				# Update every timeline
				for j in range(config.BEAM_WIDTH):
					rnn_prob[j] = val[j: j + 1]
					rnn_tmln[j] = tmln[j]
					rnn_stat[j] = tuple([tf.contrib.rnn.LSTMStateTuple(c = item[0][j], h = item[1][j]) for item in stat])
					rnn_hmap[j] = hmap[j]

			return tf.transpose(tf.stack(rnn_tmln, 0), [0, 4, 2, 3, 1]), tf.transpose(tf.stack(rnn_hmap, 0), [0, 2, 1])

			### Full Search ###
			# # current prob, time line, current state
			# rnn_prob = [tf.zeros([1])]
			# rnn_tmln = [terminal[:, 0, ...]]
			# rnn_stat = [initial_state]

			# # bfs
			# for i in range(1 + 5):
			# 	last_prob, last_tmln, last_stat = rnn_prob.pop(0), rnn_tmln.pop(0), rnn_stat.pop(0)
			# 	v_in_0 = terminal[:, 0, ...]
			# 	v_in_e = terminal[:, 1, ...]
			# 	n = last_tmln.get_shape().as_list()[-1]
			# 	v_in_1 = last_tmln[..., max(n - 1, 0): max(n - 1, 0) + 1]
			# 	v_in_2 = last_tmln[..., max(n - 2, 0): max(n - 2, 0) + 1]
			# 	inputs = tf.concat([feature, v_in_0, v_in_1, v_in_2, v_in_e], 3)
			# 	outputs, states = self.stacked_lstm(inputs = inputs, state = last_stat)
			# 	prob_new, tmln_new, _ = self.FC(rnn_output = outputs, reuse = True)
			# 	### update prob
			# 	for j in range(config.BEAM_WIDTH):
			# 		rnn_prob.append(last_prob + prob_new[j: j + 1])
			# 	### update tmln
			# 	for j in range(config.BEAM_WIDTH):
			# 		rnn_tmln.append(tf.concat([last_tmln, tmln_new[j: j + 1]], 3))
			# 	### update stat
			# 	for j in range(config.BEAM_WIDTH):
			# 		rnn_stat.append(states)

			# return tf.transpose(tf.stack(rnn_tmln, 0), [0, 4, 2, 3, 1])

			### No Beam Search ###
			# res = [terminal[:, 0, ...]]
			# prob_res = []
			# states = [initial_state]
			# for i in range(1, self.max_num_vertices + 1):
			# 	rnn_input = tf.concat([feature, terminal[:, 0, ...], res[i - 1], res[max(i - 2, 0)], terminal[:, 1, ...]], 3)
			# 	rnn_output, state = self.stacked_lstm(inputs = rnn_input, state = states[-1])
			# 	states.append(state)
			# 	next_v, prob = self.FC(rnn_output = rnn_output, reuse = True)
			# 	res.append(next_v)
			# 	prob_res.append(prob)
			# return tf.stack(res, 1), tf.stack(prob_res, 0)

	def RNN_tmp(self, feature, terminal, indices):
		batch_size = tf.concat([[tf.shape(terminal)[0]], [1, 1, 1]], 0)
		initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
			c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
			h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
		) for i in range(len(self.lstm_out_channel))])
		res = [tf.expand_dims(tf.gather(self.vertex_pool, tf.ones([1], dtype = tf.int32) * idx, axis = 0), axis = 3) for idx in indices]
		prob_res = []
		states = [initial_state]
		for i in range(1, len(indices) + 1):
			rnn_input = tf.concat([feature, terminal[:, 0, ...], res[i - 1], res[max(i - 2, 0)], terminal[:, 1, ...]], 3)
			rnn_output, state = self.stacked_lstm(inputs = rnn_input, state = states[-1])
			states.append(state)
			_, prob = self.FC(rnn_output = rnn_output, reuse = True)
			prob_res.append(prob)
		return tf.stack(res, 1), tf.stack(prob_res, 0)

	def train(self, aa, bb, vv, ii, oo, tt, ee, ll, dd):
		#
		img          = tf.reshape(aa, [config.AREA_TRAIN_BATCH, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		gt_boundary  = tf.reshape(bb, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol, 1])
		gt_vertices  = tf.reshape(vv, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol, 1])

		gt_v_in      = tf.reshape(ii, [config.TRAIN_NUM_PATH, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 1])
		gt_v_out     = tf.reshape(oo, [config.TRAIN_NUM_PATH, self.max_num_vertices, self.res_num])
		gt_terminal  = tf.reshape(tt, [config.TRAIN_NUM_PATH, 2, self.v_out_nrow, self.v_out_ncol, 1])
		gt_end       = tf.reshape(ee, [config.TRAIN_NUM_PATH, self.max_num_vertices, 1])
		gt_seq_len   = tf.reshape(ll, [config.TRAIN_NUM_PATH])
		gt_idx       = tf.reshape(dd, [config.TRAIN_NUM_PATH])
		gt_rnn_out   = tf.concat([gt_v_out, gt_end], 2)

		# PolygonRNN part
		feature, pred_boundary, pred_vertices, loss_CNN = self.CNN(img, gt_boundary, gt_vertices)
		feature_RNN = tf.concat([feature, gt_boundary, gt_vertices], axis = -1)
		logits , loss_RNN = self.RNN(feature_RNN, gt_terminal, gt_v_in, gt_rnn_out, gt_seq_len, gt_idx)

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
		feature  = tf.reshape(ff, [1, self.v_out_nrow, self.v_out_ncol, 130])
		terminal = tf.reshape(tt, [1, 2, self.v_out_nrow, self.v_out_ncol, 1])

		#
		pred_v_out, prob_res = self.RNN(feature, terminal, reuse = True)
		return pred_v_out, prob_res

	def predict_path_tmp(self, ff, tt, indices):
		#
		feature  = tf.reshape(ff, [1, self.v_out_nrow, self.v_out_ncol, 130])
		terminal = tf.reshape(tt, [1, 2, self.v_out_nrow, self.v_out_ncol, 1])

		#
		pred_v_out, prob_res = self.RNN_tmp(feature, terminal, indices)
		return pred_v_out, prob_res

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
