import numpy as np
import tensorflow as tf
from Config import *
from BasicModel import *
from UtilityBoxAnchor import *

config = Config()

class HybridModel(object):
	def __init__(self, max_num_vertices, lstm_out_channel, v_out_res, mode):
		"""
			max_num_vertices  : scalar
			lstm_out_channel  : list of int numbers
			v_out_res         : [n_row, n_col]
		"""
		# RolygonRNN parameters
		self.max_num_vertices = max_num_vertices
		self.lstm_out_channel = lstm_out_channel
		self.lstm_in_channel  = [135] + lstm_out_channel[: -1]
		self.v_out_res        = v_out_res
		self.v_out_nrow       = self.v_out_res[0]
		self.v_out_ncol       = self.v_out_res[1]
		self.res_num          = self.v_out_nrow * self.v_out_ncol
		self.mode             = mode

		# FPN parameters
		self.anchors          = generatePyramidAnchors(
			config.ANCHOR_SCALE, config.ANCHOR_RATIO, config.FEATURE_SHAPE, config.FEATURE_STRIDE
		)
		self.num_anchors      = self.anchors.shape[0]

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

	def RPN(self, img, reuse = None):
		"""
			img               : [batch_size, height, width, 3]
			logit             : [batch_size, num_anchors, 2]
			delta             : [batch_size, num_anchors, 4]
		"""
		# vgg16_res = VGG16(img, reuse)
		# p3, p4, p5, p6 = PyramidAnchorFeature(vgg16_res, reuse)
		resnet_res = ResNetV1_50('ResNetV1_50', img, reuse)
		p3, p4, p5, p6 = PyramidAnchorFeatureResNet('PyramidAnchorFeatureResNet', resnet_res, reuse)

		p3_logit, p3_delta = SingleLayerFPN(p3, len(config.ANCHOR_RATIO), reuse)
		p4_logit, p4_delta = SingleLayerFPN(p4, len(config.ANCHOR_RATIO), reuse = True)
		p5_logit, p5_delta = SingleLayerFPN(p5, len(config.ANCHOR_RATIO), reuse = True)
		p6_logit, p6_delta = SingleLayerFPN(p6, len(config.ANCHOR_RATIO), reuse = True)
		logit = tf.concat([p3_logit, p4_logit, p5_logit, p6_logit], axis = 1)
		delta = tf.concat([p3_delta, p4_delta, p5_delta, p6_delta], axis = 1)
		if self.mode == 'train':
			self.resnet_res = resnet_res
			return logit, delta
		else:
			return logit, delta, resnet_res

	def L2LossWeighted(self, gt, pred):
		batch_size = tf.cast(tf.shape(gt)[0], tf.float32)
		nt, nf = tf.reduce_sum(gt) / batch_size, tf.reduce_sum(1 - gt) / batch_size
		nt = tf.maximum(tf.minimum(nt, 0.999999), 0.000001)
		nf = tf.maximum(tf.minimum(nf, 0.999999), 0.000001)
		weights = gt * 0.5 / nt + (1 - gt) * 0.5 / nf
		weights = weights / tf.reduce_sum(weights)
		return tf.reduce_sum(tf.multiply(tf.square(gt - pred) * 33, weights))

	def CNN(self, crop_info, resnet_res = None, gt_boundary = None, gt_vertices = None, reuse = None):
		"""
			crop_info         : [batch_size, 5]
			gt_boundary       : [batch_size, height, width, 1]
			gt_vertices       : [batch_size, height, width, 1]
			combine           : [batch_size, height, width, 130]
			v_first           : 
			prob              :
		"""
		if self.mode == 'train':
			feature = SkipFeatureResNet('SkipFeatureResNet', self.resnet_res, crop_info, reuse)
		else:
			assert(resnet_res is not None)
			feature = SkipFeatureResNet('SkipFeatureResNet', resnet_res, crop_info, reuse)
		with tf.variable_scope('CNN', reuse = reuse):
			boundary = tf.layers.conv2d(inputs = feature, filters = 2, kernel_size = (1, 1), padding = 'valid', activation = None)
			combine  = tf.concat([feature, boundary], 3)
			vertices = tf.layers.conv2d(inputs = combine, filters = 2, kernel_size = (1, 1), padding = 'valid', activation = None)
		boundary_prob = tf.nn.softmax(boundary)[..., 0: 1]
		vertices_prob = tf.nn.softmax(vertices)[..., 0: 1]
		combine = tf.concat([feature, boundary, vertices], 3)
		if not reuse:
			loss  = self.L2LossWeighted(gt_boundary, boundary_prob)
			loss += self.L2LossWeighted(gt_vertices, vertices_prob)
			return combine, loss
		else:
			prob, idx = tf.nn.top_k(tf.reshape(vertices_prob, [-1, self.res_num]), k = config.BEAM_WIDTH)
			v_first = tf.gather(self.vertex_pool, idx, axis = 0)
			return combine, (v_first, prob)

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

	def RNN(self, feature, v_in = None, gt_rnn_out = None, gt_seq_len = None, v_first_with_prob = None, reuse = None):
		batch_size = tf.concat([[tf.shape(feature)[0]], [1, 1, 1]], 0)
		initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
			c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
			h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
		) for i in range(len(self.lstm_out_channel))])
		if not reuse:
			feature_rep = tf.tile(
				tf.reshape(feature, [-1, 1, self.v_out_nrow, self.v_out_ncol, 132]),
				[1, self.max_num_vertices, 1, 1, 1]
			)
			v_in_0 = tf.tile(v_in[:, 0: 1, ...], [1, self.max_num_vertices, 1, 1, 1])
			v_in_1 = v_in
			v_in_2 = tf.stack([v_in[:, 0, ...]] + tf.unstack(v_in, axis = 1)[: -1], axis = 1)
			rnn_input = tf.concat([feature_rep, v_in_0, v_in_1, v_in_2], axis = 4)
			# v_in_0:   0 0 0 0 0 ... 0
			# v_in_1:   0 1 2 3 4 ... N - 1
			# v_in_2:   0 0 1 2 3 ... N - 2
			# rnn_out:  1 2 3 4 5 ... N
			outputs, state = tf.nn.dynamic_rnn(cell = self.stacked_lstm, inputs = rnn_input,
				sequence_length = gt_seq_len, initial_state = initial_state, dtype = tf.float32
			)
			return self.FC(outputs, gt_rnn_out, gt_seq_len)
		else:
			v_first, prob = v_first_with_prob
			log_prob = tf.log(prob)

			# current prob, time line, current state
			rnn_prob = [log_prob[:, j] for j in range(config.BEAM_WIDTH)]
			rnn_time = [tf.expand_dims(v_first[:, j, ...], 3) for j in range(config.BEAM_WIDTH)]
			rnn_stat = [initial_state for j in range(config.BEAM_WIDTH)]

			# beam search
			for i in range(1, self.max_num_vertices):
				prob, time, cell = [], [], [[[], []] for item in self.lstm_out_channel]
				for j in range(config.BEAM_WIDTH):
					prob_last = tf.tile(tf.expand_dims(rnn_prob[j], 1), [1, config.BEAM_WIDTH])
					v_first = rnn_time[j][..., 0: 1]
					v_last_ = rnn_time[j][..., i - 1: i]
					v__last = rnn_time[j][..., max(i - 2, 0): max(i - 2, 0) + 1]
					inputs = tf.concat([feature, v_first, v_last_, v__last], 3)
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

	def smoothL1Loss(self, labels, predictions):
		diff = tf.abs(predictions - labels)
		val = tf.where(tf.less(diff, 1), 0.5 * tf.square(diff), diff - 0.5)
		return tf.reduce_mean(val)

	def RPNClassLoss(self, anchor_class, pred_logit):
		indices = tf.where(tf.equal(tf.reduce_sum(anchor_class, 2), 1)) # num_valid_anchors, 2
		logits = tf.gather_nd(pred_logit, indices) # num_valid_anchors, 2
		labels = tf.gather_nd(anchor_class, indices) # num_valid_anchors, 2
		pos_num = tf.reduce_sum(labels[:, 0])
		neg_num = tf.reduce_sum(labels[:, 1])
		w = labels[:, 0] * 0.5 / pos_num + labels[:, 1] * 0.5 / neg_num
		prob = tf.nn.softmax(logits)
		loss  = tf.reduce_sum(tf.losses.log_loss(labels = labels[..., 0], predictions = prob[..., 0], weights = w))
		return loss

	def RPNDeltaLoss(self, anchor_class, anchor_delta, pred_delta):
		indices = tf.where(tf.equal(anchor_class[..., 0], 1)) # num_pos_anchors, 2
		labels = tf.gather_nd(anchor_delta, indices) # num_pos_anchors, 4
		return self.smoothL1Loss(labels = labels, predictions = tf.gather_nd(pred_delta, indices))

	def train(self, aa, cc, dd, pp, ii, bb, vv, oo, ee, ll):
		#
		img          = tf.reshape(aa, [config.AREA_TRAIN_BATCH, config.AREA_SIZE[0], config.AREA_SIZE[0], 3])
		anchor_class = tf.reshape(cc, [-1, self.num_anchors, 2])
		anchor_delta = tf.reshape(dd, [-1, self.num_anchors, 4])
		crop_info    = tf.reshape(pp, [-1, 5])
		v_in         = tf.reshape(ii, [-1, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol, 1])
		gt_boundary  = tf.reshape(bb, [-1, self.v_out_nrow, self.v_out_ncol, 1])
		gt_vertices  = tf.reshape(vv, [-1, self.v_out_nrow, self.v_out_ncol, 1])
		gt_v_out     = tf.reshape(oo, [-1, self.max_num_vertices, self.res_num])
		gt_end       = tf.reshape(ee, [-1, self.max_num_vertices, 1])
		gt_seq_len   = tf.reshape(ll, [-1])
		gt_rnn_out   = tf.concat([gt_v_out, gt_end], 2)

		# RPN part
		pred_logit, pred_delta = self.RPN(img)
		loss_class = 10000 * self.RPNClassLoss(anchor_class, pred_logit)
		loss_delta = 3 * self.RPNDeltaLoss(anchor_class, anchor_delta, pred_delta)

		# PolygonRNN part
		feature, loss_CNN = self.CNN(crop_info, None, gt_boundary, gt_vertices)
		logits , loss_RNN = self.RNN(feature, v_in, gt_rnn_out, gt_seq_len)

		# 
		pred_boundary = feature[..., -2]
		pred_vertices = feature[..., -1]
		pred_rnn      = tf.nn.softmax(logits)
		pred_v_out    = tf.reshape(pred_rnn[..., 0: self.res_num], [-1, self.max_num_vertices, self.v_out_nrow, self.v_out_ncol])
		pred_end      = tf.reshape(pred_rnn[..., self.res_num], [-1, self.max_num_vertices])

		pred_pos_score = tf.nn.softmax(pred_logit)[..., 0]
		anchors_rep = tf.tile(tf.expand_dims(self.anchors, 0), [config.AREA_TRAIN_BATCH, 1, 1])
		pred_bbox  = self.applyDeltaToAnchor(anchors_rep, pred_delta)

		#
		pred_box = []
		pred_score = []
		for i in range(config.AREA_TRAIN_BATCH):
			idx_top = tf.nn.top_k(pred_pos_score[i], config.NMS_TOP_K).indices
			box_top = tf.gather(pred_bbox[i], idx_top)
			score_top = tf.gather(pred_pos_score[i], idx_top)
			idx = tf.where(score_top >= config.NMS_MIN_CONFIDENCE)
			box = tf.gather(box_top, idx)[:, 0, :]
			score = tf.gather(score_top, idx)[:, 0]
			indices = tf.image.non_max_suppression(
				boxes = box,
				scores = score,
				max_output_size = config.NMS_MAX_NUM_INS,
				iou_threshold = config.NMS_IOU_THRESHOLD,
			)
			pred_box.append(tf.gather(box, indices))
			pred_score.append(tf.gather(score, indices))

		return loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end, pred_score, pred_box

	def predict_rpn(self, aa, img_batch_size):
		#
		img          = tf.reshape(aa, [-1, config.AREA_SIZE[0], config.AREA_SIZE[1], 3])

		#
		pred_logit, pred_delta, resnet_res = self.RPN(img, reuse = True)
		pred_pos_score = tf.nn.softmax(pred_logit)[..., 0]
		anchors_rep = tf.tile(tf.expand_dims(self.anchors, 0), tf.concat([[tf.shape(img)[0]], [1, 1]], 0))
		pred_bbox  = self.applyDeltaToAnchor(anchors_rep, pred_delta)

		#
		pred_box = []
		pred_score = []
		for i in range(img_batch_size):
			idx_top = tf.nn.top_k(pred_pos_score[i], config.NMS_TOP_K).indices
			box_top = tf.gather(pred_bbox[i], idx_top)
			score_top = tf.gather(pred_pos_score[i], idx_top)
			idx = tf.where(score_top >= config.NMS_MIN_CONFIDENCE)
			box = tf.gather(box_top, idx)[:, 0, :]
			score = tf.gather(score_top, idx)[:, 0]
			indices = tf.image.non_max_suppression(
				boxes = box,
				scores = score,
				max_output_size = config.NMS_MAX_NUM_INS,
				iou_threshold = config.NMS_IOU_THRESHOLD,
			)
			pred_box.append(tf.gather(box, indices))
			pred_score.append(tf.gather(score, indices))

		return pred_score, pred_box, resnet_res

	def predict_polygon(self, pp, resnet_res):
		#
		crop_info = tf.reshape(pp, [-1, 5])
		_, conv2, conv3, conv4, conv5 = resnet_res
		conv2 = tf.reshape(conv2, [-1, config.AREA_SIZE_4 [0], config.AREA_SIZE_4 [1], 256])
		conv3 = tf.reshape(conv3, [-1, config.AREA_SIZE_8 [0], config.AREA_SIZE_8 [1], 512])
		conv4 = tf.reshape(conv4, [-1, config.AREA_SIZE_16[0], config.AREA_SIZE_16[1], 1024])
		conv5 = tf.reshape(conv5, [-1, config.AREA_SIZE_32[0], config.AREA_SIZE_32[1], 2048])
		resnet_res = (None, conv2, conv3, conv4, conv5)

		#
		feature, v_first_with_prob = self.CNN(crop_info, resnet_res, reuse = True)
		pred_v_out = self.RNN(feature, v_first_with_prob = v_first_with_prob, reuse = True)

		# Return
		pred_boundary = feature[..., -2]
		pred_vertices = feature[..., -1]
		pred_v_out = tf.transpose(pred_v_out, [1, 0, 4, 2, 3])
		# print(pred_v_out.shape) config.BEAM_WIDTH ? 24 28 28
		return pred_boundary, pred_vertices, pred_v_out

	def applyDeltaToAnchor(self, anchors, deltas):
		"""
			anchors: [batch, N, (y1, x1, y2     , x2     )]
			deltas : [batch, N, (dy, dx, log(dh), log(dw))]
		"""
		#
		h  = anchors[..., 2] - anchors[..., 0]
		w  = anchors[..., 3] - anchors[..., 1]
		cy = anchors[..., 0] + 0.5 * h
		cx = anchors[..., 1] + 0.5 * w

		# 
		cy += deltas[..., 0] * h * 0.05
		cx += deltas[..., 1] * w * 0.05
		h  *= tf.exp(deltas[..., 2] * 0.1)
		w  *= tf.exp(deltas[..., 3] * 0.1)

		# 
		y1 = cy - 0.5 * h
		x1 = cx - 0.5 * w
		y2 = y1 + h
		x2 = x1 + w

		# Return
		return tf.stack([y1, x1, y2, x2], axis = 2)

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
