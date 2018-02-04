import os, sys, glob, time
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from Model import *
import Utility as ut

ANCHOR_SCALE   = [16, 32, 64, 128]
ANCHOR_RATIO   = [0.25, 0.5, 1, 2, 4]
FEATURE_SHAPE  = [[64, 64], [32, 32], [16, 16], [8, 8]]
FEATURE_STRIDE = [4, 8, 16, 32]

CHOOSE_TOP_K = 5

class PolygonRNN(object):

	def __init__(self, max_seq_len, lstm_out_channel, v_out_res):
		"""
			max_seq_len:      scalar
			lstm_out_channel: list of int numbers
			v_out_res:        list [n_row, n_col]
		"""
		# RolygonRNN parameters
		self.max_seq_len      = max_seq_len
		self.lstm_out_channel = lstm_out_channel
		self.lstm_in_channel  = [133] + lstm_out_channel[: -1]
		self.v_out_res        = v_out_res
		self.v_out_nrow       = self.v_out_res[0]
		self.v_out_ncol       = self.v_out_res[1]
		self.res_num          = self.v_out_nrow * self.v_out_ncol

		# RPN parameters
		self.anchors          = ut.generatePyramidAnchors(ANCHOR_SCALE, ANCHOR_RATIO, FEATURE_SHAPE, FEATURE_STRIDE, 1)
		self.valid_idx        = np.array([i for i in range(self.anchors.shape[0])])
		self.num_anchors      = self.anchors.shape[0]

		# Multi-layer LSTM and inital state
		self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(
			[self.ConvLSTMCell(in_c, out_c) for in_c, out_c in zip(self.lstm_in_channel, self.lstm_out_channel)]
		)
		self.lstm_init_state = [
			tf.get_variable('conv_lstm_cell_%d_c_h' % i, [2, self.v_out_nrow, self.v_out_ncol, c_out])
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

		# Return
		return


	def ConvLSTMCell(self, input_channels, output_channels):
		"""
			input_channels: scalar
			output_channels: scalar
		"""
		# Return
		return tf.contrib.rnn.ConvLSTMCell(
			conv_ndims = 2,
			input_shape = [self.v_out_nrow, self.v_out_ncol, input_channels],
			output_channels = output_channels,
			kernel_shape = [3, 3]
		)


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
		cy += deltas[..., 0] * h * 0.1
		cx += deltas[..., 1] * w * 0.1
		h  *= tf.exp(deltas[..., 2] * 0.2)
		w  *= tf.exp(deltas[..., 3] * 0.2)

		# 
		y1 = cy - 0.5 * h
		x1 = cx - 0.5 * w
		y2 = y1 + h
		x2 = x1 + w

		# Return
		return tf.stack([y1, x1, y2, x2], axis = 2)


	def CNN(self, images, gt_boundary = None, gt_vertices = None, reuse = None):
		batch_size = tf.cast(tf.shape(images)[0], tf.float32)
		feature = PolygonRNNFeature(VGG16(images, True), reuse)
		with tf.variable_scope('CNN', reuse = reuse):
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
		if not reuse:
			loss = 0.0
			n_b = tf.reduce_sum(gt_boundary) / batch_size
			n_v = tf.reduce_sum(gt_vertices) / batch_size
			loss += tf.losses.log_loss(
				labels = gt_boundary,
				predictions = boundary,
				weights = (gt_boundary * (self.res_num - 2 * n_b) + n_b)
			)
			loss += tf.losses.log_loss(
				labels = gt_vertices,
				predictions = vertices,
				weights = (gt_vertices * (self.res_num - 2 * n_v) + n_v))
			loss /= 16
			return tf.concat([feature, boundary, vertices], 3), loss
		else:
			idx = tf.nn.top_k(tf.reshape(vertices, [-1, self.res_num]), k = CHOOSE_TOP_K).indices
			# idx = tf.argmax(tf.reshape(vertices, [-1, self.res_num]), axis = 1)
			v_first = tf.gather(self.vertex_pool, idx, axis = 0)
			return tf.concat([feature, boundary, vertices], 3), tf.transpose(v_first, perm = [1, 0, 2, 3])


	def RNN(self, feature, v_in = None, gt_rnn_out = None, gt_seq_len = None, v_first = None, reuse = None):
		batch_size_1 = tf.concat([[tf.shape(feature)[0]], [1, 1, 1]], 0)
		if not reuse:
			feature_rep = tf.tile(
				tf.reshape(feature, [-1, 1, self.v_out_nrow, self.v_out_ncol, 130]),
				[1, self.max_seq_len, 1, 1, 1]
			)
			v_in_0 = tf.tile(v_in[:, 0: 1, ...], [1, self.max_seq_len, 1, 1, 1])
			v_in_1 = v_in
			v_in_2 = tf.stack([v_in[:, 0, ...]] + tf.unstack(v_in, axis = 1)[: -1], axis = 1)
			rnn_input = tf.concat([feature_rep, v_in_0, v_in_1, v_in_2], axis = 4)
			# v_in_0:   0 0 0 0 0 ... 0
			# v_in_1:   0 1 2 3 4 ... N - 1
			# v_in_2:   0 0 1 2 3 ... N - 2
			# rnn_out:  1 2 3 4 5 ... N
			initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
				c = tf.tile(self.lstm_init_state[i][0: 1], batch_size_1),
				h = tf.tile(self.lstm_init_state[i][1: 2], batch_size_1)
			) for i in range(len(lstm_out_channel))])
			outputs, state = tf.nn.dynamic_rnn(
				cell = self.stacked_lstm,
				inputs = rnn_input,
				sequence_length = gt_seq_len,
				initial_state = initial_state,
				dtype = tf.float32
			)
			logits, loss, idx = self.FC(outputs, gt_rnn_out, gt_seq_len)
			return logits, loss
		else:
			batch_size_2 = tf.concat([[tf.shape(feature)[0]], [self.v_out_nrow, self.v_out_ncol, 1]], 0)
			v = [None for i in range(self.max_seq_len)]
			state = [None for i in range(self.max_seq_len)]
			rnn_output = [None for i in range(self.max_seq_len)]
			v[0] = tf.reshape(v_first, batch_size_2)
			state[0] = tuple([tf.contrib.rnn.LSTMStateTuple(
				c = tf.tile(self.lstm_init_state[i][0: 1], batch_size_1),
				h = tf.tile(self.lstm_init_state[i][1: 2], batch_size_1)
			) for i in range(len(lstm_out_channel))])
			for i in range(1, self.max_seq_len):
				rnn_output[i], state[i] = self.stacked_lstm(
					inputs = tf.concat([feature, v[0], v[max(i - 1, 0)], v[max(i - 2, 0)]], 3),
					state = state[i - 1]
				)
				v[i] = tf.reshape(
					self.FC(
						rnn_output = rnn_output[i],
						reuse = True,
						last_two = (v[max(i - 1, 0)], v[max(i - 2, 0)]),
					),
					[-1, self.v_out_nrow, self.v_out_ncol, 1]
				)
			return tf.stack(v, 1)


	def FC(self, rnn_output, gt_rnn_out = None, gt_seq_len = None, reuse = None, last_two = None):
		if not reuse:
			output_reshape = tf.reshape(rnn_output, [-1, self.max_seq_len, self.res_num * self.lstm_out_channel[-1]])	
		else:
			output_reshape = tf.reshape(rnn_output, [-1, 1, self.res_num * self.lstm_out_channel[-1]])
		with tf.variable_scope('FC', reuse = reuse):
			logits = tf.layers.dense(
				inputs = output_reshape,
				units = self.res_num + 1,
				activation = None
			)
		if not reuse:
			loss = tf.reduce_sum(
				tf.nn.softmax_cross_entropy_with_logits(
					labels = gt_rnn_out,
					logits = logits
				)
			) / tf.reduce_sum(gt_seq_len)
			idx = tf.argmax(logits, axis = 2)
			return logits, loss, idx
		else:
			idx = tf.argmax(logits, axis = 2)
			return tf.gather(self.vertex_pool, idx, axis = 0)


	def smoothL1Loss(self, labels, predictions):
		diff = tf.abs(predictions - labels)
		val = tf.where(tf.less(diff, 1), 0.5 * tf.square(diff), diff - 0.5)
		return tf.reduce_mean(val)


	def RPN(self, images, reuse = None):
		p2, p3, p4, p5 = PyramidAnchorFeature(VGG16(images, reuse), reuse)
		p2_logit, p2_delta = RPNSingleLayer(p2, len(ANCHOR_RATIO), reuse)
		p3_logit, p3_delta = RPNSingleLayer(p3, len(ANCHOR_RATIO), reuse = True)
		p4_logit, p4_delta = RPNSingleLayer(p4, len(ANCHOR_RATIO), reuse = True)
		p5_logit, p5_delta = RPNSingleLayer(p5, len(ANCHOR_RATIO), reuse = True)
		logit = tf.concat([p2_logit, p3_logit, p4_logit, p5_logit], axis = 1)
		delta = tf.concat([p2_delta, p3_delta, p4_delta, p5_delta], axis = 1)
		return logit, delta


	def RPNClassLoss(self, anchor_class, pred_logit):
		indices = tf.where(tf.equal(tf.reduce_sum(anchor_class, 2), 1)) # num_valid_anchors, 2
		logits = tf.gather_nd(pred_logit, indices) # num_valid_anchors, 2
		labels = tf.gather_nd(anchor_class, indices) # num_valid_anchors, 2
		pos_num = tf.reduce_sum(labels[:, 0])
		neg_num = tf.reduce_sum(labels[:, 1])
		total_num = pos_num + neg_num
		w = tf.stack([labels[:, 0] * neg_num / total_num, labels[:, 1] * pos_num / total_num], axis = 1)
		return tf.reduce_mean(tf.losses.log_loss(labels = labels, predictions = tf.nn.softmax(logits), weights = w))


	def RPNDeltaLoss(self, anchor_class, anchor_delta, pred_delta):
		indices = tf.where(tf.equal(anchor_class[..., 0], 1)) # num_pos_anchors, 2
		labels = tf.gather_nd(anchor_delta, indices) # num_pos_anchors, 4
		return self.smoothL1Loss(labels = labels, predictions = tf.gather_nd(pred_delta, indices))


	def train(self, aa, cc, dd, pp, ii, bb, vv, oo, ee, ll):
		#
		images        = tf.reshape(aa, [4, 640, 640, 3])
		img           = tf.image.resize_images(images = images, size = [256, 256])
		anchor_class  = tf.reshape(cc, [-1, self.num_anchors, 2])
		anchor_delta  = tf.reshape(dd, [-1, self.num_anchors, 4])
		patches       = tf.reshape(pp, [-1, 224, 224, 3])
		v_in          = tf.reshape(ii, [-1, self.max_seq_len, self.v_out_nrow, self.v_out_ncol, 1])
		gt_boundary   = tf.reshape(bb, [-1, self.v_out_nrow, self.v_out_ncol, 1])
		gt_vertices   = tf.reshape(vv, [-1, self.v_out_nrow, self.v_out_ncol, 1])
		gt_v_out      = tf.reshape(oo, [-1, self.max_seq_len, self.res_num])
		gt_end        = tf.reshape(ee, [-1, self.max_seq_len, 1])
		gt_seq_len    = tf.reshape(ll, [-1])
		gt_rnn_out    = tf.concat([gt_v_out, gt_end], 2)

		# RPN part
		pred_logit, pred_delta = self.RPN(img)
		loss_class = 30 * self.RPNClassLoss(anchor_class, pred_logit)
		loss_delta = 10 * self.RPNDeltaLoss(anchor_class, anchor_delta, pred_delta)

		# PolygonRNN part
		feature, loss_CNN = self.CNN(patches, gt_boundary, gt_vertices)
		logits , loss_RNN = self.RNN(feature, v_in, gt_rnn_out, gt_seq_len)

		# Return
		pred_boundary = feature[..., -2]
		pred_vertices = feature[..., -1]
		pred_rnn      = tf.nn.softmax(logits)
		pred_v_out    = tf.reshape(pred_rnn[..., 0: self.res_num],
			[-1, self.max_seq_len, self.v_out_nrow, self.v_out_ncol]
		)
		pred_end      = tf.reshape(pred_rnn[..., self.res_num],
			[-1, self.max_seq_len]
		)

		pred_score = tf.nn.softmax(pred_logit)[..., 0]
		anchors_rep = tf.tile(tf.expand_dims(self.anchors, 0), [4, 1, 1])
		pred_bbox  = self.applyDeltaToAnchor(anchors_rep, pred_delta)

		#
		pred_box = []
		for i in range(4):
			box_valid = tf.gather(pred_bbox[i], self.valid_idx)
			score_valid = tf.gather(pred_score[i], self.valid_idx)
			idx_top = tf.nn.top_k(score_valid, 500).indices
			box_top = tf.gather(box_valid, idx_top)
			score_top = tf.gather(score_valid, idx_top)
			idx = tf.where(score_top >= 0.99)
			box = tf.gather(box_top, idx)[:, 0, :]
			score = tf.gather(score_top, idx)[:, 0]
			indices = tf.image.non_max_suppression(
				boxes = box,
				scores = score,
				max_output_size = 40,
				iou_threshold = 0.15
			)
			pred_box.append(tf.gather(box, indices))

		return loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end, pred_box

	def predict_rpn(self, aa):
		#
		images     = tf.reshape(aa, [4, 640, 640, 3])
		img        = tf.image.resize_images(images = images, size = [256, 256])
		batch_size = tf.shape(images)[0]

		#
		pred_logit, pred_delta = self.RPN(img, reuse = True)
		pred_score = tf.nn.softmax(pred_logit)[..., 0]
		anchors_rep = tf.tile(tf.expand_dims(self.anchors, 0), [batch_size, 1, 1])
		pred_bbox  = self.applyDeltaToAnchor(anchors_rep, pred_delta)

		#
		pred_box = []
		for i in range(4):
			box_valid = tf.gather(pred_bbox[i], self.valid_idx)
			score_valid = tf.gather(pred_score[i], self.valid_idx)
			idx_top = tf.nn.top_k(score_valid, 500).indices
			box_top = tf.gather(box_valid, idx_top)
			score_top = tf.gather(score_valid, idx_top)
			idx = tf.where(score_top >= 0.99)
			box = tf.gather(box_top, idx)[:, 0, :]
			score = tf.gather(score_top, idx)[:, 0]
			indices = tf.image.non_max_suppression(
				boxes = box,
				scores = score,
				max_output_size = 40,
				iou_threshold = 0.15
			)
			pred_box.append(tf.gather(box, indices))
		return pred_box # pred_score, pred_bbox

	def predict_polygon(self, pp):
		#
		img  = tf.reshape(pp, [-1, 224, 224, 3])

		#
		feature, v_first = self.CNN(img, reuse = True)
		pred_v_out = tf.stack([self.RNN(feature, v_first = v_first[i], reuse = True) for i in range(CHOOSE_TOP_K)])

		# Return
		pred_boundary = feature[..., -2]
		pred_vertices = feature[..., -1]
		pred_v_out = pred_v_out[..., 0]
		# print(pred_v_out.shape) 8 ? 24 28 28
		return pred_boundary, pred_vertices, pred_v_out

















def overlay(img, mask, shape, color = (255, 0, 0)):
	org = Image.fromarray(np.array(img * 255.0, dtype = np.uint8)).convert('RGBA')
	alpha = np.array(mask * 128.0, dtype = np.uint8)
	alpha = np.concatenate(
		(
			np.ones((shape[0], shape[1], 1)) * color[0],
			np.ones((shape[0], shape[1], 1)) * color[1],
			np.ones((shape[0], shape[1], 1)) * color[2],
			np.reshape(alpha, (shape[0], shape[1], 1))
		),
		axis = 2
	)
	alpha = Image.fromarray(np.array(alpha, dtype = np.uint8), mode = 'RGBA')
	alpha = alpha.resize((224, 224), resample = Image.BICUBIC)
	merge = Image.alpha_composite(org, alpha)
	return merge

def overlayMultiMask(img, mask, shape):
	merge = Image.fromarray(np.array(img * 255.0, dtype = np.uint8)).convert('RGBA')
	merge = np.array(overlay(img, mask[0], shape)) / 255.0
	for i in range(1, mask.shape[0]):
		color = (255 * (i == 1), 128 * (i == 1) + (1 - i % 2) * 255, i % 2 * 255 - 255 * (i == 1))
		merge = np.array(overlay(merge, mask[i], shape, color)) / 255.0
	return Image.fromarray(np.array(merge * 255.0, dtype = np.uint8)).convert('RGBA')

def visualize(path, img, boundary, vertices, v_in, b_pred, v_pred, v_out_pred, end_pred, seq_len, v_out_res):
	# Clear last files
	for item in glob.glob(path + '/*'):
		os.remove(item)

	# Reshape
	shape = ((v_out_res[1], v_out_res[0]))
	blank = np.zeros(shape)

	# Polygon
	polygon = [None for i in range(img.shape[0])]
	for i in range(v_out_pred.shape[0]):
		v = v_in[i, 0]
		r, c = np.unravel_index(v.argmax(), v.shape)
		polygon[i] = [(c, r)]
		for j in range(seq_len[i] - 1):
			if end_pred[i, j] <= v.max():
				v = v_out_pred[i, j]
				r, c = np.unravel_index(v.argmax(), v.shape)
				polygon[i].append((c, r))

	# 
	for i in range(img.shape[0]):
		vv = np.concatenate((v_in[i, 0: 1], v_out_pred[i, 0: seq_len[i] - 1]), axis = 0)
		overlay(img[i], blank      , shape).save(path + '/%d-0-img.png' % i)
		overlay(img[i], boundary[i], shape).save(path + '/%d-1-bound.png' % i)
		overlay(img[i], b_pred  [i], shape).save(path + '/%d-1-bound-pred.png' % i)
		overlay(img[i], vertices[i], shape).save(path + '/%d-2-vertices.png' % i)
		overlay(img[i], v_pred  [i], shape).save(path + '/%d-2-vertices-pred.png' % i)
		overlayMultiMask(img[i], vv, shape).save(path + '/%d-3-vertices-merge.png' % i)
		# for j in range(seq_len[i]):
		# 	overlay(img[i], vv[j], shape).save(path + '/%d-3-vtx-%s.png' % (i, str(j).zfill(2)))

		link = Image.new('P', shape, color = 0)
		draw = ImageDraw.Draw(link)
		if len(polygon[i]) == 1:
			polygon[i].append(polygon[i][0])
		draw.polygon(polygon[i], fill = 0, outline = 255)
		link = np.array(link) / 255.0
		overlay(img[i], link, shape).save(path + '/%d-4-vertices-link.png' % i)

		f = open(path + '/%d-5-end-prob.txt' % i, 'w')
		for j in range(seq_len[i]):
			f.write('%.6lf\n' % end_pred[i, j])
		f.close()

	#
	return

def visualize_pred(path, img, b_pred, v_pred, v_out_pred, v_out_res):
	if not os.path.exists(path):
		os.makedirs(path)
	# Clear last files
	for item in glob.glob(path + '/*'):
		os.remove(item)

	# Reshape
	batch_size = img.shape[0]
	shape = ((v_out_res[1], v_out_res[0]))
	blank = np.zeros(shape)

	# Sequence length and polygon
	polygon = [[] for i in range(batch_size)]
	for i in range(v_out_pred.shape[0]):
		for j in range(v_out_pred.shape[1]):
			v = v_out_pred[i, j]
			if v.sum() >= 0.5:
				r, c = np.unravel_index(v.argmax(), v.shape)
				polygon[i].append((c, r))
			else:
				break
	seq_len = [len(polygon[i]) for i in range(batch_size)]

	# 
	for i in range(batch_size):
		vv = v_out_pred[i, 0: seq_len[i]]
		overlay(img[i], blank      , shape).save(path + '/%d-0-img.png' % i)
		overlay(img[i], b_pred[i]  , shape).save(path + '/%d-1-bound-pred.png' % i)
		overlay(img[i], v_pred[i]  , shape).save(path + '/%d-2-vertices-pred.png' % i)
		overlayMultiMask(img[i], vv, shape).save(path + '/%d-3-vertices-merge.png' % i)
		# for j in range(seq_len[i]):
		# 	overlay(img[i], vv[j], shape).save(path + '/%d-3-vtx-%s.png' % (i, str(j).zfill(2)))
		link = Image.new('P', shape, color = 0)
		draw = ImageDraw.Draw(link)
		if len(polygon[i]) == 1:
			polygon[i].append(polygon[i][0])
		draw.polygon(polygon[i], fill = 0, outline = 255)
		link = np.array(link) / 255.0
		overlay(img[i], link, shape).save(path + '/%d-4-vertices-link.png' % i)

	# 
	return















class Logger(object):

	def __init__(self, log_dir):
		self.writer = tf.summary.FileWriter(log_dir)
		return

	def log_scalar(self, tag, value, step):
		summary = tf.Summary(
			value = [
				tf.Summary.Value(
					tag = tag,
					simple_value = value
				)
			]
		)
		self.writer.add_summary(summary, step)
		return

	def close(self):
		self.writer.close()
		return

if __name__ == '__main__':
	# Create new folder
	if not os.path.exists('./model/'):
		os.makedirs('./model/')
	# if not os.path.exists('./val/'):
	# 	os.makedirs('./val/')
	# if not os.path.exists('./pre/'):
	# 	os.makedirs('./pre/')
	# if not os.path.exists('./tes/'):
	# 	os.makedirs('./tes/')

	# Set parameters
	n_iter = 100000
	building_path = '../../Chicago.zip'
	area_path = '/local/lizuoyue/Chicago_Area'
	max_seq_len = 24
	lr = 0.0005
	lstm_out_channel = [32, 16, 8]
	v_out_res = (28, 28)
	area_batch_size = 4
	building_batch_size = 12

	# Create data generator
	obj = ut.DataGenerator(building_path, area_path, max_seq_len, (224, 224), v_out_res)

	# Define graph
	PolyRNNGraph = PolygonRNN(
		max_seq_len = max_seq_len,
		lstm_out_channel = lstm_out_channel, 
		v_out_res = v_out_res,
	)
	aa = tf.placeholder(tf.float32)
	cc = tf.placeholder(tf.float32)
	dd = tf.placeholder(tf.float32)
	pp = tf.placeholder(tf.float32)
	ii = tf.placeholder(tf.float32)
	bb = tf.placeholder(tf.float32)
	vv = tf.placeholder(tf.float32)
	oo = tf.placeholder(tf.float32)
	ee = tf.placeholder(tf.float32)
	ll = tf.placeholder(tf.float32)

	train_res     = PolyRNNGraph.train(aa, cc, dd, pp, ii, bb, vv, oo, ee, ll)
	pred_rpn_res  = PolyRNNGraph.predict_rpn(aa)
	pred_poly_res = PolyRNNGraph.predict_polygon(pp)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	train = optimizer.minimize(train_res[0] + train_res[1] + train_res[2] + train_res[3])
	saver = tf.train.Saver(max_to_keep = 3)
	init = tf.global_variables_initializer()

	# Launch graph
	with tf.Session() as sess:
		# Create loggers
		f = open('./PolygonRNN-%d.out' % v_out_res[1], 'a')
		train_writer = Logger('./log/train/')
		valid_writer = Logger('./log/valid/')

		# Restore weights
		if len(sys.argv) > 1 and sys.argv[1] != None:
			saver.restore(sess, './model/model-%s.ckpt' % sys.argv[1])
			iter_obj = range(int(sys.argv[1]) + 1, n_iter)
		else:
			sess.run(init)
			iter_obj = range(n_iter)

		# Main loop
		for i in iter_obj:
			# Get training batch data and create feed dictionary
			img, anchor_cls, anchor_box = obj.getAreasBatch(area_batch_size, mode = 'train')
			patch, boundary, vertices, v_in, v_out, end, seq_len = obj.getBuildingsBatch(building_batch_size, mode = 'train')
			feed_dict = {
				aa: img, cc: anchor_cls, dd: anchor_box,
				pp: patch, ii: v_in, bb: boundary, vv: vertices, oo: v_out, ee: end, ll: seq_len
			}

			# Training and get result
			init_time = time.time()
			_, (loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end, pred_box) = sess.run([train, train_res], feed_dict)
			cost_time = time.time() - init_time
			train_writer.log_scalar('Loss Class', loss_class, i)
			train_writer.log_scalar('Loss Delta', loss_delta, i)
			train_writer.log_scalar('Loss CNN'  , loss_CNN  , i)
			train_writer.log_scalar('Loss RNN'  , loss_RNN  , i)
			train_writer.log_scalar('Loss RPN'  , loss_class + loss_delta, i)
			train_writer.log_scalar('Loss Poly' , loss_CNN   + loss_RNN  , i)
			train_writer.log_scalar('Loss Full' , loss_class + loss_delta + loss_CNN + loss_RNN, i)
			
			# Write loss to file
			print('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time))
			f.write('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf\n' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time))
			f.flush()

			# Visualize
			if i % 20 == 1:
				img, anchor_cls, anchor_box = obj.getAreasBatch(area_batch_size, mode = 'valid')
				feed_dict = {aa: img}
				pred_box = sess.run(pred_rpn_res, feed_dict = feed_dict)
				org_img, patch, org_info = obj.getPatchesFromAreas(pred_box)
				feed_dict = {pp: patch}
				pred_boundary, pred_vertices, pred_v_out = sess.run(pred_poly_res, feed_dict = feed_dict)
				# visualize_pred('./res-train', patch, pred_boundary, pred_vertices, pred_v_out, (28, 28))
				path = './res-train-%d' % (int((i - 1) / 20) % 10)
				if not os.path.exists(path):
					os.makedirs(path)
				for item in glob.glob(path + '/*'):
					os.remove(item)
				obj.recover(path, org_img, pred_box)
				obj.recoverGlobal(path, org_img, org_info, pred_v_out)

			# Save model
			if i % 200 == 0:
				saver.save(sess, './model/model-%d.ckpt' % i)

		# End main loop
		train_writer.close()
		valid_writer.close()
		f.close()

