import os, sys, glob, time
if os.path.exists('../Python-Lib/'):
	sys.path.insert(1, '../Python-Lib')
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from Model import *
import Utility as ut

ANCHOR_SCALE   = [16, 32, 64, 128]
ANCHOR_RATIO   = [1.0 / 3, 1.0 / 2, 1, 2, 3]
FEATURE_SHAPE  = [[64, 64], [32, 32], [16, 16], [8, 8]]
FEATURE_STRIDE = [4, 8, 16, 32]

class PolygonRNN(object):

	def __init__(self, max_seq_len, lstm_out_channel, v_out_res):
		self.anchors           = ut.generatePyramidAnchors(ANCHOR_SCALE, ANCHOR_RATIO, FEATURE_SHAPE, FEATURE_STRIDE, 1)
		self.all_idx           = np.array([i for i in range(self.anchors.shape[0])])
		self.valid_idx         = self.all_idx
		self.num_anchors       = self.anchors.shape[0]

		self.max_seq_len = max_seq_len
		self.lstm_out_channel = lstm_out_channel
		self.v_out_res = v_out_res # n_col, n_row

		# Parameters computed
		self.res_num = self.v_out_res[1] * self.v_out_res[0]
		self.lstm_in_channel = [133] + lstm_out_channel[: -1]

		# Create multi-layer LSTM and inital state
		self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(
			[self.ConvLSTMCell(in_c, out_c) for in_c, out_c in zip(self.lstm_in_channel, self.lstm_out_channel)]
		)
		self.lstm_init_state = []
		for i, c_out in enumerate(lstm_out_channel):
			self.lstm_init_state.append(tf.get_variable('conv_lstm_cell_%d_c_h' % i, [2, self.v_out_res[1], self.v_out_res[0], c_out]))

		# Create vertex pool for prediction
		self.vertex_pool = []
		for i in range(v_out_res[1]):
			for j in range(v_out_res[0]):
				self.vertex_pool.append(np.zeros((v_out_res[1], v_out_res[0]), dtype = np.float32))
				self.vertex_pool[i * v_out_res[0] + j][i, j] = 1.0
		self.vertex_pool.append(np.zeros((v_out_res[1], v_out_res[0]), dtype = np.float32))
		self.vertex_pool = np.array(self.vertex_pool)

		return

	def ConvLSTMCell(self, input_channels, output_channels):
		return tf.contrib.rnn.ConvLSTMCell(
			conv_ndims = 2,
			input_shape = [self.v_out_res[1], self.v_out_res[0], input_channels],
			output_channels = output_channels,
			kernel_shape = [3, 3]
		)

	def applyDeltaToAnchor(self, anchors, deltas):
		"""
		anchors: [N, (y1, x1, y2     , x2     )]
		deltas : [N, (dy, dx, log(dh), log(dw))]
		"""
		# 
		h  = anchors[:, 2] - anchors[:, 0]
		w  = anchors[:, 3] - anchors[:, 1]
		cy = anchors[:, 0] + 0.5 * h
		cx = anchors[:, 1] + 0.5 * w
		# 
		cy += deltas[:, 0] * h * 0.1
		cx += deltas[:, 1] * w * 0.1
		h  *= tf.exp(deltas[:, 2] * 0.2)
		w  *= tf.exp(deltas[:, 3] * 0.2)
		# 
		y1 = cy - 0.5 * h
		x1 = cx - 0.5 * w
		y2 = y1 + h
		x2 = x1 + w
		return tf.stack([y1, x1, y2, x2], axis = 1)

	def CNN(self, images, gt_boundary = None, gt_vertices = None, reuse = None):
		batch_size = tf.cast(tf.shape(images)[0], tf.float32)
		feature = PolygonFeature(VGG16(images, True), reuse)
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
			idx = tf.argmax(tf.reshape(vertices, [-1, self.res_num]), axis = 1)
			v_first = tf.gather(self.vertex_pool, idx, axis = 0)
			return tf.concat([feature, boundary, vertices], 3), v_first


	def RNN(self, feature, v_in = None, gt_rnn_out = None, gt_seq_len = None, v_first = None, reuse = None):
		batch_size = tf.concat([[tf.shape(feature)[0]], [1, 1, 1]], 0)
		if not reuse:
			feature_rep = tf.tile(
				tf.reshape(feature, [-1, 1, self.v_out_res[1], self.v_out_res[0], 130]),
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
				c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
				h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
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
			v = [None for i in range(self.max_seq_len)]
			state = [None for i in range(self.max_seq_len)]
			rnn_output = [None for i in range(self.max_seq_len)]
			v[0] = tf.reshape(v_first, [batch_size, self.v_out_res[1], self.v_out_res[0], 1])
			state[0] = tuple([tf.contrib.rnn.LSTMStateTuple(
				c = tf.tile(self.lstm_init_state[i][0: 1], batch_size),
				h = tf.tile(self.lstm_init_state[i][1: 2], batch_size)
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
					[-1, self.v_out_res[1], self.v_out_res[0], 1]
				)
			return tf.stack(v, 1)

	def FC(self, rnn_output, rnn_out_true = None, seq_len = None, reuse = None, last_two = None):
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
					labels = rnn_out_true,
					logits = logits
				)
			) / tf.reduce_sum(seq_len)
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
		p2, p3, p4, p5 = PyramidFeature(VGG16(images, reuse), reuse)
		p2_logit, p2_delta = RPNLayer(p2, len(ANCHOR_RATIO))
		p3_logit, p3_delta = RPNLayer(p3, len(ANCHOR_RATIO), reuse = True)
		p4_logit, p4_delta = RPNLayer(p4, len(ANCHOR_RATIO), reuse = True)
		p5_logit, p5_delta = RPNLayer(p5, len(ANCHOR_RATIO), reuse = True)
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
		images        = tf.reshape(aa, [-1, 256, 256, 3])
		anchor_class  = tf.reshape(cc, [-1, self.num_anchors, 2])
		anchor_delta  = tf.reshape(dd, [-1, self.num_anchors, 4])
		patches       = tf.reshape(pp, [-1, 256, 256, 3])
		v_in          = tf.reshape(ii, [-1, self.max_seq_len, self.v_out_res[1], self.v_out_res[0], 1])
		gt_boundary   = tf.reshape(bb, [-1, self.v_out_res[1], self.v_out_res[0], 1])
		gt_vertices   = tf.reshape(vv, [-1, self.v_out_res[1], self.v_out_res[0], 1])
		gt_v_out      = tf.reshape(oo, [-1, self.max_seq_len, self.res_num])
		gt_end        = tf.reshape(ee, [-1, self.max_seq_len, 1])
		gt_seq_len    = tf.reshape(ll, [-1])
		gt_rnn_out    = tf.concat([gt_v_out, gt_end], 2)

		pred_logit, pred_delta = self.RPN(images)
		loss_class = 200 * self.RPNClassLoss(anchor_class, pred_logit)
		loss_delta =  10 * self.RPNDeltaLoss(anchor_class, anchor_delta, pred_delta)
		feature, loss_CNN = self.CNN(images, gt_boundary, gt_vertices)
		logits , loss_RNN = self.RNN(feature, v_in, gt_rnn_out, gt_seq_len)

		# Return
		pred_boundary = feature[..., -2: -1]
		pred_vertices = feature[..., -1:]
		pred_rnn      = tf.nn.softmax(logits)
		pred_v_out    = tf.reshape(pred_rnn[..., 0: self.res_num],
			[-1, self.max_seq_len, self.v_out_res[1], self.v_out_res[0]]
		)
		pred_end      = tf.reshape(pred_rnn[..., self.res_num],
			[-1, self.max_seq_len, 1]
		)
		return loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end

	def predict(self, aa):
		img        = tf.reshape(xx, [self.pred_batch_size, self.img_size[1], self.img_size[0], 3])
		pred_logit, pred_delta = self.RPN(img, reuse = True)
		pred_score = tf.nn.softmax(pred_logit)[..., 0]
		pred_box   = tf.stack([self.ApplyDeltaToAnchor(self.anchors, pred_delta[i]) for i in range(self.pred_batch_size)])
		res = []
		for i in range(self.pred_batch_size):
			box_valid = tf.gather(pred_box[i], self.valid_idx)
			score_valid = tf.gather(pred_score[i], self.valid_idx)
			idx_top = tf.nn.top_k(score_valid, 500).indices
			box_top = tf.gather(box_valid, idx_top)
			score_top = tf.gather(score_valid, idx_top)
			idx = tf.where(score_top >= 0.99)
			box = tf.gather(box_top, idx)[:, 0, :]
			score = tf.gather(score_top, idx)[:, 0]
			indices = tf.image.non_max_suppression(
				boxes = box, # pred_box[i]
				scores = score, # pred_score[i]
				max_output_size = 40,
				iou_threshold = 0.15
			)
			res.append(tf.gather(box, indices)) # pred_box[i]
		img = tf.reshape(xx, [-1, 224, 224, 3])
		feature, v_first = self.CNN(img, reuse = True)
		v_out_pred = self.RNN(feature, v_first = v_first, reuse = True)
		boundary = feature[..., -2: -1]
		vertices = feature[..., -1:]
		v_out_pred = v_out_pred[..., 0]
		return boundary, vertices, v_out_pred

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
	# if not os.path.exists('./tmp/'):
	# 	os.makedirs('./tmp/')
	# if not os.path.exists('./res/'):
	# 	os.makedirs('./res/')
	# if not os.path.exists('./val/'):
	# 	os.makedirs('./val/')
	# if not os.path.exists('./pre/'):
	# 	os.makedirs('./pre/')
	# if not os.path.exists('./tes/'):
	# 	os.makedirs('./tes/')

	# Set parameters
	n_iter = 100000
	toy = False
	data_path = '../Chicago.zip'
	if not toy:
		lr = 0.0005
		max_seq_len = 24
		lstm_out_channel = [32, 16, 8]
		v_out_res = (32, 32)
		train_batch_size = 9
		pred_batch_size = 49

	# Create data generator
	# obj = ut.DataGenerator(fake = toy, data_path = data_path, max_seq_len = max_seq_len, resolution = v_out_res)

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

	result = PolyRNNGraph.train(aa, cc, dd, pp, ii, bb, vv, oo, ee, ll)
	pred = PolyRNNGraph.predict()

	for v in tf.global_variables():
		print(v.name)
	quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	train = optimizer.minimize(result[0] + result[1] + result[2])
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
			saver.restore(sess, './tmp/model-%s.ckpt' % sys.argv[1])
			iter_obj = range(int(sys.argv[1]) + 1, n_iter)
		else:
			sess.run(init)
			iter_obj = range(n_iter)

		# Main loop
		for i in iter_obj:
			# Get training batch data and create feed dictionary
			# img, boundary, vertices, v_in, v_out, end, seq_len, patch_info = obj.getDataBatch(train_batch_size, mode = 'train')
			# feed_dict = {xx: img, bb: boundary, vv: vertices, ii: v_in, oo: v_out, ee: end, ll: seq_len, angle_score: angle}

			# # Training and get result
			# sess.run(train, feed_dict)
			# loss_CNN, loss_RNN, loss_Angle, b_pred, v_pred, v_out_pred, end_pred = sess.run(result, feed_dict)
			# train_writer.log_scalar('Loss CNN' , loss_CNN, i)
			# train_writer.log_scalar('Loss RNN' , loss_RNN, i)
			# train_writer.log_scalar('Loss Angle', loss_Angle, i)
			# train_writer.log_scalar('Loss Full', loss_CNN + loss_RNN + loss_Angle, i)

			# # Write loss to file
			# print('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf' % (i, loss_CNN, loss_RNN, loss_Angle, loss_CNN + loss_RNN + loss_Angle))
			# f.write('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf\n' % (i, loss_CNN, loss_RNN, loss_Angle, loss_CNN + loss_RNN + loss_Angle))
			# f.flush()

			# # Visualize
			# visualize('./res', img, boundary, vertices, v_in, b_pred, v_pred, v_out_pred, end_pred, seq_len, v_out_res, patch_info)

			# # Save model
			# if i % 200 == 0:
			# 	saver.save(sess, './tmp/model-%d.ckpt' % i)

			# # Cross validation
			# if i % 200 == 0:
			# 	# Get validation batch data and create feed dictionary
			# 	img, boundary, vertices, v_in, v_out, end, seq_len, patch_info = obj.getDataBatch(train_batch_size, mode = 'valid')
			# 	feed_dict = {xx: img, bb: boundary, vv: vertices, ii: v_in, oo: v_out, ee: end, ll: seq_len, angle_score: angle}

			# 	# Validation and get result
			# 	loss_CNN, loss_RNN, loss_Angle, b_pred, v_pred, v_out_pred, end_pred = sess.run(result, feed_dict)
			# 	valid_writer.log_scalar('Loss CNN' , loss_CNN, i)
			# 	valid_writer.log_scalar('Loss RNN' , loss_RNN, i)
			# 	valid_writer.log_scalar('Loss Angle', loss_Angle, i)
			# 	valid_writer.log_scalar('Loss Full', loss_CNN + loss_RNN + loss_Angle, i)

			# 	# Write loss to file
			# 	print('Valid Iter %d, %.6lf, %.6lf, %.6lf, %.6lf' % (i, loss_CNN, loss_RNN, loss_Angle, loss_CNN + loss_RNN + loss_Angle))
			# 	f.write('Valid Iter %d, %.6lf, %.6lf, %.6lf, %.6lf\n' % (i, loss_CNN, loss_RNN, loss_Angle, loss_CNN + loss_RNN + loss_Angle))
			# 	f.flush()

			# 	# Visualize
			# 	visualize('./val', img, boundary, vertices, v_in, b_pred, v_pred, v_out_pred, end_pred, seq_len, v_out_res, patch_info)

			# Prediction on validation set
			if i % 2000 == 0:
				# Get validation batch data and create feed dictionary
				img, boundary, vertices, v_in, v_out, end, seq_len, patch_info = obj.getDataBatch(pred_batch_size, mode = 'valid')
				feed_dict = {xx: img, bb: boundary, vv: vertices, ii: v_in, oo: v_out, ee: end, ll: seq_len, angle_score: angle}

				# 
				b_pred, v_pred, v_out_pred = sess.run(pred, feed_dict)
				visualize_pred('./pre%d' % i, img, b_pred, v_pred, v_out_pred, v_out_res, patch_info)

			# Prediction on test set
			if i % 2000 == 0:
				# Get validation batch data and create feed dictionary
				img, boundary, vertices, v_in, v_out, end, seq_len, patch_info = obj.getDataBatch(pred_batch_size, mode = 'test')
				feed_dict = {xx: img, bb: boundary, vv: vertices, ii: v_in, oo: v_out, ee: end, ll: seq_len, angle_score: angle}

				# 
				b_pred, v_pred, v_out_pred = sess.run(pred, feed_dict)
				visualize_pred('./tes%d' % i, img, b_pred, v_pred, v_out_pred, v_out_res, patch_info)

		# End main loop
		train_writer.close()
		valid_writer.close()
		f.close()

