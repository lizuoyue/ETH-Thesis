import os, re, sys
if os.path.exists('../Python-Lib/'):
	sys.path.insert(1, '../Python-Lib')
import io, glob, time
import numpy as np
import tensorflow as tf
import Utility as ut
from PIL import Image, ImageDraw

ANCHOR_SCALE   = [16, 32, 64, 128]
ANCHOR_RATIO   = [0.25, 0.5, 1, 2, 4]
FEATURE_SHAPE  = [[64, 64], [32, 32], [16, 16], [8, 8]]
FEATURE_STRIDE = [4, 8, 16, 32]

# ANCHOR_SCALE   = [40, 80, 160, 320]
# ANCHOR_RATIO   = [0.5, 1, 2]
# FEATURE_SHAPE  = [[160, 160], [80, 80], [40, 40], [20, 20]]
# FEATURE_STRIDE = [4, 8, 16, 32]

class RPN(object):

	def __init__(self, train_batch_size, pred_batch_size, train_num_anchors):
		self.train_batch_size  = train_batch_size
		self.pred_batch_size   = pred_batch_size
		self.train_num_anchors = train_num_anchors
		# self.img_size          = (640, 640)
		self.img_size          = (256, 256)
		self.anchors           = ut.generatePyramidAnchors(ANCHOR_SCALE, ANCHOR_RATIO, FEATURE_SHAPE, FEATURE_STRIDE, 1)
		self.all_idx           = np.array([i for i in range(self.anchors.shape[0])])
		self.valid_idx         = self.all_idx[
			(self.anchors[:, 0] >= 0) &
			(self.anchors[:, 1] >= 0) &
			(self.anchors[:, 2] < self.img_size[1]) &
			(self.anchors[:, 3] < self.img_size[0])
		]
		self.num_anchors       = self.anchors.shape[0]
		print('%d of %d anchors are valid.' % (self.valid_idx.shape[0], self.num_anchors))
		return

	def VGG16(self, img, reuse = None):
		with tf.variable_scope('VGG16', reuse = reuse):
			conv1_1 = tf.layers.conv2d(
				inputs = img,
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
			return conv3_3, conv4_3, conv5_3

	def FPN(self, img, reuse = None):
		c2, c3, c4 = self.VGG16(img, reuse)
		with tf.variable_scope('FPN', reuse = reuse):
			p4 = tf.layers.conv2d(
				inputs = c4,
				filters = 256,
				kernel_size = (1, 1),
				padding = 'same',
				activation = tf.nn.relu
			)
			p3 = tf.layers.conv2d(
				inputs = c3,
				filters = 256,
				kernel_size = (1, 1),
				padding = 'same',
				activation = tf.nn.relu) + tf.image.resize_images(
				images = p4,
				size = FEATURE_SHAPE[1]
			)
			p2 = tf.layers.conv2d(
				inputs = c2,
				filters = 256,
				kernel_size = (1, 1),
				padding = 'same',
				activation = tf.nn.relu) + tf.image.resize_images(
				images = p3,
				size = FEATURE_SHAPE[0]
			)
			p2_conv = tf.layers.conv2d(
				inputs = p2,
				filters = 256,
				kernel_size = (3, 3),
				padding = 'same',
				activation = tf.nn.relu
			)
			p3_conv = tf.layers.conv2d(
				inputs = p3,
				filters = 256,
				kernel_size = (3, 3),
				padding = 'same',
				activation = tf.nn.relu
			)
			p4_conv = tf.layers.conv2d(
				inputs = p4,
				filters = 256,
				kernel_size = (3, 3),
				padding = 'same',
				activation = tf.nn.relu
			)
			p5_pool = tf.layers.max_pooling2d(
				inputs = p4_conv,
				pool_size = (2, 2),
				strides = 2
			)
		return [p2_conv, p3_conv, p4_conv, p5_pool]

	def RPNLayer(self, feature, anchors_per_pixel, reuse = None):
		"""
			feature: [batch_size, height, width, num_channels]
		"""
		num_anchors = feature.shape[1] * feature.shape[2] * anchors_per_pixel
		with tf.variable_scope('RPN', reuse = reuse):
			rpn_conv = tf.layers.conv2d(
				inputs = feature,
				filters = 256, # 512
				kernel_size = (3, 3),
				padding = 'same',
				activation = tf.nn.relu
			)
			bbox_logit = tf.layers.conv2d(
				inputs = rpn_conv,
				filters = 2 * anchors_per_pixel,
				kernel_size = (1, 1),
				padding = 'valid',
				activation = None
			)
			bbox_delta = tf.layers.conv2d(
				inputs = rpn_conv,
				filters = 4 * anchors_per_pixel,
				kernel_size = (1, 1),
				padding = 'valid',
				activation = None
			)
		return (
			tf.reshape(bbox_logit, [-1, num_anchors, 2]),
			tf.reshape(bbox_delta,  [-1, num_anchors, 4]),
		)

	def RPN(self, img, reuse = None):
		p2, p3, p4, p5     = self.FPN(img, reuse)
		p2_logit, p2_delta = self.RPNLayer(p2, len(ANCHOR_RATIO), reuse)
		p3_logit, p3_delta = self.RPNLayer(p3, len(ANCHOR_RATIO), True)
		p4_logit, p4_delta = self.RPNLayer(p4, len(ANCHOR_RATIO), True)
		p5_logit, p5_delta = self.RPNLayer(p5, len(ANCHOR_RATIO), True)
		logit = tf.concat([p2_logit, p3_logit, p4_logit, p5_logit], axis = 1)
		delta = tf.concat([p2_delta, p3_delta, p4_delta, p5_delta], axis = 1)
		return logit, delta

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
		total_num = pos_num + neg_num
		w = tf.stack([labels[:, 0] * neg_num / total_num, labels[:, 1] * pos_num / total_num], axis = 1)
		return tf.reduce_mean(tf.losses.log_loss(labels = labels, predictions = tf.nn.softmax(logits), weights = w))

	def RPNDeltaLoss(self, anchor_class, anchor_delta, pred_delta):
		indices = tf.where(tf.equal(anchor_class[..., 0], 1)) # num_pos_anchors, 2
		labels = tf.gather_nd(anchor_delta, indices) # num_pos_anchors, 4
		return self.smoothL1Loss(labels = labels, predictions = tf.gather_nd(pred_delta, indices))

	def ApplyDeltaToAnchor(self, anchors, deltas):
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

	def Train(self, xx, cc, bb):
		img          = tf.reshape(xx, [self.train_batch_size, self.img_size[1], self.img_size[0], 3])
		anchor_class = tf.reshape(cc, [self.train_batch_size, self.num_anchors, 2])
		anchor_delta = tf.reshape(bb, [self.train_batch_size, self.num_anchors, 4])
		pred_logit, pred_delta = self.RPN(img)
		loss_1 = 200 * self.RPNClassLoss(anchor_class, pred_logit)
		loss_2 = 10 * self.RPNDeltaLoss(anchor_class, anchor_delta, pred_delta)
		return loss_1, loss_2, loss_1 + loss_2

	def Predict(self, xx):
		img        = tf.reshape(xx, [self.pred_batch_size, self.img_size[1], self.img_size[0], 3])
		pred_logit, pred_delta = self.RPN(img, reuse = True)
		pred_score = tf.nn.softmax(pred_logit)[..., 0]
		pred_box   = tf.stack([self.ApplyDeltaToAnchor(self.anchors, pred_delta[i]) for i in range(self.pred_batch_size)])
		res = []
		for i in range(self.pred_batch_size):
			box_valid = tf.gather(pred_box[i], self.valid_idx)
			score_valid = tf.gather(pred_score[i], self.valid_idx)
			idx_top = tf.nn.top_k(score_valid, 1000).indices
			box_top = tf.gather(box_valid, idx_top)
			score_top = tf.gather(score_valid, idx_top)
			idx = tf.where(score_top >= 0.7)
			box = tf.gather(box_top, idx)[:, 0, :]
			score = tf.gather(score_top, idx)[:, 0]
			indices = tf.image.non_max_suppression(
				boxes = box, # pred_box[i]
				scores = score, # pred_score[i]
				max_output_size = 40,
				iou_threshold = 0.7
			)
			res.append(tf.gather(box, indices)) # pred_box[i]
		return res

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
	if not os.path.exists('./tmp/'):
		os.makedirs('./tmp/')
	if not os.path.exists('./res/'):
		os.makedirs('./res/')

	# Set parameters
	n_iter			= 100000
	lr				= 0.00002
	train_batch_size  = 9
	pred_batch_size   = 16
	train_num_anchors = 256

	# Create data generator
	obj = ut.AnchorGenerator(fake = False, data_path = '/local/lizuoyue/Chicago_Area')

	# Define graph
	RPNGraph = RPN(train_batch_size, pred_batch_size, train_num_anchors)
	xx	   = tf.placeholder(tf.float32)
	cc	   = tf.placeholder(tf.int32)
	bb	   = tf.placeholder(tf.float32)
	result   = RPNGraph.Train(xx, cc, bb)
	pred	 = RPNGraph.Predict(xx)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	train = optimizer.minimize(result[2])
	saver = tf.train.Saver(max_to_keep = 3)
	init = tf.global_variables_initializer()

	# Launch graph
	with tf.Session() as sess:
		# Create loggers
		f = open('./RPN-Loss.out', 'a')
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
			init_time = time.time()
			# Get training batch data and create feed dictionary
			img, anchor_cls, anchor_box = obj.getDataBatch(train_batch_size, mode = 'train')
			feed_dict = {xx: img, cc: anchor_cls, bb: anchor_box}
			time_1 = time.time()

			# Training and get result
			sess.run(train, feed_dict)
			time_2 = time.time()
			loss_1, loss_2, loss_3 = sess.run(result, feed_dict)
			time_3 = time.time()
			train_writer.log_scalar('Loss Class', loss_1, i)
			train_writer.log_scalar('Loss BBox' , loss_2, i)
			train_writer.log_scalar('Loss Full' , loss_3, i)

			# Write loss to file
			print('Train Iter %d, %.6lf, %.6lf, %.6lf, with time %.3lf, %.3lf, %.3lf' % (i, loss_1, loss_2, loss_3, time_1 - init_time, time_2 - time_1, time_3 - time_2))
			f.write('Train Iter %d, %.6lf, %.6lf, %.6lf, with time %.3lf, %.3lf, %.3lf\n' % (i, loss_1, loss_2, loss_3, time_1 - init_time, time_2 - time_1, time_3 - time_2))
			f.flush()

			# Save model
			if i % 100 == 0:
				saver.save(sess, './tmp/model-%d.ckpt' % i)

			if i % 100 == 0:
				img, anchor_cls, anchor_box = obj.getDataBatch(pred_batch_size, mode = 'valid')
				feed_dict = {xx: img}
				res = sess.run(pred, feed_dict)
				obj.recover('./res', img, res)

		# End main loop
		train_writer.close()
		valid_writer.close()
		f.close()

