import os, re, sys
if os.path.exists('../Python-Lib/'):
	sys.path.insert(1, '../Python-Lib')
import io, glob
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
ut = __import__('Utility')

class RPN(object):

	def __init__(self, train_batch_size, pred_batch_size, train_num_anchors):
		self.train_batch_size  = train_batch_size
		self.pred_batch_size   = pred_batch_size
		self.train_num_anchors = train_num_anchors
		self.anchors_per_pixel = len(ANCHOR_LIST)
		self.rpn_kernel_size   = (3, 3)
		self.alpha             = 10
		self.img_size          = (640, 640)
		self.img_size_s        = (int(self.img_size[0] / 16), int(self.img_size[1] / 16))
		self.valid_anchor_idx  = []
		anchor_info       = []
		# From left to right, from up to down
		idx = -1
		for i in range(self.img_size_s[1]):
			for j in range(self.img_size_s[0]):
				x = j * 16 + 8
				y = i * 16 + 8
				for k, (w, h) in enumerate(ANCHOR_LIST):
					idx += 1
					anchor_info.append([x, y, w, h])
					l, u, r, d = ut.xywh2lurd((x, y, w, h))
					if l >= 0 and u >= 0 and r < self.img_size[0] and d < self.img_size[1]:
						self.valid_anchor_idx.append(idx)
		print('Totally %d valid anchors.' % len(self.valid_anchor_idx))
		anchor_info       = np.array(anchor_info, np.float32) # num_anchors, 4
		anchor_info_rep   = np.array([anchor_info for _ in range(self.pred_batch_size)]) # pred_batch, num_anchors, 4
		self.num_anchors  = self.img_size_s[1] * self.img_size_s[0] * self.anchors_per_pixel
		self.x = anchor_info_rep[..., 0]
		self.y = anchor_info_rep[..., 1]
		self.w = anchor_info_rep[..., 2]
		self.h = anchor_info_rep[..., 3]
		self.valid_anchor_idx  = np.array(self.valid_anchor_idx, np.int32) # A list
		return

	def smoothL1Loss(self, labels, predictions):
		diff = tf.abs(predictions - labels)
		val = tf.where(tf.less(diff, 1), 0.5 * tf.square(diff), diff - 0.5)
		return tf.reduce_sum(val) / self.train_batch_size / self.train_num_anchors

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
			return conv5_3

	def RPN(self, img, reuse = None):
		feature = self.VGG16(img, reuse)
		with tf.variable_scope('RPN', reuse = reuse):
			rpn_conv = tf.layers.conv2d(
				inputs = feature,
				filters = 512,
				kernel_size = self.rpn_kernel_size,
				padding = 'same',
				activation = tf.nn.relu
			)
			bbox_logit = tf.layers.conv2d(
				inputs = rpn_conv,
				filters = 2 * self.anchors_per_pixel,
				kernel_size = (1, 1),
				padding = 'valid',
				activation = None
			)
			bbox_info = tf.layers.conv2d(
				inputs = rpn_conv,
				filters = 4 * self.anchors_per_pixel,
				kernel_size = (1, 1),
				padding = 'valid',
				activation = None
			)
		if reuse:
			batch_size = self.pred_batch_size
		else:
			batch_size = self.train_batch_size
		return (
			tf.reshape(bbox_logit, [batch_size, self.num_anchors, 2]),
			tf.reshape(bbox_info,  [batch_size, self.num_anchors, 4]),
		)

	def RPNClassLoss(self, anchor_cls, pred_logit):
		indices = tf.where(tf.equal(tf.reduce_sum(anchor_cls, 2), 1)) # num_valid_anchors, 2
		logits = tf.gather_nd(pred_logit, indices) # num_valid_anchors, 2
		labels = tf.gather_nd(anchor_cls, indices) # num_valid_anchors, 2
		prob = tf.nn.softmax(logits)
		loss = tf.losses.log_loss(predictions = prob, labels = labels)
		return tf.reduce_mean(loss)

	def RPNBoxLoss(self, anchor_cls, anchor_box, pred_info):
		indices = tf.where(tf.equal(anchor_cls[..., 0], 1)) # num_pos_anchors, 2
		labels = tf.gather_nd(anchor_box, indices) # num_pos_anchors, 4
		prob = tf.gather_nd(pred_info, indices) # num_pos_anchors, 4
		return self.smoothL1Loss(labels = labels, predictions = prob)

	def Train(self, xx, cc, bb):
		img        = tf.reshape(xx, [self.train_batch_size, self.img_size[1], self.img_size[0], 3])
		anchor_cls = tf.reshape(cc, [self.train_batch_size, self.num_anchors, 2])
		anchor_box = tf.reshape(bb, [self.train_batch_size, self.num_anchors, 4])
		pred_logit, pred_info = self.RPN(img)
		loss_1 = self.RPNClassLoss(anchor_cls, pred_logit)
		loss_2 = self.RPNBoxLoss(anchor_cls, anchor_box, pred_info)
		return loss_1, loss_2, loss_1 + self.alpha * loss_2

	def Predict(self, xx):
		img = tf.reshape(xx, [self.pred_batch_size, self.img_size[1], self.img_size[0], 3])
		pred_logit, pred_info = self.RPN(img, reuse = True)
		pred_scores = tf.nn.softmax(pred_logit)[..., 0]
		boxes_x = tf.floor(pred_info[..., 0] * self.w + self.x)
		boxes_y = tf.floor(pred_info[..., 1] * self.h + self.y)
		boxes_w = tf.floor(tf.exp(pred_info[..., 2]) * self.w)
		boxes_h = tf.floor(tf.exp(pred_info[..., 3]) * self.h)
		boxes_coo = tf.stack([
			tf.floor(boxes_y - boxes_h / 2),
			tf.floor(boxes_x - boxes_w / 2),
			tf.floor(boxes_y + boxes_h / 2),
			tf.floor(boxes_x + boxes_w / 2)
		], axis = 2)
		res = []
		for i in range(self.pred_batch_size):
			boxes = tf.gather(boxes_coo[i], self.valid_anchor_idx)
			scores = tf.gather(pred_scores[i], self.valid_anchor_idx)
			indices = tf.image.non_max_suppression(
				boxes = boxes,
				scores = scores,
				max_output_size = 50,
				iou_threshold = 0.7
			)
			res.append(tf.gather(boxes, indices))
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
	n_iter            = 100000
	lr                = 0.00005
	train_batch_size  = 4
	pred_batch_size   = 9
	train_num_anchors = 256

	# Create data generator
	ANCHOR_SCALE = [60, 120, 180, 240, 300]
	ANCHOR_RATIO = [0.5, 1, 2]
	obj = ut.AnchorGenerator(fake = False, data_path = '/local/lizuoyue/Chicago_Area', anchor_para = (ANCHOR_SCALE, ANCHOR_RATIO))

	# Define graph
	RPNGraph = RPN(train_batch_size, pred_batch_size, train_num_anchors)
	xx = tf.placeholder(tf.float32)
	cc = tf.placeholder(tf.int32)
	bb = tf.placeholder(tf.float32)
	result = RPNGraph.Train(xx, cc, bb)
	pred = RPNGraph.Predict(xx)

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
			# Get training batch data and create feed dictionary
			img, anchor_cls, anchor_box = obj.getDataBatch(train_batch_size, mode = 'train')
			feed_dict = {xx: img, cc: anchor_cls, bb: anchor_box}

			# Training and get result
			sess.run(train, feed_dict)
			loss_1, loss_2, loss_3 = sess.run(result, feed_dict)
			train_writer.log_scalar('Loss Class', loss_1, i)
			train_writer.log_scalar('Loss BBox' , loss_2, i)
			train_writer.log_scalar('Loss Full' , loss_3, i)

			# Write loss to file
			print('Train Iter %d, %.6lf, %.6lf, %.6lf' % (i, loss_1, loss_2, loss_3))
			f.write('Train Iter %d, %.6lf, %.6lf, %.6lf\n' % (i, loss_1, loss_2, loss_3))
			f.flush()

			# Save model
			if i % 200 == 0:
				saver.save(sess, './tmp/model-%d.ckpt' % i)

			if i % 200 == 0:
				img, anchor_cls, anchor_box = obj.getDataBatch(pred_batch_size, mode = 'valid')
				feed_dict = {xx: img}
				res = sess.run(pred, feed_dict)
				obj.recover('./res', img, res)

		# End main loop
		train_writer.close()
		valid_writer.close()
		f.close()

