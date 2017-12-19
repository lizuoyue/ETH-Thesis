import os, re, sys
if os.path.exists('../Python-Lib/'):
	sys.path.insert(1, '../Python-Lib')
import io, glob
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
ut = __import__('Utility')

class RPN(object):

	def __init__(self):
		self.rpn_kernel_size = (3, 3)
		self.train_batch_size = 8
		self.train_num_anchors = 256
		self.k = 9

	def smoothL1Loss(self, labels, predictions, weights):
		diff = predictions - labels
		abs_diff = tf.abs(diff)
		abs_diff_lt_1 = tf.less(abs_diff, 1)
		loss = tf.reduce_sum(
			tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
		2) * weights
		return tf.reduce_sum(loss)

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
		return feature
		with tf.variable_scope('RPN', reuse = reuse):
			rpn_conv = tf.layers.conv2d(
				inputs = feature,
				filters = 512,
				kernel_size = self.rpn_kernel_size,
				padding = 'same',
				activation = tf.nn.relu
			)
			obj_prob = tf.layers.conv2d(
				inputs = rpn_conv,
				filters = 2 * self.k,
				kernel_size = (1, 1),
				padding = 'same',
				activation = tf.sigmoid
			)
			bbox_info = tf.layers.conv2d(
				inputs = rpn_conv,
				filters = 4 * self.k,
				kernel_size = (1, 1),
				padding = 'same',
				activation = None
			)
		return (
			tf.reshape(obj_prob, [self.train_batch_size, 40, 60, self.k, 2]),
			tf.reshape(bbox_info, [self.train_batch_size, 40, 60, self.k, 4]),
		)

	def Train(self, xx, aa, pp, tt):
		img        = tf.reshape(xx, [self.train_batch_size, 640, 960, 3])
		anchor_idx = tf.reshape(aa, [self.train_batch_size, self.train_num_anchors, 3])
		anchor_cls = tf.reshape(pp, [self.train_batch_size, self.train_num_anchors, 2])
		anchor_box = tf.reshape(tt, [self.train_batch_size, self.train_num_anchors, 4])
		# obj_prob, bbox_info = self.RPN(img)
		return tf.reduce_sum(self.RPN(img))
		prob_pred = []
		bbox_pred = []
		for idx, prob, bbox in zip(tf.unstack(anchor_idx), tf.unstack(obj_prob), tf.unstack(bbox_info)):
			prob_pred.append(tf.gather_nd(prob, idx))
			bbox_pred.append(tf.gather_nd(bbox, idx))
		prob_pred = tf.stack(prob_pred)
		bbox_pred = tf.stack(bbox_pred)
		loss_1 = tf.losses.log_loss(labels = anchor_cls, predictions = prob_pred)
		loss_2 = self.smoothL1Loss(labels = anchor_box, predictions = bbox_pred, weights = anchor_cls[..., 0])
		return loss_1, loss_2

	def Predict(self, xx):
		pass

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
	lr = 0.0005

	# Create data generator
	obj = ut.AnchorGenerator()

	# Define graph
	RPNGraph = RPN()
	xx = tf.placeholder(tf.float32)
	aa = tf.placeholder(tf.int32)
	pp = tf.placeholder(tf.float32)
	tt = tf.placeholder(tf.float32)
	result = RPNGraph.Train(xx, aa, pp, tt)
	pred = RPNGraph.Predict(xx)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	train = optimizer.minimize(result)#[0] + result[1])
	saver = tf.train.Saver(max_to_keep = 3)
	init = tf.global_variables_initializer()

	# Launch graph
	with tf.Session() as sess:
		# Create loggers
		f = open('./loss.out', 'a')
		# train_writer = Logger('./log/train/')
		# valid_writer = Logger('./log/valid/')

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
			img, bbox, anchor_idx, anchor_prob, anchor_box = obj.getFakeDataBatch(1)
			# print(img.shape)
			# print(bbox.shape)
			# print(anchor_idx.shape)
			# print(anchor_prob.shape)
			# print(anchor_box.shape)
			feed_dict = {xx: img, aa: anchor_idx, pp: anchor_prob, tt: anchor_box}

			# Training and get result
			sess.run(train, feed_dict)
			loss_1, loss_2 = sess.run(result, feed_dict)
			# train_writer.log_scalar('Loss CNN' , loss_CNN, i)
			# train_writer.log_scalar('Loss RNN' , loss_RNN, i)
			# train_writer.log_scalar('Loss Full', loss_CNN + loss_RNN, i)

			# Write loss to file
			print('Train Iter %d, %.6lf, %.6lf, %.6lf' % (i, loss_1, loss_2, loss_1 + loss_2))
			f.write('Train Iter %d, %.6lf, %.6lf, %.6lf\n' % (i, loss_1, loss_2, loss_1 + loss_2))
			f.flush()

		# End main loop
		# train_writer.close()
		# valid_writer.close()
		f.close()

