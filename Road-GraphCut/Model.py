import numpy as np
import tensorflow as tf
from Config import *
from BasicModel import *

config = Config()

class Model(object):
	def __init__(self):
		# Parameters
		self.v_out_res        = config.V_OUT_RES
		self.v_out_nrow       = self.v_out_res[0]
		self.v_out_ncol       = self.v_out_res[1]
		self.res_num          = self.v_out_nrow * self.v_out_ncol

		# Vertex pool for prediction
		self.vertex_pool = []
		for i in range(self.v_out_nrow):
			for j in range(self.v_out_ncol):
				self.vertex_pool.append(np.zeros(self.v_out_res, dtype = np.float32))
				self.vertex_pool[-1][i, j] = 1.0
		self.vertex_pool.append(np.zeros(self.v_out_res, dtype = np.float32))
		self.vertex_pool = np.array(self.vertex_pool)

		#
		self.num_stages = 3
		return

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

		l1 = [Stage('L1_Stage1', skip_feature, 2, reuse = reuse)]
		l2 = [Stage('L2_Stage1', skip_feature, 2, reuse = reuse)]
		boundary_prob = [tf.nn.softmax(l1[-1])[..., 0: 1]]
		vertices_prob = [tf.nn.softmax(l2[-1])[..., 0: 1]]
		for i in range(2, self.num_stages + 1):
			stage_input = tf.concat([skip_feature, l1[-1], l2[-1]], axis = -1)
			l1.append(Stage('L1_Stage%d' % i, stage_input, 2, reuse = reuse))
			l2.append(Stage('L2_Stage%d' % i, stage_input, 2, reuse = reuse))
			boundary_prob.append(tf.nn.softmax(l1[-1])[..., 0: 1])
			vertices_prob.append(tf.nn.softmax(l2[-1])[..., 0: 1])
		if not reuse:
			loss = 0
			for item in boundary_prob:
				loss += self.weightedLogLoss(gt_boundary, item)
			for item in vertices_prob:
				loss += self.weightedLogLoss(gt_vertices, item)
			return boundary_prob[-1], vertices_prob[-1], loss
		else:
			return boundary_prob[-1], vertices_prob[-1]

	def train(self, aa, bb, vv):
		#
		img           = tf.reshape(aa, [config.AREA_TRAIN_BATCH, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		gt_boundary   = tf.reshape(bb, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol, 1])
		gt_vertices   = tf.reshape(vv, [config.AREA_TRAIN_BATCH, self.v_out_nrow, self.v_out_ncol, 1])

		# CNN part
		pred_boundary, pred_vertices, loss_CNN = self.CNN(img, gt_boundary, gt_vertices)
		return loss_CNN, pred_boundary, pred_vertices

	def predict(self, aa):
		img = tf.reshape(aa, [1, config.AREA_SIZE[1], config.AREA_SIZE[0], 3])
		pred_boundary, pred_vertices = self.CNN(img, reuse = True)
		return pred_boundary, pred_vertices

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
