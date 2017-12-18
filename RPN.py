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
		pass

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

	def RPN(self, img, anchor_true = None, reuse = None):
		feature = self.VGG16(img, reuse)
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

		if not reuse:
			loss = 0.0
			n_b = tf.reduce_sum(boundary_true) / self.train_batch_size
			n_v = tf.reduce_sum(vertices_true) / self.train_batch_size
			loss += tf.losses.log_loss(
				labels = boundary_true,
				predictions = boundary,
				weights = (boundary_true * (self.res_num - 2 * n_b) + n_b)
			)
			loss += tf.losses.log_loss(
				labels = vertices_true,
				predictions = vertices,
				weights = (vertices_true * (self.res_num - 2 * n_v) + n_v))
			loss /= 16
			return tf.concat([feature, boundary, vertices], 3), loss
		else:
			idx = tf.argmax(tf.reshape(vertices, [-1, self.res_num]), axis = 1)
			v_first = tf.gather(self.vertex_pool, idx, axis = 0)
			return tf.concat([feature, boundary, vertices], 3), v_first


	def Train(self):
		pass

	def Predict(self):
		pass

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

def visualize(path, img, boundary, vertices, v_in, b_pred, v_pred, v_out_pred, end_pred, seq_len, v_out_res, patch_info):
	# Clear last files
	for item in glob.glob(path + '/*'):
		os.remove(item)

	# Reshape
	b_pred = b_pred[..., 0]
	v_pred = v_pred[..., 0]
	end_pred = end_pred[..., 0]
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
		overlay(img[i], blank      , shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-0-img.png' % i)
		overlay(img[i], boundary[i], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-1-bound.png' % i)
		overlay(img[i], b_pred  [i], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-1-bound-pred.png' % i)
		overlay(img[i], vertices[i], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-2-vertices.png' % i)
		overlay(img[i], v_pred  [i], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-2-vertices-pred.png' % i)
		overlayMultiMask(img[i], vv, shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-3-vertices-merge.png' % i)
		# for j in range(seq_len[i]):
		# 	overlay(img[i], vv[j], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-3-vtx-%s.png' % (i, str(j).zfill(2)))

		link = Image.new('P', shape, color = 0)
		draw = ImageDraw.Draw(link)
		if len(polygon[i]) == 1:
			polygon[i].append(polygon[i][0])
		draw.polygon(polygon[i], fill = 0, outline = 255)
		link = np.array(link) / 255.0
		overlay(img[i], link, shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-4-vertices-link.png' % i)

		f = open(path + '/%d-5-end-prob.txt' % i, 'w')
		for j in range(seq_len[i]):
			f.write('%.6lf\n' % end_pred[i, j])
		f.close()

	#
	return

def visualize_pred(path, img, b_pred, v_pred, v_out_pred, v_out_res, patch_info):
	# Clear last files
	for item in glob.glob(path + '/*'):
		os.remove(item)

	# Reshape
	batch_size = img.shape[0]
	b_pred = b_pred[..., 0]
	v_pred = v_pred[..., 0]
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
		overlay(img[i], blank      , shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-0-img.png' % i)
		overlay(img[i], b_pred[i]  , shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-1-bound-pred.png' % i)
		overlay(img[i], v_pred[i]  , shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-2-vertices-pred.png' % i)
		overlayMultiMask(img[i], vv, shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-3-vertices-merge.png' % i)
		# for j in range(seq_len[i]):
		# 	overlay(img[i], vv[j], shape).save(path + '/%d-3-vtx-%s.png' % (i, str(j).zfill(2)))
		link = Image.new('P', shape, color = 0)
		draw = ImageDraw.Draw(link)
		if len(polygon[i]) == 1:
			polygon[i].append(polygon[i][0])
		draw.polygon(polygon[i], fill = 0, outline = 255)
		link = np.array(link) / 255.0
		overlay(img[i], link, shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-4-vertices-link.png' % i)

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
	if not os.path.exists('./tmp/'):
		os.makedirs('./tmp/')
	if not os.path.exists('./res/'):
		os.makedirs('./res/')
	if not os.path.exists('./val/'):
		os.makedirs('./val/')
	if not os.path.exists('./pre/'):
		os.makedirs('./pre/')
	if not os.path.exists('./tes/'):
		os.makedirs('./tes/')

	# Set parameters
	n_iter = 100000
	toy = False
	data_path = '../Chicago.zip'
	if not toy:
		lr = 0.0005
		max_seq_len = 24
		lstm_out_channel = [32, 16, 8]
		v_out_res = (28, 28)
		train_batch_size = 9
		pred_batch_size = 25
	else:
		lr = 0.0005
		max_seq_len = 12
		lstm_out_channel = [32, 16, 8] # [32, 12]
		v_out_res = (56, 56) # (28, 28)
		train_batch_size = 9
		pred_batch_size = 25

	# Create data generator
	obj = ut.DataGenerator(fake = toy, data_path = data_path, max_seq_len = max_seq_len, resolution = v_out_res)

	# Define graph
	PolyRNNGraph = PolygonRNN(
		max_seq_len = max_seq_len,
		lstm_out_channel = lstm_out_channel, 
		v_out_res = v_out_res,
		train_batch_size = train_batch_size,
		pred_batch_size = pred_batch_size
	)
	xx = tf.placeholder(tf.float32)
	bb = tf.placeholder(tf.float32)
	vv = tf.placeholder(tf.float32)
	ii = tf.placeholder(tf.float32)
	oo = tf.placeholder(tf.float32)
	ee = tf.placeholder(tf.float32)
	ll = tf.placeholder(tf.float32)
	result = PolyRNNGraph.Train(xx, bb, vv, ii, oo, ee, ll)
	pred = PolyRNNGraph.Predict(xx)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	train = optimizer.minimize(result[0] + result[1])
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
			img, boundary, vertices, v_in, v_out, end, seq_len, patch_info = obj.getDataBatch(train_batch_size, mode = 'train')
			feed_dict = {xx: img, bb: boundary, vv: vertices, ii: v_in, oo: v_out, ee: end, ll: seq_len}

			# Training and get result
			sess.run(train, feed_dict)
			loss_CNN, loss_RNN, b_pred, v_pred, v_out_pred, end_pred = sess.run(result, feed_dict)
			train_writer.log_scalar('Loss CNN' , loss_CNN, i)
			train_writer.log_scalar('Loss RNN' , loss_RNN, i)
			train_writer.log_scalar('Loss Full', loss_CNN + loss_RNN, i)

			# Write loss to file
			print('Train Iter %d, %.6lf, %.6lf, %.6lf' % (i, loss_CNN, loss_RNN, loss_CNN + loss_RNN))
			f.write('Train Iter %d, %.6lf, %.6lf, %.6lf\n' % (i, loss_CNN, loss_RNN, loss_CNN + loss_RNN))
			f.flush()

			# Visualize
			visualize('./res', img, boundary, vertices, v_in, b_pred, v_pred, v_out_pred, end_pred, seq_len, v_out_res, patch_info)

			# Save model
			if i % 200 == 0:
				saver.save(sess, './tmp/model-%d.ckpt' % i)

			# Cross validation
			if i % 200 == 0:
				# Get validation batch data and create feed dictionary
				img, boundary, vertices, v_in, v_out, end, seq_len, patch_info = obj.getDataBatch(train_batch_size, mode = 'valid')
				feed_dict = {xx: img, bb: boundary, vv: vertices, ii: v_in, oo: v_out, ee: end, ll: seq_len}

				# Validation and get result
				loss_CNN, loss_RNN, b_pred, v_pred, v_out_pred, end_pred = sess.run(result, feed_dict)
				valid_writer.log_scalar('Loss CNN' , loss_CNN, i)
				valid_writer.log_scalar('Loss RNN' , loss_RNN, i)
				valid_writer.log_scalar('Loss Full', loss_CNN + loss_RNN, i)

				# Write loss to file
				print('Valid Iter %d, %.6lf, %.6lf, %.6lf' % (i, loss_CNN, loss_RNN, loss_CNN + loss_RNN))
				f.write('Valid Iter %d, %.6lf, %.6lf, %.6lf\n' % (i, loss_CNN, loss_RNN, loss_CNN + loss_RNN))
				f.flush()

				# Visualize
				visualize('./val', img, boundary, vertices, v_in, b_pred, v_pred, v_out_pred, end_pred, seq_len, v_out_res, patch_info)

			# Prediction on validation set
			if i % 200 == 0:
				# Get validation batch data and create feed dictionary
				img, boundary, vertices, v_in, v_out, end, seq_len, patch_info = obj.getDataBatch(pred_batch_size, mode = 'valid')
				feed_dict = {xx: img, bb: boundary, vv: vertices, ii: v_in, oo: v_out, ee: end, ll: seq_len}

				# 
				b_pred, v_pred, v_out_pred = sess.run(pred, feed_dict)
				visualize_pred('./pre', img, b_pred, v_pred, v_out_pred, v_out_res, patch_info)

			# Prediction on test set
			if i % 200 == 0:
				# Get validation batch data and create feed dictionary
				img, boundary, vertices, v_in, v_out, end, seq_len, patch_info = obj.getDataBatch(pred_batch_size, mode = 'test')
				feed_dict = {xx: img, bb: boundary, vv: vertices, ii: v_in, oo: v_out, ee: end, ll: seq_len}

				# 
				b_pred, v_pred, v_out_pred = sess.run(pred, feed_dict)
				visualize_pred('./tes', img, b_pred, v_pred, v_out_pred, v_out_res, patch_info)

		# End main loop
		train_writer.close()
		valid_writer.close()
		f.close()

