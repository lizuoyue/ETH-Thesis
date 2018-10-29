import os, sys, glob, cv2
import numpy as np
import tensorflow as tf
from Config import *
from Model import *
from DataGenerator import *
from PIL import Image

config = Config()

def preserve(filename, num_lines):
	if not os.path.exists(filename):
		f = open(filename, 'w')
		f.close()
		return
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
	f = open(filename, 'w')
	for line in lines:
		n = int(line.strip().split(',')[0].split()[-1])
		if n >= num_lines:
			break
		f.write(line)
	f.close()
	return

if __name__ == '__main__':
	argv = {k: v for k, v in zip(sys.argv[1::2], sys.argv[2::2])}
	city_name = argv['--city']
	img_bias = np.array(config.PATH[city_name]['bias'])
	backbone = argv['--net']
	restore = argv['--load'] != '0'
	print(city_name, backbone, restore)

	# Define graph
	graph = Model(
		backbone = backbone,
		max_num_vertices = config.MAX_NUM_VERTICES,
		lstm_out_channel = config.LSTM_OUT_CHANNEL, 
		v_out_res = config.V_OUT_RES,
	)
	aa = tf.placeholder(tf.float32)
	bb = tf.placeholder(tf.float32)
	vv = tf.placeholder(tf.float32)
	ii = tf.placeholder(tf.float32)
	oo = tf.placeholder(tf.float32)
	tt = tf.placeholder(tf.float32)
	ee = tf.placeholder(tf.float32)
	ll = tf.placeholder(tf.int32)
	ff = tf.placeholder(tf.float32)
	dd = tf.placeholder(tf.int32)

	train_res = graph.train(aa, bb, vv, ii, oo, tt, ee, ll, dd)
	pred_mask_res = graph.predict_mask(aa)
	pred_path_res = graph.predict_path(ff, tt)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0] + train_res[1])
	saver = tf.train.Saver(max_to_keep = 5)
	init = tf.global_variables_initializer()

	obj = DataGenerator(
		city_name = city_name,
		img_size = config.AREA_SIZE,
		v_out_res = config.V_OUT_RES,
		max_seq_len = config.MAX_NUM_VERTICES,
	)

	# Create new folder
	model_path = './Model_%s_%s/' % (backbone, city_name)
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	loss_train_out = './Loss_train_%s_%s.out' % (backbone, city_name)
	loss_valid_out = './Loss_valid_%s_%s.out' % (backbone, city_name)

	# Launch graph
	with tf.Session() as sess:
		# Restore weights
		if restore:
			print('Restore pre-trained weights.')
			files = glob.glob(model_path + '*.ckpt.meta')
			files = [(int(file.replace(model_path, '').replace('.ckpt.meta', '')), file) for file in files]
			files.sort()
			num, model_to_load = files[-1]
			saver.restore(sess, model_to_load.replace('.meta', ''))
			iter_obj = range(num + 1, config.NUM_ITER)
			preserve(loss_train_out, num + 1)
			preserve(loss_valid_out, num + 1)
			train_loss = open(loss_train_out, 'a')
			valid_loss = open(loss_valid_out, 'a')
		else:
			sess.run(init)
			iter_obj = range(config.NUM_ITER)
			train_loss = open(loss_train_out, 'w')
			valid_loss = open(loss_valid_out, 'w')

		# Main loop
		for i in iter_obj:
			# Get training batch data and create feed dictionary
			img, boundary, vertices, vertex_inputs, vertex_outputs, vertex_terminals, ends, seq_lens, path_idx = getAreasBatch(config.AREA_TRAIN_BATCH, 'train')
			feed_dict = {
				aa: img - img_bias, bb: boundary, vv: vertices, ii: vertex_inputs, oo: vertex_outputs, tt: vertex_terminals, ee: ends, ll: seq_lens, dd: path_idx
			}

			# Training and get result
			init_time = time.time()
			_, (loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end) = sess.run([train, train_res], feed_dict)
			cost_time = time.time() - init_time
			
			# Write loss to file
			train_loss.write('Train Iter %d, %.6lf, %.6lf, %.3lf\n' % (i, loss_CNN, loss_RNN, cost_time))
			train_loss.flush()

			# Validation
			if i % 100 == 0:
				img, boundary, vertices, vertex_inputs, vertex_outputs, vertex_terminals, ends, seq_lens, path_idx = getAreasBatch(config.AREA_TRAIN_BATCH, 'val')
				feed_dict = {
					aa: img - img_bias, bb: boundary, vv: vertices, ii: vertex_inputs, oo: vertex_outputs, tt: vertex_terminals, ee: ends, ll: seq_lens, dd: path_idx
				}
				init_time = time.time()
				loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end = sess.run(train_res, feed_dict)
				cost_time = time.time() - init_time

				valid_loss.write('Valid Iter %d, %.6lf, %.6lf, %.3lf\n' % (i, loss_CNN, loss_RNN, cost_time))
				valid_loss.flush()

			# Save model
			if i % 5000 == 0:
				saver.save(sess, model_path + '%d.ckpt' % i)

		train_loss.close()
		valid_loss.close()

