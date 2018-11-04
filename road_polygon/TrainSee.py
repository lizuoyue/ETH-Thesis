import os, sys, glob, cv2
import numpy as np
import tensorflow as tf
from Config import *
from Model import *
from DataGenerator import *
from PIL import Image

config = Config()

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
	ee = tf.placeholder(tf.float32)
	ll = tf.placeholder(tf.int32)
	ff = tf.placeholder(tf.float32)
	dd = tf.placeholder(tf.int32)

	train_res = graph.train(aa, bb, vv, ii, oo, ee, ll, dd)
	pred_mask_res = graph.predict_mask(aa)
	pred_path_res = graph.predict_path(ff, ii)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0] + train_res[1])
	saver = tf.train.Saver(max_to_keep = 3)
	init = tf.global_variables_initializer()

	dg = DataGenerator(
		city_name = city_name,
		img_size = config.AREA_SIZE,
		v_out_res = config.V_OUT_RES,
		max_seq_len = config.MAX_NUM_VERTICES,
	)

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
		else:
			sess.run(init)
			iter_obj = range(config.NUM_ITER)

		# Main loop
		for i in iter_obj:
			# Get training batch data and create feed dictionary
			img, boundary, vertices, vertex_inputs, vertex_outputs, ends, seq_lens, path_idx = dg.getAreasBatch(config.AREA_TRAIN_BATCH, 'train')
			feed_dict = {
				aa: img - img_bias, bb: boundary, vv: vertices, ii: vertex_inputs, oo: vertex_outputs, ee: ends, ll: seq_lens, dd: path_idx
			}

			# Training and get result
			init_time = time.time()
			_, (_, _, pred_boundary, pred_vertices, pred_v_out, pred_end) = sess.run([train, train_res], feed_dict)
			cost_time = time.time() - init_time

			print(img.shape)
			print(boundary.shape, pred_boundary.shape)
			print(vertices.shape, pred_vertices.shape)
			print(vertex_inputs.shape, vertex_outputs.shape, ends.shape, seq_lens.shape, path_idx.shape)
			print(pred_v_out.shape, pred_end.shape)
			quit()
			


