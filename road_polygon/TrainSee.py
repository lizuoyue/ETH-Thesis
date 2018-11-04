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

	model_path = './Model_%s_%s/' % (backbone, city_name)
	os.popen('mkdir temp')

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

			for j in range(config.AREA_TRAIN_BATCH):
				Image.fromarray(np.array(img[i, ...], np.uint8)).save('temp/%d-0.png' % j)
				Image.fromarray(np.array(boundary[i, ...] * 255, np.uint8)).save('temp/%d-1.png' % j)
				Image.fromarray(np.array(pred_boundary[i, ..., 0] * 255, np.uint8)).save('temp/%d-1p.png' % j)
				Image.fromarray(np.array(vertices[i, ...] * 255, np.uint8)).save('temp/%d-2.png' % j)
				Image.fromarray(np.array(pred_vertices[i, ..., 0] * 255, np.uint8)).save('temp/%d-2p.png' % j)

			for j in range(config.TRAIN_NUM_PATH):
				print(seq_lens[j])
				print(ends[j])
				print(pred_end[j])
				idx = path_idx[j]
				for k in range(config.MAX_NUM_VERTICES):
					Image.fromarray(np.array(vertex_inputs[j, k, ..., 0] * 255, np.uint8)).save('temp/%d-path%d-%din1.png' % (idx, j, k))
					Image.fromarray(np.array(vertex_inputs[j, k, ..., 1] * 255, np.uint8)).save('temp/%d-path%d-%din2.png' % (idx, j, k))
					Image.fromarray(np.array(vertex_outputs[j, k] * 255, np.uint8)).save('temp/%d-path%d-%dout.png' % (idx, j, k))
					Image.fromarray(np.array(pred_v_out[j, k] * 255, np.uint8)).save('temp/%d-path%d-%dvout.png' % (idx, j, k))

			quit()
			


