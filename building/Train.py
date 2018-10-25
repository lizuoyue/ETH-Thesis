import os, sys
import numpy as np
import tensorflow as tf
from Config import *
from Model import *
from DataGenerator import *
from UtilityBoxAnchor import *

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
	np.random.seed(8888)

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
	cc = tf.placeholder(tf.float32)
	dd = tf.placeholder(tf.float32)
	pp = tf.placeholder(tf.float32)
	ii = tf.placeholder(tf.float32)
	bb = tf.placeholder(tf.float32)
	vv = tf.placeholder(tf.float32)
	oo = tf.placeholder(tf.float32)
	ee = tf.placeholder(tf.float32)
	ll = tf.placeholder(tf.float32)
	nn = [tf.placeholder(tf.float32) for _ in range(5)]

	train_res = graph.train(aa, cc, dd, pp, ii, bb, vv, oo, ee, ll)
	pred_rpn_res  = graph.predict_fpn(aa, config.AREA_VALID_BATCH)
	pred_poly_res = graph.predict_polygon(pp, nn)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0] + train_res[1] + train_res[2] + train_res[3])
	saver = tf.train.Saver(max_to_keep = 1)
	init = tf.global_variables_initializer()

	# Create data generator
	obj = DataGenerator(
		city_name = city_name,
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
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
			num, model_path = files[-1]
			saver.restore(sess, model_path.replace('.meta', ''))
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
			img, anchor_cls, anchor_box = obj.getAreasBatch(config.AREA_TRAIN_BATCH, mode = 'train')
			_, boundary, vertices, v_in, v_out, end, seq_len, crop_info = obj.getBuildingsBatch(config.BUILDING_TRAIN_BATCH, mode = 'train')
			feed_dict = {
				aa: img - img_bias, cc: anchor_cls, dd: anchor_box,
				pp: crop_info, ii: v_in, bb: boundary, vv: vertices, oo: v_out, ee: end, ll: seq_len
			}

			# Training and get result
			init_time = time.time()
			_, (loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end, pred_score, pred_box) = sess.run([train, train_res], feed_dict)
			cost_time = time.time() - init_time
			
			# Write loss to file
			train_loss.write('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf, %d\n' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time, anchor_cls[..., 0].sum()))
			train_loss.flush()

			# Validation
			if i % 100 == 0:
				img, anchor_cls, anchor_box = obj.getAreasBatch(config.AREA_TRAIN_BATCH, mode = 'valid')
				_, boundary, vertices, v_in, v_out, end, seq_len, crop_info = obj.getBuildingsBatch(config.BUILDING_TRAIN_BATCH, mode = 'valid')
				feed_dict = {
					aa: img - img_bias, cc: anchor_cls, dd: anchor_box,
					pp: crop_info, ii: v_in, bb: boundary, vv: vertices, oo: v_out, ee: end, ll: seq_len
				}
				init_time = time.time()
				loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end, pred_score, pred_box = sess.run(train_res, feed_dict)
				cost_time = time.time() - init_time
				valid_loss.write('Valid Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf\n' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time))
				valid_loss.flush()

			# Save model
			if i % 10000 == 0:
				saver.save(sess, model_path + '%d.ckpt' % i)

		# End main loop
		train_loss.close()
		valid_loss.close()

