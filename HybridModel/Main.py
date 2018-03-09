import numpy as np
import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf
from Config import *
from HybridModel import *
from DataGenerator import *
from UtilityBoxAnchor import *

config = Config()

if __name__ == '__main__':
	# Create new folder
	if not os.path.exists('./Model/'):
		os.makedirs('./Model/')

	# Create data generator
	obj = DataGenerator(
		building_path = config.PATH_B % 'Zurich',
		area_path = config.PATH_A % 'Zurich', 
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
	)

	# Define graph
	graph = HybridModel(
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

	train_res     = graph.train(aa, cc, dd, pp, ii, bb, vv, oo, ee, ll)
	pred_rpn_res  = graph.predict_rpn(aa)
	pred_poly_res = graph.predict_polygon(pp)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0] + train_res[1] + train_res[2] + train_res[3])
	saver = tf.train.Saver(max_to_keep = 3)
	init = tf.global_variables_initializer()

	# Launch graph
	with tf.Session() as sess:
		# Create loggers
		f = open('./HybridModel.out', 'a')
		train_writer = Logger('./Log/train/')
		valid_writer = Logger('./Log/valid/')

		# Restore weights
		if len(sys.argv) > 1 and sys.argv[1] != None:
			saver.restore(sess, './Model/Model-%s.ckpt' % sys.argv[1])
			iter_obj = range(int(sys.argv[1]) + 1, NUM_ITER)
		else:
			sess.run(init)
			iter_obj = range(config.NUM_ITER)

		# Main loop
		for i in iter_obj:
			# Get training batch data and create feed dictionary
			img, anchor_cls, anchor_box = obj.getAreasBatch(config.AREA_TRAIN_BATCH, mode = 'train')
			patch, boundary, vertices, v_in, v_out, end, seq_len = obj.getBuildingsBatch(config.BUILDING_TRAIN_BATCH, mode = 'train')
			feed_dict = {
				aa: img, cc: anchor_cls, dd: anchor_box,
				pp: patch, ii: v_in, bb: boundary, vv: vertices, oo: v_out, ee: end, ll: seq_len
			}

			# Training and get result
			init_time = time.time()
			_, (loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end, pred_box) = sess.run([train, train_res], feed_dict)
			cost_time = time.time() - init_time
			train_writer.log_scalar('Loss Class', loss_class, i)
			train_writer.log_scalar('Loss Delta', loss_delta, i)
			train_writer.log_scalar('Loss CNN'  , loss_CNN  , i)
			train_writer.log_scalar('Loss RNN'  , loss_RNN  , i)
			train_writer.log_scalar('Loss RPN'  , loss_class + loss_delta, i)
			train_writer.log_scalar('Loss Poly' , loss_CNN   + loss_RNN  , i)
			train_writer.log_scalar('Loss Full' , loss_class + loss_delta + loss_CNN + loss_RNN, i)
			
			# Write loss to file
			print('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time))
			f.write('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf\n' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time))
			f.flush()

			# Visualize
			if i % 200 == 1:
				img, anchor_cls, anchor_box = obj.getAreasBatch(config.AREA_PRED_BATCH, mode = 'valid')
				feed_dict = {aa: img}
				pred_box = sess.run(pred_rpn_res, feed_dict = feed_dict)
				org_img, patch, org_info = obj.getPatchesFromAreas(pred_box)
				feed_dict = {pp: patch}
				pred_boundary, pred_vertices, pred_v_out = sess.run(pred_poly_res, feed_dict = feed_dict)
				path = './Result'
				if not os.path.exists(path):
					os.makedirs(path)
				obj.recover(path, org_img, pred_box, int((i-1)/200))
				obj.recoverGlobal(path, org_img, org_info, pred_v_out, int((i-1)/200))

			# Save model
			if i % 200 == 0:
				saver.save(sess, './Model/Model-%d.ckpt' % i)

		# End main loop
		train_writer.close()
		valid_writer.close()
		f.close()

