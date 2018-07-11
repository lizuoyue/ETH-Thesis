import os, sys
import numpy as np
import tensorflow as tf
from Config import *
from HybridModel import *
from DataGenerator import *
from UtilityBoxAnchor import *

config = Config()

if __name__ == '__main__':
	assert(len(sys.argv) == 1 or len(sys.argv) == 2)

	# Define graph
	graph = HybridModel(
		max_num_vertices = config.MAX_NUM_VERTICES,
		lstm_out_channel = config.LSTM_OUT_CHANNEL, 
		v_out_res = config.V_OUT_RES,
		mode = 'train'
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
	vgg = [tf.placeholder(tf.float32) for _ in range(5)]

	train_res = graph.train(aa, cc, dd, pp, ii, bb, vv, oo, ee, ll)
	graph.mode = 'valid'
	pred_rpn_res  = graph.predict_rpn(aa, config.AREA_VALID_BATCH)
	pred_poly_res = graph.predict_polygon(pp, vgg)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0] + train_res[1] + train_res[2] + train_res[3])
	saver = tf.train.Saver(max_to_keep = 1)
	init = tf.global_variables_initializer()

	# Create data generator
	obj = DataGenerator(
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
	)

	img_bias = np.array([77.91342018,  89.78918901, 101.50963053])

	# Create new folder
	if not os.path.exists('./Model/'):
		os.makedirs('./Model/')

	# Launch graph
	with tf.Session() as sess:
		# Create loggers
		train_loss = open('./LossTrain.out', 'w')
		valid_loss = open('./LossValid.out', 'w')
		train_writer = Logger('./Log/train/')
		valid_writer = Logger('./Log/valid/')

		# Restore weights
		if len(sys.argv) == 2 and sys.argv[1] == 'restore':
			files = glob.glob('./Model/Model-*.ckpt.meta')
			files = [(int(file.replace('./Model/Model-', '').replace('.ckpt.meta', '')), file) for file in files]
			num, model_path = files[-1]
			saver.restore(sess, model_path.replace('.meta', ''))
			iter_obj = range(num + 1, config.NUM_ITER)
		else:
			sess.run(init)
			iter_obj = range(config.NUM_ITER)

		# Main loop
		for i in iter_obj:
			# Get training batch data and create feed dictionary
			img, anchor_cls, anchor_box = obj.getAreasBatch(config.AREA_TRAIN_BATCH, mode = 'train')
			patch, boundary, vertices, v_in, v_out, end, seq_len, _, crop_info = obj.getBuildingsBatch(config.BUILDING_TRAIN_BATCH, mode = 'train')
			feed_dict = {
				aa: img - img_bias, cc: anchor_cls, dd: anchor_box,
				pp: crop_info, ii: v_in, bb: boundary, vv: vertices, oo: v_out, ee: end, ll: seq_len
			}

			# Training and get result
			init_time = time.time()
			_, (loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end, pred_score, pred_box) = sess.run([train, train_res], feed_dict)
			cost_time = time.time() - init_time
			train_writer.log_scalar('Loss Class', loss_class, i)
			train_writer.log_scalar('Loss Delta', loss_delta, i)
			train_writer.log_scalar('Loss CNN'  , loss_CNN  , i)
			train_writer.log_scalar('Loss RNN'  , loss_RNN  , i)
			train_writer.log_scalar('Loss RPN'  , loss_class + loss_delta, i)
			train_writer.log_scalar('Loss Poly' , loss_CNN   + loss_RNN  , i)
			train_writer.log_scalar('Loss Full' , loss_class + loss_delta + loss_CNN + loss_RNN, i)
			
			# Write loss to file
			# print('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time))
			train_loss.write('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf, %d\n' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time, anchor_cls[..., 0].sum()))
			train_loss.flush()

			# Validation
			if i % 200 == 0:
				img, anchor_cls, anchor_box = obj.getAreasBatch(config.AREA_TRAIN_BATCH, mode = 'valid')
				patch, boundary, vertices, v_in, v_out, end, seq_len, _, crop_info = obj.getBuildingsBatch(config.BUILDING_TRAIN_BATCH, mode = 'valid')
				feed_dict = {
					aa: img - img_bias, cc: anchor_cls, dd: anchor_box,
					pp: crop_info, ii: v_in, bb: boundary, vv: vertices, oo: v_out, ee: end, ll: seq_len
				}
				init_time = time.time()
				loss_class, loss_delta, loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end, pred_score, pred_box = sess.run(train_res, feed_dict)
				cost_time = time.time() - init_time
				valid_writer.log_scalar('Loss Class', loss_class, i)
				valid_writer.log_scalar('Loss Delta', loss_delta, i)
				valid_writer.log_scalar('Loss CNN'  , loss_CNN  , i)
				valid_writer.log_scalar('Loss RNN'  , loss_RNN  , i)
				valid_writer.log_scalar('Loss RPN'  , loss_class + loss_delta, i)
				valid_writer.log_scalar('Loss Poly' , loss_CNN   + loss_RNN  , i)
				valid_writer.log_scalar('Loss Full' , loss_class + loss_delta + loss_CNN + loss_RNN, i)
				valid_loss.write('Valid Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf\n' % (i, loss_class, loss_delta, loss_CNN, loss_RNN, cost_time))
				valid_loss.flush()

			# Save model
			if i % 5000 == 0:
				saver.save(sess, './Model/Model-%d.ckpt' % i)

		# End main loop
		train_writer.close()
		valid_writer.close()
		train_loss.close()
		valid_loss.close()

