import os, sys
import numpy as np
import tensorflow as tf
from Config import *
from Model import *
from RoadData import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob

config = Config()

if __name__ == '__main__':
	assert(len(sys.argv) == 2 or len(sys.argv) == 3)

	# Define graph
	graph = Model(
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
	ll = tf.placeholder(tf.float32)
	ff = tf.placeholder(tf.float32)

	train_res = graph.train(aa, bb, vv, ii, oo, tt, ee, ll)
	pred_mask_res = graph.predict_mask(aa)
	pred_path_res = graph.predict_path(ff, tt)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0] + train_res[1])
	saver = tf.train.Saver(max_to_keep = 1)
	init = tf.global_variables_initializer()

	# Create new folder
	if not os.path.exists('./Model/'):
		os.makedirs('./Model/')

	if not os.path.exists('./test_res/'):
		os.makedirs('./test_res/')

	# Launch graph
	with tf.Session() as sess:
		# Create loggers
		train_loss = open('./LossTrain.out', 'w')
		valid_loss = open('./LossValid.out', 'w')
		train_writer = Logger('./Log/train/')
		valid_writer = Logger('./Log/valid/')

		# Restore weights
		if len(sys.argv) == 3 and sys.argv[2] == 'restore':
			files = glob.glob('./Model/Model-*.ckpt.meta')
			files = [(int(file.replace('./Model/Model-', '').replace('.ckpt.meta', '')), file) for file in files]
			files.sort()
			num, model_path = files[-1]
			saver.restore(sess, model_path.replace('.meta', ''))
			iter_obj = range(num + 1, config.NUM_ITER)
		else:
			sess.run(init)
			iter_obj = range(config.NUM_ITER)

		# Main loop
		for i in iter_obj:
			# Get training batch data and create feed dictionary
			img, boundary, vertices, vertex_inputs, vertex_outputs, vertex_terminals, ends, seq_lens = getDataBatch(config.AREA_TRAIN_BATCH)
			# for j in range(config.AREA_TRAIN_BATCH):
			# 	plt.imsave('0-img.png', img[j])
			# 	plt.imsave('1-b.png', boundary[j])
			# 	plt.imsave('2-v.png', vertices[j])
			# 	plt.imsave('3-s.png', vertex_terminals[j, 0, 0])
			# 	plt.imsave('3-t.png', vertex_terminals[j, 0, 1])
			# 	print('seq_len', seq_lens[j, 0])
			# 	for k in range(config.MAX_NUM_VERTICES):
			# 		plt.imsave('4-%d-vi.png'%k, vertex_inputs[j,0,k])
			# 		plt.imsave('4-%d-vo.png'%k, vertex_outputs[j,0,k])
			# 	print(ends[j,0])
			# print('press enter to continue')
			# input()
			# continue
			feed_dict = {
				aa: img, bb: boundary, vv: vertices, ii: vertex_inputs, oo: vertex_outputs, tt: vertex_terminals, ee: ends, ll: seq_lens
			}

			# Training and get result
			init_time = time.time()
			_, (loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end) = sess.run([train, train_res], feed_dict)
			cost_time = time.time() - init_time
			train_writer.log_scalar('Loss CNN'  , loss_CNN  , i)
			train_writer.log_scalar('Loss RNN'  , loss_RNN  , i)
			train_writer.log_scalar('Loss Full' , loss_CNN + loss_RNN, i)
			
			# Write loss to file
			train_loss.write('Train Iter %d, %.6lf, %.6lf, %.3lf\n' % (i, loss_CNN, loss_RNN, cost_time))
			train_loss.flush()

			# Validation
			if i % 200 == 0:
				img, boundary, vertices, vertex_inputs, vertex_outputs, vertex_terminals, ends, seq_lens = getDataBatch(config.AREA_TRAIN_BATCH)
				feed_dict = {
					aa: img, bb: boundary, vv: vertices, ii: vertex_inputs, oo: vertex_outputs, tt: vertex_terminals, ee: ends, ll: seq_lens
				}
				init_time = time.time()
				loss_CNN, loss_RNN, pred_boundary, pred_vertices, pred_v_out, pred_end = sess.run(train_res, feed_dict)
				cost_time = time.time() - init_time
				valid_writer.log_scalar('Loss CNN'  , loss_CNN  , i)
				valid_writer.log_scalar('Loss RNN'  , loss_RNN  , i)
				valid_writer.log_scalar('Loss Full' , loss_CNN + loss_RNN, i)
				valid_loss.write('Valid Iter %d, %.6lf, %.6lf, %.3lf\n' % (i, loss_CNN, loss_RNN, cost_time))
				valid_loss.flush()

			# Test
			if i % 1000 == 0:
				img, _, _, _, _, terminal_gt, _, _ = getDataBatch(1)
				feature, pred_boundary, pred_vertices = sess.run(pred_mask_res, feed_dict = {aa: img})

				path = 'test_res/'
				plt.imsave(path + '%d-0.png' % i, img[0])
				plt.imsave(path + '%d-1.png' % i, pred_boundary[0] * 255)
				plt.imsave(path + '%d-2.png' % i, pred_vertices[0] * 255)

				# terminal = getAllTerminal(pred_vertices[0])
				multi_roads = []
				for j in range(terminal_gt.shape[1]):
					road = [vertex_terminals[0, j, 0]]
					pred_v_out = sess.run(pred_path_res, feed_dict = {ff: feature, tt: terminal_gt[0, j]})
					for k in range(config.MAX_NUM_VERTICES):
						road.append(vertex_outputs[0, j, k])
					multi_roads.append(road)

				newImg = recoverMultiPath(img[0], np.array(multi_roads))
				plt.imsave(path + '%d-3.png' % i, newImg)

			# Save model
			if i % 2000 == 0:
				saver.save(sess, './Model/Model-%d.ckpt' % i)

		# End main loop
		train_writer.close()
		valid_writer.close()
		train_loss.close()
		valid_loss.close()

