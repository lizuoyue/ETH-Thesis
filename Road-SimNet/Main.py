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

def savePNG(mat1, mat2, filename):
	if mat2.shape[0] < mat1.shape[0]:
		import cv2
		mat2 = cv2.resize(mat2, (0, 0), fx = 8, fy = 8, interpolation = cv2.INTER_NEAREST) 
	plt.imshow(mat1)
	plt.imshow(mat2, alpha = 0.5)
	plt.axis('off')
	plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
	return

if __name__ == '__main__':
	assert(len(sys.argv) == 2 or len(sys.argv) == 3)

	# Define graph
	graph = Model()
	aa = tf.placeholder(tf.float32)
	bb = tf.placeholder(tf.float32)
	vv = tf.placeholder(tf.float32)
	ii = tf.placeholder(tf.float32)
	dd = tf.placeholder(tf.int32)
	oo = tf.placeholder(tf.float32)
	ff = tf.placeholder(tf.float32)

	train_res = graph.train(aa, bb, vv, ii, dd, oo)
	pred_mask_res = graph.predict_mask(aa)
	pred_sim_res = graph.predict_sim(ff, ii, dd)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0] + train_res[1] + train_res[2])
	saver = tf.train.Saver(max_to_keep = 3)
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
			if i % 1 == -1:
				img, boundary, vertices, sim_in, sim_idx, sim_out = getDataBatch(config.AREA_TRAIN_BATCH, 'train')
				feed_dict = {aa: img, bb: boundary, vv: vertices, ii: sim_in, dd: sim_idx, oo: sim_out}

				# Training and get result
				init_time = time.time()
				_, (loss_B, loss_V, loss_SIM, pred_boundary, pred_vertices, pred_sim) = sess.run([train, train_res], feed_dict)
				cost_time = time.time() - init_time

				acc = np.zeros((2, 2), np.int32)
				for j in range(pred_sim.shape[0]):
					acc[int(pred_sim[j] > 0.5), int(sim_out[j])] += 1
				acc = (acc[0, 0] + acc[1, 1]) / np.sum(acc)

				train_writer.log_scalar('Loss B'   , loss_B  , i)
				train_writer.log_scalar('Loss V'   , loss_V  , i)
				train_writer.log_scalar('Loss SIM' , loss_SIM, i)
				train_writer.log_scalar('Loss Full', loss_B + loss_V + loss_SIM, i)

				# Write loss to file
				train_loss.write('Train Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf\n' % (i, loss_B, loss_V, loss_SIM, acc, cost_time))
				train_loss.flush()

			# Validation
			if i % 100 == -1:
				img, boundary, vertices, sim_in, sim_idx, sim_out = getDataBatch(config.AREA_TRAIN_BATCH, 'val')
				feed_dict = {
					aa: img, bb: boundary, vv: vertices, ii: sim_in, dd: sim_idx, oo: sim_out
				}
				init_time = time.time()
				loss_B, loss_V, loss_SIM, pred_boundary, pred_vertices, pred_sim = sess.run(train_res, feed_dict)
				cost_time = time.time() - init_time

				acc = np.zeros((2, 2), np.int32)
				for j in range(pred_sim.shape[0]):
					acc[int(pred_sim[j] > 0.5), int(sim_out[j])] += 1
				acc = (acc[0, 0] + acc[1, 1]) / np.sum(acc)
				
				valid_writer.log_scalar('Loss B'   , loss_B  , i)
				valid_writer.log_scalar('Loss V'   , loss_V  , i)
				valid_writer.log_scalar('Loss SIM' , loss_SIM, i)
				valid_writer.log_scalar('Loss Full', loss_B + loss_V + loss_SIM, i)
				valid_loss.write('Valid Iter %d, %.6lf, %.6lf, %.6lf, %.6lf, %.3lf\n' % (i, loss_B, loss_V, loss_SIM, acc, cost_time))
				valid_loss.flush()

			# Test
			if i % 1000 == 0:
				img, _, _, _, _, _ = getDataBatch(config.AREA_TEST_BATCH, 'val')
				feature, pred_boundary, pred_vertices = sess.run(pred_mask_res, feed_dict = {aa: img})

				path = 'test_res/'
				ii_feed, dd_feed = [], []
				edges_idx_list = []
				peaks_with_score_list = []
				for j in range(config.AREA_TEST_BATCH):
					savePNG(img[j], pred_boundary[j] * 255, path + '%d-0.png' % j)
					savePNG(img[j], pred_vertices[j] * 255, path + '%d-1.png' % j)

					edges, edges_idx, peaks_with_score, peaks_map = getAllEdges(pred_boundary[j], pred_vertices[j])
					ii_feed.append(edges)
					dd_feed.append(j * np.ones([edges.shape[0]], np.int32))
					edges_idx_list.append(edges_idx)
					peaks_with_score_list.append(peaks_with_score)

					savePNG(img[j], peaks_map, path + '%d-2.png' % j)

				ii_feed = np.concatenate(ii_feed, axis = 0)
				dd_feed = np.concatenate(dd_feed, axis = 0)
				pred_sim_prob = sess.run(predict_sim, feed_dict = {ff: feature, ii: ii_feed, dd: dd_feed})

				for j in range(config.AREA_TEST_BATCH):
					prob = pred_sim_prob[dd_feed == j]
					pathImg = recover(prob, edges_idx_list[j], peaks_with_score_list[j])
					savePNG(img[j], pathImg, path + '%d-3.png' % j)

			# Save model
			if i % 5000 == 0:
				saver.save(sess, './Model/Model-%d.ckpt' % i)

		# End main loop
		train_writer.close()
		valid_writer.close()
		train_loss.close()
		valid_loss.close()

