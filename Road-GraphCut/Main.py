import os, sys
import numpy as np
import tensorflow as tf
from Config import *
from Model import *
# from RoadData import *
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
	# assert(len(sys.argv) == 2 or len(sys.argv) == 3)

	# Define graph
	graph = Model()
	aa = tf.placeholder(tf.float32)
	bb = tf.placeholder(tf.float32)
	vv = tf.placeholder(tf.float32)

	train_res = graph.train(aa, bb, vv)
	pred_res = graph.predict(aa)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0])
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
			img, boundary, vertices = getDataBatch(config.AREA_TRAIN_BATCH, 'train')
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
			feed_dict = {aa: img, bb: boundary, vv: vertices}

			# Training and get result
			init_time = time.time()
			_, (loss_CNN, pred_boundary, pred_vertices) = sess.run([train, train_res], feed_dict)
			cost_time = time.time() - init_time
			train_writer.log_scalar('Loss CNN'  , loss_CNN  , i)
			
			# Write loss to file
			train_loss.write('Train Iter %d, %.6lf, %.3lf\n' % (i, loss_CNN, cost_time))
			train_loss.flush()

			# Validation
			if i % 100 == 0:
				img, boundary, vertices = getDataBatch(config.AREA_TRAIN_BATCH, 'val')
				feed_dict = {aa: img, bb: boundary, vv: vertices}
				init_time = time.time()
				loss_CNN, pred_boundary, pred_vertices = sess.run(train_res, feed_dict)
				cost_time = time.time() - init_time
				valid_writer.log_scalar('Loss CNN'  , loss_CNN  , i)

				valid_loss.write('Valid Iter %d, %.6lf, %.3lf\n' % (i, loss_CNN, cost_time))
				valid_loss.flush()

			# Test
			if i % 500 == 1:
				for j in range(30):
					img, boundary, vertices = getDataBatch(1, 'val')
					pred_boundary, pred_vertices = sess.run(pred_res, feed_dict = {aa: img})

					# peaks, v_in, v_in_vis = getAllTerminal(pred_vertices[0], pred_boundary[0])

					path = 'test_res/'
					savePNG(img[0], pred_boundary[0] * 255, path + '%d-0.png' % j)
					savePNG(img[0], pred_vertices[0] * 255, path + '%d-1.png' % j)

					# from scipy.ndimage.filters import gaussian_filter
					# savePNG(img[0], gaussian_filter(pred_vertices[0] * 255, 1), path + '%d-1-sigma.png' % j)

					# savePNG(img[0], v_in_vis, path + '%d-2.png' % j)

					# pred_v_out = sess.run(pred_path_res, feed_dict = {ff: feature, ii: v_in})

					# newImg = recoverMultiPath(img[0], v_in, pred_v_out, peaks)
					# savePNG(img[0], newImg, path + '%d-3.png' % j)
					# for k in range(v_in.shape[0]):
					# 	savePNG(img[0], v_in[k], path + '%d-4-%d-in.png' % (j, k))
					# 	savePNG(img[0], pred_v_out[k], path + '%d-4-%d-out.png' % (j, k))

			# Save model
			if i % 5000 == 0:
				saver.save(sess, './Model/Model-%d.ckpt' % i)

		# End main loop
		train_writer.close()
		valid_writer.close()
		train_loss.close()
		valid_loss.close()

