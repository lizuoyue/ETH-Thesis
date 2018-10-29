import os, sys
import numpy as np
import tensorflow as tf
from Config import *
from Model import *
from RoadData import *
import cv2, json, glob

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(NumpyEncoder, self).default(obj)

config = Config()

if __name__ == '__main__':
	city_name = sys.argv[1]

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

	saver = tf.train.Saver(max_to_keep = 1)
	files = glob.glob('./Model/Model%s/Model-*.ckpt.meta' % city_name)
	files = [(int(file.replace('./Model/Model%s/Model-' % city_name, '').replace('.ckpt.meta', '')), file) for file in files]
	model_path = './Model/Model%s/Model-%d.ckpt' % (city_name, files[-1][0])

	path = './test_res/'
	os.popen('mkdir %s' % path.replace('./', ''))
	# Launch graph
	with tf.Session() as sess:
		with open('Eval.out', 'w') as f:
			# Restore weights
			saver.restore(sess, model_path)
			for i in range(10):
				time_res = [i]
				img, _, _, _, _, terminal_gt, _, _ = getDataBatch(1)

				t = time.time()
				pred_boundary, pred_vertices, feature = sess.run(pred_mask_res, feed_dict = {aa: img})
				time_res.append(time.time() - t)

				cv2.imwrite(path + '%d-0.png' % i, img[0])
				cv2.imwrite(path + '%d-2.png' % i, pred_boundary[0] * 255)
				cv2.imwrite(path + '%d-3.png' % i, pred_vertices[0] * 255)

				terminal = getAllTerminal(pred_vertices[0])

				t = time.time()
				res = []
				for j in range(terminal.shape[0]):
					pred_v_out = sess.run(pred_path_res, feed_dict = {ff: feature, tt: terminal_gt})
					res.append(pred_v_out[0, 0])
				if terminal.shape[0] == 0:
					time_res.append(0)
				else:
					time_res.append((time.time() - t) / terminal.shape[0])

				newImg = recoverMultiPath(img[0], np.array(res))
				cv2.imwrite(path + '%d-1.png' % i, newImg)

				f.write('%d,%.3lf,%.3lf\n' % tuple(time_res))
				f.flush()







			# Test
			if i % 1 == choose_test:
				img, _, _, _, _, _, _, _, _ = getDataBatch(1, 'val')
				# if i < 45 or i > 45:
				# 	continue
				print(i)
				feature, pred_boundary, pred_vertices = sess.run(pred_mask_res, feed_dict = {aa: img})

				path = 'test_res%s/' % city_name
				savePNG(img[0], np.zeros(config.AREA_SIZE), path + '%d-0.png' % i)
				savePNG(img[0], pred_boundary[0, ..., 0] * 255, path + '%d-1.png' % i)
				savePNG(img[0], pred_vertices[0, ..., 0] * 255, path + '%d-2.png' % i)

				map_b, map_v, all_terminal, indices = getAllTerminal(pred_boundary[0], pred_vertices[0])
				feature = np.concatenate([feature, map_b[np.newaxis, ..., np.newaxis], map_v[np.newaxis, ..., np.newaxis]], axis = -1)

				savePNG(img[0], map_b, path + '%d-3.png' % i)
				savePNG(img[0], map_v, path + '%d-4.png' % i)

				multi_roads = []
				prob_res_li = []
				for terminal_1, terminal_2 in all_terminal:
					pred_v_out_1, prob_res_1, rnn_prob_1 = sess.run(pred_path_res, feed_dict = {ff: feature, tt: terminal_1})
					pred_v_out_2, prob_res_2, rnn_prob_2 = sess.run(pred_path_res, feed_dict = {ff: feature, tt: terminal_2})
					if rnn_prob_1[0] >= rnn_prob_2[0]:
						multi_roads.append(pred_v_out_1[0])
						prob_res_li.append(prob_res_1[0])
					else:
						multi_roads.append(pred_v_out_2[0])
						prob_res_li.append(prob_res_2[0])

				paths, pathImgs = recoverMultiPath(img[0].shape[0: 2], multi_roads)
				paths[paths > 1e-3] = 1.0
				savePNG(img[0], paths, path + '%d-5.png' % i)
				os.makedirs('./test_res%s/%d' % (city_name, i))
				for j, pathImg in enumerate(pathImgs):
					savePNG(img[0], pathImg, path + '%d/%d-%d.png' % ((i,) + indices[j]))
					np.save(path + '%d/%d-%d.npy' % ((i,) + indices[j]), prob_res_li[j])







def savePNG(mat1, mat2, filename):
	if mat2.shape[0] < mat1.shape[0]:
		mat2 = cv2.resize(mat2, (0, 0), fx = 8, fy = 8, interpolation = cv2.INTER_NEAREST)
	if mat2.max() > 0:
		mat2 = mat2 / mat2.max()
	m1 = Image.fromarray(mat1, mode = 'RGB')
	m1.putalpha(255)
	m2 = Image.fromarray(np.array(cmap(mat2) * 255.0, np.uint8)).convert(mode = 'RGB')
	m2.putalpha(255)
	m2 = np.array(m2)
	m2[..., 3] = np.array(mat2 * 255.0, np.uint8)
	m2 = Image.fromarray(m2)
	Image.alpha_composite(m1, m2).save(filename)
	return

