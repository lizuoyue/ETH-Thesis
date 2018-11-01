import os, sys
import numpy as np
import tensorflow as tf
from Config import *
from Model import *
from DataGenerator import *
import cv2, json, glob
import matplotlib.pyplot as plt
from PIL import Image

config = Config()
cmap = plt.get_cmap('viridis')

class NumpyEncoder(json.JSONEncoder):
	""" Special json encoder for numpy types """
	def default(self, obj):
		if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
			np.int16, np.int32, np.int64, np.uint8,
			np.uint16, np.uint32, np.uint64)):
			return int(obj)
		elif isinstance(obj, (np.float_, np.float16, np.float32, 
			np.float64)):
			return float(obj)
		elif isinstance(obj,(np.ndarray,)):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

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

if __name__ == '__main__':
	argv = {k: v for k, v in zip(sys.argv[1::2], sys.argv[2::2])}
	city_name = argv['--city']
	img_bias = np.array(config.PATH[city_name]['bias'])
	backbone = argv['--net']
	mode = argv['--mode']
	vis = argv['--vis'] != '0'
	assert(mode in ['val', 'test'])
	print(city_name, backbone, mode, vis)

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
	train = optimizer.minimize(train_res[0] + train_res[1] + train_res[2] + train_res[3])

	saver = tf.train.Saver(max_to_keep = 1)
	model_path = './Model_%s_%s/' % (backbone, city_name)
	files = glob.glob(model_path + '*.ckpt.meta')
	files = [(int(file.replace(model_path, '').replace('.ckpt.meta', '')), file) for file in files]
	files.sort()
	_, model_to_load = files[-1]

	test_path = './Test_Result_%s_%s' % (backbone, city_name)
	if vis:
		if not os.path.exists(test_path):
			os.popen('mkdir %s' % test_path.replace('./', ''))

	eval_files = glob.glob(config.PATH[city_name]['img-%s' % mode] + '/*')
	eval_files.sort()

	# Launch graph
	with tf.Session() as sess:
		with open('Eval_%s_%s_%s.out' % (city_name, backbone, mode), 'w') as f:
			# Restore weights
			saver.restore(sess, model_to_load[:-5])
			for img_seq, img_file in enumerate(eval_files):

				img_id = int(img_file.split('/')[-1].split('.')[0])
				img = np.array(Image.open(img_file).resize(config.AREA_SIZE))[..., 0: 3]
				time_res = [img_seq, img_id]

				t = time.time()
				feature, pred_boundary, pred_vertices = sess.run(pred_mask_res, feed_dict = {aa: img - img_bias})
				time_res.append(time.time() - t)

				if vis:
					savePNG(img, np.zeros(config.AREA_SIZE), test_path + '/%d-0.png' % img_id)
					savePNG(img, pred_boundary[0, ..., 0] * 255, test_path + '/%d-1.png' % img_id)
					savePNG(img, pred_vertices[0, ..., 0] * 255, test_path + '/%d-2.png' % img_id)

				print(pred_boundary[0].shape, pred_vertices[0].shape)
				map_b, map_v, all_terminal, indices = getAllTerminal(pred_boundary[0], pred_vertices[0])
				feature = np.concatenate([feature, map_b[np.newaxis, ..., np.newaxis], map_v[np.newaxis, ..., np.newaxis]], axis = -1)

				if vis:
					savePNG(img, map_b, test_path + '/%d-3.png' % img_id)
					savePNG(img, map_v, test_path + '/%d-4.png' % img_id)

				t = time.time()
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
				if terminal.shape[0] == 0:
					time_res.append(0)
				else:
					time_res.append((time.time() - t) / terminal.shape[0])

				paths, pathImgs = recoverMultiPath(img[0].shape[0: 2], multi_roads)
				paths[paths > 1e-3] = 1.0

				if vis:
					savePNG(img, paths, test_path + '/%d-5.png' % img_id)
					os.makedirs(test_path + '/%d' % img_id)
					for i, pathImg in enumerate(pathImgs):
						savePNG(img, pathImg, test_path + '/%d/%d-%d.png' % ((img_id,) + indices[i]))
						np.save(test_path + '/%d/%d-%d.npy' % ((img_id,) + indices[i]), prob_res_li[i])

				f.write('%d, %d, %.3lf, %.3lf\n' % tuple(time_res))
				f.flush()

