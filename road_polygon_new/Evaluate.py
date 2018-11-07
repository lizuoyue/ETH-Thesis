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

	result = []
	total_time = 0
	test_file_path = config.PATH[city_name]['img-%s' % mode]
	test_info = json.load(open(config.PATH[city_name]['ann-%s' % mode]))

	# Launch graph
	with tf.Session() as sess:
		with open('Eval_%s_%s_%s.out' % (city_name, backbone, mode), 'w') as f:
			# Restore weights
			saver.restore(sess, model_to_load[:-5])
			for img_seq, img_info in enumerate(test_info['images']):

				if not img_info['tile_file'].startswith('chicago'):
					continue

				img_file = test_file_path + '/' + img_info['file_name']
				img_id = img_info['id']
				img = np.array(Image.open(img_file).resize(config.AREA_SIZE))[..., 0: 3]
				img_bias = img.mean(axis = (0, 1))
				time_res = [img_seq, img_id]

				t = time.time()
				feature, pred_boundary, pred_vertices = sess.run(pred_mask_res, feed_dict = {aa: img - img_bias})

				if vis:
					savePNG(img, np.zeros(config.AREA_SIZE), test_path + '/%d-0.png' % img_id)
					savePNG(img, pred_boundary[0, ..., 0] * 255, test_path + '/%d-1.png' % img_id)
					savePNG(img, pred_vertices[0, ..., 0] * 255, test_path + '/%d-2.png' % img_id)

				map_b, map_v, pairs, peaks_with_score, v_val2idx, score_table = getVerticesPairs(pred_boundary[0], pred_vertices[0])
				feature = np.concatenate([feature, map_b[np.newaxis, ..., np.newaxis], map_v[np.newaxis, ..., np.newaxis]], axis = -1)

				if vis:
					savePNG(img, map_b, test_path + '/%d-3.png' % img_id)
					savePNG(img, map_v, test_path + '/%d-4.png' % img_id)

				edges = set()
				multi_roads = []
				prob_res_li = []
				rnn_probs = []
				for pair in pairs:
					pred_v_out, prob_res, rnn_prob = sess.run(pred_path_res, feed_dict = {ff: feature, ii: pair})
					multi_roads.append(pred_v_out[0])
					temp = np.reshape(pred_v_out[0, 0: 2, ..., 0], [2, 784])
					temp = np.concatenate([temp, np.zeros((2, 1))], axis = -1)
					prob_res_li.append(np.concatenate([temp, prob_res[0, 1:]], axis = 0))
					rnn_probs.append(rnn_prob[0])

					edges.update(recoverEdges(pred_v_out[0], v_val2idx))
				edges = [item + (score_table[item], ) for item in list(edges)]

				if vis:
					paths, pathImgs, smallImgs = recoverMultiPath(img.shape[0: 2], multi_roads)
					paths[paths > 1e-3] = 1.0
					savePNG(img, paths, test_path + '/%d-5.png' % img_id)
					if not os.path.exists(test_path + '/%d' % img_id):
						os.makedirs(test_path + '/%d' % img_id)
					for i, pathImg in enumerate(pathImgs):
						score = (pred_boundary[0][smallImgs[i] > 0.5] < 0.3).mean()
						savePNG(img, pathImg, test_path + '/%d/%d-%.6lf.png' % (img_id, i, score)) # rnn_probs[i]
						np.save(test_path + '/%d/%d.npy' % (img_id, i), prob_res_li[i])

				time_res.append(time.time() - t)
				total_time += time_res[-1]
				print(time_res)
				f.write('%d, %d, %.3lf\n' % tuple(time_res))
				f.flush()

				result.append({
					'image_id': img_id,
					'vertices': peaks_with_score,
					'edges': edges
				})
				print(result[0])
				break

				if img_seq % 100 == 0:
					with open('predictions_%s_%s_%s.json' % (city_name, backbone, mode), 'w') as fp:
						fp.write(json.dumps(result, cls = NumpyEncoder))
						fp.close()

			print(total_time, 's')
			with open('predictions_%s_%s_%s.json' % (city_name, backbone, mode), 'w') as fp:
				fp.write(json.dumps(result, cls = NumpyEncoder))
				fp.close()



