import numpy as np
import os, sys, time, json, glob
import tensorflow as tf
from Config import *
from Model import *
from DataGenerator import *
from UtilityBoxAnchor import *

config = Config()

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

if __name__ == '__main__':
	np.random.seed(8888)

	argv = {k: v for k, v in zip(sys.argv[1::2], sys.argv[2::2])}
	city_name = argv['--city']
	img_bias = np.array(config.PATH[city_name]['bias'])
	backbone = argv['--net']
	mode = argv['--mode']
	vis = argv['--vis'] != '0'
	print(city_name, backbone, mode, vis)

	# Create data generator
	obj = DataGenerator(
		city_name = city_name,
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
		mode = mode
	)

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
	pred_rpn_res  = graph.predict_fpn(aa, config.AREA_TEST_BATCH)
	pred_poly_res = graph.predict_polygon(pp, nn)

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
	if not os.path.exists(test_path):
		os.popen('mkdir %s' % test_path.replace('./', ''))

	# Launch graph
	with tf.Session() as sess:
		with open('Eval.out', 'w') as f:
			# Restore weights
			saver.restore(sess, model_to_load.replace('.meta', ''))
			i = 0
			while obj.TEST_FLAG:
				time_res = [i]
				img = obj.getAreasBatch(config.AREA_TEST_BATCH, mode = 'test')
				feed_dict = {aa: img - img_bias}

				t = time.time()
				pred_score, pred_box, backbone_result = sess.run(pred_rpn_res, feed_dict = feed_dict)
				time_res.append(time.time() - t)

				backbone_result = list(backbone_result)
				crop_info, patch_info, box_info = obj.getPatchesFromAreas(pred_score, pred_box)
				feed_dict = {k: v for k, v in zip(nn, backbone_result)}
				feed_dict[pp] = crop_info

				t = time.time()
				_, _, pred_v_out = sess.run(pred_poly_res, feed_dict = feed_dict)
				if pred_v_out.shape[1] > 0:
					time_res.append((time.time() - t) / pred_v_out.shape[1])
				else:
					time_res.append(0)
				time_res.append((time.time() - t))
				
				obj.recoverBoxPolygon(patch_info, box_info, pred_v_out, mode = 'test', visualize = vis, path = test_path, batch_idx = i)

				f.write('%d, %.3lf, %.3lf, %.3lf\n' % tuple(time_res))
				f.flush()

				print(i)
				i += 1

			with open('predictions.json', 'w') as fp:
				fp.write(json.dumps(obj.TEST_RESULT, cls = NumpyEncoder))
				fp.close()

