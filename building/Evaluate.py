import numpy as np
import os, sys, time, json, glob
import tensorflow as tf
from Config import *
from HybridModel import *
from DataGenerator import *
from UtilityBoxAnchor import *

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
	assert(len(sys.argv) == 2)
	if sys.argv[1] == 'test':
		mode = 'test'
	if sys.argv[1] == 'valid':
		mode = 'test-val'
	# Create data generator
	obj = DataGenerator(
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
		mode = mode
	)

	img_bias = np.array([77.91342018,  89.78918901, 101.50963053])

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
	graph.mode = 'test'
	pred_rpn_res  = graph.predict_rpn(aa, config.AREA_TEST_BATCH)
	pred_poly_res = graph.predict_polygon(pp, vgg)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	saver = tf.train.Saver(max_to_keep = 1)
	files = glob.glob('./Model/Model-*.ckpt.meta')
	files = [(int(file.replace('./Model/Model-', '').replace('.ckpt.meta', '')), file) for file in files]
	model_path = './Model/Model-%d.ckpt' % files[-1][0]

	# Launch graph
	with tf.Session() as sess:
		with open('Eval.out', 'w') as f:
			# Restore weights
			saver.restore(sess, model_path)
			i = 0
			while obj.TEST_FLAG:
				time_res = [i]
				img = obj.getAreasBatch(config.AREA_TEST_BATCH, mode = 'test')
				feed_dict = {aa: img - img_bias}

				t = time.time()
				pred_score, pred_box, vgg16_res = sess.run(pred_rpn_res, feed_dict = feed_dict)
				time_res.append(time.time() - t)

				vgg16_res = list(vgg16_res)
				crop_info, patch_info, box_info = obj.getPatchesFromAreas(pred_score, pred_box)
				feed_dict = {k: v for k, v in zip(vgg, vgg16_res)}
				feed_dict[pp] = crop_info

				t = time.time()
				_, _, pred_v_out = sess.run(pred_poly_res, feed_dict = feed_dict)
				if pred_v_out.shape[1] > 0:
					time_res.append((time.time() - t) / pred_v_out.shape[1])
				else:
					time_res.append(0)
				time_res.append((time.time() - t))

				path = './test_res'
				os.popen('mkdir %s' % path.replace('./', ''))
				obj.recoverBoxPolygon(patch_info, box_info, pred_v_out, mode = 'test', visualize = True, path = path, batch_idx = i)

				f.write('%d,%.3lf,%.3lf,%.3lf\n' % tuple(time_res))
				f.flush()

				print(i)
				i += 1

			with open('predictions.json', 'w') as fp:
				fp.write(json.dumps(obj.TEST_RESULT, cls = NumpyEncoder))
				fp.close()

