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
	pred_mask_res  = graph.predict_mask(aa)
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
			for i in range(100):
				time_res = [i]
				img, _, _, _, _, _, _, _ = getDataBatch(1)

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
					pred_v_out = sess.run(pred_path_res, feed_dict = {ff: feature, tt: terminal[j]})
					res.append(pred_v_out[0, 0])
				time_res.append((time.time() - t) / terminal.shape[0])

				newImg = recoverMultiPath(img[0], np.array(res))
				cv2.imwrite(path + '%d-1.png' % i, newImg)

				f.write('%d,%.3lf,%.3lf\n' % tuple(time_res))
				f.flush()







