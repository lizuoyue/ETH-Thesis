import numpy as np
import os, sys
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf
from Config import *
from HybridModel import *
from DataGenerator import *
from UtilityBoxAnchor import *
from PIL import Image, ImageDraw

config = Config()

def overlay(img, mask, shape, color = (255, 0, 0)):
	org = Image.fromarray(np.array(img * 255.0, dtype = np.uint8)).convert('RGBA')
	alpha = np.array(mask * 128.0, dtype = np.uint8)
	alpha = np.concatenate(
		(
			np.ones((shape[0], shape[1], 1)) * color[0],
			np.ones((shape[0], shape[1], 1)) * color[1],
			np.ones((shape[0], shape[1], 1)) * color[2],
			np.reshape(alpha, (shape[0], shape[1], 1))
		),
		axis = 2
	)
	alpha = Image.fromarray(np.array(alpha, dtype = np.uint8), mode = 'RGBA')
	alpha = alpha.resize((224, 224), resample = Image.BICUBIC)
	merge = Image.alpha_composite(org, alpha)
	return merge

def overlayMultiMask(img, mask, shape):
	merge = Image.fromarray(np.array(img * 255.0, dtype = np.uint8)).convert('RGBA')
	merge = np.array(overlay(img, mask[0], shape)) / 255.0
	for i in range(1, mask.shape[0]):
		color = (255 * (i == 1), 128 * (i == 1) + (1 - i % 2) * 255, i % 2 * 255 - 255 * (i == 1))
		merge = np.array(overlay(merge, mask[i], shape, color)) / 255.0
	return Image.fromarray(np.array(merge * 255.0, dtype = np.uint8)).convert('RGBA')

def visualize(path, img, boundary, vertices, v_in, b_pred, v_pred, v_out_pred, end_pred, seq_len, v_out_res, patch_info):
	# Clear last files
	for item in glob.glob(path + '/*'):
		os.remove(item)

	# Reshape
	b_pred = b_pred[..., 0]
	v_pred = v_pred[..., 0]
	end_pred = end_pred[..., 0]
	shape = ((v_out_res[1], v_out_res[0]))
	blank = np.zeros(shape)

	# Polygon
	polygon = [None for i in range(img.shape[0])]
	for i in range(v_out_pred.shape[0]):
		v = v_in[i, 0]
		r, c = np.unravel_index(v.argmax(), v.shape)
		polygon[i] = [(c, r)]
		for j in range(seq_len[i] - 1):
			if end_pred[i, j] <= v.max():
				v = v_out_pred[i, j]
				r, c = np.unravel_index(v.argmax(), v.shape)
				polygon[i].append((c, r))

	# 
	for i in range(img.shape[0]):
		vv = np.concatenate((v_in[i, 0: 1], v_out_pred[i, 0: seq_len[i] - 1]), axis = 0)
		overlay(img[i], blank      , shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-0-img.png' % i)
		overlay(img[i], boundary[i], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-1-bound.png' % i)
		overlay(img[i], b_pred  [i], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-1-bound-pred.png' % i)
		overlay(img[i], vertices[i], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-2-vertices.png' % i)
		overlay(img[i], v_pred  [i], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-2-vertices-pred.png' % i)
		overlayMultiMask(img[i], vv, shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-3-vertices-merge.png' % i)
		# for j in range(seq_len[i]):
		# 	overlay(img[i], vv[j], shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-3-vtx-%s.png' % (i, str(j).zfill(2)))

		link = Image.new('P', shape, color = 0)
		draw = ImageDraw.Draw(link)
		if len(polygon[i]) == 1:
			polygon[i].append(polygon[i][0])
		draw.polygon(polygon[i], fill = 0, outline = 255)
		link = np.array(link) / 255.0
		overlay(img[i], link, shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-4-vertices-link.png' % i)

		f = open(path + '/%d-5-end-prob.txt' % i, 'w')
		for j in range(seq_len[i]):
			f.write('%.6lf\n' % end_pred[i, j])
		f.close()

	#
	return

def visualize_pred(path, img, b_pred, v_pred, v_out_pred, v_out_res, patch_info):
	if not os.path.exists(path):
		os.makedirs(path)

	# Reshape
	batch_size = img.shape[0]
	b_pred = b_pred[..., 0]
	v_pred = v_pred[..., 0]
	shape = ((v_out_res[1], v_out_res[0]))
	blank = np.zeros(shape)

	# Sequence length and polygon
	polygon = [[] for i in range(batch_size)]
	for i in range(v_out_pred.shape[0]):
		for j in range(v_out_pred.shape[1]):
			v = v_out_pred[i, j]
			if v.sum() >= 0.5:
				r, c = np.unravel_index(v.argmax(), v.shape)
				polygon[i].append((c, r))
			else:
				break
	seq_len = [len(polygon[i]) for i in range(batch_size)]

	# 
	for i in range(batch_size):
		vv = v_out_pred[i, 0: seq_len[i]]
		overlay(img[i], blank      , shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-0-img.png' % i)
		overlay(img[i], b_pred[i]  , shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-1-bound-pred.png' % i)
		overlay(img[i], v_pred[i]  , shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-2-vertices-pred.png' % i)
		overlayMultiMask(img[i], vv, shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-3-vertices-merge.png' % i)
		# for j in range(seq_len[i]):
		# 	overlay(img[i], vv[j], shape).save(path + '/%d-3-vtx-%s.png' % (i, str(j).zfill(2)))
		link = Image.new('P', shape, color = 0)
		draw = ImageDraw.Draw(link)
		if len(polygon[i]) == 1:
			polygon[i].append(polygon[i][0])
		draw.polygon(polygon[i], fill = 0, outline = 255)
		link = np.array(link) / 255.0
		overlay(img[i], link, shape).resize(size = tuple(patch_info[i][0: 2]), resample = Image.BICUBIC).rotate(-patch_info[i][2]).save(path + '/%d-4-vertices-link.png' % i)

	# 
	return

def recoverPolygon(vv):
	batch_size = img.shape[0]
	polygon = [[] for i in range(batch_size)]
	for i in range(vv.shape[0]):
		for j in range(vv.shape[1]):
			v = vv[i, j]
			if v.sum() >= 0.5:
				r, c = np.unravel_index(v.argmax(), v.shape)
				polygon[i].append((c, r))
			else:
				break
	return polygon

def scoreIoU(org_info, gt_vv, vv):
	gt_poly = recoverPolygon(gt_vv)
	poly = recoverPolygon(vv)
	assert(len(gt_poly) == len(poly))
	batch_size = len(poly)
	res = []
	for i in range(batch_size):
		gt_mask = Image.new('P', config.V_OUT_RES, color = 0)
		draw = ImageDraw.Draw(gt_mask)
		draw.polygon(gt_poly[i], fill = 0, outline = 255)
		gt_mask = gt_mask.resize(size = tuple(org_info[i][0: 2]), resample = Image.BICUBIC).rotate(-org_info[i][2])
		mask = Image.new('P', config.V_OUT_RES, color = 0)
		draw = ImageDraw.Draw(mask)
		if len(poly[i]) == 1:
			poly[i].append(poly[i][0])
		draw.polygon(poly[i], fill = 0, outline = 255)
		mask = mask.resize(size = tuple(org_info[i][0: 2]), resample = Image.BICUBIC).rotate(-org_info[i][2])
		gt_mask = np.array(np.array(gt_mask) / 255.0, np.bool)
		mask = np.array(np.array(mask) / 255.0, np.bool)
		res.append(np.sum(gt_mask & mask) / np.sum(gt_mask | mask))
	return res

if __name__ == '__main__':
	# Create new folder
	assert(os.path.exists('./Model/'))

	# Create data generator
	obj = DataGenerator(
		building_path = config.PATH_B % 'Chicago',
		area_path = config.PATH_A % 'Chicago', 
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
	)

	# Define graph
	graph = HybridModel(
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

	train_res     = graph.train(aa, cc, dd, pp, ii, bb, vv, oo, ee, ll)
	pred_rpn_res  = graph.predict_rpn(aa)
	pred_poly_res = graph.predict_polygon(pp)

	# for v in tf.global_variables():
	# 	print(v.name)
	# quit()

	optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	train = optimizer.minimize(train_res[0] + train_res[1] + train_res[2] + train_res[3])
	saver = tf.train.Saver(max_to_keep = 3)
	init = tf.global_variables_initializer()

	# Launch graph
	with tf.Session() as sess:
		# Restore weights
		assert(len(sys.argv) > 1)
		saver.restore(sess, './Model/Model-%s.ckpt' % sys.argv[1])

		for i in range(50):
			print('Round %d' % i)
			img, anchor_cls, anchor_box = obj.getAreasBatch(config.AREA_PRED_BATCH, mode = 'test', idx = i)
			feed_dict = {aa: img}
			pred_box = sess.run(pred_rpn_res, feed_dict = feed_dict)
			org_img, patch, org_info = obj.getPatchesFromAreas(pred_box)
			feed_dict = {pp: patch}
			pred_boundary, pred_vertices, pred_v_out = sess.run(pred_poly_res, feed_dict = feed_dict)
			path = './EvalAreaResult'
			if not os.path.exists(path):
				os.makedirs(path)
			obj.recover(path, org_img, pred_box, i)
			obj.recoverGlobal(path, org_img, org_info, pred_v_out, i)

			img, boundary, vertices, vertex_input, vertex_output, end, seq_len, org_info = obj.getBuildingsBatch(config.BUILDING_PRED_BATCH, mode = 'test', idx = i)
			feed_dict = {pp: img}
			pred_boundary, pred_vertices, pred_v_out = sess.run(pred_poly_res, feed_dict = feed_dict)
			path = './EvalBuildingResult%d' % i
			if not os.path.exists(path):
				os.makedirs(path)
			visualize_pred(path, img, pred_boundary, pred_vertices, pred_v_out[0], config.V_OUT_RES, org_info)
			print(scoreIoU(org_info, vertex_input, pred_v_out[0]))

