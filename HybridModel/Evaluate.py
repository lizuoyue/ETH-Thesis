import numpy as np
import os, sys, time
if os.path.exists('../../Python-Lib/'):
	sys.path.insert(1, '../../Python-Lib')
import tensorflow as tf
from Config import *
from HybridModel import *
from DataGenerator import *
from UtilityBoxAnchor import *
from PIL import Image, ImageDraw

config = Config()
global city_name

def overlay(img, mask, shape, color = (255, 0, 0)):
	org = Image.fromarray(np.array(img, np.uint8)).convert('RGBA')
	alpha = np.array(mask * 128.0, np.uint8)
	alpha = np.concatenate(
		(
			np.ones((shape[0], shape[1], 1)) * color[0],
			np.ones((shape[0], shape[1], 1)) * color[1],
			np.ones((shape[0], shape[1], 1)) * color[2],
			np.reshape(alpha, (shape[0], shape[1], 1))
		),
		axis = 2
	)
	alpha = Image.fromarray(np.array(alpha, np.uint8), mode = 'RGBA')
	alpha = alpha.resize((224, 224), resample = Image.BICUBIC)
	merge = Image.alpha_composite(org, alpha)
	return merge

def overlayMultiMask(img, mask, shape):
	merge = Image.fromarray(np.array(img, np.uint8)).convert('RGBA')
	merge = np.array(overlay(img, mask[0], shape))
	for i in range(1, mask.shape[0]):
		color = (255 * (i == 1), 128 * (i == 1) + (1 - i % 2) * 255, i % 2 * 255 - 255 * (i == 1))
		merge = np.array(overlay(merge, mask[i], shape, color), np.uint8)
	return Image.fromarray(merge).convert('RGBA')

def visualize(path, img, boundary, vertices, v_in, b_pred, v_pred, v_out_pred, end_pred, seq_len, v_out_res, patch_info):
	# Clear last files
	for item in glob.glob(path + '/*'):
		os.remove(item)

	# Reshape
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

def visualize_pred(path, img, b_pred, v_pred, v_out_pred, v_out_res, patch_info, idx = None):
	if not os.path.exists(path):
		os.makedirs(path)

	# Reshape
	batch_size = img.shape[0]
	img = img + config.COLOR_MEAN['Buildings'][city_name]

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
		w, h = tuple(patch_info[i, 0: 2])
		w_rate = w / v_out_res[0]
		h_rate = h / v_out_res[1]
		mask = Image.new('RGB', (w, h), color = (0, 0, 0))
		draw = ImageDraw.Draw(mask)
		if len(polygon[i]) == 1:
			polygon[i].append(polygon[i][0])
		recover = [(int(x * w_rate), int(y * h_rate)) for x, y in polygon[i]]
		c = np.random.randint(0, len(config.TABLEAU20))
		draw.polygon(recover, fill = config.TABLEAU20[c], outline = config.TABLEAU20_DEEP[c])
		res = DataGenerator.overlay(Image.fromarray(np.array(img[i], np.uint8)).resize(size = (w, h), resample = Image.BICUBIC), mask)
		res.rotate(-patch_info[i, 2]).save(path + '/%d.png' % (idx + i))
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

def score(org_info, gt_vv, vv):
	gt_poly = recoverPolygon(gt_vv)
	poly = recoverPolygon(vv)
	assert(len(gt_poly) == len(poly))
	batch_size = len(poly)
	acc, pre, rec, f1s, iou = [], [], [], [], []
	for i in range(batch_size):
		gt_mask = Image.new('P', config.V_OUT_RES, color = 0)
		draw = ImageDraw.Draw(gt_mask)
		draw.polygon(gt_poly[i], fill = 255, outline = 255)
		# gt_mask = gt_mask.resize(size = tuple(org_info[i, 0: 2]), resample = Image.BICUBIC).rotate(-org_info[i, 2])
		gt_mask = gt_mask.rotate(-org_info[i, 2])
		mask = Image.new('P', config.V_OUT_RES, color = 0)
		draw = ImageDraw.Draw(mask)
		if len(poly[i]) == 1:
			poly[i].append(poly[i][0])
		draw.polygon(poly[i], fill = 255, outline = 255)
		# mask = mask.resize(size = tuple(org_info[i, 0: 2]), resample = Image.BICUBIC).rotate(-org_info[i, 2])
		mask = mask.rotate(-org_info[i, 2])
		gt_mask = np.array(np.array(gt_mask) / 255.0, np.bool)
		mask = np.array(np.array(mask) / 255.0, np.bool)

		w_0, w_1 = 0.5 / np.sum(gt_mask == False), 0.5 / np.sum(gt_mask == True)
		tp = np.sum((gt_mask == True) & (mask == True))
		fp = np.sum((gt_mask == False) & (mask == True))
		fn = np.sum((gt_mask == True) & (mask == False))
		tn = np.sum((gt_mask == False) & (mask == False))
		acc.append(w_0 * tn + w_1 * tp)
		pre.append(tp / (tp + fp))
		rec.append(tp / (tp + fn))
		f1s.append(2 / (1/pre[-1] + 1/rec[-1]))
		iou.append(np.sum(gt_mask & mask) / np.sum(gt_mask | mask))
	return [[acc[i], pre[i], rec[i], f1s[i], iou[i]] for i in range(batch_size)]

if __name__ == '__main__':
	# Create new folder
	city_name = sys.argv[1]
	assert(os.path.exists('./Model%s/' % (city_name + sys.argv[2])))
	assert(len(sys.argv) == 3)

	# Create data generator
	obj = DataGenerator(
		city_name = city_name,
		img_size = config.PATCH_SIZE,
		v_out_res = config.V_OUT_RES,
		max_num_vertices = config.MAX_NUM_VERTICES,
	)

	# Define graph
	graph = HybridModel(
		max_num_vertices = config.MAX_NUM_VERTICES,
		lstm_out_channel = config.LSTM_OUT_CHANNEL, 
		v_out_res = config.V_OUT_RES,
		two_step = (sys.argv[2] == '2'),
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

	# optimizer = tf.train.AdamOptimizer(learning_rate = config.LEARNING_RATE)
	# train = optimizer.minimize(train_res[0] + train_res[1] + train_res[2] + train_res[3])
	saver = tf.train.Saver(max_to_keep = 3)
	# init = tf.global_variables_initializer()
	city_name = city_name + sys.argv[2]

	# Launch graph
	with tf.Session() as sess:
		# Restore weights
		assert(len(sys.argv) > 1)
		saver.restore(sess, './Model%s/Model%s-24800.ckpt' % (city_name, city_name))

		f = open('Eval%s.csv' % city_name, 'w')
		f.write('id,acc,pre,rec,f1s,iou\n')

		ff = open('Time%s.csv' % city_name, 'w')
		ff.write('round,area,num,building,fixbuilding\n')

		for i in range(0, 300):
			time_res = [i]
			img, anchor_cls, anchor_box = obj.getAreasBatch(config.AREA_PRED_BATCH, mode = 'test', idx = i)
			feed_dict = {aa: img}

			t = time.time()
			pred_box = sess.run(pred_rpn_res, feed_dict = feed_dict)
			time_res.append(len(pred_box))
			time_res.append(time.time() - t)

			org_img, patch, org_info = obj.getPatchesFromAreas(pred_box)
			feed_dict = {pp: patch}

			t = time.time()
			pred_boundary, pred_vertices, pred_v_out = sess.run(pred_poly_res, feed_dict = feed_dict)
			time_res.append(time.time() - t)

			path = './EvalAreaResult%s' % city_name
			if not os.path.exists(path):
				os.makedirs(path)
			obj.recoverGlobal(path, org_img, org_info, pred_v_out, pred_box, i)

			img, boundary, vertices, vertex_input, vertex_output, end, seq_len, org_info = obj.getBuildingsBatch(config.BUILDING_PRED_BATCH, mode = 'test', idx = i)
			feed_dict = {pp: img}

			t = time.time()
			pred_boundary, pred_vertices, pred_v_out = sess.run(pred_poly_res, feed_dict = feed_dict)
			time_res.append(time.time() - t)

			path = './EvalBuildingResult%s' % city_name
			if not os.path.exists(path):
				os.makedirs(path)
			visualize_pred(path, img, pred_boundary, pred_vertices, pred_v_out[0], config.V_OUT_RES, org_info, i * config.BUILDING_PRED_BATCH)
			path = './EvalBuildingResultGT%s' % city_name
			if not os.path.exists(path):
				os.makedirs(path)
			visualize_pred(path, img, boundary, vertices, vertex_input, config.V_OUT_RES, org_info, i * config.BUILDING_PRED_BATCH)
			res = score(org_info, vertex_input, pred_v_out[0])
			for j, line in enumerate(res):
				f.write('%d,%.6lf,%.6lf,%.6lf,%.6lf,%.6lf\n' % tuple([i * config.BUILDING_PRED_BATCH + j] + line))
			f.flush()

			ff.write('%d,%.3lf,%d,%.3lf,%.3lf' % tuple(time_res))
			ff.flush()

		f.close()
		f.close()

